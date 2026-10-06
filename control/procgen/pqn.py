"""PQN baseline for fixed-level continual Procgen curricula.

The optimization follows CleanRL's PQN implementation: collect a short batch
from synchronous actors, construct frozen backward Q(lambda) targets, then
shuffle and reuse that batch for multiple epochs.  There is no replay buffer or
target network.
"""

from argparse import ArgumentParser
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from config import add_common_arguments, epsilon_at, resolve_and_validate
from diagnostics import ContinualDiagnostics
from model import make_pqn_network, parameter_count
from procgen_env import make_training_env
from utils import (
	EpisodeReturnTracker,
	describe_protocol,
	prepare_output_directories,
	seed_everything,
	serializable_config,
)


def channels_first_uint8(observations, device):
	array = np.ascontiguousarray(observations.transpose(0, 3, 1, 2))
	return torch.as_tensor(array, dtype=torch.uint8, device=device)


class RolloutStateSampler:
	"""Expose the latest PQN rollout through the diagnostics replay interface."""

	def __init__(self, seed):
		self.rng = np.random.RandomState(seed)
		self.states = None

	def update(self, states):
		self.states = states.reshape(-1, 3, 64, 64)

	def sample_states(self, count):
		if self.states is None or len(self.states) == 0:
			return None
		indices = self.rng.randint(
			0, len(self.states), size=min(count, len(self.states))
		)
		indices = torch.as_tensor(indices, device=self.states.device)
		return self.states[indices]


class PQNAgent:
	def __init__(self, num_actions, args, device):
		self.args = args
		self.device = device
		self.online = make_pqn_network(num_actions).to(device)
		self.optimizer = optim.RAdam(
			self.online.parameters(),
			lr=args.learning_rate,
			weight_decay=args.weight_decay,
		)

	@torch.no_grad()
	def actions_and_values(
		self, states, epsilon, num_actions, return_q_values=False
	):
		q_values = self.online(states)
		actions = q_values.argmax(dim=1).cpu().numpy()
		random_mask = np.random.random(len(actions)) < epsilon
		actions[random_mask] = np.random.randint(
			0, num_actions, size=random_mask.sum()
		)
		state_values = q_values.max(dim=1).values
		if return_q_values:
			return actions, state_values, q_values
		return actions, state_values

	@torch.no_grad()
	def lambda_returns(self, rewards, done, values, next_states):
		"""Backward Q(lambda) recursion from Algorithm 1 and CleanRL."""
		returns = torch.empty_like(rewards)
		next_values = self.online(next_states).max(dim=1).values
		returns[-1] = rewards[-1] + (
			self.args.gamma * (~done[-1]).float() * next_values
		)
		for step in range(len(rewards) - 2, -1, -1):
			bootstrap = (
				self.args.q_lambda * returns[step + 1]
				+ (1.0 - self.args.q_lambda) * values[step + 1]
			)
			returns[step] = rewards[step] + (
				self.args.gamma * (~done[step]).float() * bootstrap
			)
		return returns

	def train(self, states, actions, targets):
		states = states.reshape(-1, 3, 64, 64)
		actions = actions.reshape(-1)
		targets = targets.reshape(-1)
		batch_size = len(states)
		minibatch_size = min(self.args.pqn_minibatch_size, batch_size)
		indices = np.arange(batch_size)
		losses, predicted_q = [], []
		gradient_updates = 0

		for _ in range(self.args.update_epochs):
			np.random.shuffle(indices)
			for start in range(0, batch_size, minibatch_size):
				minibatch = torch.as_tensor(
					indices[start:start + minibatch_size], device=self.device
				)
				predictions = self.online(states[minibatch]).gather(
					1, actions[minibatch, None]
				).squeeze(1)
				loss = F.mse_loss(predictions, targets[minibatch])
				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(
					self.online.parameters(), self.args.gradient_clip
				)
				self.optimizer.step()
				losses.append(loss.item())
				predicted_q.append(predictions.detach().mean().item())
				gradient_updates += 1
		return (
			float(np.mean(losses)),
			float(np.mean(predicted_q)),
			gradient_updates,
		)


def add_pqn_arguments(parser):
	"""Add the exact optimization arguments shared by all PQN variants."""
	add_common_arguments(parser)
	parser.add_argument(
		"--learning-rate", "--lr1", dest="learning_rate", type=float,
		default=2.5e-4,
	)
	parser.add_argument(
		"--rollout-steps", type=int, default=32,
		help="transitions collected per actor before each PQN update",
	)
	parser.add_argument("--num-minibatches", type=int, default=16)
	parser.add_argument("--update-epochs", type=int, default=2)
	parser.add_argument("--q-lambda", type=float, default=0.65)
	parser.add_argument(
		"--weight-decay", type=float, default=0.0,
		help="optional L2 regularization; empirical Atari PQN uses LayerNorm alone",
	)
	# The paper uses 128 actors and 32 minibatches.  With Procgen's shared
	# 64-actor protocol, 16 minibatches preserves its 128-sample minibatch and
	# two epochs preserve one gradient update per 64 environment transitions.
	parser.set_defaults(final_epsilon=0.01)
	return parser


def resolve_pqn_args(args):
	"""Resolve shared Procgen settings and validate PQN-specific arguments."""
	args = resolve_and_validate(args)

	positive = (
		"learning_rate", "rollout_steps", "num_minibatches", "update_epochs",
	)
	for name in positive:
		if getattr(args, name) <= 0:
			raise ValueError(f"--{name.replace('_', '-')} must be positive")
	if not 0 <= args.q_lambda <= 1:
		raise ValueError("--q-lambda must be in [0, 1]")
	if args.weight_decay < 0:
		raise ValueError("--weight-decay cannot be negative")
	args.rollout_batch_size = args.num_envs * args.rollout_steps
	if args.rollout_batch_size % args.num_minibatches != 0:
		raise ValueError(
			"--num-minibatches must divide num-envs * rollout-steps exactly"
		)
	args.pqn_minibatch_size = (
		args.rollout_batch_size // args.num_minibatches
	)
	args.effective_transitions_per_update = (
		args.rollout_batch_size
		/ (args.num_minibatches * args.update_epochs)
	)
	return args


def parse_args(arguments=None):
	parser = ArgumentParser(description="PQN on continual Procgen")
	add_pqn_arguments(parser)
	return resolve_pqn_args(parser.parse_args(arguments))


def main():
	args = parse_args()
	prepare_output_directories(args)
	seed_everything(args.seed)
	describe_protocol(args, uses_replay=False)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	current_stage = 0
	task_index = 0
	env = make_training_env(args, task_index)
	if env.num_actions != 15:
		raise ValueError(
			f"expected Procgen's 15-action space, got {env.num_actions}"
		)
	agent = PQNAgent(env.num_actions, args, device)
	print(f"PQN behavior parameters: {parameter_count(agent.online):,}")
	print(
		f"PQN rollout={args.rollout_batch_size:,}, "
		f"minibatch={args.pqn_minibatch_size:,}, epochs={args.update_epochs}"
	)

	diagnostics = ContinualDiagnostics(args)
	rollout_sampler = RolloutStateSampler(args.seed + 17)
	return_tracker = EpisodeReturnTracker(args.task_sequence, args.num_envs)
	observations = env.reset()

	algorithm_name = "PQN"
	filename = (
		f"PQN_Procgen_{args.distribution_mode}_{args.num_tasks}tasks"
		f"_{args.cycles}cycles_switch_{args.switch_steps}"
		f"_lambda_{args.q_lambda}_lr_{args.learning_rate}_seed_{args.seed}"
	)
	wandb.init(
		project=args.wandb_project,
		name=filename,
		group=algorithm_name,
		mode=args.wandb_mode,
		config=serializable_config(
			args, algorithm_name,
			{
				"network_parameters": parameter_count(agent.online),
				"uses_replay": False,
				"uses_target_network": False,
			},
		),
	)

	vector_steps = args.t_steps // args.num_envs
	returns_array = np.zeros(vector_steps, dtype=np.float32)
	next_action_gap = args.target_update_frequency
	next_evaluation = args.eval_frequency
	next_log = args.log_frequency
	global_steps = 0
	gradient_updates = 0
	episodes_completed = 0
	last_loss = None
	last_q = None

	try:
		while global_steps < args.t_steps:
			stage = global_steps // args.switch_steps
			if stage != current_stage:
				env.close()
				current_stage = stage
				task_index = stage % args.num_tasks
				env = make_training_env(args, task_index)
				observations = env.reset()
				return_tracker.switch_task()
				diagnostics.start_training_phase()

			# End a rollout at every evaluation or task boundary.  This keeps
			# reference probes at exactly the same lifetime steps as DQN and
			# prevents Q(lambda) targets from crossing known task boundaries.
			stage_remaining = (
				(current_stage + 1) * args.switch_steps - global_steps
			) // args.num_envs
			evaluation_remaining = (
				(next_evaluation - global_steps) // args.num_envs
			)
			total_remaining = (args.t_steps - global_steps) // args.num_envs
			rollout_length = min(
				args.rollout_steps,
				stage_remaining,
				evaluation_remaining,
				total_remaining,
			)
			if rollout_length <= 0:
				raise RuntimeError("invalid zero-length PQN rollout")

			states = torch.empty(
				(rollout_length, args.num_envs, 3, 64, 64),
				dtype=torch.uint8,
				device=device,
			)
			actions = torch.empty(
				(rollout_length, args.num_envs), dtype=torch.long, device=device
			)
			rewards = torch.empty(
				(rollout_length, args.num_envs), dtype=torch.float32, device=device
			)
			done = torch.empty(
				(rollout_length, args.num_envs), dtype=torch.bool, device=device
			)
			values = torch.empty_like(rewards)

			game = args.task_sequence[task_index]
			last_epsilon = args.initial_epsilon
			for rollout_step in range(rollout_length):
				stage_steps = global_steps % args.switch_steps
				last_epsilon = epsilon_at(args, global_steps, stage_steps)
				state_tensor = channels_first_uint8(observations, device)
				selected_actions, state_values = agent.actions_and_values(
					state_tensor, last_epsilon, env.num_actions
				)
				next_observations, raw_rewards, terminal, _ = env.step(
					selected_actions
				)
				train_rewards = raw_rewards
				if args.reward_clip > 0:
					train_rewards = np.clip(
						raw_rewards, -args.reward_clip, args.reward_clip
					)

				states[rollout_step].copy_(state_tensor)
				actions[rollout_step].copy_(torch.as_tensor(
					selected_actions, device=device
				))
				rewards[rollout_step].copy_(torch.as_tensor(
					train_rewards, device=device
				))
				done[rollout_step].copy_(torch.as_tensor(
					terminal, device=device
				))
				values[rollout_step].copy_(state_values)

				observations = next_observations
				completed = return_tracker.update(game, raw_rewards, terminal)
				episodes_completed += len(completed)
				global_steps += args.num_envs
				returns_array[global_steps // args.num_envs - 1] = (
					return_tracker.ema[game]
				)

			with torch.no_grad():
				targets = agent.lambda_returns(
					rewards,
					done,
					values,
					channels_first_uint8(observations, device),
				)
			last_loss, last_q, new_updates = agent.train(
				states, actions, targets
			)
			gradient_updates += new_updates
			rollout_sampler.update(states)

			metrics = {}
			action_gap_due = False
			while global_steps >= next_action_gap:
				action_gap_due = True
				next_action_gap += args.target_update_frequency
			if action_gap_due:
				metrics.update(diagnostics.action_gap_metrics(
					agent.online, rollout_sampler, device, game
				))

			while global_steps >= next_evaluation:
				completed_stages = global_steps // args.switch_steps
				metrics.update(diagnostics.evaluation_metrics(
					agent.online, device, completed_stages
				))
				next_evaluation += args.eval_frequency

			log_due = global_steps >= next_log
			while global_steps >= next_log:
				next_log += args.log_frequency
			if log_due:
				metrics.update({
					"train_return": return_tracker.ema[game],
					"episodic_return_10000": return_tracker.ema[game],
					"epsilon": last_epsilon,
					"task_index": task_index,
					"task_stage": current_stage,
					"cycle_index": current_stage // args.num_tasks,
					"gradient_updates": gradient_updates,
					"pqn/rollout_size": rollout_length * args.num_envs,
					"episodes_completed": episodes_completed,
					"loss/td": last_loss,
					"pqn/q_values": last_q,
				})
			if metrics:
				wandb.log(metrics, step=global_steps)
	finally:
		env.close()
		wandb.finish()

	if args.save_model:
		torch.save(
			agent.online.state_dict(),
			args.output_dir / "models" / f"{filename}_Net.pt",
		)
	if args.save:
		with (args.output_dir / "results" / f"{filename}_returns.pkl").open("wb") as file:
			pickle.dump(returns_array, file)
	if args.plot:
		import matplotlib.pyplot as plt
		plt.plot(np.arange(vector_steps) * args.num_envs, returns_array)
		plt.xlabel("lifetime environment transitions")
		plt.ylabel("training return EMA")
		plt.show()


if __name__ == "__main__":
	main()
