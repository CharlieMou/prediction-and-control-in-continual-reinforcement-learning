"""Parameter-matched permanent-transient DQN for continual Procgen."""

from argparse import ArgumentParser
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from config import (
	DQN_CHANNELS,
	DQN_HIDDEN_SIZE,
	PT_CHANNELS,
	add_common_arguments,
	epsilon_at,
	network_parameter_count,
	pt_hidden_size,
	resolve_and_validate,
)
from diagnostics import ContinualDiagnostics
from model import CombinedQNetwork, make_pt_network, parameter_count
from procgen_env import make_training_env
from replay import PermanentReplayBuffer, ReplayBuffer
from utils import (
	EpisodeReturnTracker,
	describe_protocol,
	observation_tensor,
	prepare_output_directories,
	seed_everything,
	serializable_config,
)


class PTDQNAgent:
	def __init__(self, num_actions, args, device):
		self.args = args
		self.device = device
		self.permanent = make_pt_network(num_actions).to(device)
		self.transient = make_pt_network(num_actions).to(device)
		self.target_transient = make_pt_network(num_actions).to(device)
		self.target_transient.load_state_dict(self.transient.state_dict())
		self.target_transient.requires_grad_(False)
		self.behavior = CombinedQNetwork(self.permanent, self.transient)
		self.permanent_optimizer = optim.SGD(
			self.permanent.parameters(), lr=args.lr1
		)
		self.transient_optimizer = optim.Adam(
			self.transient.parameters(), lr=args.lr2, eps=args.adam_eps
		)

	@torch.no_grad()
	def actions(self, observations, epsilon, num_actions):
		states = observation_tensor(observations, self.device)
		permanent_values = self.permanent(states)
		q_values = permanent_values + self.transient(states)
		actions = q_values.argmax(dim=1).cpu().numpy()
		random_mask = np.random.random(len(actions)) < epsilon
		actions[random_mask] = np.random.randint(
			0, num_actions, size=random_mask.sum()
		)
		indices = torch.arange(len(actions), device=self.device)
		action_tensor = torch.as_tensor(actions, device=self.device)
		old_permanent = permanent_values[indices, action_tensor].cpu().numpy()
		return old_permanent, actions

	def train_transient(self, replay):
		states, actions, next_states, rewards, done = replay.sample()
		with torch.no_grad():
			next_values = (
				self.permanent(next_states) + self.target_transient(next_states)
			).max(dim=1, keepdim=True).values
			permanent_predictions = self.permanent(states).gather(1, actions)
			targets = rewards + (1.0 - done) * self.args.gamma * next_values
		transient_predictions = self.transient(states).gather(1, actions)
		loss = F.smooth_l1_loss(
			permanent_predictions + transient_predictions, targets
		)
		self.transient_optimizer.zero_grad()
		loss.backward()
		gradient_norm = torch.nn.utils.clip_grad_norm_(
			self.transient.parameters(), self.args.gradient_clip
		)
		self.transient_optimizer.step()
		return loss.item(), float(gradient_norm)

	def consolidate(self, permanent_replay):
		loss_sum = 0.0
		updates = 0
		for states, actions, old_permanent in permanent_replay.batches(
			self.args.batch_size, self.device
		):
			with torch.no_grad():
				targets = (
					old_permanent + self.transient(states).gather(1, actions)
				)
			predictions = self.permanent(states).gather(1, actions)
			loss = F.smooth_l1_loss(predictions, targets)
			self.permanent_optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(
				self.permanent.parameters(), self.args.gradient_clip
			)
			self.permanent_optimizer.step()
			loss_sum += loss.item()
			updates += 1

		with torch.no_grad():
			for parameter in self.transient.parameters():
				parameter.mul_(self.args.decay)
		# A target that predates transient decay gives an inconsistent P+T target.
		self.update_target()
		return loss_sum / max(updates, 1), updates

	def update_target(self):
		self.target_transient.load_state_dict(self.transient.state_dict())


def parse_args():
	parser = ArgumentParser(description="PT-DQN on continual Procgen")
	add_common_arguments(parser)
	parser.add_argument(
		"--lr1", type=float, default=1e-5,
		help="PT-specific permanent-network SGD learning rate",
	)
	parser.add_argument(
		"--lr2", type=float, default=2.5e-4,
		help="transient Adam rate, shared with the DQN baseline",
	)
	parser.add_argument(
		"--update", "--permanent-update-frequency", dest="update", type=int,
		default=100_000, help="permanent consolidation interval in transitions",
	)
	parser.add_argument(
		"--decay", type=float, default=0.95,
		help="multiplicative transient decay after consolidation",
	)
	args = resolve_and_validate(parser.parse_args())
	if args.lr1 <= 0 or args.lr2 <= 0:
		raise ValueError("--lr1 and --lr2 must be positive")
	if args.update <= args.batch_size:
		raise ValueError("--update must be greater than --batch-size")
	if not 0 <= args.decay <= 1:
		raise ValueError("--decay must be in [0, 1]")
	return args


def main():
	args = parse_args()
	prepare_output_directories(args)
	seed_everything(args.seed)
	describe_protocol(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	current_stage = 0
	task_index = 0
	env = make_training_env(args, task_index)
	if env.num_actions != 15:
		raise ValueError(f"expected Procgen's 15-action space, got {env.num_actions}")
	agent = PTDQNAgent(env.num_actions, args, device)
	dqn_parameters = network_parameter_count(
		DQN_CHANNELS, DQN_HIDDEN_SIZE, env.num_actions
	)
	component_parameters = parameter_count(agent.permanent)
	pt_parameters = 2 * component_parameters
	relative_budget_error = abs(pt_parameters - dqn_parameters) / dqn_parameters
	if parameter_count(agent.transient) != component_parameters:
		raise RuntimeError("permanent and transient networks must have equal size")
	if relative_budget_error > 1e-3:
		raise RuntimeError("PT-DQN behavior parameter budget is not DQN-matched")
	print(
		f"PT component architecture: channels={PT_CHANNELS}, "
		f"hidden={pt_hidden_size(env.num_actions)}"
	)
	print(
		f"Parameters: DQN={dqn_parameters:,}; P={component_parameters:,}; "
		f"T={component_parameters:,}; PT total={pt_parameters:,}; "
		f"relative error={relative_budget_error:.6%}"
	)

	replay = ReplayBuffer(args.replay_size, args.batch_size, device, args.seed + 17)
	permanent_replay = PermanentReplayBuffer(args.update)
	diagnostics = ContinualDiagnostics(args)
	return_tracker = EpisodeReturnTracker(args.task_sequence, args.num_envs)
	observations = env.reset()

	algorithm_name = "PT-DQN"
	filename = (
		f"PT_DQN_half_Procgen_{args.distribution_mode}_{args.num_tasks}tasks"
		f"_{args.cycles}cycles_switch_{args.switch_steps}_update_{args.update}"
		f"_decay_{args.decay}_lrP_{args.lr1}_lrT_{args.lr2}_seed_{args.seed}"
	)
	wandb.init(
		project=args.wandb_project,
		name=filename,
		group=algorithm_name,
		mode=args.wandb_mode,
		config=serializable_config(args, algorithm_name, {
			"dqn_reference_parameters": dqn_parameters,
			"permanent_parameters": component_parameters,
			"transient_parameters": component_parameters,
			"pt_behavior_parameters": pt_parameters,
			"parameter_budget_relative_error": relative_budget_error,
		}),
	)

	vector_steps = args.t_steps // args.num_envs
	returns_array = np.zeros(vector_steps, dtype=np.float32)
	next_update = args.warmup_steps
	next_target_update = args.target_update_frequency
	next_permanent_update = args.update
	next_evaluation = args.eval_frequency
	next_log = args.log_frequency
	gradient_updates = 0
	permanent_updates = 0
	episodes_completed = 0
	last_transient_loss = None
	last_permanent_loss = None
	last_gradient_norm = None

	try:
		for vector_step in range(vector_steps):
			steps_before = vector_step * args.num_envs
			stage = steps_before // args.switch_steps
			if stage != current_stage:
				env.close()
				current_stage = stage
				task_index = stage % args.num_tasks
				env = make_training_env(args, task_index)
				observations = env.reset()
				return_tracker.switch_task()
				diagnostics.start_training_phase()
				if args.clear_replay_on_switch:
					replay.clear()
					permanent_replay.clear()

			game = args.task_sequence[task_index]
			stage_steps = steps_before % args.switch_steps
			epsilon = 1.0 if steps_before < args.warmup_steps else epsilon_at(
				args, steps_before, stage_steps
			)
			old_permanent, actions = agent.actions(
				observations, epsilon, env.num_actions
			)
			next_observations, raw_rewards, done, _ = env.step(actions)
			train_rewards = raw_rewards
			if args.reward_clip > 0:
				train_rewards = np.clip(
					raw_rewards, -args.reward_clip, args.reward_clip
				)
			replay.store_batch(
				observations, actions, next_observations, train_rewards, done
			)
			permanent_replay.store_batch(
				observations, actions, old_permanent
			)
			observations = next_observations
			completed = return_tracker.update(game, raw_rewards, done)
			episodes_completed += len(completed)
			global_steps = steps_before + args.num_envs

			while global_steps >= next_update:
				if replay.size() >= args.batch_size:
					last_transient_loss, last_gradient_norm = agent.train_transient(
						replay
					)
					gradient_updates += 1
				next_update += args.update_frequency

			metrics = {}
			target_updated = False
			while global_steps >= next_target_update:
				agent.update_target()
				target_updated = True
				next_target_update += args.target_update_frequency
			while global_steps >= next_permanent_update:
				if global_steps >= args.warmup_steps:
					last_permanent_loss, updates = agent.consolidate(
						permanent_replay
					)
					permanent_updates += updates
					target_updated = True
				next_permanent_update += args.update
			if target_updated:
				metrics.update(diagnostics.action_gap_metrics(
					agent.behavior, replay, device, game
				))

			while global_steps >= next_evaluation:
				completed_stages = global_steps // args.switch_steps
				metrics.update(diagnostics.evaluation_metrics(
					agent.behavior, device, completed_stages
				))
				next_evaluation += args.eval_frequency

			log_due = global_steps >= next_log
			while global_steps >= next_log:
				next_log += args.log_frequency
			if log_due:
				metrics.update({
					"train_return": return_tracker.ema[game],
					"episodic_return_10000": return_tracker.ema[game],
					"epsilon": epsilon,
					"task_index": task_index,
					"task_stage": current_stage,
					"cycle_index": current_stage // args.num_tasks,
					"gradient_updates": gradient_updates,
					"permanent_updates": permanent_updates,
					"replay_size": replay.size(),
					"pt/permanent_replay_size": permanent_replay.size(),
					"episodes_completed": episodes_completed,
				})
				if last_transient_loss is not None:
					metrics["loss/transient_td"] = last_transient_loss
				if last_permanent_loss is not None:
					metrics["loss/permanent"] = last_permanent_loss
			if metrics:
				wandb.log(metrics, step=global_steps)

			returns_array[vector_step] = return_tracker.ema[game]
	finally:
		env.close()
		wandb.finish()

	if args.save_model:
		torch.save(
			agent.behavior.state_dict(),
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
