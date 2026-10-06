"""Fully streaming PQN with GAC on continual Procgen.

Every behavior state and its concurrently computed action gap are considered
by a lifetime reservoir immediately.  Targets are frozen at generation time
and are never retrospectively revalued with a later network.
"""

from argparse import ArgumentParser
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from config import epsilon_at
from diagnostics import ContinualDiagnostics
from model import parameter_count
from pqn import (
	PQNAgent,
	RolloutStateSampler,
	add_pqn_arguments,
	channels_first_uint8,
	resolve_pqn_args,
)
from pqn_auxiliary import StreamingAuxiliaryMemory
from procgen_env import make_training_env
from utils import (
	EpisodeReturnTracker,
	describe_protocol,
	prepare_output_directories,
	seed_everything,
	serializable_config,
)


class PQNGACAgent(PQNAgent):
	def train(self, states, actions, targets, auxiliary_memory):
		states = states.reshape(-1, 3, 64, 64)
		actions = actions.reshape(-1)
		targets = targets.reshape(-1)
		batch_size = len(states)
		minibatch_size = min(self.args.pqn_minibatch_size, batch_size)
		indices = np.arange(batch_size)
		total_losses, td_losses, gac_losses, predicted_q = [], [], [], []
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
				td_loss = F.mse_loss(predictions, targets[minibatch])

				gac_loss = torch.zeros((), device=self.device)
				if (
					auxiliary_memory.size() >= self.args.pqn_minibatch_size
					and self.args.lambda_gac > 0
				):
					memory_states, memory_actions, margins = (
						auxiliary_memory.sample(len(minibatch))
					)
					memory_q = self.online(memory_states)
					greedy_q = memory_q.gather(1, memory_actions)
					competitor_q = memory_q.scatter(
						1, memory_actions, -torch.inf
					).max(dim=1, keepdim=True).values
					gac_loss = torch.relu(
						competitor_q - greedy_q + self.args.rho * margins
					).mean()

				loss = td_loss + self.args.lambda_gac * gac_loss
				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(
					self.online.parameters(), self.args.gradient_clip
				)
				self.optimizer.step()
				total_losses.append(loss.item())
				td_losses.append(td_loss.item())
				gac_losses.append(gac_loss.item())
				predicted_q.append(predictions.detach().mean().item())
				gradient_updates += 1

		return (
			float(np.mean(total_losses)),
			float(np.mean(td_losses)),
			float(np.mean(gac_losses)),
			float(np.mean(predicted_q)),
			gradient_updates,
		)


def parse_args(arguments=None):
	parser = ArgumentParser(description="PQN with streaming GAC on continual Procgen")
	add_pqn_arguments(parser)
	parser.add_argument(
		"--revaluation-size", "--auxiliary-memory-size",
		dest="revaluation_size", type=int, default=10_000,
		help="maximum number of streaming greedy-action constraints",
	)
	parser.add_argument(
		"--rho", type=float, default=0.5,
		help="fraction of each historical action gap to preserve",
	)
	parser.add_argument(
		"--lambda-gac", type=float, default=1.0,
		help="GAC regularization coefficient",
	)
	args = resolve_pqn_args(parser.parse_args(arguments))
	if args.revaluation_size <= 0:
		raise ValueError("--revaluation-size must be positive")
	if args.revaluation_size < args.pqn_minibatch_size:
		raise ValueError(
			"--revaluation-size must be at least one PQN minibatch "
			f"({args.pqn_minibatch_size})"
		)
	if not 0 <= args.rho <= 1:
		raise ValueError("--rho must be in [0, 1]")
	if args.lambda_gac < 0:
		raise ValueError("--lambda-gac cannot be negative")
	return args


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
	agent = PQNGACAgent(env.num_actions, args, device)
	print(f"PQN GAC behavior parameters: {parameter_count(agent.online):,}")
	print(
		f"PQN rollout={args.rollout_batch_size:,}, "
		f"minibatch={args.pqn_minibatch_size:,}, epochs={args.update_epochs}"
	)

	auxiliary_memory = StreamingAuxiliaryMemory(
		args.revaluation_size,
		args.pqn_minibatch_size,
		env.num_actions,
		device,
		args.seed + 31,
		"gac",
	)
	diagnostics = ContinualDiagnostics(args)
	rollout_sampler = RolloutStateSampler(args.seed + 17)
	return_tracker = EpisodeReturnTracker(args.task_sequence, args.num_envs)
	observations = env.reset()

	state_mib = args.revaluation_size * 3 * 64 * 64 / 2**20
	print(
		f"Streaming GAC state storage reserves approximately {state_mib:.1f} MiB"
	)
	algorithm_name = "PQN w/ Streaming GAC"
	filename = (
		f"PQN_Streaming_GAC_Procgen_{args.distribution_mode}_{args.num_tasks}tasks"
		f"_{args.cycles}cycles_switch_{args.switch_steps}"
		f"_qlambda_{args.q_lambda}_lr_{args.learning_rate}"
		f"_lambda_{args.lambda_gac}_rho_{args.rho}"
		f"_memory_{args.revaluation_size}_seed_{args.seed}"
	)
	wandb.init(
		project=args.wandb_project,
		name=filename,
		group=algorithm_name,
		mode=args.wandb_mode,
		config=serializable_config(args, algorithm_name, {
			"network_parameters": parameter_count(agent.online),
			"uses_replay": False,
			"uses_target_network": False,
			"boundary_information_used_by_auxiliary_memory": False,
			"auxiliary_target_time": "behavior_generation",
			"auxiliary_memory_state_mib": state_mib,
		}),
	)

	vector_steps = args.t_steps // args.num_envs
	returns_array = np.zeros(vector_steps, dtype=np.float32)
	next_action_gap = args.target_update_frequency
	next_evaluation = args.eval_frequency
	next_log = args.log_frequency
	global_steps = 0
	gradient_updates = 0
	episodes_completed = 0
	last_losses = None
	last_q = None

	try:
		while global_steps < args.t_steps:
			reservoir_acceptances = 0
			stage = global_steps // args.switch_steps
			if stage != current_stage:
				env.close()
				current_stage = stage
				task_index = stage % args.num_tasks
				env = make_training_env(args, task_index)
				observations = env.reset()
				return_tracker.switch_task()
				diagnostics.start_training_phase()

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
				selected_actions, state_values, behavior_q = (
					agent.actions_and_values(
						state_tensor, last_epsilon, env.num_actions,
						return_q_values=True,
					)
				)
				reservoir_acceptances += auxiliary_memory.update(
					state_tensor, behavior_q
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
			last_total, last_td, last_gac, last_q, new_updates = agent.train(
				states, actions, targets, auxiliary_memory
			)
			last_losses = (last_total, last_td, last_gac)
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
					"gac/revaluation_size": auxiliary_memory.size(),
					"gac/auxiliary_memory_size": auxiliary_memory.size(),
					"gac/states_seen": auxiliary_memory.num_seen,
					"gac/reservoir_acceptances": reservoir_acceptances,
					"episodes_completed": episodes_completed,
					"loss/total": last_losses[0],
					"loss/td": last_losses[1],
					"loss/gac": last_losses[2],
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
		with (
			args.output_dir / "results" / f"{filename}_returns.pkl"
		).open("wb") as file:
			pickle.dump(returns_array, file)
	if args.plot:
		import matplotlib.pyplot as plt
		plt.plot(np.arange(vector_steps) * args.num_envs, returns_array)
		plt.xlabel("lifetime environment transitions")
		plt.ylabel("training return EMA")
		plt.show()


if __name__ == "__main__":
	main()
