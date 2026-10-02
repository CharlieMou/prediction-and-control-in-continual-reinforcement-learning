"""DQN with task-boundary Value Cloning on continual Procgen."""

from argparse import ArgumentParser
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from config import add_common_arguments, epsilon_at, resolve_and_validate
from diagnostics import ContinualDiagnostics
from model import make_dqn_network, parameter_count
from procgen_env import make_training_env
from replay import ReplayBuffer
from utils import (
	EpisodeReturnTracker,
	describe_protocol,
	observation_tensor,
	prepare_output_directories,
	seed_everything,
	serializable_config,
)


class ValueCloningMemory:
	"""Reservoir of uint8 states and frozen full-Q-vector targets."""

	def __init__(self, max_size, batch_size, num_actions, device, seed):
		self.max_size = int(max_size)
		self.batch_size = batch_size
		self.device = device
		self.rng = np.random.RandomState(seed)
		self.states = np.empty(
			(self.max_size, 3, 64, 64), dtype=np.uint8
		)
		self.q_values = np.empty(
			(self.max_size, num_actions), dtype=np.float32
		)
		self.length = 0
		self.num_seen = 0

	@torch.no_grad()
	def store_replay_targets(self, replay, net):
		"""Add a uniform reservoir sample of the completed task's replay."""
		# Decide the final reservoir contents before evaluating Q. Since the
		# network is fixed throughout this task-boundary operation, this is
		# equivalent to the MinAtar implementation's state-by-state updates but
		# avoids forwarding every entry in Procgen's much larger replay.
		replacements = {}
		for replay_index in range(replay.size()):
			seen = self.num_seen + replay_index + 1
			if self.length < self.max_size:
				slot = self.length
				self.length += 1
			else:
				slot = int(self.rng.randint(0, seen))
				if slot >= self.max_size:
					continue
			replacements[slot] = replay_index
		self.num_seen += replay.size()

		if not replacements:
			return 0
		slots = np.fromiter(replacements.keys(), dtype=np.int64)
		replay_indices = np.fromiter(replacements.values(), dtype=np.int64)
		for start in range(0, len(slots), self.batch_size):
			batch_slots = slots[start:start + self.batch_size]
			batch_indices = replay_indices[start:start + self.batch_size]
			states = replay.states[batch_indices]
			state_tensor = torch.as_tensor(
				states, device=self.device
			).float().div_(255.0)
			q_targets = net(state_tensor).cpu().numpy()
			self.states[batch_slots] = states
			self.q_values[batch_slots] = q_targets
		return len(slots)

	def sample(self):
		count = min(self.length, self.batch_size)
		if count == 0:
			raise RuntimeError("cannot sample an empty value-cloning memory")
		indices = self.rng.choice(self.length, size=count, replace=False)
		states = torch.as_tensor(
			self.states[indices], device=self.device
		).float().div_(255.0)
		q_values = torch.as_tensor(
			self.q_values[indices], device=self.device
		)
		return states, q_values

	def size(self):
		return self.length


class ValueCloningAgent:
	def __init__(self, num_actions, args, device):
		self.args = args
		self.device = device
		self.online = make_dqn_network(num_actions).to(device)
		self.target = make_dqn_network(num_actions).to(device)
		self.target.load_state_dict(self.online.state_dict())
		self.target.requires_grad_(False)
		self.optimizer = optim.Adam(
			self.online.parameters(), lr=args.lr1, eps=args.adam_eps
		)

	@torch.no_grad()
	def actions(self, observations, epsilon, num_actions):
		q_values = self.online(observation_tensor(observations, self.device))
		actions = q_values.argmax(dim=1).cpu().numpy()
		random_mask = np.random.random(len(actions)) < epsilon
		actions[random_mask] = np.random.randint(
			0, num_actions, size=random_mask.sum()
		)
		return actions

	def train(self, replay, value_memory):
		states, actions, next_states, rewards, done = replay.sample()
		with torch.no_grad():
			next_values = self.target(next_states).max(dim=1, keepdim=True).values
			targets = rewards + (1.0 - done) * self.args.gamma * next_values
		predictions = self.online(states).gather(1, actions)
		td_loss = F.smooth_l1_loss(predictions, targets)

		value_cloning_loss = torch.zeros((), device=self.device)
		if value_memory.size() > 0 and self.args.lambda_vc > 0:
			cloning_states, cloning_targets = value_memory.sample()
			value_cloning_loss = F.smooth_l1_loss(
				self.online(cloning_states), cloning_targets
			)
		loss = td_loss + self.args.lambda_vc * value_cloning_loss
		self.optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.online.parameters(), self.args.gradient_clip
		)
		self.optimizer.step()
		return loss.item(), td_loss.item(), value_cloning_loss.item()

	def update_target(self):
		self.target.load_state_dict(self.online.state_dict())


def parse_args():
	parser = ArgumentParser(
		description="DQN with Value Cloning on continual Procgen"
	)
	add_common_arguments(parser)
	parser.add_argument(
		"--lr1", type=float, default=2.5e-4,
		help="Adam learning rate, identical to the DQN baseline",
	)
	parser.add_argument(
		"--value-memory-size", type=int, default=10_000,
		help="maximum number of historical state/Q-vector pairs",
	)
	parser.add_argument(
		"--lambda-vc", type=float, default=1.0,
		help="value-cloning loss coefficient",
	)
	args = resolve_and_validate(parser.parse_args())
	if args.lr1 <= 0:
		raise ValueError("--lr1 must be positive")
	if args.value_memory_size <= 0:
		raise ValueError("--value-memory-size must be positive")
	if args.lambda_vc < 0:
		raise ValueError("--lambda-vc cannot be negative")
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
	agent = ValueCloningAgent(env.num_actions, args, device)
	print(f"DQN Value Cloning behavior parameters: {parameter_count(agent.online):,}")
	value_memory_mib = args.value_memory_size * 3 * 64 * 64 / 2**20
	print(
		f"Value-cloning state storage reserves approximately "
		f"{value_memory_mib:.1f} MiB"
	)

	replay = ReplayBuffer(args.replay_size, args.batch_size, device, args.seed + 17)
	value_memory = ValueCloningMemory(
		args.value_memory_size, args.batch_size, env.num_actions,
		device, args.seed + 31,
	)
	diagnostics = ContinualDiagnostics(args)
	return_tracker = EpisodeReturnTracker(args.task_sequence, args.num_envs)
	observations = env.reset()

	algorithm_name = "DQN w/ Value Cloning"
	filename = (
		f"DQN_ValueCloning_Procgen_{args.distribution_mode}"
		f"_{args.num_tasks}tasks_{args.cycles}cycles"
		f"_switch_{args.switch_steps}_lr_{args.lr1}"
		f"_lambda_{args.lambda_vc}_memory_{args.value_memory_size}"
		f"_seed_{args.seed}"
	)
	wandb.init(
		project=args.wandb_project,
		name=filename,
		group=algorithm_name,
		mode=args.wandb_mode,
		config=serializable_config(args, algorithm_name, {
			"network_parameters": parameter_count(agent.online),
			"value_memory_state_mib": value_memory_mib,
		}),
	)

	vector_steps = args.t_steps // args.num_envs
	returns_array = np.zeros(vector_steps, dtype=np.float32)
	next_update = args.warmup_steps
	next_target_update = args.target_update_frequency
	next_evaluation = args.eval_frequency
	next_log = args.log_frequency
	gradient_updates = 0
	episodes_completed = 0
	last_losses = None

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

			game = args.task_sequence[task_index]
			stage_steps = steps_before % args.switch_steps
			epsilon = 1.0 if steps_before < args.warmup_steps else epsilon_at(
				args, steps_before, stage_steps
			)
			if steps_before < args.warmup_steps:
				actions = np.random.randint(
					0, env.num_actions, size=args.num_envs
				)
			else:
				actions = agent.actions(observations, epsilon, env.num_actions)

			next_observations, raw_rewards, done, _ = env.step(actions)
			train_rewards = raw_rewards
			if args.reward_clip > 0:
				train_rewards = np.clip(
					raw_rewards, -args.reward_clip, args.reward_clip
				)
			replay.store_batch(
				observations, actions, next_observations, train_rewards, done
			)
			observations = next_observations
			completed = return_tracker.update(game, raw_rewards, done)
			episodes_completed += len(completed)
			global_steps = steps_before + args.num_envs

			while global_steps >= next_update:
				if replay.size() >= args.batch_size:
					last_losses = agent.train(replay, value_memory)
					gradient_updates += 1
				next_update += args.update_frequency

			# Match the MinAtar implementation: clone the current network's full
			# Q-vector targets immediately after completing each non-final stage.
			if (
				global_steps % args.switch_steps == 0
				and global_steps < args.t_steps
			):
				value_memory.store_replay_targets(replay, agent.online)

			metrics = {}
			target_updated = False
			while global_steps >= next_target_update:
				agent.update_target()
				target_updated = True
				next_target_update += args.target_update_frequency
			if target_updated:
				metrics.update(diagnostics.action_gap_metrics(
					agent.online, replay, device, game
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
					"epsilon": epsilon,
					"task_index": task_index,
					"task_stage": current_stage,
					"cycle_index": current_stage // args.num_tasks,
					"gradient_updates": gradient_updates,
					"replay_size": replay.size(),
					"vc/value_memory_size": value_memory.size(),
					"episodes_completed": episodes_completed,
				})
				if last_losses is not None:
					total_loss, td_loss, value_cloning_loss = last_losses
					metrics.update({
						"loss/total": total_loss,
						"loss/td": td_loss,
						"loss/value_cloning": value_cloning_loss,
					})
			if metrics:
				wandb.log(metrics, step=global_steps)

			returns_array[vector_step] = return_tracker.ema[game]
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
