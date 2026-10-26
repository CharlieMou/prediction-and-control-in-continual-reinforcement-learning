"""DQN baseline for fixed-level continual Procgen curricula."""

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


class DQNAgent:
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

	def train(self, replay):
		states, actions, next_states, rewards, done = replay.sample()
		with torch.no_grad():
			next_values = self.target(next_states).max(dim=1, keepdim=True).values
			targets = rewards + (1.0 - done) * self.args.gamma * next_values
		predictions = self.online(states).gather(1, actions)
		loss = F.smooth_l1_loss(predictions, targets)
		self.optimizer.zero_grad()
		loss.backward()
		gradient_norm = torch.nn.utils.clip_grad_norm_(
			self.online.parameters(), self.args.gradient_clip
		)
		self.optimizer.step()
		return loss.item(), float(gradient_norm)

	def update_target(self):
		self.target.load_state_dict(self.online.state_dict())


def parse_args():
	parser = ArgumentParser(description="DQN on continual Procgen")
	add_common_arguments(parser)
	parser.add_argument(
		"--lr1", type=float, default=2.5e-4,
		help="Adam learning rate from the off-policy Procgen study",
	)
	args = resolve_and_validate(parser.parse_args())
	if args.lr1 <= 0:
		raise ValueError("--lr1 must be positive")
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
	agent = DQNAgent(env.num_actions, args, device)
	print(f"DQN behavior parameters: {parameter_count(agent.online):,}")

	replay = ReplayBuffer(args.replay_size, args.batch_size, device, args.seed + 17)
	diagnostics = ContinualDiagnostics(args)
	return_tracker = EpisodeReturnTracker(args.task_sequence, args.num_envs)
	observations = env.reset()

	algorithm_name = "DQN"
	filename = (
		f"DQN_Procgen_{args.distribution_mode}_{args.num_tasks}tasks"
		f"_{args.cycles}cycles_switch_{args.switch_steps}_lr_{args.lr1}"
		f"_seed_{args.seed}"
	)
	wandb.init(
		project=args.wandb_project,
		name=filename,
		group=algorithm_name,
		mode=args.wandb_mode,
		config=serializable_config(
			args, algorithm_name,
			{"network_parameters": parameter_count(agent.online)},
		),
	)

	vector_steps = args.t_steps // args.num_envs
	returns_array = np.zeros(vector_steps, dtype=np.float32)
	next_update = args.warmup_steps
	next_target_update = args.target_update_frequency
	next_evaluation = args.eval_frequency
	next_log = args.log_frequency
	gradient_updates = 0
	episodes_completed = 0
	last_loss = None
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

			game = args.task_sequence[task_index]
			stage_steps = steps_before % args.switch_steps
			epsilon = 1.0 if steps_before < args.warmup_steps else epsilon_at(
				args, steps_before, stage_steps
			)
			if steps_before < args.warmup_steps:
				actions = np.random.randint(0, env.num_actions, size=args.num_envs)
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
					last_loss, last_gradient_norm = agent.train(replay)
					gradient_updates += 1
				next_update += args.update_frequency

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
					"episodes_completed": episodes_completed,
				})
				if last_loss is not None:
					metrics["loss/td"] = last_loss
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
