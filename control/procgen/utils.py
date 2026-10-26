"""Training utilities shared by the Procgen DQN baselines."""

from pathlib import Path
import random

import numpy as np
import torch


def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def observation_tensor(observations, device):
	array = np.ascontiguousarray(observations.transpose(0, 3, 1, 2))
	return torch.as_tensor(array, device=device).float().div_(255.0)


class EpisodeReturnTracker:
	def __init__(self, task_sequence, num_envs):
		self.task_sequence = task_sequence
		self.num_envs = num_envs
		self.ema = {task: 0.0 for task in task_sequence}
		self.current = np.zeros(num_envs, dtype=np.float64)

	def switch_task(self):
		self.current.fill(0.0)

	def update(self, game, rewards, done):
		self.current += rewards
		completed = self.current[done]
		for episode_return in completed:
			self.ema[game] = 0.99 * self.ema[game] + 0.01 * episode_return
		self.current[done] = 0.0
		return completed


def serializable_config(args, algorithm, extra=None):
	config = {}
	for key, value in vars(args).items():
		if isinstance(value, Path):
			value = str(value)
		elif isinstance(value, tuple):
			value = list(value)
		config[key] = value
	config["algorithm"] = algorithm
	if extra:
		config.update(extra)
	return config


def prepare_output_directories(args):
	for name in ("results", "models", "logs"):
		(args.output_dir / name).mkdir(parents=True, exist_ok=True)


def describe_protocol(args, uses_replay=True):
	train_stop = args.train_start_level + args.num_train_levels
	test_stop = args.test_start_level + args.num_test_levels
	print(f"Tasks: {', '.join(args.task_sequence)}; cycles={args.cycles}")
	print(
		f"Procgen mode={args.distribution_mode}; "
		f"train levels=[{args.train_start_level}, {train_stop}); "
		f"test levels=[{args.test_start_level}, {test_stop})"
	)
	print(
		f"Transitions={args.t_steps:,}; switch every {args.switch_steps:,}; "
		f"parallel envs={args.num_envs}"
	)
	if uses_replay:
		replay_gib = args.replay_size * 2 * 3 * 64 * 64 / 2**30
		print(
			f"Replay observation storage reserves approximately {replay_gib:.1f} GiB"
		)
