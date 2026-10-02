"""Small, dependency-light wrapper around Procgen's Baselines VecEnv API."""

import numpy as np


class ProcgenVectorEnv:
	def __init__(self, env_name, num_envs, start_level, num_levels,
			distribution_mode, rand_seed, num_threads):
		try:
			from procgen import ProcgenEnv
		except ImportError as error:
			raise ImportError(
				"Procgen is not installed. Install control/procgen/requirements.txt "
				"in a Python 3.10 environment."
			) from error

		self.env_name = env_name
		self.num_envs = num_envs
		self.start_level = start_level
		self.num_levels = num_levels
		self._env = ProcgenEnv(
			num_envs=num_envs,
			env_name=env_name,
			start_level=start_level,
			num_levels=num_levels,
			distribution_mode=distribution_mode,
			rand_seed=rand_seed,
			num_threads=num_threads,
			use_sequential_levels=False,
		)
		self.action_space = self._env.action_space
		self.num_actions = self.action_space.n

	@staticmethod
	def _pixels(observation):
		pixels = observation["rgb"] if isinstance(observation, dict) else observation
		if pixels.dtype != np.uint8 or pixels.shape[-3:] != (64, 64, 3):
			raise ValueError(
				f"expected uint8 Procgen observations shaped (*, 64, 64, 3), "
				f"got {pixels.dtype} {pixels.shape}"
			)
		return pixels

	def reset(self):
		return self._pixels(self._env.reset())

	def step(self, actions):
		actions = np.asarray(actions, dtype=np.int32)
		observation, reward, done, info = self._env.step(actions)
		return (
			self._pixels(observation),
			np.asarray(reward, dtype=np.float32),
			np.asarray(done, dtype=np.bool_),
			info,
		)

	def close(self):
		self._env.close()


def make_training_env(args, task_index):
	return ProcgenVectorEnv(
		env_name=args.task_sequence[task_index],
		num_envs=args.num_envs,
		start_level=args.train_start_level,
		num_levels=args.num_train_levels,
		distribution_mode=args.distribution_mode,
		rand_seed=args.level_seed + 1009 * task_index,
		num_threads=args.num_threads,
	)


def make_evaluation_env(args, task_index):
	return ProcgenVectorEnv(
		env_name=args.task_sequence[task_index],
		num_envs=min(args.eval_num_envs, args.eval_episodes),
		start_level=args.test_start_level,
		num_levels=args.num_test_levels,
		distribution_mode=args.distribution_mode,
		rand_seed=args.level_seed + 100_003 + 1009 * task_index,
		num_threads=args.num_threads,
	)


def make_training_evaluation_env(args, task_index):
	"""Fixed evaluation stream drawn from the task's training-level set."""
	return ProcgenVectorEnv(
		env_name=args.task_sequence[task_index],
		num_envs=min(args.eval_num_envs, args.eval_episodes),
		start_level=args.train_start_level,
		num_levels=args.num_train_levels,
		distribution_mode=args.distribution_mode,
		rand_seed=args.level_seed + 50_003 + 1009 * task_index,
		num_threads=args.num_threads,
	)
