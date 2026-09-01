"""Paper-compatible Forager environment used by the control experiments.

This adapter recreates ``ForagaxSquareWaveTwoBiome-v11`` (the environment
used for Figure 5 of the Forager paper) with the Numba Forager source vendored
in ``control/forager/forager``.  It exposes the small subset of the Gym API
used by the existing DQN implementations without adding a Gym dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# Make the vendored project importable when these scripts are launched from
# either the repository root or control/forager.
_FORAGER_SOURCE = Path(__file__).resolve().parent / "forager"
if str(_FORAGER_SOURCE) not in sys.path:
	sys.path.insert(0, str(_FORAGER_SOURCE))

import forager._utils.numba as nbu  # noqa: E402
import forager.grid as grid  # noqa: E402
from forager.Env import ForagerEnv  # noqa: E402
from forager.config import ForagerConfig  # noqa: E402
from forager.objects import ForagerObject, Wall  # noqa: E402


WORLD_SIZE = (24, 15)
DEFAULT_APERTURE = 9
DEFAULT_REWARD_PERIOD = 500_000
DEFAULT_EXPIRY_TIME = 500
DEFAULT_REGEN_RANGE = (9, 11)

PURPLE = np.array((124, 61, 81), dtype=np.uint8)
YELLOW = np.array((218, 145, 0), dtype=np.uint8)
BLACK = np.array((0, 0, 0), dtype=np.uint8)


class _Space:
	def __init__(self, shape):
		self.shape = tuple(shape)


class _Discrete:
	def __init__(self, n: int, seed: int):
		self.n = n
		self.seed(seed)

	def seed(self, seed: int):
		self._rng = np.random.default_rng(seed)

	def sample(self) -> int:
		return int(self._rng.integers(self.n))


class SquareWaveMushroom(ForagerObject):
	"""Collectable whose reward follows the paper's hidden square wave."""

	def __init__(
			self,
			name: str,
			color: np.ndarray,
			base_reward: float,
			amplitude: float,
			period: int,
			bounds,
			loc=None,
	):
		super().__init__(name=name)
		self.color = color
		self.base_reward = base_reward
		self.amplitude = amplitude
		self.period = period
		self.bounds = bounds
		self.target_location = loc

	def reward(self, rng: np.random.Generator, clock: int) -> float:
		position = clock % self.period
		if position == 0:
			square = 0.0
		elif position < self.period // 2:
			square = 1.0
		else:
			square = -1.0
		return self.base_reward + self.amplitude * square

	def regen_delay(self, rng: np.random.Generator, clock: int) -> int:
		# The wrapper chooses a free random cell in the same biome at respawn.
		self.target_location = None
		low, high = DEFAULT_REGEN_RANGE
		return int(rng.integers(low, high + 1))


class ExpiringForagerEnv(ForagerEnv):
	"""Forager core with same-biome respawn and mushroom expiry."""

	def __init__(self, config: ForagerConfig, expiry_time: int):
		self._spawn_times = {}
		self._expiry_time = expiry_time
		super().__init__(config)

	def add_object(self, obj, start=None, stop=None):
		super().add_object(obj, start, stop)
		idx = nbu.ravel(obj.current_location, self._size)
		self._spawn_times[idx] = self._clock

	def remove_object(self, coords):
		idx = nbu.ravel(coords, self._size)
		self._spawn_times.pop(idx, None)
		return super().remove_object(coords)

	def _sample_free(self, bounds):
		start, stop = bounds
		width = stop[0] - start[0]
		height = stop[1] - start[1]
		candidates = np.arange(width * height)
		self.rng.shuffle(candidates)
		for candidate in candidates:
			x = start[0] + int(candidate % width)
			y = start[1] + int(candidate // width)
			coords = (x, y)
			if nbu.ravel(coords, self._size) not in self._obj_store.idx_to_name:
				return coords
		return None

	def _respawn(self):
		if self._clock not in self._to_respawn:
			return
		for obj in self._to_respawn[self._clock]:
			if isinstance(obj, SquareWaveMushroom):
				obj.target_location = self._sample_free(obj.bounds)
				if obj.target_location is None:
					self._to_respawn[self._clock + 1].append(obj)
					continue
			self.add_object(obj, (0, 0), self._size)
		del self._to_respawn[self._clock]

	def _expire_objects(self):
		expired = []
		for idx, spawn_time in tuple(self._spawn_times.items()):
			if self._clock - spawn_time < self._expiry_time:
				continue
			obj = self._obj_store.get_object(idx)
			if isinstance(obj, SquareWaveMushroom):
				expired.append(idx)
		for idx in expired:
			self.remove_object(nbu.unravel(idx, self._size))

	def step(self, action):
		next_state = grid.step(self._state, self._size, action)
		idx = nbu.ravel(next_state, self._size)
		reward = 0.0
		if self._obj_store.has_object(idx):
			obj = self._obj_store.get_object(idx)
			reward = obj.collision(self.rng, self._clock)
			if obj.blocking:
				next_state = self._state
			if obj.collectable:
				self.remove_object(next_state)

		self._state = next_state
		self._clock += 1
		self._respawn()
		self._expire_objects()
		return self._get_observation(self._state), float(reward)


class TwoBiomeForagerEnv:
	"""Continuing two-biome task from Figures 4--6 of the paper."""

	game_name = "ForagerSquareWaveTwoBiome-v11"

	def __init__(
			self,
			seed: int = 0,
			aperture_size: int = DEFAULT_APERTURE,
			reward_period: int = DEFAULT_REWARD_PERIOD,
			expiry_time: int = DEFAULT_EXPIRY_TIME,
	):
		if aperture_size <= 0 or aperture_size % 2 == 0:
			raise ValueError("aperture_size must be a positive odd integer")
		if reward_period <= 1 or reward_period % 2 != 0:
			raise ValueError("reward_period must be an even integer greater than one")
		if expiry_time <= 0:
			raise ValueError("expiry_time must be positive")

		self.seed = seed
		self.aperture_size = aperture_size
		self.reward_period = reward_period
		self.expiry_time = expiry_time
		self.action_space = _Discrete(4, seed)
		# 9x9x3 color one-hot plus previous-action one-hot and previous reward.
		self.image_shape = (aperture_size, aperture_size, 3)
		self.observation_space = _Space((int(np.prod(self.image_shape)) + 5,))
		self._build()

	@property
	def clock(self) -> int:
		return self._env._clock

	@property
	def task_phase(self) -> int:
		position = self.clock % self.reward_period
		return 0 if position == 0 or position < self.reward_period // 2 else 1

	def _factory(self, name, color, amplitude, bounds):
		period = self.reward_period
		return lambda: SquareWaveMushroom(
			name=name,
			color=color,
			base_reward=-5.0,
			amplitude=amplitude,
			period=period,
			bounds=bounds,
		)

	def _build(self):
		left = ((4, 0), (8, 15))
		right = ((16, 0), (20, 15))
		factories = {
			"left_oyster": self._factory("left_oyster", PURPLE, 9.0, left),
			"left_chanterelle": self._factory(
				"left_chanterelle", YELLOW, 3.0, left
			),
			"right_oyster": self._factory("right_oyster", PURPLE, -3.0, right),
			"right_chanterelle": self._factory(
				"right_chanterelle", YELLOW, -9.0, right
			),
			"wall": Wall,
		}
		config = ForagerConfig(
			size=WORLD_SIZE,
			object_types=factories,
			seed=self.seed,
			observation_mode="objects",
			aperture=self.aperture_size,
		)
		self._env = ExpiringForagerEnv(config, self.expiry_time)
		self._populate(factories, left, right)

	def _region_cells(self, start, stop):
		return [(x, y) for x in range(start[0], stop[0]) for y in range(start[1], stop[1])]

	def _place_region(self, cells, specifications):
		cells = [
			coords for coords in cells
			if nbu.ravel(coords, WORLD_SIZE) not in self._env._obj_store.idx_to_name
		]
		self._env.rng.shuffle(cells)
		cursor = 0
		for factory, count in specifications:
			if cursor + count > len(cells):
				raise RuntimeError("not enough free cells to populate Forager biome")
			for coords in cells[cursor:cursor + count]:
				obj = factory()
				obj.target_location = coords
				self._env.add_object(obj)
			cursor += count

	def _populate(self, factories, left, right):
		left_cells = self._region_cells(*left)
		right_cells = self._region_cells(*right)
		void_cells = self._region_cells((0, 0), WORLD_SIZE)
		# v11 first places 24 background walls across the complete world,
		# then adds 15+15 mushrooms and four walls in each food biome.
		self._place_region(void_cells, ((factories["wall"], 24),))
		self._place_region(
			left_cells,
			((factories["left_oyster"], 15),
			 (factories["left_chanterelle"], 15),
			 (factories["wall"], 4)),
		)
		self._place_region(
			right_cells,
			((factories["right_oyster"], 15),
			 (factories["right_chanterelle"], 15),
			 (factories["wall"], 4)),
		)

	def _paper_observation(self, raw, last_action: int | None, last_reward: float):
		mapping = self._env._names_to_dims
		purple = raw[..., mapping["left_oyster"]] | raw[..., mapping["right_oyster"]]
		yellow = (
			raw[..., mapping["left_chanterelle"]]
			| raw[..., mapping["right_chanterelle"]]
		)
		wall = raw[..., mapping["wall"]]
		image = np.stack((purple, yellow, wall), axis=-1).astype(np.float32)
		action = np.zeros(4, dtype=np.float32)
		if last_action is not None:
			action[last_action] = 1.0
		return np.concatenate(
			(image.reshape(-1), action, np.array([last_reward], dtype=np.float32))
		)

	def reset(self):
		self._build()
		return self._paper_observation(self._env.start(), None, 0.0)

	def step(self, action: int):
		raw, reward = self._env.step(action)
		obs = self._paper_observation(raw, action, reward)
		return obs, reward, False, {"task_phase": self.task_phase, "clock": self.clock}

	def close(self):
		return None


def add_forager_arguments(parser):
	parser.add_argument("--aperture-size", type=int, default=DEFAULT_APERTURE)
	parser.add_argument("--reward-period", type=int, default=DEFAULT_REWARD_PERIOD,
			help="full square-wave period; the hidden reward task flips every half-period")
	parser.add_argument("--expiry-time", type=int, default=DEFAULT_EXPIRY_TIME)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--epsilon", type=float, default=0.1)
	parser.add_argument("--replay-size", type=int, default=1000)
	parser.add_argument("--minimum-replay-history", type=int, default=50)
	parser.add_argument("--update-frequency", type=int, default=4)
	parser.add_argument("--target-update-frequency", type=int, default=128,
			help="target refresh interval measured in gradient updates")
	parser.add_argument("--return-smoothing-window", type=int, default=10000)
	parser.add_argument("--log-frequency", type=int, default=1000)
	parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
			default="online")


def validate_forager_arguments(args):
	if args.t_steps <= 0:
		raise ValueError("t-steps must be positive")
	if not 0.0 <= args.epsilon <= 1.0:
		raise ValueError("epsilon must be in [0, 1]")
	if not 0.0 <= args.gamma <= 1.0:
		raise ValueError("gamma must be in [0, 1]")
	if args.replay_size < args.batch_size:
		raise ValueError("replay-size must be at least batch-size")
	if not args.batch_size <= args.minimum_replay_history <= args.replay_size:
		raise ValueError(
			"minimum-replay-history must be between batch-size and replay-size"
		)
	for name in (
			"update_frequency", "target_update_frequency",
			"return_smoothing_window", "log_frequency"):
		if getattr(args, name) <= 0:
			raise ValueError(f"{name.replace('_', '-')} must be positive")


def make_forager_env(args):
	return TwoBiomeForagerEnv(
		seed=args.seed,
		aperture_size=args.aperture_size,
		reward_period=args.reward_period,
		expiry_time=args.expiry_time,
	)
