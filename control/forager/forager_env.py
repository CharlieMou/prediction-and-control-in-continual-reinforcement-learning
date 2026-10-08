"""Paper-compatible Forager environments used by the control experiments.

This adapter recreates ``ForagaxSquareWaveTwoBiome-v11`` (the environment
used for Figure 5 of the Forager paper) and ``ForagaxUnendingTasks-v1`` (the
environment used for Figure 7) with the Numba Forager source vendored in
``control/forager/forager``.  It exposes the small subset of the Gym API used
by the existing DQN implementations without adding a Gym dependency.
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

UNENDING_WORLD_SIZE = (28, 28)
UNENDING_BIOME_WIDTH = 9
UNENDING_BIOME_GAP = 5
UNENDING_BIOME_OFFSET = 3
UNENDING_FOOD_FREQUENCY = 0.2
UNENDING_WALL_FREQUENCY = 0.1
UNENDING_LOCAL_WALL_FREQUENCY = 0.6
DEFAULT_FOURIER_TERMS = 10
DEFAULT_REWARD_REPEAT = 1_000
DEFAULT_CONSUMPTION_THRESHOLD = 10_000
DEFAULT_CUE_PERIOD = 100
DEFAULT_CUE_DURATION = 10
FOURIER_PERIOD_RANGE = (10, 1_000)

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


def _fourier_reward(series, clock: int) -> float:
	position = (clock // series["reward_repeat"]) % series["period"]
	angle = 2.0 * np.pi * position / series["period"]
	frequencies = np.arange(1, len(series["a_coeffs"]) + 1)
	value = np.sum(
		series["a_coeffs"] * np.cos(frequencies * angle)
		+ series["b_coeffs"] * np.sin(frequencies * angle)
	)
	reward_range = series["y_max"] - series["y_min"]
	if abs(reward_range) < 1e-8:
		return 0.0
	return float(2.0 * (value - series["y_min"]) / reward_range - 1.0)


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


class FourierMushroom(ForagerObject):
	"""One generation of a biome's Fourier-modulated mushroom species."""

	def __init__(self, name: str, biome: int, bounds, series, color):
		super().__init__(name=name)
		self.biome = biome
		self.bounds = bounds
		self.series = series
		self.period = series["period"]
		self.reward_repeat = series["reward_repeat"]
		self.a_coeffs = series["a_coeffs"]
		self.b_coeffs = series["b_coeffs"]
		self.y_min = series["y_min"]
		self.y_max = series["y_max"]
		self.color = color

	def reward(self, rng: np.random.Generator, clock: int) -> float:
		return _fourier_reward(self.series, clock)

	def regen_delay(self, rng: np.random.Generator, clock: int) -> int:
		self.target_location = None
		low, high = DEFAULT_REGEN_RANGE
		# Match jax.random.randint(key, (), 9, 11) in ForagaxBig-v5.
		return int(rng.integers(low, high))


class BlackWall(Wall):
	"""Wall with the fixed black RGB value used by ForagaxBig-v5."""

	def __init__(self):
		super().__init__()
		self.color = BLACK


class UnendingForagerCore(ForagerEnv):
	"""Numba Forager core with centered rewards and dynamic species."""

	def __init__(
		self,
		config: ForagerConfig,
		mushroom_names,
		biome_bounds,
		consumption_threshold: int,
		reward_for_biome,
		renew_biome,
	):
		self.mushroom_names = tuple(mushroom_names)
		self.name_to_biome = {
			name: biome for biome, name in enumerate(self.mushroom_names)
		}
		self.biome_bounds = tuple(biome_bounds)
		self.consumption_threshold = consumption_threshold
		self.consumption_counts = np.zeros(len(biome_bounds), dtype=np.int64)
		self.generations = np.zeros(len(biome_bounds), dtype=np.int64)
		self._reward_for_biome = reward_for_biome
		self._renew_biome = renew_biome
		super().__init__(config)

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
			if isinstance(obj, FourierMushroom):
				obj.target_location = self._sample_free(obj.bounds)
				if obj.target_location is None:
					self._to_respawn[self._clock + 1].append(obj)
					continue
			self.add_object(obj, (0, 0), self._size)
		del self._to_respawn[self._clock]

	def _global_mean_reward(self, clock: int) -> float:
		total = 0.0
		count = 0
		for name in self._obj_store.idx_to_name.values():
			if name in self.name_to_biome:
				total += self._reward_for_biome(self.name_to_biome[name], clock)
				count += 1
		return total / count if count else 0.0

	def _replace_species(self, biome: int):
		name = self.mushroom_names[biome]
		for idx, object_name in tuple(self._obj_store.idx_to_name.items()):
			if object_name == name:
				self._obj_store.remove_object(idx)

		for respawn_time, objects in tuple(self._to_respawn.items()):
			remaining = [
				obj for obj in objects
				if not isinstance(obj, FourierMushroom) or obj.biome != biome
			]
			if remaining:
				self._to_respawn[respawn_time] = remaining
			else:
				del self._to_respawn[respawn_time]

		self.consumption_counts[biome] = 0
		self.generations[biome] += 1
		self._renew_biome(biome)

	def step(self, action):
		next_state = grid.step(self._state, self._size, action)
		idx = nbu.ravel(next_state, self._size)
		reward = 0.0
		collected_biome = None
		if self._obj_store.has_object(idx):
			obj = self._obj_store.get_object(idx)
			mean_reward = self._global_mean_reward(self._clock)
			reward = obj.collision(self.rng, self._clock)
			if isinstance(obj, FourierMushroom):
				reward -= mean_reward
				collected_biome = obj.biome
			if obj.blocking:
				next_state = self._state
			if obj.collectable:
				self.remove_object(next_state)

		if collected_biome is not None:
			self.consumption_counts[collected_biome] += 1
			if (
				self.consumption_counts[collected_biome]
				>= self.consumption_threshold
			):
				self._replace_species(collected_biome)

		self._state = next_state
		self._clock += 1
		self._respawn()
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


class UnendingFourBiomeForagerEnv:
	"""Continuing four-biome task from Figure 7 of the Forager paper."""

	game_name = "ForagaxUnendingTasks-v1"

	def __init__(
		self,
		seed: int = 0,
		aperture_size: int = DEFAULT_APERTURE,
		fourier_terms: int = DEFAULT_FOURIER_TERMS,
		reward_repeat: int = DEFAULT_REWARD_REPEAT,
		consumption_threshold: int = DEFAULT_CONSUMPTION_THRESHOLD,
		cue_period: int = DEFAULT_CUE_PERIOD,
		cue_duration: int = DEFAULT_CUE_DURATION,
	):
		if aperture_size <= 0 or aperture_size % 2 == 0:
			raise ValueError("aperture_size must be a positive odd integer")
		if fourier_terms <= 0:
			raise ValueError("fourier_terms must be positive")
		if reward_repeat <= 0:
			raise ValueError("reward_repeat must be positive")
		if consumption_threshold <= 0:
			raise ValueError("consumption_threshold must be positive")
		if cue_period <= 0:
			raise ValueError("cue_period must be positive")
		if not 0 < cue_duration <= cue_period:
			raise ValueError("cue_duration must be in [1, cue_period]")

		self.seed = seed
		self.aperture_size = aperture_size
		self.fourier_terms = fourier_terms
		self.reward_repeat = reward_repeat
		self.consumption_threshold = consumption_threshold
		self.cue_period = cue_period
		self.cue_duration = cue_duration
		self.action_space = _Discrete(4, seed)
		self.image_shape = (aperture_size, aperture_size, 3)
		# RGB image, previous action, previous reward, and four-biome cue.
		self.scalar_observation_size = 9
		self.observation_space = _Space((
			int(np.prod(self.image_shape)) + self.scalar_observation_size,
		))
		self.mushrooms_per_biome = int(np.ceil(
			UNENDING_BIOME_WIDTH ** 2 * UNENDING_FOOD_FREQUENCY
		))
		self._build()

	@property
	def clock(self) -> int:
		return self._env._clock

	@property
	def task_phase(self) -> int:
		return int(self._env.generations.sum())

	@property
	def biome_generations(self):
		return tuple(int(value) for value in self._env.generations)

	@property
	def cue_active(self) -> bool:
		return self.clock % self.cue_period < self.cue_duration

	@property
	def best_biome(self) -> int:
		return int(np.argmax(self.raw_biome_rewards()))

	def raw_biome_rewards(self, clock: int | None = None):
		if clock is None:
			clock = self.clock
		return np.asarray([
			_fourier_reward(series, clock) for series in self._species
		], dtype=np.float32)

	def biome_rewards(self, clock: int | None = None):
		rewards = self.raw_biome_rewards(clock)
		return rewards - rewards.mean()

	def _biome_layout(self):
		first = UNENDING_BIOME_OFFSET
		second = first + UNENDING_BIOME_WIDTH + UNENDING_BIOME_GAP
		starts = (
			(first, first),
			(second, first),
			(first, second),
			(second, second),
		)
		return tuple(
			(start, (start[0] + UNENDING_BIOME_WIDTH,
				start[1] + UNENDING_BIOME_WIDTH))
			for start in starts
		)

	def _wall_layout(self):
		offset = UNENDING_BIOME_WIDTH // 2 - 1
		return tuple(
			((start[0] + offset, start[1] + offset),
			 (start[0] + offset + 3, start[1] + offset + 3))
			for start, _ in self._biome_bounds
		)

	def _factory(self, biome: int):
		return lambda: FourierMushroom(
			name=self._mushroom_names[biome],
			biome=biome,
			bounds=self._biome_bounds[biome],
			series=self._species[biome],
			color=self._species[biome]["color"],
		)

	def _build(self):
		self._biome_bounds = self._biome_layout()
		self._mushroom_names = tuple(f"biome_{index}" for index in range(4))
		self._species = [None] * len(self._biome_bounds)
		factories = {
			name: self._factory(index)
			for index, name in enumerate(self._mushroom_names)
		}
		factories["wall"] = BlackWall
		config = ForagerConfig(
			size=UNENDING_WORLD_SIZE,
			object_types=factories,
			seed=self.seed,
			observation_mode="objects",
			aperture=self.aperture_size,
		)
		self._env = UnendingForagerCore(
			config=config,
			mushroom_names=self._mushroom_names,
			biome_bounds=self._biome_bounds,
			consumption_threshold=self.consumption_threshold,
			reward_for_biome=self._raw_biome_reward,
			renew_biome=self._renew_biome,
		)
		for biome in range(len(self._biome_bounds)):
			self._sample_species(biome)
		self._populate(factories)

	def _sample_species(self, biome: int):
		low, high = FOURIER_PERIOD_RANGE
		period = int(self._env.rng.integers(low, high + 1))
		frequencies = np.arange(1, self.fourier_terms + 1, dtype=np.float64)
		a_coeffs = self._env.rng.normal(size=self.fourier_terms) / frequencies
		b_coeffs = self._env.rng.normal(size=self.fourier_terms) / frequencies
		angles = np.linspace(0.0, 2.0 * np.pi, 1_000)
		values = np.zeros_like(angles)
		for index, frequency in enumerate(frequencies):
			values += (
				a_coeffs[index] * np.cos(frequency * angles)
				+ b_coeffs[index] * np.sin(frequency * angles)
			)
		self._species[biome] = {
			"period": period,
			"reward_repeat": self.reward_repeat,
			"a_coeffs": a_coeffs,
			"b_coeffs": b_coeffs,
			"y_min": float(values.min()),
			"y_max": float(values.max()),
			"color": self._env.rng.integers(
				0, 256, size=3, dtype=np.uint8
			),
		}

	def _raw_biome_reward(self, biome: int, clock: int) -> float:
		return _fourier_reward(self._species[biome], clock)

	def _region_cells(self, start, stop):
		return [
			(x, y)
			for x in range(start[0], stop[0])
			for y in range(start[1], stop[1])
		]

	def _clear_region(self, cells):
		for coords in cells:
			idx = nbu.ravel(coords, UNENDING_WORLD_SIZE)
			if self._env._obj_store.has_object(idx):
				self._env._obj_store.remove_object(idx)

	def _place_region(self, cells, factory, count):
		cells = [
			coords for coords in cells
			if nbu.ravel(coords, UNENDING_WORLD_SIZE)
			not in self._env._obj_store.idx_to_name
		]
		self._env.rng.shuffle(cells)
		if count > len(cells):
			raise RuntimeError("not enough free cells to populate Forager biome")
		for coords in cells[:count]:
			obj = factory()
			obj.target_location = coords
			self._env.add_object(obj)

	def _populate_biome(self, biome: int, factories):
		cells = self._region_cells(*self._biome_bounds[biome])
		self._place_region(
			cells, factories[self._mushroom_names[biome]],
			self.mushrooms_per_biome,
		)

	def _populate(self, factories):
		world_cells = self._region_cells((0, 0), UNENDING_WORLD_SIZE)
		background_walls = int(np.ceil(
			np.prod(UNENDING_WORLD_SIZE) * UNENDING_WALL_FREQUENCY
		))
		self._place_region(world_cells, factories["wall"], background_walls)

		local_walls = int(np.ceil(9 * UNENDING_LOCAL_WALL_FREQUENCY))
		for bounds in self._wall_layout():
			cells = self._region_cells(*bounds)
			# Smaller wall biomes overwrite the background biome in Foragax.
			self._clear_region(cells)
			self._place_region(cells, factories["wall"], local_walls)

		for biome in range(len(self._biome_bounds)):
			self._populate_biome(biome, factories)

	def _renew_biome(self, biome: int):
		self._sample_species(biome)
		factory = self._env._obj_store.factories[self._mushroom_names[biome]]
		cells = self._region_cells(*self._biome_bounds[biome])
		self._place_region(cells, factory, self.mushrooms_per_biome)

	def _cue(self):
		cue = np.zeros(len(self._biome_bounds), dtype=np.float32)
		if self.cue_active:
			cue[self.best_biome] = 1.0
		return cue

	def _paper_observation(self, raw, last_action: int | None, last_reward: float):
		mapping = self._env._names_to_dims
		image = np.ones(self.image_shape, dtype=np.float32)
		for biome, name in enumerate(self._mushroom_names):
			mask = raw[..., mapping[name]]
			image[mask] = self._species[biome]["color"].astype(np.float32) / 255.0
		image[raw[..., mapping["wall"]]] = 0.0
		action = np.zeros(4, dtype=np.float32)
		if last_action is not None:
			action[last_action] = 1.0
		return np.concatenate((
			image.reshape(-1),
			action,
			np.array([last_reward], dtype=np.float32),
			self._cue(),
		))

	def reset(self):
		self._build()
		return self._paper_observation(self._env.start(), None, 0.0)

	def step(self, action: int):
		raw, reward = self._env.step(action)
		obs = self._paper_observation(raw, action, reward)
		info = {
			"task_phase": self.task_phase,
			"clock": self.clock,
			"best_biome": self.best_biome,
			"cue_active": self.cue_active,
			"biome_generations": self.biome_generations,
		}
		return obs, reward, False, info

	def close(self):
		return None


class VectorizedUnendingFourBiomeForagerEnv:
	"""Synchronous independent copies of the unending Forager benchmark."""

	game_name = UnendingFourBiomeForagerEnv.game_name

	def __init__(
		self,
		num_envs: int,
		seed: int = 0,
		aperture_size: int = DEFAULT_APERTURE,
		fourier_terms: int = DEFAULT_FOURIER_TERMS,
		reward_repeat: int = DEFAULT_REWARD_REPEAT,
		consumption_threshold: int = DEFAULT_CONSUMPTION_THRESHOLD,
		cue_period: int = DEFAULT_CUE_PERIOD,
		cue_duration: int = DEFAULT_CUE_DURATION,
	):
		if num_envs <= 0:
			raise ValueError("num_envs must be positive")
		self.num_envs = num_envs
		self.envs = tuple(
			UnendingFourBiomeForagerEnv(
				seed=seed + index,
				aperture_size=aperture_size,
				fourier_terms=fourier_terms,
				reward_repeat=reward_repeat,
				consumption_threshold=consumption_threshold,
				cue_period=cue_period,
				cue_duration=cue_duration,
			)
			for index in range(num_envs)
		)
		prototype = self.envs[0]
		self.single_observation_space = prototype.observation_space
		self.single_action_space = prototype.action_space
		self.observation_space = _Space(
			(num_envs,) + self.single_observation_space.shape
		)
		self.action_space = self.single_action_space
		self.image_shape = prototype.image_shape

	def reset(self):
		return np.stack([env.reset() for env in self.envs])

	def step(self, actions):
		actions = np.asarray(actions).reshape(-1)
		if len(actions) != self.num_envs:
			raise ValueError(
				f"expected {self.num_envs} actions, got {len(actions)}"
			)
		transitions = [
			env.step(int(action)) for env, action in zip(self.envs, actions)
		]
		observations, rewards, done, infos = zip(*transitions)
		return (
			np.stack(observations),
			np.asarray(rewards, dtype=np.float32),
			np.asarray(done, dtype=np.bool_),
			infos,
		)

	def close(self):
		for env in self.envs:
			env.close()


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


def add_unending_environment_arguments(parser):
	parser.add_argument("--aperture-size", type=int, default=DEFAULT_APERTURE)
	parser.add_argument("--fourier-terms", type=int,
			default=DEFAULT_FOURIER_TERMS)
	parser.add_argument("--reward-repeat", type=int,
			default=DEFAULT_REWARD_REPEAT,
			help="environment steps represented by one Fourier-series time point")
	parser.add_argument("--consumption-threshold", type=int,
			default=DEFAULT_CONSUMPTION_THRESHOLD,
			help="consumptions before a biome's mushroom species is replaced")
	parser.add_argument("--cue-period", type=int, default=DEFAULT_CUE_PERIOD)
	parser.add_argument("--cue-duration", type=int, default=DEFAULT_CUE_DURATION)
	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--return-smoothing-window", type=int, default=10000)
	parser.add_argument("--log-frequency", type=int, default=1000)
	parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"),
			default="online")


def add_unending_forager_arguments(parser):
	add_unending_environment_arguments(parser)
	parser.add_argument("--epsilon", type=float, default=0.1)
	parser.add_argument("--replay-size", type=int, default=1000)
	parser.add_argument("--minimum-replay-history", type=int, default=50)
	parser.add_argument("--update-frequency", type=int, default=4)
	parser.add_argument("--target-update-frequency", type=int, default=128,
			help="target refresh interval measured in gradient updates")


def validate_unending_environment_arguments(args):
	if args.t_steps <= 0:
		raise ValueError("t-steps must be positive")
	if not 0.0 <= args.gamma <= 1.0:
		raise ValueError("gamma must be in [0, 1]")
	for name in ("return_smoothing_window", "log_frequency"):
		if getattr(args, name) <= 0:
			raise ValueError(f"{name.replace('_', '-')} must be positive")
	for name in ("fourier_terms", "reward_repeat", "consumption_threshold",
				 "cue_period", "cue_duration"):
		if getattr(args, name) <= 0:
			raise ValueError(f"{name.replace('_', '-')} must be positive")
	if args.cue_duration > args.cue_period:
		raise ValueError("cue-duration must not exceed cue-period")


def validate_unending_forager_arguments(args):
	validate_unending_environment_arguments(args)
	if not 0.0 <= args.epsilon <= 1.0:
		raise ValueError("epsilon must be in [0, 1]")
	if args.replay_size < args.batch_size:
		raise ValueError("replay-size must be at least batch-size")
	if not args.batch_size <= args.minimum_replay_history <= args.replay_size:
		raise ValueError(
			"minimum-replay-history must be between batch-size and replay-size"
		)
	for name in ("update_frequency", "target_update_frequency"):
		if getattr(args, name) <= 0:
			raise ValueError(f"{name.replace('_', '-')} must be positive")


def make_unending_forager_env(args):
	return UnendingFourBiomeForagerEnv(
		seed=args.seed,
		aperture_size=args.aperture_size,
		fourier_terms=args.fourier_terms,
		reward_repeat=args.reward_repeat,
		consumption_threshold=args.consumption_threshold,
		cue_period=args.cue_period,
		cue_duration=args.cue_duration,
	)


def make_vectorized_unending_forager_env(args):
	return VectorizedUnendingFourBiomeForagerEnv(
		num_envs=args.num_envs,
		seed=args.seed,
		aperture_size=args.aperture_size,
		fourier_terms=args.fourier_terms,
		reward_repeat=args.reward_repeat,
		consumption_threshold=args.consumption_threshold,
		cue_period=args.cue_period,
		cue_duration=args.cue_duration,
	)
