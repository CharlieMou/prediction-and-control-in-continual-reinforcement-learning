"""State sampling and auxiliary reservoirs for replay-free PQN methods."""

import numpy as np
import torch


class StreamingAuxiliaryMemory:
	"""Reservoir states together with targets produced at interaction time.

	Every state in the behavior stream is considered exactly once by Algorithm R.
	The accompanying Q-vector is never recomputed: Value Cloning stores it
	directly, while GAC stores its greedy action and top-two action gap.
	"""

	def __init__(
		self, max_size, batch_size, num_actions, device, seed, target_type
	):
		if target_type not in ("value_cloning", "gac"):
			raise ValueError("target_type must be 'value_cloning' or 'gac'")
		self.max_size = int(max_size)
		self.batch_size = int(batch_size)
		self.num_actions = int(num_actions)
		if self.max_size <= 0 or self.batch_size <= 0 or self.num_actions <= 1:
			raise ValueError("memory, batch, and action sizes must be valid")
		self.device = device
		self.target_type = target_type
		self.rng = np.random.RandomState(seed)
		self.states = np.empty(
			(self.max_size, 3, 64, 64), dtype=np.uint8
		)
		if target_type == "value_cloning":
			self.q_values = np.empty(
				(self.max_size, self.num_actions), dtype=np.float32
			)
		else:
			self.actions = np.empty((self.max_size, 1), dtype=np.uint8)
			self.margins = np.empty((self.max_size, 1), dtype=np.float32)
		self.length = 0
		self.num_seen = 0

	def size(self):
		return self.length

	@torch.no_grad()
	def update(self, states, q_values):
		"""Consider a behavior batch and freeze only accepted state/Q pairs."""
		if (
			states.dtype != torch.uint8
			or states.ndim != 4
			or tuple(states.shape[1:]) != (3, 64, 64)
		):
			raise ValueError("expected uint8 states shaped (N, 3, 64, 64)")
		if q_values.shape != (len(states), self.num_actions):
			raise ValueError("Q-value batch does not match states/num_actions")

		# Simulate Algorithm R in stream order.  A later item in this vector
		# batch may replace a slot accepted earlier, so only materialize the
		# final write for each slot while retaining the exact sequential law.
		writes = {}
		accepted = 0
		for source in range(len(states)):
			self.num_seen += 1
			if self.length < self.max_size:
				slot = self.length
				self.length += 1
			else:
				slot = int(self.rng.randint(self.num_seen))
				if slot >= self.max_size:
					continue
			writes[slot] = source
			accepted += 1

		if not writes:
			return 0
		slots = np.fromiter(writes.keys(), dtype=np.int64)
		sources = np.fromiter(writes.values(), dtype=np.int64)
		indices = torch.as_tensor(sources, dtype=torch.long, device=states.device)
		selected_states = states.index_select(0, indices).cpu().numpy()
		selected_q = q_values.index_select(0, indices)
		self.states[slots] = selected_states
		if self.target_type == "value_cloning":
			self.q_values[slots] = selected_q.cpu().numpy()
		else:
			top = selected_q.topk(2, dim=1)
			self.actions[slots] = top.indices[:, :1].cpu().numpy()
			self.margins[slots] = (
				top.values[:, :1] - top.values[:, 1:2]
			).cpu().numpy()
		return accepted

	def sample(self, count=None):
		if self.length == 0:
			return None
		requested = self.batch_size if count is None else int(count)
		count = min(self.length, requested)
		indices = self.rng.choice(self.length, size=count, replace=False)
		states = torch.as_tensor(
			self.states[indices], dtype=torch.uint8, device=self.device
		)
		if self.target_type == "value_cloning":
			q_values = torch.as_tensor(
				self.q_values[indices], dtype=torch.float32, device=self.device
			)
			return states, q_values
		actions = torch.as_tensor(
			self.actions[indices], dtype=torch.long, device=self.device
		)
		margins = torch.as_tensor(
			self.margins[indices], dtype=torch.float32, device=self.device
		)
		return states, actions, margins


class StageStateReservoir:
	"""Uniform state reservoir over an eligible suffix of a fixed-size window.

	The states are collected from PQN's normal interaction stream but never used
	for TD learning.  ``tail_fraction=1`` samples the full window, while smaller
	values sample only the final fraction so the candidates better represent the
	policy near the end of the window.  The caller chooses whether a window is a
	known task stage (semi-continual) or a task-agnostic online update interval.
	"""

	def __init__(self, max_size, stage_size, seed, tail_fraction):
		self.max_size = int(max_size)
		self.stage_size = int(stage_size)
		self.tail_fraction = float(tail_fraction)
		if self.max_size <= 0 or self.stage_size <= 0:
			raise ValueError("max_size and stage_size must be positive")
		if not 0 < self.tail_fraction <= 1:
			raise ValueError("tail_fraction must be in (0, 1]")
		self.population_size = max(
			1, int(np.ceil(self.stage_size * self.tail_fraction))
		)
		self.population_start = self.stage_size - self.population_size
		self.candidate_size = min(self.max_size, self.population_size)
		self.rng = np.random.RandomState(seed)
		self.states = np.empty(
			(self.candidate_size, 3, 64, 64), dtype=np.uint8
		)
		self.clear()

	def add_batch(self, observations):
		"""Store any pre-sampled stream positions in this observation batch."""
		if observations.dtype != np.uint8 or observations.shape[-3:] != (64, 64, 3):
			raise ValueError("expected uint8 observations shaped (N, 64, 64, 3)")
		count = len(observations)
		stop = self.num_seen + count
		if stop > self.stage_size:
			raise RuntimeError("state sampler received more than one window")
		left = np.searchsorted(self.positions, self.num_seen, side="left")
		right = np.searchsorted(self.positions, stop, side="left")
		if right > left:
			offsets = self.positions[left:right] - self.num_seen
			selected = observations[offsets].transpose(0, 3, 1, 2)
			self.states[left:right] = selected
		self.num_seen = stop
		self.length = right
		return right - left

	def size(self):
		return self.length

	def completed_states(self):
		if self.num_seen != self.stage_size:
			raise RuntimeError("cannot consolidate an incomplete state window")
		if self.length != min(self.max_size, self.population_size):
			raise RuntimeError("window-local state sample is incomplete")
		return self.states[:self.length]

	def clear(self):
		count = self.candidate_size
		if count == self.population_size:
			self.positions = np.arange(
				self.population_start, self.stage_size, dtype=np.int64
			)
		else:
			self.positions = self.population_start + np.sort(
				self.rng.choice(self.population_size, size=count, replace=False)
			)
		self.length = 0
		self.num_seen = 0


class AuxiliaryMemory:
	"""Lifetime reservoir of frozen Value Cloning targets.

	A hypergeometric reservoir merge combines a window-local uniform sample
	with the existing lifetime memory.  This is distributionally equivalent to
	running reservoir sampling over every eligible training state, while only
	evaluating states selected for insertion with the current value function.
	"""

	def __init__(self, max_size, batch_size, num_actions, device, seed):
		self.max_size = int(max_size)
		self.batch_size = int(batch_size)
		self.num_actions = int(num_actions)
		self.device = device
		self.rng = np.random.RandomState(seed)
		self.states = np.empty(
			(self.max_size, 3, 64, 64), dtype=np.uint8
		)
		self.q_values = np.empty(
			(self.max_size, self.num_actions), dtype=np.float32
		)
		self.length = 0
		# Unlike length, num_seen counts the eligible state population represented
		# by the reservoir, not merely the selected task-local candidates.
		self.num_seen = 0

	def size(self):
		return self.length

	@torch.no_grad()
	def _label(self, states, network):
		labels = np.empty((len(states), self.num_actions), dtype=np.float32)
		for start in range(0, len(states), self.batch_size):
			stop = min(start + self.batch_size, len(states))
			inputs = torch.as_tensor(
				states[start:stop], dtype=torch.uint8, device=self.device
			)
			labels[start:stop] = network(inputs).cpu().numpy()
		return labels

	def commit(self, stage_sampler, network):
		"""Merge one completed window and freeze selected full-Q targets."""
		stage_sampler.completed_states()
		new_seen = int(stage_sampler.population_size)
		if new_seen <= 0:
			return 0

		old_seen = self.num_seen
		final_size = min(self.max_size, old_seen + new_seen)
		if old_seen == 0:
			new_count = final_size
		elif new_seen == 0:
			new_count = 0
		else:
			new_count = int(self.rng.hypergeometric(
				new_seen, old_seen, final_size
			))
		old_count = final_size - new_count
		if old_count > self.length or new_count > stage_sampler.size():
			raise RuntimeError("invalid window reservoir merge")

		new_indices = self.rng.choice(
			stage_sampler.size(), size=new_count, replace=False
		)

		if self.length == self.max_size:
			# At capacity, retaining a uniform subset of the old reservoir is
			# equivalent to replacing uniformly selected slots.
			slots = self.rng.choice(
				self.length, size=new_count, replace=False
			)
		else:
			# This branch is reached before the first time memory fills.  Compact
			# a uniform old subset in place, then append the selected new states.
			old_indices = np.sort(self.rng.choice(
				self.length, size=old_count, replace=False
			))
			for destination, source in enumerate(old_indices):
				if destination != source:
					self.states[destination] = self.states[source]
					self.q_values[destination] = self.q_values[source]
			slots = np.arange(old_count, final_size)

		# Copy and label in inference-sized chunks. Avoiding a full advanced-
		# indexing copy keeps peak memory close to the two reservoirs.
		for start in range(0, new_count, self.batch_size):
			stop = min(start + self.batch_size, new_count)
			batch_slots = slots[start:stop]
			batch_states = stage_sampler.states[new_indices[start:stop]]
			self.states[batch_slots] = batch_states
			self.q_values[batch_slots] = self._label(batch_states, network)

		self.length = final_size
		self.num_seen += new_seen
		return new_count

	def sample(self, count=None):
		if self.length == 0:
			return None
		requested = self.batch_size if count is None else int(count)
		count = min(self.length, requested)
		indices = self.rng.choice(self.length, size=count, replace=False)
		states = torch.as_tensor(
			self.states[indices], dtype=torch.uint8, device=self.device
		)
		q_values = torch.as_tensor(
			self.q_values[indices], dtype=torch.float32, device=self.device
		)
		return states, q_values
