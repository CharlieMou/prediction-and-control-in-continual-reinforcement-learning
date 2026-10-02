"""Compact uint8 replay buffers for 64x64 Procgen observations."""

import numpy as np
import torch


class ReplayBuffer:
	def __init__(self, max_size, batch_size, device, seed):
		self.max_size = int(max_size)
		self.batch_size = batch_size
		self.device = device
		self.rng = np.random.RandomState(seed)
		self.states = np.empty((self.max_size, 3, 64, 64), dtype=np.uint8)
		self.next_states = np.empty_like(self.states)
		self.actions = np.empty((self.max_size, 1), dtype=np.uint8)
		self.rewards = np.empty((self.max_size, 1), dtype=np.float32)
		self.done = np.empty((self.max_size, 1), dtype=np.uint8)
		self.position = 0
		self.length = 0

	@staticmethod
	def _channels_first(observations):
		return np.ascontiguousarray(observations.transpose(0, 3, 1, 2))

	def store_batch(self, states, actions, next_states, rewards, done):
		count = len(states)
		if count > self.max_size:
			states = states[-self.max_size:]
			actions = actions[-self.max_size:]
			next_states = next_states[-self.max_size:]
			rewards = rewards[-self.max_size:]
			done = done[-self.max_size:]
			count = self.max_size
		indices = (self.position + np.arange(count)) % self.max_size
		self.states[indices] = self._channels_first(states)
		self.next_states[indices] = self._channels_first(next_states)
		self.actions[indices, 0] = np.asarray(actions, dtype=np.uint8)
		self.rewards[indices, 0] = np.asarray(rewards, dtype=np.float32)
		self.done[indices, 0] = np.asarray(done, dtype=np.uint8)
		self.position = (self.position + count) % self.max_size
		self.length = min(self.length + count, self.max_size)

	def _state_tensor(self, array):
		return torch.as_tensor(array, device=self.device).float().div_(255.0)

	def sample(self):
		if self.length < self.batch_size:
			raise RuntimeError("not enough transitions for a replay batch")
		indices = self.rng.randint(0, self.length, size=self.batch_size)
		return (
			self._state_tensor(self.states[indices]),
			torch.as_tensor(self.actions[indices], device=self.device).long(),
			self._state_tensor(self.next_states[indices]),
			torch.as_tensor(self.rewards[indices], device=self.device),
			torch.as_tensor(self.done[indices], device=self.device).float(),
		)

	def sample_states(self, count):
		if self.length == 0:
			return None
		indices = self.rng.randint(0, self.length, size=min(count, self.length))
		return self._state_tensor(self.states[indices])

	def size(self):
		return self.length

	def clear(self):
		self.position = 0
		self.length = 0


class PermanentReplayBuffer:
	"""Recent states and pre-update permanent values used for consolidation."""

	def __init__(self, max_size):
		self.max_size = int(max_size)
		self.states = np.empty((self.max_size, 3, 64, 64), dtype=np.uint8)
		self.actions = np.empty((self.max_size, 1), dtype=np.uint8)
		self.old_values = np.empty((self.max_size, 1), dtype=np.float32)
		self.position = 0
		self.length = 0

	def store_batch(self, states, actions, old_values):
		count = len(states)
		if count > self.max_size:
			states = states[-self.max_size:]
			actions = actions[-self.max_size:]
			old_values = old_values[-self.max_size:]
			count = self.max_size
		indices = (self.position + np.arange(count)) % self.max_size
		self.states[indices] = np.ascontiguousarray(states.transpose(0, 3, 1, 2))
		self.actions[indices, 0] = np.asarray(actions, dtype=np.uint8)
		self.old_values[indices, 0] = np.asarray(old_values, dtype=np.float32)
		self.position = (self.position + count) % self.max_size
		self.length = min(self.length + count, self.max_size)

	def batches(self, batch_size, device):
		usable = self.length - self.length % batch_size
		if usable == 0:
			return
		start = (self.position - self.length) % self.max_size
		ordered = (start + np.arange(usable)) % self.max_size
		for offset in range(0, usable, batch_size):
			indices = ordered[offset:offset + batch_size]
			yield (
				torch.as_tensor(self.states[indices], device=device).float().div_(255.0),
				torch.as_tensor(self.actions[indices], device=device).long(),
				torch.as_tensor(self.old_values[indices], device=device),
			)

	def size(self):
		return self.length

	def clear(self):
		self.position = 0
		self.length = 0
