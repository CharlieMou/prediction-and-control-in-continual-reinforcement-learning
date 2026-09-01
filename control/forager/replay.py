from collections import deque
import random

import numpy as np
import torch


def observation_tensor(obs):
	return torch.as_tensor(np.asarray(obs), dtype=torch.float32)


class expReplay:
	def __init__(self, batch_size, device, max_size=1000):
		self.memory = deque(maxlen=max_size)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, next_obs, reward, done):
		state = observation_tensor(obs)
		next_state = observation_tensor(next_obs)
		action = torch.tensor([action], dtype=torch.int64)
		reward = torch.tensor([reward], dtype=torch.float32)
		done = torch.tensor([float(done)], dtype=torch.float32)
		self.memory.append((state, action, next_state, reward, done))

	def sample(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		state, action, next_state, reward, done = map(torch.stack, zip(*batch))
		return (
			state.to(self.device),
			action.to(self.device),
			next_state.to(self.device),
			reward.to(self.device),
			done.to(self.device),
		)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()


class expReplay_PM:
	def __init__(self, max_size, batch_size, device):
		self.memory = deque(maxlen=max_size)
		self.batch_size = batch_size
		self.device = device

	def store(self, obs, action, val_p):
		state = observation_tensor(obs)
		action = torch.tensor([action], dtype=torch.int64)
		value = val_p.detach().cpu().item() if torch.is_tensor(val_p) else float(val_p)
		val_p = torch.tensor([value], dtype=torch.float32)
		self.memory.append((state, action, val_p))

	def sample(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		state, action, val_p = map(torch.stack, zip(*batch))
		return state.to(self.device), action.to(self.device), val_p.to(self.device)

	def size(self):
		return len(self.memory)

	def delete(self):
		self.memory.clear()
