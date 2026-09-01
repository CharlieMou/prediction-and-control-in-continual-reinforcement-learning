"""Networks for the Figure 5 Forager experiments."""

import math

import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	"""Flattened input and two 64-unit LayerNorm/ReLU hidden layers.

	The historical class name is retained so the three agent implementations do
	not need an algorithm-level rewrite.
	"""

	def __init__(self, input_dim, num_actions, hidden=64):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, hidden)
		self.ln1 = nn.LayerNorm(hidden)
		self.fc2 = nn.Linear(hidden, hidden)
		self.ln2 = nn.LayerNorm(hidden)
		self.output = nn.Linear(hidden, num_actions)
		self._initialize()

	def _initialize(self):
		for layer in (self.fc1, self.fc2):
			nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
			nn.init.zeros_(layer.bias)
		nn.init.trunc_normal_(
			self.output.weight, std=1.0 / math.sqrt(self.fc2.out_features)
		)
		nn.init.zeros_(self.output.bias)

	def forward(self, x):
		x = x.reshape(x.size(0), -1)
		x = F.relu(self.ln1(self.fc1(x)))
		x = F.relu(self.ln2(self.fc2(x)))
		return self.output(x)


class CNN_half(CNN):
	"""Half-width network used for each PT-DQN component."""

	def __init__(self, input_dim, num_actions):
		super().__init__(input_dim, num_actions, hidden=32)
