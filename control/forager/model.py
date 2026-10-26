"""Networks for the Figure 5 Forager experiments."""

import math

import torch
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


class UnendingCNN(nn.Module):
	"""Paper convolutional encoder plus the existing LayerNorm DQN head."""

	def __init__(
		self,
		input_dim,
		num_actions,
		image_shape,
		hidden=64,
		conv_channels=16,
	):
		super().__init__()
		self.image_shape = tuple(image_shape)
		self.image_size = math.prod(self.image_shape)
		self.scalar_size = input_dim - self.image_size
		if self.scalar_size < 0:
			raise ValueError("input_dim is smaller than the RGB image")

		channels = self.image_shape[-1]
		self.conv1 = nn.Conv2d(channels, conv_channels, kernel_size=1, stride=1)
		self.conv_ln1 = nn.LayerNorm(conv_channels)
		self.conv2 = nn.Conv2d(
			conv_channels, conv_channels, kernel_size=3, stride=1, padding=1
		)
		self.conv_ln2 = nn.LayerNorm(conv_channels)
		vision_size = self.image_shape[0] * self.image_shape[1] * conv_channels
		self.fc1 = nn.Linear(vision_size + self.scalar_size, hidden)
		self.ln1 = nn.LayerNorm(hidden)
		self.fc2 = nn.Linear(hidden, hidden)
		self.ln2 = nn.LayerNorm(hidden)
		self.output = nn.Linear(hidden, num_actions)
		self._initialize()

	def _initialize(self):
		for layer in (self.conv1, self.conv2, self.fc1, self.fc2):
			nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
			nn.init.zeros_(layer.bias)
		nn.init.trunc_normal_(
			self.output.weight, std=1.0 / math.sqrt(self.fc2.out_features)
		)
		nn.init.zeros_(self.output.bias)

	def _conv_layer(self, x, conv, layer_norm):
		x = conv(x).permute(0, 2, 3, 1)
		x = F.relu(layer_norm(x))
		return x.permute(0, 3, 1, 2)

	def forward(self, inputs):
		image = inputs[:, :self.image_size].reshape(
			inputs.size(0), *self.image_shape
		).permute(0, 3, 1, 2)
		scalars = inputs[:, self.image_size:]
		vision = self._conv_layer(image, self.conv1, self.conv_ln1)
		vision = self._conv_layer(vision, self.conv2, self.conv_ln2)
		x = torch.cat((vision.reshape(inputs.size(0), -1), scalars), dim=1)
		x = F.relu(self.ln1(self.fc1(x)))
		x = F.relu(self.ln2(self.fc2(x)))
		return self.output(x)


class UnendingCNN_half(UnendingCNN):
	"""Half-width UnendingCNN used for each PT-DQN component."""

	def __init__(self, input_dim, num_actions, image_shape):
		super().__init__(input_dim, num_actions, image_shape, hidden=32)
