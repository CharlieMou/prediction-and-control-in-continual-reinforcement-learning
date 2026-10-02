"""IMPALA-style value networks used by both Procgen baselines."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
	DQN_CHANNELS,
	DQN_HIDDEN_SIZE,
	PT_CHANNELS,
	pt_hidden_size,
)


class ResidualBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

	def forward(self, inputs):
		x = self.conv1(F.relu(inputs))
		x = self.conv2(F.relu(x))
		return inputs + x


class ImpalaStage(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.residual1 = ResidualBlock(out_channels)
		self.residual2 = ResidualBlock(out_channels)

	def forward(self, inputs):
		x = self.conv(inputs)
		x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
		x = self.residual1(x)
		return self.residual2(x)


class ProcgenQNetwork(nn.Module):
	"""The DQN architecture from the off-policy Procgen study."""

	def __init__(self, num_actions, channels=DQN_CHANNELS,
			hidden_size=DQN_HIDDEN_SIZE):
		super().__init__()
		self.channels = tuple(channels)
		self.hidden_size = hidden_size
		self.features = nn.Sequential(
			ImpalaStage(3, channels[0]),
			ImpalaStage(channels[0], channels[1]),
			ImpalaStage(channels[1], channels[2]),
		)
		self.hidden = nn.Linear(8 * 8 * channels[-1], hidden_size)
		self.output = nn.Linear(hidden_size, num_actions)
		self.apply(self._initialize)

	@staticmethod
	def _initialize(module):
		if isinstance(module, (nn.Conv2d, nn.Linear)):
			nn.init.xavier_uniform_(module.weight)
			if module.bias is not None:
				nn.init.zeros_(module.bias)

	def forward(self, inputs):
		x = self.features(inputs)
		x = F.relu(x.flatten(start_dim=1))
		x = F.relu(self.hidden(x))
		return self.output(x)


def make_dqn_network(num_actions):
	return ProcgenQNetwork(num_actions)


def make_pt_network(num_actions):
	return ProcgenQNetwork(
		num_actions,
		channels=PT_CHANNELS,
		hidden_size=pt_hidden_size(num_actions),
	)


def parameter_count(network):
	return sum(parameter.numel() for parameter in network.parameters())


class CombinedQNetwork(nn.Module):
	"""Behavior value function Q = P + T used by PT-DQN."""

	def __init__(self, permanent_net, transient_net):
		super().__init__()
		self.permanent_net = permanent_net
		self.transient_net = transient_net

	def forward(self, inputs):
		return self.permanent_net(inputs) + self.transient_net(inputs)
