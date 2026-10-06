"""IMPALA-style value networks used by the Procgen baselines."""

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


class LayerNormResidualBlock(nn.Module):
	"""IMPALA residual block with the per-feature LayerNorm used by PQN."""

	def __init__(self, channels, spatial_size):
		super().__init__()
		shape = (channels, spatial_size, spatial_size)
		self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.norm1 = nn.LayerNorm(shape)
		self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
		self.norm2 = nn.LayerNorm(shape)

	def forward(self, inputs):
		x = self.norm1(self.conv1(F.relu(inputs)))
		x = self.norm2(self.conv2(F.relu(x)))
		return inputs + x


class LayerNormImpalaStage(nn.Module):
	def __init__(self, in_channels, out_channels, spatial_size):
		super().__init__()
		pooled_size = (spatial_size + 1) // 2
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
		self.norm = nn.LayerNorm((out_channels, spatial_size, spatial_size))
		self.residual1 = LayerNormResidualBlock(out_channels, pooled_size)
		self.residual2 = LayerNormResidualBlock(out_channels, pooled_size)

	def forward(self, inputs):
		x = self.norm(self.conv(inputs))
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


class ProcgenPQNNetwork(nn.Module):
	"""Layer-normalized IMPALA network adapted from CleanRL's Atari PQN."""

	def __init__(self, num_actions, channels=DQN_CHANNELS,
			hidden_size=DQN_HIDDEN_SIZE):
		super().__init__()
		self.features = nn.Sequential(
			LayerNormImpalaStage(3, channels[0], 64),
			LayerNormImpalaStage(channels[0], channels[1], 32),
			LayerNormImpalaStage(channels[1], channels[2], 16),
		)
		self.hidden = nn.Linear(8 * 8 * channels[-1], hidden_size)
		self.hidden_norm = nn.LayerNorm(hidden_size)
		self.output = nn.Linear(hidden_size, num_actions)
		self.apply(self._initialize)

	@staticmethod
	def _initialize(module):
		if isinstance(module, (nn.Conv2d, nn.Linear)):
			nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
			if module.bias is not None:
				nn.init.zeros_(module.bias)

	def forward(self, inputs):
		# Training rollouts remain uint8 to keep PQN's on-policy buffer compact;
		# shared diagnostics already provide normalized floating-point tensors.
		if not torch.is_floating_point(inputs):
			inputs = inputs.float().div(255.0)
		x = self.features(inputs)
		x = F.relu(x.flatten(start_dim=1))
		x = F.relu(self.hidden_norm(self.hidden(x)))
		return self.output(x)


def make_dqn_network(num_actions):
	return ProcgenQNetwork(num_actions)


def make_pqn_network(num_actions):
	return ProcgenPQNNetwork(num_actions)


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
