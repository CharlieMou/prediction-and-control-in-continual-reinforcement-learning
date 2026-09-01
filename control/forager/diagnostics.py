"""Diagnostics that do not require task IDs or task boundaries."""

import random
from collections import deque

import torch


class ContinualDiagnostics:
	def __init__(self, seed):
		self.gap_rng = random.Random(seed + 200000)
		self.recent_average_q = deque(maxlen=1000)

	@torch.no_grad()
	def action_gap_metrics(self, net, replay, device):
		if replay.size() == 0:
			return {}
		batch_size = min(replay.batch_size, replay.size())
		batch = self.gap_rng.sample(list(replay.memory), batch_size)
		states = torch.stack([transition[0] for transition in batch]).to(device)
		q_values = net(states)
		top_two = q_values.topk(2, dim=1).values
		mean_gap = (top_two[:, 0] - top_two[:, 1]).mean().item()
		mean_q = q_values.mean().item()
		self.recent_average_q.append(mean_q)
		average_q = sum(self.recent_average_q) / len(self.recent_average_q)
		return {
			"action_gap/mean": mean_gap,
			"average_q/recent": average_q,
			"relative_action_gap/mean": mean_gap / (abs(average_q) + 1e-8),
		}
