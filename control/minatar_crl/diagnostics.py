import random
from collections import deque

import numpy as np
import torch

from CL_envs_fixed import TASK_SEQUENCE
from gym_wrapper import BaseEnv


class ContinualDiagnostics:
	"""Task performance, forgetting, action-gap, and overestimation diagnostics."""

	def __init__(self, seed, gamma, eval_episodes, eval_max_steps,
				eval_epsilon, probe_size=2048):
		self.seed = seed
		self.gamma = gamma
		self.eval_episodes = eval_episodes
		self.eval_max_steps = eval_max_steps
		self.eval_epsilon = eval_epsilon
		self.probe_size = probe_size
		self.best_performance = {game: None for game in TASK_SEQUENCE}
		self.references = {}
		self.last_completed_stages = 0
		self.gap_rng = random.Random(seed + 200000)
		self.start_training_phase()

	def start_training_phase(self):
		self.phase_gap_sum = 0.0
		self.phase_gap_count = 0
		self.recent_average_q = deque(maxlen=1000)

	@staticmethod
	def _q_values(net, observations, device):
		states = np.stack([np.moveaxis(obs, 2, 0) for obs in observations])
		states = torch.tensor(states, dtype=torch.float, device=device).contiguous()
		return net(states)

	@torch.no_grad()
	def action_gap_metrics(self, net, replay, device, game):
		if replay.size() == 0:
			return {}
		batch_size = min(replay.batch_size, replay.size())
		batch = self.gap_rng.sample(list(replay.memory), batch_size)
		states = torch.stack([transition[0] for transition in batch]).to(device)
		q_values = net(states)
		top_two = q_values.topk(2, dim=1).values
		mean_gap = (top_two[:, 0] - top_two[:, 1]).mean().item()
		mean_q = q_values.mean().item()
		self.phase_gap_sum += mean_gap
		self.phase_gap_count += 1
		self.recent_average_q.append(mean_q)
		average_gap = self.phase_gap_sum / self.phase_gap_count
		average_q = sum(self.recent_average_q) / len(self.recent_average_q)
		return {
			f"action_gap/{game}": average_gap,
			f"average_q/{game}": average_q,
			f"relative_action_gap/{game}": average_gap / (abs(average_q) + 1e-8),
		}

	@torch.no_grad()
	def _evaluate_game(self, net, device, game, game_index):
		episode_returns = []
		overestimation_errors = []
		probe_states, probe_actions, probe_q = [], [], []
		probe_seen = 0
		probe_rng = random.Random(self.seed + 400000 + game_index)

		for episode in range(self.eval_episodes):
			eval_seed = self.seed + 100000 + 1000 * game_index + episode
			env = BaseEnv(game, seed=eval_seed, use_minimal_action_set=False,
						use_minimal_observation=False)
			env.action_space.seed(eval_seed)
			action_rng = np.random.RandomState(eval_seed)
			obs = env.reset()
			rewards, predictions = [], []
			episode_return = 0.0
			done = False

			for _ in range(self.eval_max_steps):
				q_values = self._q_values(net, [obs], device)[0]
				greedy_action = q_values.argmax().item()
				action = env.action_space.sample() if action_rng.random_sample() < self.eval_epsilon \
						else greedy_action
				predictions.append(q_values[action].item())

				probe_seen += 1
				probe = (obs.copy(), greedy_action, q_values[greedy_action].item())
				if len(probe_states) < self.probe_size:
					probe_states.append(probe[0])
					probe_actions.append(probe[1])
					probe_q.append(probe[2])
				else:
					replace = probe_rng.randrange(probe_seen)
					if replace < self.probe_size:
						probe_states[replace], probe_actions[replace], probe_q[replace] = probe

				obs, reward, done, _ = env.step(action)
				rewards.append(reward)
				episode_return += reward
				if done:
					break

			bootstrap = 0.0
			if not done:
				bootstrap = self._q_values(net, [obs], device).max().item()
			return_to_go = bootstrap
			empirical_returns = [0.0] * len(rewards)
			for index in range(len(rewards) - 1, -1, -1):
				return_to_go = rewards[index] + self.gamma * return_to_go
				empirical_returns[index] = return_to_go
			overestimation_errors.extend(
				prediction - empirical_return
				for prediction, empirical_return in zip(predictions, empirical_returns)
			)
			episode_returns.append(episode_return)
			env.close()

		probe = (
			np.asarray(probe_states, dtype=np.bool_),
			np.asarray(probe_actions, dtype=np.int64),
			np.asarray(probe_q, dtype=np.float32),
		)
		return float(np.mean(episode_returns)), float(np.mean(overestimation_errors)), probe

	@torch.no_grad()
	def _reference_forgetting(self, net, device, game):
		states, old_actions, old_q = self.references[game]
		q_values = self._q_values(net, states, device)
		old_actions_tensor = torch.tensor(old_actions, dtype=torch.long, device=device)
		indices = torch.arange(len(old_actions), device=device)
		current_old_action_q = q_values[indices, old_actions_tensor].cpu().numpy()
		policy_forgetting = (q_values.argmax(dim=1).cpu().numpy() != old_actions).mean()
		q_forgetting = np.mean((current_old_action_q - old_q) ** 2)
		return float(policy_forgetting), float(q_forgetting)

	@torch.no_grad()
	def evaluation_metrics(self, net, device, completed_stages):
		was_training = net.training
		net.eval()
		performances, probes = {}, {}
		metrics = {}

		for game_index, game in enumerate(TASK_SEQUENCE):
			performance, overestimation, probe = self._evaluate_game(
				net, device, game, game_index
			)
			performances[game] = performance
			probes[game] = probe
			metrics[f"performance/{game}"] = performance
			metrics[f"overestimation/{game}"] = overestimation

		learned_tasks = set(TASK_SEQUENCE[index % len(TASK_SEQUENCE)]
				for index in range(completed_stages))
		return_forgetting = []
		policy_forgetting = []
		q_forgetting = []
		for game in learned_tasks:
			best = self.best_performance[game]
			forgetting = 0.0 if best is None else best - performances[game]
			metrics[f"forgetting/{game}"] = forgetting
			return_forgetting.append(forgetting)
			self.best_performance[game] = performances[game] if best is None \
					else max(best, performances[game])

			if game in self.references:
				policy_cf, q_cf = self._reference_forgetting(net, device, game)
				metrics[f"policy_forgetting/{game}"] = policy_cf
				metrics[f"q_forgetting/{game}"] = q_cf
				policy_forgetting.append(policy_cf)
				q_forgetting.append(q_cf)

		if len(return_forgetting) == len(TASK_SEQUENCE):
			metrics["forgetting/average"] = float(np.mean(return_forgetting))
		if len(policy_forgetting) == len(TASK_SEQUENCE):
			metrics["policy_forgetting/average"] = float(np.mean(policy_forgetting))
		if len(q_forgetting) == len(TASK_SEQUENCE):
			metrics["q_forgetting/average"] = float(np.mean(q_forgetting))

		if completed_stages > self.last_completed_stages:
			completed_game = TASK_SEQUENCE[(completed_stages - 1) % len(TASK_SEQUENCE)]
			self.references[completed_game] = probes[completed_game]
			self.last_completed_stages = completed_stages

		if was_training:
			net.train()
		return metrics
