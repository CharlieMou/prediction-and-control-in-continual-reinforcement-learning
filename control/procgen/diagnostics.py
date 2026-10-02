"""Continual evaluation and forgetting diagnostics on fixed held-out levels."""

import random
from collections import deque

import numpy as np
import torch

from procgen_env import make_evaluation_env, make_training_evaluation_env


PROCGEN_RETURN_BOUNDS = {
	"hard": {
		"coinrun": (5, 10), "starpilot": (1.5, 35),
		"caveflyer": (2, 13.4), "dodgeball": (1.5, 19),
		"fruitbot": (-0.5, 27.2), "chaser": (0.5, 14.2),
		"miner": (1.5, 20), "jumper": (1, 10), "leaper": (1.5, 10),
		"maze": (4, 10), "bigfish": (0, 40), "heist": (2, 10),
		"climber": (1, 12.6), "plunder": (3, 30), "ninja": (2, 10),
		"bossfight": (0.5, 13),
	},
	"easy": {
		"coinrun": (5, 10), "starpilot": (2.5, 64),
		"caveflyer": (3.5, 12), "dodgeball": (1.5, 19),
		"fruitbot": (-1.5, 32.4), "chaser": (0.5, 13),
		"miner": (1.5, 13), "jumper": (3, 10), "leaper": (3, 10),
		"maze": (5, 10), "bigfish": (1, 40), "heist": (3.5, 10),
		"climber": (2, 12.6), "plunder": (4.5, 30),
		"ninja": (3.5, 10), "bossfight": (0.5, 13),
	},
}


def observation_tensor(observations, device):
	array = np.ascontiguousarray(observations.transpose(0, 3, 1, 2))
	return torch.as_tensor(array, device=device).float().div_(255.0)


class ContinualDiagnostics:
	def __init__(self, args):
		self.args = args
		self.references = {}
		self.last_completed_stages = 0
		self.start_training_phase()

	def start_training_phase(self):
		self.phase_gap_sum = 0.0
		self.phase_gap_count = 0
		self.recent_average_q = deque(maxlen=1000)

	@staticmethod
	def _normalise_return(game, value, mode):
		minimum, maximum = PROCGEN_RETURN_BOUNDS[mode][game]
		return (value - minimum) / (maximum - minimum)

	@torch.no_grad()
	def _q_values(self, net, observations, device, chunk_size=256):
		outputs = []
		for start in range(0, len(observations), chunk_size):
			batch = observation_tensor(observations[start:start + chunk_size], device)
			outputs.append(net(batch))
		return torch.cat(outputs)

	@torch.no_grad()
	def action_gap_metrics(self, net, replay, device, game):
		states = replay.sample_states(self.args.batch_size)
		if states is None:
			return {}
		q_values = net(states)
		top_two = q_values.topk(2, dim=1).values
		gap = (top_two[:, 0] - top_two[:, 1]).mean().item()
		mean_q = q_values.mean().item()
		self.phase_gap_sum += gap
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
	def _evaluate_task(self, net, device, task_index,
			env_factory=make_evaluation_env):
		game = self.args.task_sequence[task_index]
		env = env_factory(self.args, task_index)
		rng = np.random.RandomState(
			self.args.level_seed + 200_003 + 1009 * task_index
		)
		probe_rng = random.Random(
			self.args.level_seed + 300_007 + 1009 * task_index
		)
		observations = env.reset()
		num_envs = env.num_envs
		returns = np.zeros(num_envs, dtype=np.float64)
		path_predictions = [[] for _ in range(num_envs)]
		path_rewards = [[] for _ in range(num_envs)]
		episode_returns = []
		overestimation = []
		probe_states, probe_actions, probe_q = [], [], []
		probe_seen = 0

		try:
			for _ in range(self.args.eval_max_steps):
				q_values = net(observation_tensor(observations, device))
				greedy_actions = q_values.argmax(dim=1).cpu().numpy()
				actions = greedy_actions.copy()
				random_mask = rng.random_sample(num_envs) < self.args.eval_epsilon
				actions[random_mask] = rng.randint(
					0, env.num_actions, size=random_mask.sum()
				)

				for actor in range(num_envs):
					probe_seen += 1
					probe = (
						observations[actor].copy(),
						int(greedy_actions[actor]),
						float(q_values[actor, greedy_actions[actor]].item()),
					)
					if len(probe_states) < self.args.probe_size:
						probe_states.append(probe[0])
						probe_actions.append(probe[1])
						probe_q.append(probe[2])
					else:
						replace = probe_rng.randrange(probe_seen)
						if replace < self.args.probe_size:
							probe_states[replace] = probe[0]
							probe_actions[replace] = probe[1]
							probe_q[replace] = probe[2]

				next_observations, rewards, done, _ = env.step(actions)
				returns += rewards
				for actor in range(num_envs):
					path_predictions[actor].append(
						float(q_values[actor, actions[actor]].item())
					)
					path_rewards[actor].append(float(rewards[actor]))
					if done[actor]:
						if len(episode_returns) < self.args.eval_episodes:
							episode_returns.append(float(returns[actor]))
							return_to_go = 0.0
							empirical = [0.0] * len(path_rewards[actor])
							for index in range(len(empirical) - 1, -1, -1):
								return_to_go = (
									path_rewards[actor][index]
									+ self.args.gamma * return_to_go
								)
								empirical[index] = return_to_go
							overestimation.extend(
								prediction - target for prediction, target in zip(
									path_predictions[actor], empirical
								)
							)
						returns[actor] = 0.0
						path_predictions[actor].clear()
						path_rewards[actor].clear()
				observations = next_observations
				if len(episode_returns) >= self.args.eval_episodes:
					break
		finally:
			env.close()

		if len(episode_returns) < self.args.eval_episodes:
			raise RuntimeError(
				f"only completed {len(episode_returns)}/{self.args.eval_episodes} "
				f"evaluation episodes for {game}; increase --eval-max-steps"
			)
		probe = (
			np.asarray(probe_states, dtype=np.uint8),
			np.asarray(probe_actions, dtype=np.int64),
			np.asarray(probe_q, dtype=np.float32),
		)
		return (
			float(np.mean(episode_returns)),
			float(np.mean(overestimation)),
			probe,
		)

	@torch.no_grad()
	def _reference_forgetting(self, net, device, game):
		states, old_actions, old_q = self.references[game]
		q_values = self._q_values(net, states, device)
		actions = torch.as_tensor(old_actions, dtype=torch.long, device=device)
		indices = torch.arange(len(actions), device=device)
		current_old_q = q_values[indices, actions].cpu().numpy()
		policy_forgetting = (
			q_values.argmax(dim=1).cpu().numpy() != old_actions
		).mean()
		q_forgetting = np.mean((current_old_q - old_q) ** 2)
		return float(policy_forgetting), float(q_forgetting)

	@torch.no_grad()
	def evaluation_metrics(self, net, device, completed_stages):
		was_training = net.training
		net.eval()
		performances, probes = {}, {}
		metrics = {}

		for task_index, game in enumerate(self.args.task_sequence):
			performance, overestimation, probe = self._evaluate_task(
				net, device, task_index
			)
			training_performance, _, _ = self._evaluate_task(
				net, device, task_index, make_training_evaluation_env
			)
			performances[game] = performance
			probes[game] = probe
			metrics[f"performance/{game}"] = performance
			metrics[f"training_performance/{game}"] = training_performance
			metrics[f"performance_normalized/{game}"] = self._normalise_return(
				game, performance, self.args.distribution_mode
			)
			metrics[f"overestimation/{game}"] = overestimation

		metrics["performance/average"] = float(np.mean(list(performances.values())))
		metrics["training_performance/average"] = float(np.mean([
			metrics[f"training_performance/{game}"]
			for game in self.args.task_sequence
		]))
		metrics["performance_normalized/average"] = float(np.mean([
			metrics[f"performance_normalized/{game}"]
			for game in self.args.task_sequence
		]))
		metrics["overestimation/average"] = float(np.mean([
			metrics[f"overestimation/{game}"] for game in self.args.task_sequence
		]))

		learned_tasks = set(
			self.args.task_sequence[index % self.args.num_tasks]
			for index in range(completed_stages)
		)
		policy_forgetting, q_forgetting = [], []
		for game in learned_tasks:
			if game in self.references:
				policy_cf, q_cf = self._reference_forgetting(net, device, game)
				metrics[f"policy_forgetting/{game}"] = policy_cf
				metrics[f"q_forgetting/{game}"] = q_cf
				policy_forgetting.append(policy_cf)
				q_forgetting.append(q_cf)

		if len(policy_forgetting) == self.args.num_tasks:
			metrics["policy_forgetting/average"] = float(np.mean(policy_forgetting))
		if len(q_forgetting) == self.args.num_tasks:
			metrics["q_forgetting/average"] = float(np.mean(q_forgetting))

		if completed_stages > self.last_completed_stages:
			completed_game = self.args.task_sequence[
				(completed_stages - 1) % self.args.num_tasks
			]
			self.references[completed_game] = probes[completed_game]
			self.last_completed_stages = completed_stages

		if was_training:
			net.train()
		return metrics
