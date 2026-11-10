"""Empirical diagnostics for GAC behavioral-forgetting and return bounds."""

from dataclasses import dataclass

import numpy as np
import torch
import wandb

from gym_wrapper import BaseEnv


RETURN_EVALUATION_EPISODES = 100
REWARD_BOUNDS = {"breakout": 1.0, "freeway": 1.0, "space_invaders": 1.0}


@dataclass(frozen=True)
class HistoricalReference:
	states: np.ndarray
	actions: np.ndarray
	gaps: np.ndarray
	deltas: np.ndarray
	ecdf: np.ndarray
	discounted_return: float


def empirical_bound_infimum(gaps, regularizer, rho):
	"""Compute the exact empirical infimum, including ECDF left limits."""
	if rho <= 0:
		return float("nan"), float("nan"), "undefined"

	sorted_gaps = np.sort(np.asarray(gaps, dtype=np.float64))
	positive_gaps = np.unique(sorted_gaps[sorted_gaps > 0])
	if positive_gaps.size == 0:
		return 1.0, float("inf"), "infinity"

	penalty = regularizer / rho
	left_ecdf = np.searchsorted(
		sorted_gaps, positive_gaps, side="left") / sorted_gaps.size
	right_ecdf = np.searchsorted(
		sorted_gaps, positive_gaps, side="right") / sorted_gaps.size
	values = np.concatenate((
		left_ecdf + penalty / positive_gaps,
		right_ecdf + penalty / positive_gaps,
		[1.0],
	))
	best = int(values.argmin())
	if best == 2 * positive_gaps.size:
		return 1.0, float("inf"), "infinity"
	side = "left_limit" if best < positive_gaps.size else "breakpoint"
	return float(values[best]), float(positive_gaps[best % positive_gaps.size]), side


class Theorem1Analyzer:
	"""Maintain fixed historical references and log Theorem 1 diagnostics."""

	def __init__(self, seed, gamma, rho, device, task_sequence, switch_steps,
			eval_max_steps, reference_size=10000,
			evaluation_start=3000000, evaluation_interval=250000,
			threshold_count=256, q_batch_size=1024):
		self.seed = seed
		self.gamma = gamma
		self.rho = rho
		self.device = device
		self.games = tuple(dict.fromkeys(task_sequence))
		self.switch_steps = switch_steps
		self.eval_max_steps = eval_max_steps
		self.reference_size = reference_size
		self.evaluation_start = evaluation_start
		self.evaluation_interval = evaluation_interval
		self.threshold_count = threshold_count
		self.q_batch_size = q_batch_size
		self.return_scales = {
			game: 2 * REWARD_BOUNDS[game] / (1 - gamma) ** 2
			for game in self.games}
		self.references = {}
		wandb.config.update({
			"theorem1_reference_size": reference_size,
			"theorem1_evaluation_start": evaluation_start,
			"theorem1_evaluation_interval": evaluation_interval,
			"theorem1_threshold_count": threshold_count,
			"theorem1_reward_bounds": REWARD_BOUNDS,
			"theorem1_return_evaluation_episodes": RETURN_EVALUATION_EPISODES,
		})

		self.gap_table = wandb.Table(
			columns=["game", "state_index", "historical_gap"],
			log_mode="INCREMENTAL")
		self.ecdf_table = wandb.Table(
			columns=["game", "delta", "empirical_cdf"],
			log_mode="INCREMENTAL")
		self.summary_table = wandb.Table(columns=[
			"step", "game", "regularizer", "behavioral_forgetting",
			"bound_star", "delta_star", "delta_star_side",
			"historical_return", "current_return", "return_discrepancy",
			"return_discrepancy_over_C", "return_bound_star",
		], log_mode="INCREMENTAL")
		self.bound_tables = {
			game: wandb.Table(columns=[
				"step", "delta", "empirical_cdf", "regularizer",
				"empirical_bound", "evaluation",
			], log_mode="INCREMENTAL")
			for game in self.games
		}
		self.tightness_tables = {
			game: wandb.Table(
				columns=["step", "metric", "value"],
				log_mode="INCREMENTAL")
			for game in self.games
		}

	@staticmethod
	def _state_tensor(observations, device):
		states = np.moveaxis(np.asarray(observations), -1, 1)
		return torch.as_tensor(states, dtype=torch.float32, device=device)

	def _q_values(self, net, states):
		outputs = []
		for start in range(0, len(states), self.q_batch_size):
			batch = self._state_tensor(
				states[start:start + self.q_batch_size], self.device)
			outputs.append(net(batch).cpu().numpy())
		return np.concatenate(outputs)

	def _greedy_action(self, net, observation):
		state = self._state_tensor([observation], self.device)
		return int(net(state).argmax(dim=1).item())

	def _sample_discounted_occupancy(self, net, game, game_index):
		"""Sample from the reset-on-terminal discounted greedy occupancy."""
		seed = self.seed + 500000 + 1000 * game_index
		env = BaseEnv(game, seed=seed, use_minimal_action_set=False,
				use_minimal_observation=False)
		rng = np.random.RandomState(seed)
		states = []
		for _ in range(self.reference_size):
			observation = env.reset()
			horizon = rng.geometric(1 - self.gamma) - 1
			for _ in range(horizon):
				action = self._greedy_action(net, observation)
				observation, _, done, _ = env.step(action)
				if done:
					observation = env.reset()
			states.append(observation.copy())
		env.close()
		return np.asarray(states, dtype=np.bool_)

	def _discounted_return(self, net, game, game_index):
		returns = []
		for episode in range(RETURN_EVALUATION_EPISODES):
			seed = self.seed + 700000 + 1000 * game_index + episode
			env = BaseEnv(game, seed=seed, use_minimal_action_set=False,
					use_minimal_observation=False)
			observation = env.reset()
			discount = 1.0
			total = 0.0
			for _ in range(self.eval_max_steps):
				action = self._greedy_action(net, observation)
				observation, reward, done, _ = env.step(action)
				total += discount * reward
				discount *= self.gamma
				if done:
					break
			returns.append(total)
			env.close()
		return float(np.mean(returns))

	def _capture_reference(self, net, game):
		game_index = self.games.index(game)
		states = self._sample_discounted_occupancy(net, game, game_index)
		q_values = self._q_values(net, states)
		actions = q_values.argmax(axis=1)
		competitors = q_values.copy()
		competitors[np.arange(len(states)), actions] = -np.inf
		gaps = q_values[np.arange(len(states)), actions] - competitors.max(axis=1)

		positive = gaps[gaps > 0]
		if positive.size:
			lower = max(float(positive.min()) / 10, 1e-12)
			upper = max(float(positive.max()) * 10, lower * 10)
		else:
			lower, upper = 1e-8, 1.0
		deltas = np.geomspace(lower, upper, self.threshold_count)
		sorted_gaps = np.sort(gaps)
		ecdf = np.searchsorted(sorted_gaps, deltas, side="right") / len(gaps)
		reference = HistoricalReference(
			states, actions, gaps, deltas, ecdf,
			self._discounted_return(net, game, game_index))
		self.references[game] = reference

		for index, gap in enumerate(gaps):
			self.gap_table.add_data(game, index, float(gap))
		for delta, probability in zip(deltas, ecdf):
			self.ecdf_table.add_data(game, float(delta), float(probability))

		return {
			f"theorem1/{game}/historical_discounted_return":
				reference.discounted_return,
			"theorem1/data/historical_gaps": self.gap_table,
			"theorem1/data/historical_gap_ecdf": self.ecdf_table,
		}

	def _evaluate_reference(self, net, game, step):
		reference = self.references[game]
		q_values = self._q_values(net, reference.states)
		rows = np.arange(len(reference.states))
		competitors = q_values.copy()
		competitors[rows, reference.actions] = -np.inf
		continual_gaps = (
			q_values[rows, reference.actions] - competitors.max(axis=1))
		violations = np.maximum(
			self.rho * reference.gaps - continual_gaps, 0.0)
		regularizer = float(violations.mean())
		forgetting = float((q_values.argmax(axis=1) != reference.actions).mean())
		bound_star, delta_star, delta_side = empirical_bound_infimum(
			reference.gaps, regularizer, self.rho)
		bounds = reference.ecdf + regularizer / (self.rho * reference.deltas) \
				if self.rho > 0 else np.full_like(reference.deltas, np.nan)

		game_index = self.games.index(game)
		current_return = self._discounted_return(net, game, game_index)
		return_discrepancy = abs(
			current_return - reference.discounted_return)
		return_scale = self.return_scales[game]
		normalized_discrepancy = return_discrepancy / return_scale
		finite_delta_star = delta_star if np.isfinite(delta_star) else None

		for delta, probability, bound in zip(
				reference.deltas, reference.ecdf, bounds):
			self.bound_tables[game].add_data(
				step, float(delta), float(probability), regularizer,
				float(bound), str(step))
		self.summary_table.add_data(
			step, game, regularizer, forgetting, bound_star,
			finite_delta_star, delta_side, reference.discounted_return,
			current_return, return_discrepancy, normalized_discrepancy,
			return_scale * bound_star)
		for metric, value in (
				("behavioral_forgetting", forgetting),
				("empirical_bound_star", bound_star),
				("return_discrepancy_over_C", normalized_discrepancy)):
			self.tightness_tables[game].add_data(step, metric, value)

		prefix = f"theorem1/{game}"
		metrics = {
			f"{prefix}/regularizer": regularizer,
			f"{prefix}/behavioral_forgetting": forgetting,
			f"{prefix}/bound_star": bound_star,
			f"{prefix}/current_discounted_return": current_return,
			f"{prefix}/return_discrepancy": return_discrepancy,
			f"{prefix}/return_discrepancy_over_C": normalized_discrepancy,
			f"{prefix}/return_bound_star": return_scale * bound_star,
			f"{prefix}/delta_star_at_infinity": float(delta_side == "infinity"),
			"theorem1/data/summary": self.summary_table,
			f"theorem1/data/bound/{game}": self.bound_tables[game],
			f"theorem1/data/tightness/{game}": self.tightness_tables[game],
		}
		if finite_delta_star is not None:
			metrics[f"{prefix}/delta_star"] = finite_delta_star
		return metrics, (regularizer, forgetting, bound_star,
				normalized_discrepancy)

	@torch.no_grad()
	def update(self, step, net, game):
		"""Capture first-visit references and evaluate on scheduled steps."""
		should_capture = step % self.switch_steps == 0 and game not in self.references
		should_evaluate = (
			len(self.references) + int(should_capture) == len(self.games)
			and step >= self.evaluation_start
			and (step - self.evaluation_start) % self.evaluation_interval == 0)
		if not should_capture and not should_evaluate:
			return {}

		was_training = net.training
		net.eval()
		metrics = {}
		try:
			if should_capture:
				metrics.update(self._capture_reference(net, game))

			if should_evaluate:
				summaries = []
				for historical_game in self.games:
					game_metrics, summary = self._evaluate_reference(
						net, historical_game, step)
					metrics.update(game_metrics)
					summaries.append(summary)
				values = np.asarray(summaries, dtype=np.float64)
				for index, name in enumerate((
						"regularizer", "behavioral_forgetting", "bound_star",
						"return_discrepancy_over_C")):
					if not np.isnan(values[:, index]).all():
						metrics[f"theorem1/average/{name}"] = float(
							np.nanmean(values[:, index]))
		finally:
			if was_training:
				net.train()
		return metrics


class AuxiliaryMemoryBoundAnalyzer:
	"""Evaluate the empirical forgetting bound on the GAC memory mixture."""

	def __init__(self, rho, device, evaluation_start=3000000,
			evaluation_interval=1000000, frequent_log_interval=250000,
			threshold_count=256, q_batch_size=1024):
		self.rho = rho
		self.device = device
		self.evaluation_start = evaluation_start
		self.evaluation_interval = evaluation_interval
		self.frequent_log_interval = frequent_log_interval
		self.threshold_count = threshold_count
		self.q_batch_size = q_batch_size
		wandb.config.update({
			"empirical_bound_distribution": "uniform_gac_auxiliary_memory",
			"empirical_bound_delta_selection": "finite_log_grid_only",
			"empirical_bound_evaluation_start": evaluation_start,
			"empirical_bound_evaluation_interval": evaluation_interval,
			"empirical_bound_frequent_log_interval": frequent_log_interval,
			"empirical_bound_threshold_count": threshold_count,
		})

		self.gap_table = wandb.Table(columns=[
			"step", "state_index", "historical_gap",
		], log_mode="INCREMENTAL")
		self.bound_table = wandb.Table(columns=[
			"step", "delta", "empirical_cdf", "regularizer",
			"empirical_bound", "evaluation",
		], log_mode="INCREMENTAL")
		self.summary_table = wandb.Table(columns=[
			"step", "memory_size", "regularizer", "behavioral_forgetting",
			"bound_star", "delta_star", "delta_star_side", "bound_slack",
		], log_mode="INCREMENTAL")

	def _q_values(self, net, memory):
		outputs = []
		for start in range(0, len(memory), self.q_batch_size):
			batch = torch.stack([
				state for state, _, _ in
				memory[start:start + self.q_batch_size]
			]).to(self.device)
			outputs.append(net(batch).cpu().numpy())
		return np.concatenate(outputs)

	@torch.no_grad()
	def update(self, step, net, auxiliary_memory):
		"""Evaluate Lemma D.1.1 under the current empirical memory measure."""
		should_log_frequent = (
			step >= self.evaluation_start
			and (step - self.evaluation_start) % self.frequent_log_interval == 0)
		should_log_full = (
			step >= self.evaluation_start
			and (step - self.evaluation_start) % self.evaluation_interval == 0)
		if (not should_log_frequent and not should_log_full) \
				or auxiliary_memory.size() == 0:
			return {}

		was_training = net.training
		net.eval()
		try:
			memory = auxiliary_memory.memory
			q_values = self._q_values(net, memory)
			actions = torch.stack([
				action for _, action, _ in memory
			]).view(-1).numpy()
			gaps = torch.stack([
				gap for _, _, gap in memory
			]).view(-1).numpy()

			rows = np.arange(len(memory))
			competitors = q_values.copy()
			competitors[rows, actions] = -np.inf
			continual_gaps = (
				q_values[rows, actions] - competitors.max(axis=1))
			violations = np.maximum(
				self.rho * gaps - continual_gaps, 0.0)
			regularizer = float(violations.mean())
			forgetting = float((q_values.argmax(axis=1) != actions).mean())

			positive = gaps[gaps > 0]
			if positive.size:
				lower = max(float(positive.min()) / 10, 1e-12)
				upper = max(float(positive.max()) * 10, lower * 10)
			else:
				lower, upper = 1e-8, 1.0
			deltas = np.geomspace(lower, upper, self.threshold_count)
			sorted_gaps = np.sort(gaps)
			ecdf = np.searchsorted(
				sorted_gaps, deltas, side="right") / len(gaps)
			bounds = ecdf + regularizer / (self.rho * deltas) \
				if self.rho > 0 else np.full_like(deltas, np.nan)
			if self.rho > 0:
				best = int(np.argmin(bounds))
				bound_star = float(bounds[best])
				delta_star = float(deltas[best])
				delta_side = "finite_grid"
			else:
				bound_star = float("nan")
				delta_star = float("nan")
				delta_side = "undefined"
			finite_delta_star = delta_star \
				if np.isfinite(delta_star) else None
			bound_slack = bound_star - forgetting

			prefix = "theorem1/memory"
			metrics = {
				f"{prefix}/behavioral_forgetting": forgetting,
				f"{prefix}/bound_star": bound_star,
			}
			if not should_log_full:
				return metrics

			for index, gap in enumerate(gaps):
				self.gap_table.add_data(step, index, float(gap))
			for delta, probability, bound in zip(deltas, ecdf, bounds):
				self.bound_table.add_data(
					step, float(delta), float(probability), regularizer,
					float(bound), str(step))
			self.summary_table.add_data(
				step, len(memory), regularizer, forgetting, bound_star,
				finite_delta_star, delta_side, bound_slack)

			metrics.update({
				f"{prefix}/regularizer": regularizer,
				f"{prefix}/bound_slack": bound_slack,
				f"{prefix}/delta_star_at_infinity":
					float(delta_side == "infinity"),
				f"{prefix}/memory_size": len(memory),
				"theorem1/data/memory_gaps": self.gap_table,
				"theorem1/data/memory_bound": self.bound_table,
				"theorem1/data/memory_summary": self.summary_table,
			})
			if finite_delta_star is not None:
				metrics[f"{prefix}/delta_star"] = finite_delta_star
			return metrics
		finally:
			if was_training:
				net.train()
