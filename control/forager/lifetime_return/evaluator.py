"""Non-invasive historical reward-mapping evaluation and CSV logging."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from forager_env import TwoBiomeForagerEnv
from replay import observation_tensor


OUTPUT_DIR = Path(__file__).resolve().parent
METRIC_NAMES = (
	"lifetime_return/historical",
	"lifetime_return/historical_std",
	"lifetime_return/historical_average_reward",
	"lifetime_return/training_phase",
	"lifetime_return/historical_phase",
	"lifetime_return/evaluation_steps",
	"lifetime_return/evaluation_rollouts",
)


def add_lifetime_return_arguments(parser):
	parser.add_argument("--lifetime-eval-frequency", type=int, default=50_000,
		help="training steps between historical reward-mapping evaluations")
	parser.add_argument("--lifetime-eval-steps", type=int, default=10_000,
		help="continuing-environment steps in each evaluation rollout")
	parser.add_argument("--lifetime-eval-rollouts", type=int, default=5,
		help="independent fixed-mapping rollouts per evaluation point")
	parser.add_argument("--lifetime-eval-epsilon", type=float, default=0.001,
		help="epsilon used only by the historical evaluation policy")
	parser.add_argument("--lifetime-output-dir", type=Path, default=OUTPUT_DIR,
		help="directory for per-run lifetime-return CSV files")


def validate_lifetime_return_arguments(args):
	for name in (
		"lifetime_eval_frequency", "lifetime_eval_steps",
		"lifetime_eval_rollouts",
	):
		if getattr(args, name) <= 0:
			raise ValueError(f"{name.replace('_', '-')} must be positive")
	if not 0.0 <= args.lifetime_eval_epsilon <= 1.0:
		raise ValueError("lifetime-eval-epsilon must be in [0, 1]")
	if args.lifetime_eval_frequency % args.log_frequency != 0:
		raise ValueError(
			"lifetime-eval-frequency must be divisible by log-frequency"
		)


class HistoricalTaskEvaluator:
	"""Evaluate only the completed mapping opposite to the training mapping."""

	def __init__(self, args, method, series, run_filename):
		self.args = args
		self.method = method
		self.series = series
		self.current_phase = 0
		self.completed_phases = set()

		args.lifetime_output_dir.mkdir(parents=True, exist_ok=True)
		ignored = {
			"seed", "lifetime_output_dir", "wandb_mode", "save", "save_model",
			"plot",
		}
		configuration = {
			key: str(value) for key, value in vars(args).items()
			if key not in ignored
		}
		self.configuration_id = hashlib.blake2s(
			json.dumps(configuration, sort_keys=True).encode("utf-8"),
			digest_size=5,
		).hexdigest()
		self.csv_path = (
			args.lifetime_output_dir
			/ f"{run_filename}_cfg_{self.configuration_id}_lifetime_return.csv"
		)
		self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
		self._fieldnames = (
			"step", "method", "series", "configuration", "seed", *METRIC_NAMES,
		)
		self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
		self._writer.writeheader()
		self._csv_file.flush()

	def observe_phase(self, phase):
		phase = int(phase)
		if phase != self.current_phase:
			self.completed_phases.add(self.current_phase)
			self.current_phase = phase

	@torch.no_grad()
	def _evaluate_phase(self, net, device, phase):
		was_training = net.training
		net.eval()
		returns = []
		try:
			for rollout in range(self.args.lifetime_eval_rollouts):
				eval_seed = (
					self.args.seed + 1_000_000 + 10_000 * phase + rollout
				)
				env = TwoBiomeForagerEnv(
					seed=eval_seed,
					aperture_size=self.args.aperture_size,
					reward_period=self.args.reward_period,
					expiry_time=self.args.expiry_time,
					fixed_task_phase=phase,
				)
				rng = np.random.default_rng(eval_seed + 5_000_000)
				observation = env.reset()
				rollout_return = 0.0
				for _ in range(self.args.lifetime_eval_steps):
					state = observation_tensor(observation).to(device)
					q_values = net(state.unsqueeze(0))
					if rng.random() <= self.args.lifetime_eval_epsilon:
						action = int(rng.integers(env.action_space.n))
					else:
						action = q_values.argmax(dim=1).item()
					observation, reward, _, _ = env.step(action)
					rollout_return += reward
				env.close()
				returns.append(rollout_return)
		finally:
			if was_training:
				net.train()
		return np.asarray(returns, dtype=np.float64)

	def metrics(self, step, net, device):
		if step % self.args.lifetime_eval_frequency != 0:
			return {}
		historical_phase = 1 - self.current_phase
		if historical_phase not in self.completed_phases:
			return {}

		returns = self._evaluate_phase(net, device, historical_phase)
		metrics = {
			"lifetime_return/historical": float(returns.mean()),
			"lifetime_return/historical_std": float(returns.std(ddof=0)),
			"lifetime_return/historical_average_reward": float(
				returns.mean() / self.args.lifetime_eval_steps
			),
			"lifetime_return/training_phase": self.current_phase,
			"lifetime_return/historical_phase": historical_phase,
			"lifetime_return/evaluation_steps": self.args.lifetime_eval_steps,
			"lifetime_return/evaluation_rollouts": self.args.lifetime_eval_rollouts,
		}
		row = {
			"step": step,
			"method": self.method,
			"series": self.series,
			"configuration": self.configuration_id,
			"seed": self.args.seed,
			**metrics,
		}
		self._writer.writerow(row)
		self._csv_file.flush()
		return metrics

	def close(self):
		if not self._csv_file.closed:
			self._csv_file.close()
