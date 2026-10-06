"""Task-boundary-free online PQN with GAC on continual Procgen.

PQN remains replay-free for TD learning.  Its normal interaction stream is
split into task-agnostic update windows.  A uniform state sample from each
window is labelled by the current Q-network and hypergeometrically merged into
the lifetime GAC memory.  No task switch triggers consolidation.
"""

from argparse import ArgumentParser
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from config import epsilon_at
from diagnostics import ContinualDiagnostics
from model import parameter_count
from pqn import (
	PQNAgent,
	RolloutStateSampler,
	add_pqn_arguments,
	channels_first_uint8,
	resolve_pqn_args,
)
from pqn_auxiliary import StageStateReservoir
from procgen_env import make_training_env
from utils import (
	EpisodeReturnTracker,
	describe_protocol,
	prepare_output_directories,
	seed_everything,
	serializable_config,
)


class AuxiliaryMemory:
	"""Lifetime reservoir of online greedy-action constraints."""

	def __init__(self, max_size, batch_size, num_actions, device, seed):
		self.max_size = int(max_size)
		self.batch_size = int(batch_size)
		self.num_actions = int(num_actions)
		self.device = device
		self.rng = np.random.RandomState(seed)
		self.states = np.empty(
			(self.max_size, 3, 64, 64), dtype=np.uint8
		)
		self.actions = np.empty((self.max_size, 1), dtype=np.uint8)
		self.margins = np.empty((self.max_size, 1), dtype=np.float32)
		self.length = 0
		# num_seen counts the eligible state population represented by the
		# reservoir, rather than only the selected task-local candidates.
		self.num_seen = 0

	def size(self):
		return self.length

	@torch.no_grad()
	def _label(self, states, network):
		actions = np.empty((len(states), 1), dtype=np.uint8)
		margins = np.empty((len(states), 1), dtype=np.float32)
		for start in range(0, len(states), self.batch_size):
			stop = min(start + self.batch_size, len(states))
			inputs = torch.as_tensor(
				states[start:stop], dtype=torch.uint8, device=self.device
			)
			top = network(inputs).topk(2, dim=1)
			actions[start:stop] = top.indices[:, :1].cpu().numpy()
			margins[start:stop] = (
				top.values[:, :1] - top.values[:, 1:2]
			).cpu().numpy()
		return actions, margins

	def commit(self, stage_sampler, network):
		"""Merge one completed update window and freeze selected GAC targets."""
		stage_sampler.completed_states()
		new_seen = int(stage_sampler.population_size)
		if new_seen <= 0:
			return 0

		old_seen = self.num_seen
		final_size = min(self.max_size, old_seen + new_seen)
		if old_seen == 0:
			new_count = final_size
		elif new_seen == 0:
			new_count = 0
		else:
			new_count = int(self.rng.hypergeometric(
				new_seen, old_seen, final_size
			))
		old_count = final_size - new_count
		if old_count > self.length or new_count > stage_sampler.size():
			raise RuntimeError("invalid update-window reservoir merge")

		new_indices = self.rng.choice(
			stage_sampler.size(), size=new_count, replace=False
		)

		if self.length == self.max_size:
			# At capacity, retaining a uniform subset of the old reservoir is
			# equivalent to replacing uniformly selected slots.
			slots = self.rng.choice(
				self.length, size=new_count, replace=False
			)
		else:
			# Before memory first fills, compact a uniform old subset and append
			# the selected new states.
			old_indices = np.sort(self.rng.choice(
				self.length, size=old_count, replace=False
			))
			for destination, source in enumerate(old_indices):
				if destination != source:
					self.states[destination] = self.states[source]
					self.actions[destination] = self.actions[source]
					self.margins[destination] = self.margins[source]
			slots = np.arange(old_count, final_size)

		for start in range(0, new_count, self.batch_size):
			stop = min(start + self.batch_size, new_count)
			batch_slots = slots[start:stop]
			batch_states = stage_sampler.states[new_indices[start:stop]]
			batch_actions, batch_margins = self._label(batch_states, network)
			self.states[batch_slots] = batch_states
			self.actions[batch_slots] = batch_actions
			self.margins[batch_slots] = batch_margins

		self.length = final_size
		self.num_seen += new_seen
		return new_count

	def sample(self, count=None):
		if self.length == 0:
			return None
		requested = self.batch_size if count is None else int(count)
		count = min(self.length, requested)
		indices = self.rng.choice(self.length, size=count, replace=False)
		states = torch.as_tensor(
			self.states[indices], dtype=torch.uint8, device=self.device
		)
		actions = torch.as_tensor(
			self.actions[indices], dtype=torch.long, device=self.device
		)
		margins = torch.as_tensor(
			self.margins[indices], dtype=torch.float32, device=self.device
		)
		return states, actions, margins


class PQNGACAgent(PQNAgent):
	def train(self, states, actions, targets, auxiliary_memory):
		states = states.reshape(-1, 3, 64, 64)
		actions = actions.reshape(-1)
		targets = targets.reshape(-1)
		batch_size = len(states)
		minibatch_size = min(self.args.pqn_minibatch_size, batch_size)
		indices = np.arange(batch_size)
		total_losses, td_losses, gac_losses, predicted_q = [], [], [], []
		gradient_updates = 0

		for _ in range(self.args.update_epochs):
			np.random.shuffle(indices)
			for start in range(0, batch_size, minibatch_size):
				minibatch = torch.as_tensor(
					indices[start:start + minibatch_size], device=self.device
				)
				predictions = self.online(states[minibatch]).gather(
					1, actions[minibatch, None]
				).squeeze(1)
				td_loss = F.mse_loss(predictions, targets[minibatch])

				gac_loss = torch.zeros((), device=self.device)
				if (
					auxiliary_memory.size() >= self.args.pqn_minibatch_size
					and self.args.lambda_gac > 0
				):
					revalued_states, revalued_actions, margins = (
						auxiliary_memory.sample(len(minibatch))
					)
					revalued_q = self.online(revalued_states)
					greedy_q = revalued_q.gather(1, revalued_actions)
					competitor_q = revalued_q.scatter(
						1, revalued_actions, -torch.inf
					).max(dim=1, keepdim=True).values
					gac_loss = torch.relu(
						competitor_q - greedy_q + self.args.rho * margins
					).mean()

				loss = td_loss + self.args.lambda_gac * gac_loss
				self.optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(
					self.online.parameters(), self.args.gradient_clip
				)
				self.optimizer.step()
				total_losses.append(loss.item())
				td_losses.append(td_loss.item())
				gac_losses.append(gac_loss.item())
				predicted_q.append(predictions.detach().mean().item())
				gradient_updates += 1

		return (
			float(np.mean(total_losses)),
			float(np.mean(td_losses)),
			float(np.mean(gac_losses)),
			float(np.mean(predicted_q)),
			gradient_updates,
		)


def parse_args(arguments=None):
	parser = ArgumentParser(description="PQN with online GAC on continual Procgen")
	add_pqn_arguments(parser)
	parser.add_argument(
		"--update-interval", type=int, default=None,
		help=(
			"auxiliary-memory update interval in environment transitions; "
			"default: one PQN rollout, 1 means every vector-environment step"
		),
	)
	parser.add_argument(
		"--revaluation-size", type=int, default=10_000,
		help="maximum number of retrospectively revalued decisions",
	)
	parser.add_argument(
		"--stage-sampling-tail-fraction", type=float, default=1.0,
		help=(
			"fraction of each update window eligible for auxiliary-memory "
			"sampling, counted backward from its end; 1.0 is CLEAR-like"
		),
	)
	parser.add_argument(
		"--rho", type=float, default=0.5,
		help="fraction of each historical action gap to preserve",
	)
	parser.add_argument(
		"--lambda-gac", type=float, default=1.0,
		help="GAC regularization coefficient",
	)
	args = resolve_pqn_args(parser.parse_args(arguments))
	if args.update_interval is None:
		args.update_interval = args.rollout_batch_size
	if args.update_interval <= 0:
		raise ValueError("--update-interval must be positive")
	if args.update_interval == 1:
		# One synchronous Procgen interaction yields num_envs transitions.
		args.effective_update_interval = args.num_envs
	elif args.update_interval < args.num_envs:
		raise ValueError(
			"--update-interval values below --num-envs are not representable "
			"by the synchronous vector environment; use 1 for every vector step"
		)
	elif args.update_interval % args.num_envs != 0:
		raise ValueError(
			"--num-envs must divide --update-interval (or set it to 1)"
		)
	else:
		args.effective_update_interval = args.update_interval
	if args.effective_update_interval > args.t_steps:
		raise ValueError("--update-interval cannot exceed --t-steps")
	args.auxiliary_updates_align_with_task_boundaries = (
		args.effective_update_interval == args.switch_steps
	)
	if args.revaluation_size <= 0:
		raise ValueError("--revaluation-size must be positive")
	if args.revaluation_size < args.pqn_minibatch_size:
		raise ValueError(
			"--revaluation-size must be at least one PQN minibatch "
			f"({args.pqn_minibatch_size})"
		)
	if not 0 < args.stage_sampling_tail_fraction <= 1:
		raise ValueError("--stage-sampling-tail-fraction must be in (0, 1]")
	if not 0 <= args.rho <= 1:
		raise ValueError("--rho must be in [0, 1]")
	if args.lambda_gac < 0:
		raise ValueError("--lambda-gac cannot be negative")
	return args


def main():
	args = parse_args()
	prepare_output_directories(args)
	seed_everything(args.seed)
	describe_protocol(args, uses_replay=False)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	current_stage = 0
	task_index = 0
	env = make_training_env(args, task_index)
	if env.num_actions != 15:
		raise ValueError(
			f"expected Procgen's 15-action space, got {env.num_actions}"
		)
	agent = PQNGACAgent(env.num_actions, args, device)
	print(f"PQN GAC behavior parameters: {parameter_count(agent.online):,}")
	print(
		f"PQN rollout={args.rollout_batch_size:,}, "
		f"minibatch={args.pqn_minibatch_size:,}, epochs={args.update_epochs}"
	)

	update_sampler = StageStateReservoir(
		args.revaluation_size,
		args.effective_update_interval,
		args.seed + 23,
		args.stage_sampling_tail_fraction,
	)
	auxiliary_memory = AuxiliaryMemory(
		args.revaluation_size,
		args.pqn_minibatch_size,
		env.num_actions,
		device,
		args.seed + 31,
	)
	diagnostics = ContinualDiagnostics(args)
	rollout_sampler = RolloutStateSampler(args.seed + 17)
	return_tracker = EpisodeReturnTracker(args.task_sequence, args.num_envs)
	observations = env.reset()

	state_mib = args.revaluation_size * 3 * 64 * 64 / 2**20
	candidate_mib = len(update_sampler.states) * 3 * 64 * 64 / 2**20
	print(
		f"GAC state storage reserves approximately {state_mib:.1f} MiB; "
		f"update-window candidates reserve {candidate_mib:.1f} MiB"
	)
	print(
		"Online GAC updates every "
		f"{args.effective_update_interval:,} transitions and samples the final "
		f"{args.stage_sampling_tail_fraction:.1%} of each task-agnostic window "
		f"({update_sampler.population_size:,} eligible transitions/window)"
	)
	algorithm_name = "PQN w/ Online GAC"
	filename = (
		f"PQN_Online_GAC_Procgen_{args.distribution_mode}_{args.num_tasks}tasks"
		f"_{args.cycles}cycles_switch_{args.switch_steps}"
		f"_qlambda_{args.q_lambda}_lr_{args.learning_rate}"
		f"_lambda_{args.lambda_gac}_rho_{args.rho}"
		f"_update_{args.update_interval}"
		f"_tail_{args.stage_sampling_tail_fraction:g}"
		f"_memory_{args.revaluation_size}_seed_{args.seed}"
	)
	wandb.init(
		project=args.wandb_project,
		name=filename,
		group=algorithm_name,
		mode=args.wandb_mode,
		config=serializable_config(args, algorithm_name, {
			"network_parameters": parameter_count(agent.online),
			"uses_replay": False,
			"uses_target_network": False,
			"boundary_information_used_by_auxiliary_memory": (
				args.auxiliary_updates_align_with_task_boundaries
			),
			"effective_update_interval": args.effective_update_interval,
			"revaluation_memory_state_mib": state_mib,
			"stage_candidate_state_mib": candidate_mib,
		}),
	)

	vector_steps = args.t_steps // args.num_envs
	returns_array = np.zeros(vector_steps, dtype=np.float32)
	next_action_gap = args.target_update_frequency
	next_evaluation = args.eval_frequency
	next_log = args.log_frequency
	global_steps = 0
	gradient_updates = 0
	episodes_completed = 0
	last_losses = None
	last_q = None

	try:
		while global_steps < args.t_steps:
			window_updates = 0
			window_candidates = 0
			window_new_samples = 0
			stage = global_steps // args.switch_steps
			if stage != current_stage:
				env.close()
				current_stage = stage
				task_index = stage % args.num_tasks
				env = make_training_env(args, task_index)
				observations = env.reset()
				return_tracker.switch_task()
				diagnostics.start_training_phase()

			stage_remaining = (
				(current_stage + 1) * args.switch_steps - global_steps
			) // args.num_envs
			evaluation_remaining = (
				(next_evaluation - global_steps) // args.num_envs
			)
			total_remaining = (args.t_steps - global_steps) // args.num_envs
			rollout_length = min(
				args.rollout_steps,
				stage_remaining,
				evaluation_remaining,
				total_remaining,
			)
			if rollout_length <= 0:
				raise RuntimeError("invalid zero-length PQN rollout")

			states = torch.empty(
				(rollout_length, args.num_envs, 3, 64, 64),
				dtype=torch.uint8,
				device=device,
			)
			actions = torch.empty(
				(rollout_length, args.num_envs), dtype=torch.long, device=device
			)
			rewards = torch.empty(
				(rollout_length, args.num_envs), dtype=torch.float32, device=device
			)
			done = torch.empty(
				(rollout_length, args.num_envs), dtype=torch.bool, device=device
			)
			values = torch.empty_like(rewards)

			game = args.task_sequence[task_index]
			last_epsilon = args.initial_epsilon
			for rollout_step in range(rollout_length):
				stage_steps = global_steps % args.switch_steps
				last_epsilon = epsilon_at(args, global_steps, stage_steps)
				update_sampler.add_batch(observations)
				state_tensor = channels_first_uint8(observations, device)
				selected_actions, state_values = agent.actions_and_values(
					state_tensor, last_epsilon, env.num_actions
				)
				next_observations, raw_rewards, terminal, _ = env.step(
					selected_actions
				)
				train_rewards = raw_rewards
				if args.reward_clip > 0:
					train_rewards = np.clip(
						raw_rewards, -args.reward_clip, args.reward_clip
					)

				states[rollout_step].copy_(state_tensor)
				actions[rollout_step].copy_(torch.as_tensor(
					selected_actions, device=device
				))
				rewards[rollout_step].copy_(torch.as_tensor(
					train_rewards, device=device
				))
				done[rollout_step].copy_(torch.as_tensor(
					terminal, device=device
				))
				values[rollout_step].copy_(state_values)

				observations = next_observations
				completed = return_tracker.update(game, raw_rewards, terminal)
				episodes_completed += len(completed)
				global_steps += args.num_envs
				returns_array[global_steps // args.num_envs - 1] = (
					return_tracker.ema[game]
				)
				if update_sampler.num_seen == update_sampler.stage_size:
					candidates = update_sampler.size()
					new_samples = auxiliary_memory.commit(
						update_sampler, agent.online
					)
					window_updates += 1
					window_candidates += candidates
					window_new_samples += new_samples
					update_sampler.clear()

			with torch.no_grad():
				targets = agent.lambda_returns(
					rewards,
					done,
					values,
					channels_first_uint8(observations, device),
				)
			last_total, last_td, last_gac, last_q, new_updates = agent.train(
				states, actions, targets, auxiliary_memory
			)
			last_losses = (last_total, last_td, last_gac)
			gradient_updates += new_updates
			rollout_sampler.update(states)

			metrics = {}
			if window_updates:
				metrics.update({
					"gac/update_count": window_updates,
					"gac/update_candidates": window_candidates,
					"gac/update_new_samples": window_new_samples,
				})

			action_gap_due = False
			while global_steps >= next_action_gap:
				action_gap_due = True
				next_action_gap += args.target_update_frequency
			if action_gap_due:
				metrics.update(diagnostics.action_gap_metrics(
					agent.online, rollout_sampler, device, game
				))

			while global_steps >= next_evaluation:
				completed_stages = global_steps // args.switch_steps
				metrics.update(diagnostics.evaluation_metrics(
					agent.online, device, completed_stages
				))
				next_evaluation += args.eval_frequency

			log_due = global_steps >= next_log
			while global_steps >= next_log:
				next_log += args.log_frequency
			if log_due:
				metrics.update({
					"train_return": return_tracker.ema[game],
					"episodic_return_10000": return_tracker.ema[game],
					"epsilon": last_epsilon,
					"task_index": task_index,
					"task_stage": current_stage,
					"cycle_index": current_stage // args.num_tasks,
					"gradient_updates": gradient_updates,
					"pqn/rollout_size": rollout_length * args.num_envs,
					"gac/revaluation_size": auxiliary_memory.size(),
					"gac/update_window_size": update_sampler.size(),
					"gac/eligible_states_seen": auxiliary_memory.num_seen,
					"episodes_completed": episodes_completed,
					"loss/total": last_losses[0],
					"loss/td": last_losses[1],
					"loss/gac": last_losses[2],
					"pqn/q_values": last_q,
				})
			if metrics:
				wandb.log(metrics, step=global_steps)
	finally:
		env.close()
		wandb.finish()

	if args.save_model:
		torch.save(
			agent.online.state_dict(),
			args.output_dir / "models" / f"{filename}_Net.pt",
		)
	if args.save:
		with (
			args.output_dir / "results" / f"{filename}_returns.pkl"
		).open("wb") as file:
			pickle.dump(returns_array, file)
	if args.plot:
		import matplotlib.pyplot as plt
		plt.plot(np.arange(vector_steps) * args.num_envs, returns_array)
		plt.xlabel("lifetime environment transitions")
		plt.ylabel("training return EMA")
		plt.show()


if __name__ == "__main__":
	main()
