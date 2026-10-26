"""DQN with task-boundary Greedy Action Consolidation on continual Procgen."""

from argparse import ArgumentParser
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from config import add_common_arguments, epsilon_at, resolve_and_validate
from diagnostics import ContinualDiagnostics
from model import make_dqn_network, parameter_count
from procgen_env import make_training_env
from replay import ReplayBuffer
from utils import (
	EpisodeReturnTracker,
	describe_protocol,
	observation_tensor,
	prepare_output_directories,
	seed_everything,
	serializable_config,
)


def str2bool(value):
	if isinstance(value, bool):
		return value
	if value.lower() in ("true", "1", "yes"):
		return True
	if value.lower() in ("false", "0", "no"):
		return False
	raise ValueError("align must be true or false")


class RetrospectiveRevaluation:
	"""Reservoir of states, historical greedy actions, and action gaps."""

	def __init__(self, max_size, batch_size, device, seed):
		self.max_size = int(max_size)
		self.batch_size = batch_size
		self.device = device
		self.rng = np.random.RandomState(seed)
		self.states = np.empty(
			(self.max_size, 3, 64, 64), dtype=np.uint8
		)
		self.actions = np.empty((self.max_size, 1), dtype=np.uint8)
		self.margins = np.empty((self.max_size, 1), dtype=np.float32)
		self.length = 0
		self.num_seen = 0

	@torch.no_grad()
	def revalue_replay(self, replay, online, target, align):
		"""Retrospectively convert replay states into GAC decisions."""
		replacements = {}
		for start in range(0, replay.size(), self.batch_size):
			stop = min(start + self.batch_size, replay.size())
			states = replay.states[start:stop]
			state_tensor = torch.as_tensor(
				states, device=self.device
			).float().div_(255.0)

			online_top = online(state_tensor).topk(2, dim=1)
			actions = online_top.indices[:, :1]
			margins = online_top.values[:, :1] - online_top.values[:, 1:2]
			keep = torch.ones(
				len(states), dtype=torch.bool, device=self.device
			)
			if align:
				target_top = target(state_tensor).topk(2, dim=1)
				keep = actions.squeeze(1) == target_top.indices[:, 0]
				target_margins = (
					target_top.values[:, :1] - target_top.values[:, 1:2]
				)
				margins = torch.minimum(margins, target_margins)

			kept_offsets = keep.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
			kept_actions = actions[keep].squeeze(1).cpu().numpy()
			kept_margins = margins[keep].squeeze(1).cpu().numpy()
			for offset, action, margin in zip(
				kept_offsets, kept_actions, kept_margins
			):
				self.num_seen += 1
				if self.length < self.max_size:
					slot = self.length
					self.length += 1
				else:
					slot = int(self.rng.randint(0, self.num_seen))
					if slot >= self.max_size:
						continue
				replacements[slot] = (
					start + int(offset), int(action), float(margin)
				)

		for slot, (replay_index, action, margin) in replacements.items():
			self.states[slot] = replay.states[replay_index]
			self.actions[slot, 0] = action
			self.margins[slot, 0] = margin
		return len(replacements)

	def sample(self):
		count = min(self.length, self.batch_size)
		if count == 0:
			raise RuntimeError("cannot sample an empty GAC memory")
		indices = self.rng.choice(self.length, size=count, replace=False)
		states = torch.as_tensor(
			self.states[indices], device=self.device
		).float().div_(255.0)
		actions = torch.as_tensor(
			self.actions[indices], device=self.device
		).long()
		margins = torch.as_tensor(
			self.margins[indices], device=self.device
		)
		return states, actions, margins

	def size(self):
		return self.length


class GACAgent:
	def __init__(self, num_actions, args, device):
		self.args = args
		self.device = device
		self.online = make_dqn_network(num_actions).to(device)
		self.target = make_dqn_network(num_actions).to(device)
		self.target.load_state_dict(self.online.state_dict())
		self.target.requires_grad_(False)
		self.optimizer = optim.Adam(
			self.online.parameters(), lr=args.lr1, eps=args.adam_eps
		)

	@torch.no_grad()
	def actions(self, observations, epsilon, num_actions):
		q_values = self.online(observation_tensor(observations, self.device))
		actions = q_values.argmax(dim=1).cpu().numpy()
		random_mask = np.random.random(len(actions)) < epsilon
		actions[random_mask] = np.random.randint(
			0, num_actions, size=random_mask.sum()
		)
		return actions

	def train(self, replay, revaluation):
		states, actions, next_states, rewards, done = replay.sample()
		with torch.no_grad():
			next_values = self.target(next_states).max(dim=1, keepdim=True).values
			targets = rewards + (1.0 - done) * self.args.gamma * next_values
		predictions = self.online(states).gather(1, actions)
		td_loss = F.smooth_l1_loss(predictions, targets)

		gac_loss = torch.zeros((), device=self.device)
		if revaluation.size() > 0 and self.args.lambda_gac > 0:
			revalued_states, revalued_actions, margins = revaluation.sample()
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
		return loss.item(), td_loss.item(), gac_loss.item()

	def update_target(self):
		self.target.load_state_dict(self.online.state_dict())


def parse_args():
	parser = ArgumentParser(description="DQN with GAC on continual Procgen")
	add_common_arguments(parser)
	parser.add_argument(
		"--lr1", type=float, default=2.5e-4,
		help="Adam learning rate, identical to the DQN baseline",
	)
	parser.add_argument(
		"--revaluation-size", type=int, default=10_000,
		help="maximum number of retrospectively revalued decisions",
	)
	parser.add_argument(
		"--rho", type=float, default=0.5,
		help="fraction of each historical action gap to preserve",
	)
	parser.add_argument(
		"--lambda-gac", type=float, default=1.0,
		help="GAC regularization coefficient",
	)
	parser.add_argument(
		"--align", type=str2bool, default=True,
		help="require online/target greedy-action agreement",
	)
	args = resolve_and_validate(parser.parse_args())
	if args.lr1 <= 0:
		raise ValueError("--lr1 must be positive")
	if args.revaluation_size <= 0:
		raise ValueError("--revaluation-size must be positive")
	if not 0 <= args.rho <= 1:
		raise ValueError("--rho must be in [0, 1]")
	if args.lambda_gac < 0:
		raise ValueError("--lambda-gac cannot be negative")
	return args


def main():
	args = parse_args()
	prepare_output_directories(args)
	seed_everything(args.seed)
	describe_protocol(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	current_stage = 0
	task_index = 0
	env = make_training_env(args, task_index)
	if env.num_actions != 15:
		raise ValueError(f"expected Procgen's 15-action space, got {env.num_actions}")
	agent = GACAgent(env.num_actions, args, device)
	print(f"DQN GAC behavior parameters: {parameter_count(agent.online):,}")
	revaluation_mib = args.revaluation_size * 3 * 64 * 64 / 2**20
	print(
		f"GAC state storage reserves approximately {revaluation_mib:.1f} MiB"
	)

	replay = ReplayBuffer(args.replay_size, args.batch_size, device, args.seed + 17)
	revaluation = RetrospectiveRevaluation(
		args.revaluation_size, args.batch_size, device, args.seed + 31
	)
	diagnostics = ContinualDiagnostics(args)
	return_tracker = EpisodeReturnTracker(args.task_sequence, args.num_envs)
	observations = env.reset()

	algorithm_name = "DQN w/ GAC"
	filename = (
		f"DQN_GAC_Procgen_{args.distribution_mode}_{args.num_tasks}tasks"
		f"_{args.cycles}cycles_switch_{args.switch_steps}_lr_{args.lr1}"
		f"_lambda_{args.lambda_gac}_rho_{args.rho}_align_{args.align}"
		f"_memory_{args.revaluation_size}_seed_{args.seed}"
	)
	wandb.init(
		project=args.wandb_project,
		name=filename,
		group=algorithm_name,
		mode=args.wandb_mode,
		config=serializable_config(args, algorithm_name, {
			"network_parameters": parameter_count(agent.online),
			"revaluation_memory_state_mib": revaluation_mib,
		}),
	)

	vector_steps = args.t_steps // args.num_envs
	returns_array = np.zeros(vector_steps, dtype=np.float32)
	next_update = args.warmup_steps
	next_target_update = args.target_update_frequency
	next_evaluation = args.eval_frequency
	next_log = args.log_frequency
	gradient_updates = 0
	episodes_completed = 0
	last_losses = None

	try:
		for vector_step in range(vector_steps):
			steps_before = vector_step * args.num_envs
			stage = steps_before // args.switch_steps
			if stage != current_stage:
				env.close()
				current_stage = stage
				task_index = stage % args.num_tasks
				env = make_training_env(args, task_index)
				observations = env.reset()
				return_tracker.switch_task()
				diagnostics.start_training_phase()
				if args.clear_replay_on_switch:
					replay.clear()

			game = args.task_sequence[task_index]
			stage_steps = steps_before % args.switch_steps
			epsilon = 1.0 if steps_before < args.warmup_steps else epsilon_at(
				args, steps_before, stage_steps
			)
			if steps_before < args.warmup_steps:
				actions = np.random.randint(
					0, env.num_actions, size=args.num_envs
				)
			else:
				actions = agent.actions(observations, epsilon, env.num_actions)

			next_observations, raw_rewards, done, _ = env.step(actions)
			train_rewards = raw_rewards
			if args.reward_clip > 0:
				train_rewards = np.clip(
					raw_rewards, -args.reward_clip, args.reward_clip
				)
			replay.store_batch(
				observations, actions, next_observations, train_rewards, done
			)
			observations = next_observations
			completed = return_tracker.update(game, raw_rewards, done)
			episodes_completed += len(completed)
			global_steps = steps_before + args.num_envs

			while global_steps >= next_update:
				if replay.size() >= args.batch_size:
					last_losses = agent.train(replay, revaluation)
					gradient_updates += 1
				next_update += args.update_frequency

			# Consolidate before a coincident target update so --align compares
			# the final online policy with a genuinely lagged target policy.
			if (
				global_steps % args.switch_steps == 0
				and global_steps < args.t_steps
			):
				print(
					f"GAC revaluation at step {global_steps:,}: "
					f"scanning {replay.size():,} replay states...",
					flush=True,
				)
				seen_before = revaluation.num_seen
				refreshed = revaluation.revalue_replay(
					replay, agent.online, agent.target, args.align
				)
				print(
					f"GAC revaluation complete: "
					f"{revaluation.num_seen - seen_before:,} aligned decisions, "
					f"{refreshed:,} reservoir slots refreshed, "
					f"memory={revaluation.size():,}",
					flush=True,
				)

			metrics = {}
			target_updated = False
			while global_steps >= next_target_update:
				agent.update_target()
				target_updated = True
				next_target_update += args.target_update_frequency
			if target_updated:
				metrics.update(diagnostics.action_gap_metrics(
					agent.online, replay, device, game
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
					"epsilon": epsilon,
					"task_index": task_index,
					"task_stage": current_stage,
					"cycle_index": current_stage // args.num_tasks,
					"gradient_updates": gradient_updates,
					"replay_size": replay.size(),
					"gac/revaluation_size": revaluation.size(),
					"episodes_completed": episodes_completed,
				})
				if last_losses is not None:
					total_loss, td_loss, gac_loss = last_losses
					metrics.update({
						"loss/total": total_loss,
						"loss/td": td_loss,
						"loss/gac": gac_loss,
					})
			if metrics:
				wandb.log(metrics, step=global_steps)

			returns_array[vector_step] = return_tracker.ema[game]
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
