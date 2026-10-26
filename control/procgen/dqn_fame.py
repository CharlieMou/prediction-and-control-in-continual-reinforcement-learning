"""DQN with Fast and Meta knowledge learners (FAME) on continual Procgen."""

from argparse import ArgumentParser
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
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


class MetaReplayBuffer:
	"""Fixed-size state/action replay for FAME's softmax-MLE update."""

	def __init__(self, max_size, batch_size, device, seed):
		self.max_size = int(max_size)
		self.batch_size = batch_size
		self.device = device
		self.rng = np.random.RandomState(seed)
		self.states = np.empty((self.max_size, 3, 64, 64), dtype=np.uint8)
		self.actions = np.empty((self.max_size, 1), dtype=np.uint8)
		self.position = 0
		self.length = 0

	def _store_channels_first(self, states, actions):
		count = len(states)
		if count > self.max_size:
			states = states[-self.max_size:]
			actions = actions[-self.max_size:]
			count = self.max_size
		indices = (self.position + np.arange(count)) % self.max_size
		self.states[indices] = states
		self.actions[indices, 0] = np.asarray(actions, dtype=np.uint8).reshape(-1)
		self.position = (self.position + count) % self.max_size
		self.length = min(self.length + count, self.max_size)

	def store_batch(self, observations, actions):
		states = np.ascontiguousarray(observations.transpose(0, 3, 1, 2))
		self._store_channels_first(states, actions)

	def sample(self):
		if self.length < self.batch_size:
			raise RuntimeError("not enough state-action pairs for a meta batch")
		indices = self.rng.randint(0, self.length, size=self.batch_size)
		states = torch.as_tensor(
			self.states[indices], device=self.device
		).float().div_(255.0)
		actions = torch.as_tensor(
			self.actions[indices], device=self.device
		).long()
		return states, actions

	def copy_to(self, target, chunk_size=4096):
		start = (self.position - self.length) % self.max_size
		for offset in range(0, self.length, chunk_size):
			count = min(chunk_size, self.length - offset)
			indices = (start + offset + np.arange(count)) % self.max_size
			target._store_channels_first(
				self.states[indices], self.actions[indices]
			)

	def size(self):
		return self.length

	def clear(self):
		self.position = 0
		self.length = 0


class FAMEAgent:
	def __init__(self, num_actions, args, device):
		self.args = args
		self.device = device
		self.num_actions = num_actions
		self.fast = make_dqn_network(num_actions).to(device)
		self.random = make_dqn_network(num_actions).to(device)
		self.meta = make_dqn_network(num_actions).to(device)
		self.target = make_dqn_network(num_actions).to(device)
		self.target.load_state_dict(self.fast.state_dict())
		self.target.requires_grad_(False)
		self.fast_optimizer = self._new_fast_optimizer()
		self.last_total_loss = None
		self.last_td_loss = None
		self.last_bc_loss = None
		self.last_meta_loss = None

	def _new_fast_optimizer(self):
		return optim.Adam(
			self.fast.parameters(), lr=self.args.lr1, eps=self.args.adam_eps
		)

	@torch.no_grad()
	def actions(self, observations, epsilon, learner=None):
		learner = self.fast if learner is None else learner
		q_values = learner(observation_tensor(observations, self.device))
		actions = q_values.argmax(dim=1).cpu().numpy()
		random_mask = np.random.random(len(actions)) < epsilon
		actions[random_mask] = np.random.randint(
			0, self.num_actions, size=random_mask.sum()
		)
		return actions

	def train_fast(self, replay, regularizer=None):
		states, actions, next_states, rewards, done = replay.sample()
		with torch.no_grad():
			next_values = self.target(next_states).max(dim=1, keepdim=True).values
			targets = rewards + (1.0 - done) * self.args.gamma * next_values
		predictions = self.fast(states).gather(1, actions)
		td_loss = F.smooth_l1_loss(predictions, targets)
		bc_loss = torch.zeros((), device=self.device)
		if regularizer is not None and self.args.lambda_reg > 0:
			with torch.no_grad():
				soft_targets = F.softmax(
					regularizer(states) / self.args.temperature, dim=-1
				)
			bc_loss = F.kl_div(
				F.log_softmax(self.fast(states) / self.args.temperature, dim=-1),
				soft_targets,
				reduction="batchmean",
			)
		loss = td_loss + self.args.lambda_reg * bc_loss
		self.fast_optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.fast.parameters(), self.args.gradient_clip
		)
		self.fast_optimizer.step()
		self.last_total_loss = loss.item()
		self.last_td_loss = td_loss.item()
		self.last_bc_loss = bc_loss.item()

	def train_meta(self, old_stage_count, recent_replay, meta_replay):
		if recent_replay.size() < self.args.batch_size:
			return None, 0
		if old_stage_count > 0 and meta_replay.size() < self.args.batch_size:
			return None, 0
		training_size = (
			meta_replay.size() if old_stage_count > 0 else recent_replay.size()
		)
		updates_per_epoch = training_size // self.args.batch_size - 1
		if updates_per_epoch <= 0:
			return None, 0

		optimizer = optim.Adam(self.meta.parameters(), lr=self.args.meta_lr)
		scheduler = ExponentialLR(optimizer, gamma=self.args.meta_lr_decay)
		last_loss = None
		updates = 0
		for epoch in range(self.args.epoch_meta):
			for update_index in range(updates_per_epoch):
				if old_stage_count == 0:
					states, actions = recent_replay.sample()
					loss = F.nll_loss(
						F.log_softmax(
							self.meta(states) / self.args.temperature, dim=-1
						),
						actions.view(-1),
					)
				else:
					states, actions = meta_replay.sample()
					loss = F.nll_loss(
						F.log_softmax(
							self.meta(states) / self.args.temperature, dim=-1
						),
						actions.view(-1),
					)
					if update_index % old_stage_count == 0:
						new_states, new_actions = recent_replay.sample()
						loss = loss + F.nll_loss(
							F.log_softmax(
								self.meta(new_states) / self.args.temperature,
								dim=-1,
							),
							new_actions.view(-1),
						)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				last_loss = loss.item()
				updates += 1
			if (epoch + 1) % self.args.meta_lr_decay_frequency == 0:
				scheduler.step()
		self.last_meta_loss = last_loss
		return last_loss, updates

	def select_random_learner(self):
		self.fast = self.random
		self.fast_optimizer = self._new_fast_optimizer()

	def refresh_random_learner(self):
		self.random = make_dqn_network(self.num_actions).to(self.device)

	def update_target(self):
		self.target.load_state_dict(self.fast.state_dict())

	def reset_stage_losses(self):
		self.last_total_loss = None
		self.last_td_loss = None
		self.last_bc_loss = None
		self.last_meta_loss = None


def parse_args():
	parser = ArgumentParser(description="DQN with FAME on continual Procgen")
	add_common_arguments(parser)
	parser.add_argument(
		"--lr1", type=float, default=2.5e-4,
		help="Fast learner Adam rate, identical to the DQN baseline",
	)
	parser.add_argument(
		"--meta-lr", type=float, default=2.5e-4,
		help="Meta MLE rate; follows the paper's pixel-based Atari setting",
	)
	parser.add_argument(
		"--size-fast2meta", type=int, default=None,
		help="final-stage transitions retained for Meta; default: 2%% of a stage",
	)
	parser.add_argument(
		"--size-meta", type=int, default=None,
		help="Meta replay capacity; default: the DQN replay capacity",
	)
	parser.add_argument(
		"--detection-step", type=int, default=None,
		help="transitions per candidate; default: 600 vector steps",
	)
	parser.add_argument("--epoch-meta", type=int, default=200)
	parser.add_argument(
		"--reset", type=int, choices=(0, 1), default=1,
		help="include a fresh random learner in adaptive warm-up",
	)
	parser.add_argument(
		"--warmstep", type=int, default=None,
		help="BC-regularized stage prefix; default: 10%% of a stage",
	)
	parser.add_argument("--lambda-reg", type=float, default=1.0)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--use-ttest", type=int, choices=(0, 1), default=0)
	parser.add_argument("--meta-lr-decay", type=float, default=0.95)
	parser.add_argument("--meta-lr-decay-frequency", type=int, default=2)
	args = resolve_and_validate(parser.parse_args())

	if args.size_fast2meta is None:
		args.size_fast2meta = max(
			args.num_envs,
			int(0.02 * args.switch_steps) // args.num_envs * args.num_envs,
		)
	if args.size_meta is None:
		args.size_meta = args.replay_size
	if args.detection_step is None:
		args.detection_step = 600 * args.num_envs
	if args.warmstep is None:
		args.warmstep = int(0.10 * args.switch_steps)

	positive = (
		"lr1", "meta_lr", "size_fast2meta", "size_meta", "detection_step",
		"epoch_meta", "warmstep", "temperature", "meta_lr_decay",
		"meta_lr_decay_frequency",
	)
	for name in positive:
		if getattr(args, name) <= 0:
			raise ValueError(f"--{name.replace('_', '-')} must be positive")
	if args.lambda_reg < 0:
		raise ValueError("--lambda-reg cannot be negative")
	if args.size_fast2meta > min(args.switch_steps, args.size_meta):
		raise ValueError("--size-fast2meta cannot exceed a stage or Meta replay")
	if args.warmstep > args.switch_steps:
		raise ValueError("--warmstep cannot exceed --switch-steps")
	if args.detection_step % args.num_envs != 0:
		raise ValueError("--num-envs must divide --detection-step")
	candidates = 3 if args.reset else 2
	if candidates * args.detection_step >= args.switch_steps:
		raise ValueError("adaptive warm-up rollouts must fit within one task stage")
	if not 0 < args.meta_lr_decay <= 1:
		raise ValueError("--meta-lr-decay must be in (0, 1]")
	return args


def is_better(args, samples_a, samples_b, mean_a, mean_b):
	if not args.use_ttest or len(samples_a) < 2 or len(samples_b) < 2:
		return mean_a > mean_b
	try:
		from scipy import stats
	except ImportError as error:
		raise ImportError("--use-ttest=1 requires scipy") from error
	result = stats.ttest_ind(
		samples_a, samples_b, alternative="greater", equal_var=False
	)
	return result.pvalue < 0.05


def dual_evaluation_metrics(agent, fast_diagnostics, meta_diagnostics,
		device, completed_stages):
	"""Evaluate both learners, using Fast's probes as both CF references."""
	fast_metrics = fast_diagnostics.evaluation_metrics(
		agent.fast, device, completed_stages
	)
	meta_diagnostics.references = fast_diagnostics.references
	meta_diagnostics.last_completed_stages = fast_diagnostics.last_completed_stages
	meta_metrics = meta_diagnostics.evaluation_metrics(
		agent.meta, device, completed_stages
	)

	metrics = {}
	dual_families = {
		"performance", "training_performance",
		"policy_forgetting", "q_forgetting",
	}
	for learner_name, learner_metrics in (
		("meta", meta_metrics), ("fast", fast_metrics)
	):
		for key, value in learner_metrics.items():
			family, separator, metric_name = key.partition("/")
			if separator and family in dual_families:
				metrics[f"{family}_{learner_name}/{metric_name}"] = value

	# Keep FAME on the existing baseline panels. The paper evaluates retained
	# performance and policy knowledge on Meta, while Q diagnostics follow Fast.
	metrics.update({
		key: value for key, value in meta_metrics.items()
		if key.startswith((
			"performance/", "performance_normalized/",
			"training_performance/", "policy_forgetting/",
		))
	})
	metrics.update({
		key: value for key, value in fast_metrics.items()
		if key.startswith(("overestimation/", "q_forgetting/"))
	})
	return metrics


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
	agent = FAMEAgent(env.num_actions, args, device)
	print(
		f"FAME parameters per learner: {parameter_count(agent.fast):,}; "
		f"Fast-to-Meta={args.size_fast2meta:,}; Meta replay={args.size_meta:,}"
	)

	fast_replay = ReplayBuffer(
		args.replay_size, args.batch_size, device, args.seed + 17
	)
	fast2meta_replay = MetaReplayBuffer(
		args.size_fast2meta, args.batch_size, device, args.seed + 29
	)
	meta_replay = MetaReplayBuffer(
		args.size_meta, args.batch_size, device, args.seed + 43
	)
	fast_diagnostics = ContinualDiagnostics(args)
	meta_diagnostics = ContinualDiagnostics(args)
	return_tracker = EpisodeReturnTracker(args.task_sequence, args.num_envs)
	observations = env.reset()

	algorithm_name = "DQN w/ FAME"
	filename = (
		f"DQN_FAME_Procgen_{args.distribution_mode}_{args.num_tasks}tasks"
		f"_{args.cycles}cycles_switch_{args.switch_steps}_lrF_{args.lr1}"
		f"_lrM_{args.meta_lr}_N_{args.size_fast2meta}"
		f"_detect_{args.detection_step}_L_{args.warmstep}_seed_{args.seed}"
	)
	wandb.init(
		project=args.wandb_project,
		name=filename,
		group=algorithm_name,
		mode=args.wandb_mode,
		config=serializable_config(args, algorithm_name, {
			"fast_parameters": parameter_count(agent.fast),
			"meta_parameters": parameter_count(agent.meta),
			"fame_estimation_fraction": args.size_fast2meta / args.switch_steps,
			"fame_detection_vector_steps": args.detection_step // args.num_envs,
		}),
	)

	vector_steps = args.t_steps // args.num_envs
	returns_array = np.zeros(vector_steps, dtype=np.float32)
	global_steps = 0
	next_update = args.warmup_steps
	next_target_update = args.target_update_frequency
	next_evaluation = args.eval_frequency
	next_log = args.log_frequency
	gradient_updates = 0
	meta_updates = 0
	episodes_completed = 0
	meta_warmup = False
	stage_choices = []
	pending_metrics = {}

	def record_step(game, epsilon, detection=False, extra_metrics=None):
		nonlocal next_target_update, next_evaluation, next_log
		metrics = {} if extra_metrics is None else dict(extra_metrics)
		target_updated = False
		while global_steps >= next_target_update:
			agent.update_target()
			target_updated = True
			next_target_update += args.target_update_frequency
		if target_updated:
			metrics.update(fast_diagnostics.action_gap_metrics(
				agent.fast, fast_replay, device, game
			))

		while global_steps >= next_evaluation:
			metrics.update(dual_evaluation_metrics(
				agent, fast_diagnostics, meta_diagnostics, device,
				global_steps // args.switch_steps,
			))
			next_evaluation += args.eval_frequency

		log_due = global_steps >= next_log
		while global_steps >= next_log:
			next_log += args.log_frequency
		if log_due:
			metrics.update(pending_metrics)
			pending_metrics.clear()
			metrics.update({
				"train_return": return_tracker.ema[game],
				"episodic_return_10000": return_tracker.ema[game],
				"epsilon": epsilon,
				"task_index": task_index,
				"task_stage": current_stage,
				"cycle_index": current_stage // args.num_tasks,
				"gradient_updates": gradient_updates,
				"meta_updates": meta_updates,
				"replay_size": fast_replay.size(),
				"fame/fast2meta_replay_size": fast2meta_replay.size(),
				"fame/meta_replay_size": meta_replay.size(),
				"fame/meta_warmup": int(meta_warmup),
				"fame/detection": int(detection),
				"episodes_completed": episodes_completed,
			})
			if agent.last_total_loss is not None:
				metrics.update({
					"loss/total": agent.last_total_loss,
					"loss/td": agent.last_td_loss,
					"loss/behavior_cloning_kl": agent.last_bc_loss,
				})
			if agent.last_meta_loss is not None:
				metrics["loss/meta_mle"] = agent.last_meta_loss
		if metrics:
			wandb.log(metrics, step=global_steps)
		returns_array[global_steps // args.num_envs - 1] = return_tracker.ema[game]

	def detection_rollout(learner, game):
		nonlocal env, observations, global_steps, next_update, episodes_completed
		# Procgen's VecEnv ignores a second manual reset. Re-create it so every
		# candidate sees the same fixed training-level stream from its beginning.
		env.close()
		env = make_training_env(args, task_index)
		observations = env.reset()
		return_tracker.switch_task()
		candidate_returns = []
		for _ in range(args.detection_step // args.num_envs):
			actions = agent.actions(observations, 0.0, learner)
			next_observations, raw_rewards, done, _ = env.step(actions)
			train_rewards = raw_rewards
			if args.reward_clip > 0:
				train_rewards = np.clip(
					raw_rewards, -args.reward_clip, args.reward_clip
				)
			fast_replay.store_batch(
				observations, actions, next_observations, train_rewards, done
			)
			observations = next_observations
			completed = return_tracker.update(game, raw_rewards, done)
			candidate_returns.extend(float(value) for value in completed)
			episodes_completed += len(completed)
			global_steps += args.num_envs
			# Policy evaluation consumes interaction budget but performs no TD update.
			while global_steps >= next_update:
				next_update += args.update_frequency
			record_step(game, 0.0, detection=True)
		return candidate_returns

	try:
		while global_steps < args.t_steps:
			if global_steps > 0 and global_steps % args.switch_steps == 0:
				env.close()
				current_stage = global_steps // args.switch_steps
				task_index = current_stage % args.num_tasks
				env = make_training_env(args, task_index)
				game = args.task_sequence[task_index]
				observations = env.reset()
				return_tracker.switch_task()
				fast_diagnostics.start_training_phase()
				agent.reset_stage_losses()
				meta_warmup = False

				fast_returns = detection_rollout(agent.fast, game)
				meta_returns = detection_rollout(agent.meta, game)
				random_returns = (
					detection_rollout(agent.random, game) if args.reset else []
				)
				mean_fast = np.mean(fast_returns) if fast_returns else -1000.0
				mean_meta = np.mean(meta_returns) if meta_returns else -1000.0
				mean_random = np.mean(random_returns) if random_returns else -1000.0

				meta_beats_fast = is_better(
					args, meta_returns, fast_returns, mean_meta, mean_fast
				)
				fast_beats_meta = is_better(
					args, fast_returns, meta_returns, mean_fast, mean_meta
				)
				if meta_beats_fast and mean_meta > mean_random:
					choice = "Meta"
					meta_warmup = True
				elif fast_beats_meta and mean_fast > mean_random:
					choice = "Fast"
				else:
					choice = "Random" if args.reset else "Fast"
					if args.reset:
						agent.select_random_learner()
				stage_choices.append(choice)
				agent.update_target()
				if args.reset:
					agent.refresh_random_learner()
				env.close()
				env = make_training_env(args, task_index)
				observations = env.reset()
				return_tracker.switch_task()
				pending_metrics.update({
					"fame/detection_fast_return": mean_fast,
					"fame/detection_meta_return": mean_meta,
					"fame/detection_random_return": mean_random,
					"fame/selected_meta": int(choice == "Meta"),
					"fame/selected_fast": int(choice == "Fast"),
					"fame/selected_random": int(choice == "Random"),
				})
				if global_steps >= args.t_steps:
					break

			game = args.task_sequence[task_index]
			stage_steps = global_steps % args.switch_steps
			epsilon = 1.0 if global_steps < args.warmup_steps else epsilon_at(
				args, global_steps, stage_steps
			)
			if global_steps < args.warmup_steps:
				actions = np.random.randint(
					0, env.num_actions, size=args.num_envs
				)
			else:
				actions = agent.actions(observations, epsilon)

			next_observations, raw_rewards, done, _ = env.step(actions)
			train_rewards = raw_rewards
			if args.reward_clip > 0:
				train_rewards = np.clip(
					raw_rewards, -args.reward_clip, args.reward_clip
				)
			fast_replay.store_batch(
				observations, actions, next_observations, train_rewards, done
			)
			if stage_steps >= args.switch_steps - args.size_fast2meta:
				fast2meta_replay.store_batch(observations, actions)
			observations = next_observations
			completed = return_tracker.update(game, raw_rewards, done)
			episodes_completed += len(completed)
			global_steps += args.num_envs

			while global_steps >= next_update:
				if fast_replay.size() >= args.batch_size:
					regularizer = (
						agent.meta if meta_warmup
						and stage_steps < args.warmstep
						and args.lambda_reg > 0 else None
					)
					agent.train_fast(fast_replay, regularizer)
					gradient_updates += 1
				next_update += args.update_frequency

			stage_ending = global_steps % args.switch_steps == 0
			extra_metrics = None
			if stage_ending:
				meta_loss, updates = agent.train_meta(
					current_stage, fast2meta_replay, meta_replay
				)
				meta_updates += updates
				if meta_loss is not None:
					extra_metrics = {"loss/meta_mle": meta_loss}
			record_step(game, epsilon, extra_metrics=extra_metrics)

			if stage_ending:
				fast2meta_replay.copy_to(meta_replay)
				fast2meta_replay.clear()
				fast_replay.clear()
				if args.save_model:
					torch.save(
						agent.meta.state_dict(),
						args.output_dir / "models"
						/ f"{filename}_Meta_stage_{current_stage}.pt",
					)
	finally:
		env.close()
		wandb.finish()

	if args.save_model:
		torch.save(
			agent.fast.state_dict(),
			args.output_dir / "models" / f"{filename}_Fast.pt",
		)
		torch.save(
			agent.meta.state_dict(),
			args.output_dir / "models" / f"{filename}_Meta.pt",
		)
	if args.save:
		with (args.output_dir / "results" / f"{filename}_returns.pkl").open("wb") as file:
			pickle.dump(returns_array, file)
	if args.plot:
		import matplotlib.pyplot as plt
		plt.plot(np.arange(vector_steps) * args.num_envs, returns_array)
		plt.xlabel("lifetime environment transitions")
		plt.ylabel("training return EMA")
		plt.show()
	print("DQN w/ FAME initialization choices:", stage_choices)


if __name__ == "__main__":
	main()
