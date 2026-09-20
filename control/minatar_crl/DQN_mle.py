import os
import pickle
import random
from argparse import ArgumentParser
from collections import deque
from configparser import ConfigParser

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from scipy import stats
from torch.optim.lr_scheduler import ExponentialLR

from model import CNN
from replay import expReplay
from CL_envs_fixed import CL_envs_func, NUM_TASK_STAGES, TASK_SEQUENCE
from diagnostics import ContinualDiagnostics


class MetaReplay:
	"""State/action replay used to fit the meta policy by MLE."""

	def __init__(self, max_size, batch_size, device):
		self.memory = deque(maxlen=max_size)
		self.batch_size = batch_size
		self.device = device

	def store(self, observation, action):
		state = torch.tensor(np.moveaxis(observation, 2, 0), dtype=torch.float)
		action = torch.tensor([action], dtype=torch.int64)
		self.memory.append((state, action))

	def sample(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		states, actions = map(torch.stack, zip(*batch))
		return states.to(self.device), actions.to(self.device)

	def copy_to(self, target):
		target.memory.extend(self.memory)

	def clear(self):
		self.memory.clear()

	def size(self):
		return len(self.memory)


parser = ArgumentParser(description="DQN w/ MLE for the fixed MinAtar curriculum")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=NUM_TASK_STAGES * 500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=1e-3, help="learning rate for meta learner")
parser.add_argument('--lr2', type=float, default=1e-5, help="learning rate for fast learner")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--size-fast2meta', type=int, default=12000,
		help="number of final-stage policy samples retained for meta learning")
parser.add_argument('--size-meta', type=int, default=100000, help="meta replay capacity")
# The paper, README command, and run.sh use n=600. FAME.py's 1200 default is stale.
parser.add_argument('--detection-step', type=int, default=600,
		help="environment steps used to evaluate each candidate policy")
parser.add_argument('--epoch-meta', type=int, default=200, help="meta MLE epochs")
parser.add_argument('--reset', type=int, choices=(0, 1), default=1,
		help="allow random reinitialization when neither learned policy transfers")
parser.add_argument('--warmstep', type=int, default=50000,
		help="KL-regularized fast-learning steps after selecting the meta policy")
parser.add_argument('--lambda-reg', type=float, default=1.0,
		help="behavior-cloning KL coefficient")
parser.add_argument('--temperature', type=float, default=1.0,
		help="softmax temperature for the meta policy")
parser.add_argument('--use-ttest', type=int, choices=(0, 1), default=0,
		help="use a one-sided Welch test instead of empirical mean ranking")
parser.add_argument('--eval-frequency', type=int, default=50000, help="diagnostic evaluation frequency")
parser.add_argument('--eval-episodes', type=int, default=5, help="episodes per evaluation task")
parser.add_argument('--eval-max-steps', type=int, default=10000, help="maximum evaluation episode length")
parser.add_argument('--eval-epsilon', type=float, default=0.001, help="near-greedy evaluation epsilon")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")

args = parser.parse_args()
if args.t_steps != NUM_TASK_STAGES * args.switch:
	raise ValueError("t-steps must equal switch * NUM_TASK_STAGES")
if min(args.batch_size, args.size_fast2meta, args.size_meta, args.detection_step,
		args.epoch_meta, args.warmstep) <= 0:
	raise ValueError("buffer, detection, epoch, warm-up, and batch parameters must be positive")
if args.size_fast2meta > min(args.switch, args.size_meta):
	raise ValueError("size-fast2meta cannot exceed switch or size-meta")
if args.warmstep > args.switch:
	raise ValueError("warmstep cannot exceed switch")
if args.lr1 <= 0 or args.lr2 <= 0:
	raise ValueError("learning rates must be positive")
if args.lambda_reg < 0:
	raise ValueError("lambda-reg must be non-negative")
if args.temperature <= 0:
	raise ValueError("temperature must be positive")
if args.eval_frequency <= 0 or args.eval_episodes <= 0 or args.eval_max_steps <= 0:
	raise ValueError("evaluation parameters must be positive")
if args.switch % args.eval_frequency != 0:
	raise ValueError("eval-frequency must divide switch to evaluate every task boundary")
if not 0 <= args.eval_epsilon <= 1:
	raise ValueError("eval-epsilon must be in [0, 1]")

config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config[str(args.env_name)]
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def greedy_action(observation, learner):
	state = torch.tensor(np.moveaxis(observation, 2, 0), dtype=torch.float,
			device=device).unsqueeze(0)
	with torch.no_grad():
		q_values = learner(state)
	action = q_values.argmax(dim=1).item()
	return action, q_values[0, action].item()


def behavior_action(observation):
	action, _ = greedy_action(observation, Fast_Learner)
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	return action


def train_fast(regularizer=None):
	states, actions, next_states, rewards, done = fast_replay.sample()
	with torch.no_grad():
		next_values = Target_net(next_states).max(1)[0].reshape(-1, 1)
	targets = rewards + (1 - done) * gamma * next_values
	predictions = Fast_Learner(states).gather(1, actions)
	td_loss = fast_criterion(predictions, targets)
	bc_loss = torch.zeros((), device=device)
	if regularizer is not None and args.lambda_reg > 0:
		with torch.no_grad():
			soft_targets = F.softmax(
					regularizer(states) / args.temperature, dim=-1)
		bc_loss = F.kl_div(
				F.log_softmax(Fast_Learner(states) / args.temperature, dim=-1),
				soft_targets, reduction='batchmean')
	loss = td_loss + args.lambda_reg * bc_loss
	fast_optimizer.zero_grad()
	loss.backward()
	fast_optimizer.step()
	return loss.item(), td_loss.item(), bc_loss.item()


def train_meta(number_of_old_tasks):
	"""Optimize the equal-task-weighted maximum-likelihood objective."""
	if fast2meta_replay.size() < args.batch_size:
		return None
	if number_of_old_tasks > 0 and meta_replay.size() < args.batch_size:
		return None
	training_size = meta_replay.size() if number_of_old_tasks > 0 \
			else fast2meta_replay.size()
	updates_per_epoch = (training_size // args.batch_size) - 1
	if updates_per_epoch <= 0:
		return None
	last_loss = None
	for epoch in range(args.epoch_meta):
		for update_index in range(updates_per_epoch):
			if number_of_old_tasks == 0:
				new_states, new_actions = fast2meta_replay.sample()
				loss = F.nll_loss(F.log_softmax(
						Meta_Learner(new_states) / args.temperature, dim=-1),
						new_actions.view(-1))
			else:
				old_states, old_actions = meta_replay.sample()
				loss = F.nll_loss(F.log_softmax(
						Meta_Learner(old_states) / args.temperature, dim=-1),
						old_actions.view(-1))
				# Sampling the new task every k-1 updates gives each of the k
				# encountered task stages equal expected objective weight.
				if update_index % number_of_old_tasks == 0:
					new_states, new_actions = fast2meta_replay.sample()
					loss = loss + F.nll_loss(F.log_softmax(
							Meta_Learner(new_states) / args.temperature, dim=-1),
							new_actions.view(-1))
			meta_optimizer.zero_grad()
			loss.backward()
			meta_optimizer.step()
			last_loss = loss.item()
		if (epoch + 1) % 2 == 0:
			meta_scheduler.step()
	return last_loss


def is_better(samples_a, samples_b, mean_a, mean_b):
	if not args.use_ttest or len(samples_a) < 2 or len(samples_b) < 2:
		return mean_a > mean_b
	result = stats.ttest_ind(samples_a, samples_b, alternative='greater',
			equal_var=False)
	return result.pvalue < 0.05


filename = (f"DQN_MLE_env_name_{args.env_name}_gamma_{misc_param['gamma']}"
		f"_steps_{args.t_steps}_switch_{args.switch}_batch_{args.batch_size}"
		f"_lr1_{args.lr1}_lr2_{args.lr2}_size_fast2meta_{args.size_fast2meta}"
		f"_size_meta_{args.size_meta}_detection_step_{args.detection_step}"
		f"_epoch_meta_{args.epoch_meta}_warmstep_{args.warmstep}"
		f"_lambda_reg_{args.lambda_reg}_seed_{args.seed}")

return_smoothing_window = 10000
algorithm_name = "DQN w/ MLE"
wandb.init(
	project="MinAtar_testbed",
	name=filename,
	group=algorithm_name,
	config={**vars(args), "algorithm": algorithm_name,
			"task_sequence": list(TASK_SEQUENCE),
			"gamma": gamma, "epsilon": epsilon,
			"return_smoothing_window": return_smoothing_window,
			"meta_lr_decay": 0.95,
			"meta_lr_decay_frequency": 2}
)

task_index = 0
env = CL_envs_func(args.env_name, args.seed, task_index)
in_channels = env.observation_space.shape[2]
num_actions = env.action_space.n

Fast_Learner = CNN(in_channels, num_actions).to(device)
fast_optimizer = optim.Adam(Fast_Learner.parameters(), lr=args.lr2)
fast_criterion = torch.nn.MSELoss()

Random_Learner = CNN(in_channels, num_actions).to(device)
Meta_Learner = CNN(in_channels, num_actions).to(device)
meta_optimizer = optim.Adam(Meta_Learner.parameters(), lr=args.lr1)
meta_scheduler = ExponentialLR(meta_optimizer, gamma=0.95)

Target_net = CNN(in_channels, num_actions).to(device)
Target_net.load_state_dict(Fast_Learner.state_dict())

fast_replay = expReplay(batch_size=args.batch_size, device=device)
fast2meta_replay = MetaReplay(args.size_fast2meta, args.batch_size, device)
meta_replay = MetaReplay(args.size_meta, args.batch_size, device)
fast_diagnostics = ContinualDiagnostics(args.seed, gamma, args.eval_episodes,
		args.eval_max_steps, args.eval_epsilon)
meta_diagnostics = ContinualDiagnostics(args.seed, gamma, args.eval_episodes,
		args.eval_max_steps, args.eval_epsilon)

returns_array = np.zeros(args.t_steps)
return_window_sum = 0.0
avg_return = 0.0
episode_return = 0.0
observation = env.reset()
global_step = 0
meta_warmup = False
stage_choices = []
last_total_loss = None
last_td_loss = None
last_bc_loss = None
pending_metrics = {}


def record_step(step, extra_metrics=None):
	"""Record one environment interaction using the other baselines' convention."""
	global return_window_sum, pending_metrics
	returns_array[step] = avg_return
	return_window_sum += returns_array[step]
	if step >= return_smoothing_window:
		return_window_sum -= returns_array[step-return_smoothing_window]
	smoothed_return = returns_array[step] / return_smoothing_window \
			if step < return_smoothing_window - 1 \
			else return_window_sum / return_smoothing_window

	metrics = pending_metrics
	pending_metrics = {}
	if extra_metrics:
		metrics.update(extra_metrics)
	if (step+1)%1000 == 0:
		Target_net.load_state_dict(Fast_Learner.state_dict())
		metrics.update(fast_diagnostics.action_gap_metrics(
			Fast_Learner, fast_replay, device, env.game_name
		))
	if (step+1)%args.eval_frequency == 0:
		completed_stages = (step+1)//args.switch
		fast_metrics = fast_diagnostics.evaluation_metrics(
			Fast_Learner, device, completed_stages)
		# FAME consolidates the task-end Fast policy. Use Fast's probe states,
		# greedy actions, and Q values as the common forgetting reference for
		# both learners, including Meta's intentionally cross-scale Q metric.
		meta_diagnostics.references = fast_diagnostics.references
		meta_diagnostics.last_completed_stages = \
			fast_diagnostics.last_completed_stages
		meta_metrics = meta_diagnostics.evaluation_metrics(
			Meta_Learner, device, completed_stages)
		dual_metric_families = {"performance", "forgetting",
				"policy_forgetting", "q_forgetting"}
		for learner_name, learner_metrics in (("meta", meta_metrics),
				("fast", fast_metrics)):
			for key, value in learner_metrics.items():
				family, separator, metric_name = key.partition("/")
				if separator and family in dual_metric_families:
					metrics[f"{family}_{learner_name}/{metric_name}"] = value
		# Meta stores the cross-task policy, while Fast is the TD-trained Q.
		# Keep the established unsuffixed metrics for dashboard compatibility:
		# return/policy metrics follow Meta, while Q metrics follow Fast.
		metrics.update({key: value for key, value in meta_metrics.items()
			if key.startswith(("performance/", "forgetting/",
					"policy_forgetting/"))})
		metrics.update({key: value for key, value in fast_metrics.items()
			if key.startswith(("overestimation/", "q_forgetting/"))})
	metrics.update({"episodic_return_10000": smoothed_return,
			"task_index": task_index,
			"mle/meta_warmup": int(meta_warmup and
					step % args.switch < args.warmstep),
			"mle/fast_replay_size": fast_replay.size(),
			"mle/meta_replay_size": meta_replay.size()})
	if last_total_loss is not None:
		metrics.update({"loss/total": last_total_loss,
				"loss/td": last_td_loss,
				"loss/behavior_cloning_kl": last_bc_loss})
	wandb.log(metrics, step=step)


def detection_rollout(learner, number_of_steps):
	"""Evaluate greedily and retain the interactions for fast learning."""
	global observation, global_step, avg_return
	local_return = 0.0
	local_length = 0
	completed_returns = []
	for _ in range(number_of_steps):
		if global_step >= args.t_steps:
			break
		action, _ = greedy_action(observation, learner)
		next_observation, reward, done, _ = env.step(action)
		fast_replay.store(observation, action, next_observation, reward, done)
		local_return += reward
		local_length += 1
		observation = next_observation
		if done or local_length > 300:
			observation = env.reset()
			avg_return = 0.99 * avg_return + 0.01 * local_return
			completed_returns.append(local_return)
			local_return = 0.0
			local_length = 0
		record_step(global_step)
		global_step += 1
	return completed_returns


while global_step < args.t_steps:
	if global_step > 0 and global_step%args.switch == 0:
		task_index += 1
		env = CL_envs_func(args.env_name, args.seed, task_index)
		observation = env.reset()
		avg_return = 0.0
		episode_return = 0.0
		fast_diagnostics.start_training_phase()
		meta_warmup = False
		last_total_loss = None
		last_td_loss = None
		last_bc_loss = None

		fast_returns = detection_rollout(Fast_Learner, args.detection_step)
		observation = env.reset()
		meta_returns = detection_rollout(Meta_Learner, args.detection_step)
		random_returns = []
		if args.reset:
			observation = env.reset()
			random_returns = detection_rollout(
					Random_Learner, args.detection_step)
		mean_fast = np.mean(fast_returns) if fast_returns else -1000.0
		mean_meta = np.mean(meta_returns) if meta_returns else -1000.0
		# Candidate policies must be compared in the same units.  The original
		# FAME implementation compared episodic returns for Fast/Meta against a
		# single Q value for Random, which is not a meaningful ranking.
		mean_random = np.mean(random_returns) if random_returns else -1000.0

		meta_beats_fast = is_better(meta_returns, fast_returns,
				mean_meta, mean_fast)
		fast_beats_meta = is_better(fast_returns, meta_returns,
				mean_fast, mean_meta)
		if meta_beats_fast and mean_meta > mean_random:
			choice = "Meta"
			meta_warmup = True
		elif fast_beats_meta and mean_fast > mean_random:
			choice = "Fast"
		else:
			choice = "Random" if args.reset else "Fast"
			if args.reset:
				# Reuse the exact random network whose return was evaluated above.
				Fast_Learner = Random_Learner
				fast_optimizer = optim.Adam(Fast_Learner.parameters(), lr=args.lr2)
		stage_choices.append(choice)
		Target_net.load_state_dict(Fast_Learner.state_dict())
		if args.reset:
			# Keep an independent, untouched candidate for the next task switch.
			Random_Learner = CNN(in_channels, num_actions).to(device)
		observation = env.reset()
		pending_metrics.update({"mle/detection_fast_return": mean_fast,
				"mle/detection_meta_return": mean_meta,
				"mle/detection_random_return": mean_random,
				"mle/selected_meta": int(choice == "Meta"),
				"mle/selected_fast": int(choice == "Fast"),
				"mle/selected_random": int(choice == "Random")})
		if global_step >= args.t_steps:
			break

	action = behavior_action(observation)
	next_observation, reward, done, _ = env.step(action)
	episode_return += reward
	fast_replay.store(observation, action, next_observation, reward, done)
	stage_step = global_step % args.switch
	if stage_step >= args.switch - args.size_fast2meta:
		fast2meta_replay.store(observation, action)

	if fast_replay.size() >= args.batch_size:
		regularizer = Meta_Learner if (meta_warmup and
				stage_step < args.warmstep and args.lambda_reg > 0) else None
		last_total_loss, last_td_loss, last_bc_loss = train_fast(regularizer)

	observation = next_observation
	if done:
		observation = env.reset()
		avg_return = 0.99 * avg_return + 0.01 * episode_return
		episode_return = 0.0

	stage_ending = (global_step+1) % args.switch == 0
	meta_loss = None
	if stage_ending:
		meta_optimizer = optim.Adam(Meta_Learner.parameters(), lr=args.lr1)
		meta_scheduler = ExponentialLR(meta_optimizer, gamma=0.95)
		meta_loss = train_meta(task_index)
	record_step(global_step, {"loss/meta_mle": meta_loss}
			if meta_loss is not None else None)
	global_step += 1

	if stage_ending:
		fast2meta_replay.copy_to(meta_replay)
		fast2meta_replay.clear()
		fast_replay.delete()
		if args.save_model:
			os.makedirs("models", exist_ok=True)
			torch.save(Meta_Learner.state_dict(),
					f"models/{filename}_Meta{task_index}.pt")

if args.save_model:
	os.makedirs("models", exist_ok=True)
	torch.save(Fast_Learner.state_dict(), f"models/{filename}_Fast.pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(returns_array, 'b')
	plt.show()

if args.save:
	os.makedirs("results", exist_ok=True)
	with open(f"results/{filename}_returns.pkl", "wb") as output:
		pickle.dump(returns_array, output)

print("DQN w/ MLE initialization choices:", stage_choices)
wandb.finish()
