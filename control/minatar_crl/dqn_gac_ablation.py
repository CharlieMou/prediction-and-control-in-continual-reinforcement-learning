import time

program_start_time = time.perf_counter()

import numpy as np
import pickle
import torch
import torch.optim as optim
import copy
import random
import wandb

from model import *
from replay import *
from CL_envs_fixed import *
from diagnostics import ContinualDiagnostics

from argparse import ArgumentParser
from configparser import ConfigParser


def str2bool(value):
	if isinstance(value, bool):
		return value
	if value.lower() in ('true', '1', 'yes'):
		return True
	if value.lower() in ('false', '0', 'no'):
		return False
	raise ValueError("align must be true or false")


class RetrospectiveRevaluation:
	"""Fixed-size reservoir of retrospectively revalued decisions."""

	def __init__(self, max_size, batch_size, device):
		self.memory = []
		self.max_size = max_size
		self.batch_size = batch_size
		self.device = device
		self.num_seen = 0
		self.mean_margin = torch.zeros(())

	def reserve(self, num_candidates):
		"""Choose the final reservoir slots before GAC targets are computed."""
		selected = {}
		final_size = len(self.memory)
		for candidate_index in range(num_candidates):
			self.num_seen += 1
			if final_size < self.max_size:
				index = final_size
				final_size += 1
			else:
				index = random.randrange(self.num_seen)
				if index >= self.max_size:
					continue
			selected[index] = candidate_index
		return selected, final_size

	def commit(self, selected, final_size):
		previous_size = len(self.memory)
		new_decisions = []
		for index, revalued_decision in selected.items():
			if index < previous_size:
				self.memory[index] = revalued_decision
			else:
				new_decisions.append(revalued_decision)
		self.memory.extend(new_decisions[:final_size - previous_size])
		if self.memory:
			self.mean_margin = torch.stack([
				margin for _, _, margin in self.memory
			]).mean()

	@staticmethod
	def reservoir_sample(candidates, sample_size):
		"""Uniformly sample candidates without materializing their metrics."""
		reservoir = []
		for num_seen, candidate in enumerate(candidates, start=1):
			if len(reservoir) < sample_size:
				reservoir.append(candidate)
			else:
				index = random.randrange(num_seen)
				if index < sample_size:
					reservoir[index] = candidate
		return reservoir

	def sample(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		states, actions, margins = map(torch.stack, zip(*batch))
		return (states.to(self.device), actions.to(self.device),
				margins.to(self.device))

	def size(self):
		return len(self.memory)

	def memory_footprint(self):
		"""Return the tensor data stored by GAC, in bytes."""
		state_bytes = sum(state.numel() * state.element_size()
				for state, _, _ in self.memory)
		action_bytes = sum(action.numel() * action.element_size()
				for _, action, _ in self.memory)
		margin_bytes = sum(margin.numel() * margin.element_size()
				for _, _, margin in self.memory)
		return {"states": state_bytes, "actions": action_bytes,
				"margins": margin_bytes,
				"total": state_bytes + action_bytes + margin_bytes}


parser = ArgumentParser(description="Parameters for the code - DQN-GAC")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=NUM_TASK_STAGES * 500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for DQN")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--revaluation-size', type=int, default=10000,
				help="maximum number of retrospectively revalued decisions")
parser.add_argument('--rho', type=float, default=0.5,
				help="fraction of each historical action gap to preserve")
parser.add_argument('--lambda-gac', type=float, default=1.0,
				help="GAC loss coefficient")
parser.add_argument('--align', type=str2bool, default=True,
				help="require online/target greedy-action agreement (true or false)")
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
if args.revaluation_size <= 0:
	raise ValueError("revaluation-size must be positive")
if not 0 <= args.rho <= 1:
	raise ValueError("rho must be in [0, 1]")
if args.lambda_gac < 0:
	raise ValueError("lambda-gac must be non-negative")
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


def train_Net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states)
		next_pred = next_pred.max(1)[0]
	pred = Net(states)
	pred = pred.gather(1, actions)
	targets = rewards + (1 - done) * gamma * next_pred.reshape(-1, 1)
	td_loss = criterion(pred, targets)
	gac_loss = torch.zeros((), device=device)
	if retrospective_revaluation.size() > 0:
		revalued_states, revalued_actions, _ = retrospective_revaluation.sample()
		revalued_q = Net(revalued_states)
		greedy_q = revalued_q.gather(1, revalued_actions)
		competitor_q = revalued_q.scatter(1, revalued_actions, -torch.inf).max(
				dim=1, keepdim=True)[0]
		constant_margin = args.rho * retrospective_revaluation.mean_margin.to(device)
		gac_loss = torch.relu(competitor_q - greedy_q + constant_margin).mean()
	loss = td_loss + args.lambda_gac * gac_loss
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item(), td_loss.item(), gac_loss.item()


@torch.no_grad()
def revalue():
	"""Retrospectively revalue replay states into historical decisions."""
	transitions = list(exp_replay.memory)
	reservoir_slots, final_size = retrospective_revaluation.reserve(
		len(transitions))
	reservoir_slots = list(reservoir_slots.items())
	selected = {}

	def evaluate(candidate_indices):
		states = torch.stack([
			transitions[candidate_index][0]
			for candidate_index in candidate_indices
		]).to(device)
		online_q = Net(states)
		online_top = online_q.topk(2, dim=1)
		actions = online_top.indices[:, :1]
		margins = online_top.values[:, :1] - online_top.values[:, 1:2]
		keep = torch.ones(states.shape[0], dtype=torch.bool, device=device)
		if args.align:
			target_q = Target_net(states)
			target_top = target_q.topk(2, dim=1)
			keep = actions.squeeze(1) == target_top.indices[:, 0]
			target_margins = target_top.values[:, :1] - target_top.values[:, 1:2]
			margins = torch.minimum(margins, target_margins)
		return states, actions, margins, keep.cpu().tolist()

	for start in range(0, len(reservoir_slots), args.batch_size):
		batch = reservoir_slots[start:start + args.batch_size]
		states, actions, margins, keep = evaluate([
			candidate_index for _, candidate_index in batch
		])
		for (slot, _), state, action, margin, accepted in zip(
				batch, states, actions, margins, keep):
			if accepted:
				selected[slot] = (state.cpu(), action.cpu(), margin.cpu())

	retrospective_revaluation.commit(selected, final_size)
	return len(selected)


def get_action(c_obs):
	c_obs = np.moveaxis(c_obs, 2, 0)
	c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
	with torch.no_grad():
		curr_Q_vals = Net(c_obs.unsqueeze(0))
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return curr_Q_vals[0][action], action


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("mps")#
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def synchronize_device():
	if device.type == "cuda":
		torch.cuda.synchronize(device)


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

filename = "DQN_GAC_without_hist_action_gaps_lambda_"+str(args.lambda_gac)+\
		"_rho_"+str(args.rho)+\
		"_align_"+str(args.align)+\
		"_env_name_"+args.env_name+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+\
		"_revaluation_"+str(args.revaluation_size)+"_batch_"+str(args.batch_size)+\
		"_lr1_"+str(args.lr1)+"_seed_"+str(args.seed)

return_smoothing_window = 10000
resource_log_frequency = 500000
algorithm_name = "DQN+GAC w/o hist. action gaps"
wandb.init(
	project="MinAtar_testbed",
	name=filename,
	group=algorithm_name,
	config={**vars(args), "algorithm": algorithm_name,
			"task_sequence": list(TASK_SEQUENCE),
			"gamma": gamma, "epsilon": epsilon,
			"return_smoothing_window": return_smoothing_window}
)

environment_interaction_time = 0.0
gradient_update_time = 0.0
auxiliary_memory_time = 0.0

task_index = 0
environment_start_time = time.perf_counter()
env = CL_envs_func(args.env_name, args.seed, task_index)
environment_interaction_time += time.perf_counter() - environment_start_time
in_channels = env.observation_space.shape[2]
num_actions = env.action_space.n

Net = CNN(in_channels, num_actions).to(device)
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()

Target_net = CNN(in_channels, num_actions).to(device)
Target_net.load_state_dict(Net.state_dict())

exp_replay = expReplay(batch_size=args.batch_size, device=device)
retrospective_revaluation = RetrospectiveRevaluation(
		args.revaluation_size, args.batch_size, device)
diagnostics = ContinualDiagnostics(args.seed, gamma, args.eval_episodes,
		args.eval_max_steps, args.eval_epsilon)

returns_array = np.zeros(args.t_steps)

avg_return = 0
epi_return = 0
done = False
environment_start_time = time.perf_counter()
cs = env.reset()
environment_interaction_time += time.perf_counter() - environment_start_time
return_window_sum = 0.0

for step in range(args.t_steps):
	wandb_metrics = {}

	if step > 0 and step%args.switch == 0:
		task_index += 1
		environment_start_time = time.perf_counter()
		env = CL_envs_func(args.env_name, args.seed, task_index)
		cs = env.reset()
		environment_interaction_time += time.perf_counter() - environment_start_time
		avg_return = 0
		epi_return = 0
		diagnostics.start_training_phase()

	environment_start_time = time.perf_counter()
	_, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	environment_interaction_time += time.perf_counter() - environment_start_time
	epi_return += rew
	exp_replay.store(cs, c_action, ns, rew, done)

	if exp_replay.size() >= args.batch_size:
		synchronize_device()
		gradient_start_time = time.perf_counter()
		loss, td_loss, gac_loss = train_Net()
		synchronize_device()
		gradient_update_time += time.perf_counter() - gradient_start_time

	cs = ns

	# Consolidate before the coincident target update so --align compares the
	# final online policy with a genuinely lagged target policy.
	if (step+1)%args.switch == 0 and step+1 < args.t_steps:
		synchronize_device()
		auxiliary_memory_start_time = time.perf_counter()
		wandb_metrics["gac/greedy_actions_aligned"] = revalue()
		synchronize_device()
		auxiliary_memory_time += (
			time.perf_counter() - auxiliary_memory_start_time)

	if (step+1)%1000 == 0:
		Target_net.load_state_dict(Net.state_dict())
		wandb_metrics.update(diagnostics.action_gap_metrics(
			Net, exp_replay, device, env.game_name
		))

	if done:
		environment_start_time = time.perf_counter()
		cs = env.reset()
		environment_interaction_time += time.perf_counter() - environment_start_time
		avg_return = 0.99 * avg_return + 0.01 * epi_return
		epi_return = 0

	returns_array[step] = copy.copy(avg_return)
	return_window_sum += returns_array[step]
	if step >= return_smoothing_window:
		return_window_sum -= returns_array[step-return_smoothing_window]
	smoothed_return = returns_array[step] / return_smoothing_window \
			if step < return_smoothing_window - 1 \
			else return_window_sum / return_smoothing_window
	if (step+1)%args.eval_frequency == 0:
		completed_stages = (step+1)//args.switch
		wandb_metrics.update(diagnostics.evaluation_metrics(
			Net, device, completed_stages
		))
	wandb_metrics.update({"episodic_return_10000": smoothed_return,
			"task_index": task_index,
			"gac/revaluation_size": retrospective_revaluation.size()})
	if (step+1)%resource_log_frequency == 0:
		gac_memory_footprint = retrospective_revaluation.memory_footprint()
		wandb_metrics.update({
			"memory/gac_states_bytes": gac_memory_footprint["states"],
			"memory/gac_actions_bytes": gac_memory_footprint["actions"],
			"memory/gac_margins_bytes": gac_memory_footprint["margins"],
			"memory/gac_total_bytes": gac_memory_footprint["total"],
			"memory/total_bytes": gac_memory_footprint["total"],
			"wall_clock/environment_interaction_seconds": environment_interaction_time,
			"wall_clock/gradient_updates_seconds": gradient_update_time,
			"wall_clock/auxiliary_memory_seconds": auxiliary_memory_time,
			"wall_clock/total_seconds": time.perf_counter() - program_start_time})
	if exp_replay.size() >= args.batch_size:
		wandb_metrics.update({"loss/total": loss, "loss/td": td_loss,
				"loss/gac": gac_loss})
	wandb.log(wandb_metrics, step=step)

if args.save_model:
	torch.save(Net.state_dict(), "models/"+filename+"_Net"+".pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(returns_array, 'b')
	plt.show()

if args.save:
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(returns_array, f)

synchronize_device()
total_wall_clock_time = time.perf_counter() - program_start_time
print("Wall-clock time (seconds): "
		f"environment interaction={environment_interaction_time:.6f}, "
		f"gradient updates={gradient_update_time:.6f}, "
		f"auxiliary memory={auxiliary_memory_time:.6f}, "
		f"total={total_wall_clock_time:.6f}", flush=True)
wandb.finish()
