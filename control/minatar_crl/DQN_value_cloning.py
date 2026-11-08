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


class ValueCloningMemory:
	"""Fixed-size reservoir of states and Q-vector cloning targets."""

	def __init__(self, max_size, batch_size, device):
		self.memory = []
		self.max_size = max_size
		self.batch_size = batch_size
		self.device = device
		self.num_seen = 0

	def reserve(self, num_candidates):
		"""Choose the final reservoir slots before cloning targets are computed."""
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
		self.memory.extend([None] * (final_size - len(self.memory)))
		for index, cloning_example in selected.items():
			self.memory[index] = cloning_example

	def sample(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		states, q_values = map(torch.stack, zip(*batch))
		return states.to(self.device), q_values.to(self.device)

	def size(self):
		return len(self.memory)


parser = ArgumentParser(description="Parameters for DQN with Value Cloning (DQN-VC)")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=NUM_TASK_STAGES * 500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for DQN")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--value-memory-size', type=int, default=10000,
				help="maximum number of stored historical Q-vectors")
parser.add_argument('--lambda-vc', type=float, default=1.0,
				help="value-cloning loss coefficient")
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
if args.value_memory_size <= 0:
	raise ValueError("value-memory-size must be positive")
if args.lambda_vc < 0:
	raise ValueError("lambda-vc must be non-negative")
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
		next_pred = Target_net(next_states).max(1)[0]
	pred = Net(states).gather(1, actions)
	targets = rewards + (1 - done) * gamma * next_pred.reshape(-1, 1)
	td_loss = criterion(pred, targets)
	value_cloning_loss = torch.zeros((), device=device)
	if value_cloning_memory.size() > 0:
		cloning_states, cloning_q_targets = value_cloning_memory.sample()
		value_cloning_loss = criterion(Net(cloning_states), cloning_q_targets)
	loss = td_loss + args.lambda_vc * value_cloning_loss
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item(), td_loss.item(), value_cloning_loss.item()


@torch.no_grad()
def store_value_cloning_targets():
	"""Store the completed task's Q-vectors as value-cloning targets."""
	transitions = list(exp_replay.memory)
	reservoir_slots, final_size = value_cloning_memory.reserve(len(transitions))
	selected = {}
	reservoir_slots = list(reservoir_slots.items())
	for start in range(0, len(reservoir_slots), args.batch_size):
		batch = reservoir_slots[start:start + args.batch_size]
		states = torch.stack([
			transitions[candidate_index][0]
			for _, candidate_index in batch
		]).to(device)
		q_values = Net(states)
		for (slot, _), state, cloning_q_target in zip(batch, states, q_values):
			selected[slot] = (state.cpu(), cloning_q_target.cpu())
	value_cloning_memory.commit(selected, final_size)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def synchronize_device():
	if device.type == "cuda":
		torch.cuda.synchronize(device)


torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

filename = "DQN_valuecloning_lambda_"+str(args.lambda_vc)+\
		"_env_name_"+args.env_name+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+\
		"_value_memory_"+str(args.value_memory_size)+"_batch_"+str(args.batch_size)+\
		"_lr1_"+str(args.lr1)+"_seed_"+str(args.seed)

return_smoothing_window = 10000
algorithm_name = "DQN-valuecloning"
wandb.init(project="MinAtar_testbed", name=filename, group=algorithm_name,
		config={**vars(args), "algorithm": algorithm_name,
			"task_sequence": list(TASK_SEQUENCE), "gamma": gamma,
			"epsilon": epsilon, "return_smoothing_window": return_smoothing_window})

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
value_cloning_memory = ValueCloningMemory(
		args.value_memory_size, args.batch_size, device)
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
		loss, td_loss, value_cloning_loss = train_Net()
		synchronize_device()
		gradient_update_time += time.perf_counter() - gradient_start_time
	cs = ns
	if (step+1)%args.switch == 0 and step+1 < args.t_steps:
		synchronize_device()
		auxiliary_memory_start_time = time.perf_counter()
		store_value_cloning_targets()
		synchronize_device()
		auxiliary_memory_time += (
			time.perf_counter() - auxiliary_memory_start_time)
	if (step+1)%1000 == 0:
		Target_net.load_state_dict(Net.state_dict())
		wandb_metrics.update(diagnostics.action_gap_metrics(
			Net, exp_replay, device, env.game_name))
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
			Net, device, completed_stages))
	wandb_metrics.update({"episodic_return_10000": smoothed_return,
			"task_index": task_index,
			"vc/value_memory_size": value_cloning_memory.size(),
			"wall_clock/environment_interaction_seconds": environment_interaction_time,
			"wall_clock/gradient_updates_seconds": gradient_update_time,
			"wall_clock/auxiliary_memory_seconds": auxiliary_memory_time,
			"wall_clock/total_seconds": time.perf_counter() - program_start_time})
	if exp_replay.size() >= args.batch_size:
		wandb_metrics.update({"loss/total": loss, "loss/td": td_loss,
				"loss/value_cloning": value_cloning_loss})
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
wandb.finish()
synchronize_device()
total_wall_clock_time = time.perf_counter() - program_start_time
print("Wall-clock time (seconds): "
		f"environment interaction={environment_interaction_time:.6f}, "
		f"gradient updates={gradient_update_time:.6f}, "
		f"auxiliary memory={auxiliary_memory_time:.6f}, "
		f"total={total_wall_clock_time:.6f}")
