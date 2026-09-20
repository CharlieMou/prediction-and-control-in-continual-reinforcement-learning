import numpy as np
import pickle
import itertools
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


class CombinedQNetwork(torch.nn.Module):
	"""Expose the PT decomposition as the behavior Q-function Q = P + T."""

	def __init__(self, permanent_net, transient_net):
		super().__init__()
		self.permanent_net = permanent_net
		self.transient_net = transient_net

	def forward(self, inputs):
		return self.permanent_net(inputs) + self.transient_net(inputs)


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

	def store(self, state, action, margin):
		revalued_decision = (state.cpu(), action.cpu(), margin.cpu())
		self.num_seen += 1
		if len(self.memory) < self.max_size:
			self.memory.append(revalued_decision)
		else:
			index = random.randrange(self.num_seen)
			if index < self.max_size:
				self.memory[index] = revalued_decision

	def sample(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		states, actions, margins = map(torch.stack, zip(*batch))
		return (states.to(self.device), actions.to(self.device),
				margins.to(self.device))

	def size(self):
		return len(self.memory)


parser = ArgumentParser(description="Parameters for the code - PT-DQN-GAC")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=NUM_TASK_STAGES * 500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for weights")
parser.add_argument('--lr2', type=float, default=0.1, help="learning rate for transient values")
parser.add_argument('--update', type=int, default=50000, help="PM update frequency")
parser.add_argument('--decay', type=float, default=0, help="decay transient weights after transfer")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--revaluation-size', type=int, default=10000,
				help="maximum number of retrospectively revalued decisions")
parser.add_argument('--rho', type=float, default=0.5,
				help="fraction of each historical action gap to preserve")
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
if args.update <= args.batch_size:
	raise ValueError("update must be greater than batch-size")
if args.revaluation_size <= 0:
	raise ValueError("revaluation-size must be positive")
if not 0 <= args.rho <= 1:
	raise ValueError("rho must be in [0, 1]")
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


def train_T_Net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		T_next_pred = Target_net(next_states)
		P_next_pred = P_Net(next_states)
		P_pred = P_Net(states).gather(1, actions)
	T_pred = T_Net(states).gather(1, actions)
	targets = rewards + (1 - done) * gamma * ((P_next_pred + T_next_pred).max(1)[0]).reshape(-1, 1)
	td_loss = T_criterion(T_pred + P_pred, targets)
	gac_loss = torch.zeros((), device=device)
	if retrospective_revaluation.size() > 0:
		revalued_states, revalued_actions, margins = retrospective_revaluation.sample()
		with torch.no_grad():
			permanent_q = P_Net(revalued_states)
		revalued_q = permanent_q + T_Net(revalued_states)
		greedy_q = revalued_q.gather(1, revalued_actions)
		competitor_q = revalued_q.scatter(1, revalued_actions, -torch.inf).max(
				dim=1, keepdim=True)[0]
		gac_loss = torch.relu(competitor_q - greedy_q + args.rho * margins).mean()
	loss = td_loss + gac_loss
	T_opt.zero_grad()
	loss.backward()
	T_opt.step()
	return loss.item(), td_loss.item(), gac_loss.item()


def train_P_Net():
	loss_u = 0
	u_steps = (exp_replay_PM.size()//args.batch_size) - 1
	for p_update in range(u_steps):
		curr_batch = list(itertools.islice(exp_replay_PM.memory,
				p_update*args.batch_size, (p_update+1)*args.batch_size))
		states, actions, old_p_vals = map(torch.stack, zip(*curr_batch))
		states = states.to(device)
		actions = actions.to(device)
		old_p_vals = old_p_vals.to(device)
		with torch.no_grad():
			T_pred = T_Net(states).gather(1, actions)
		P_pred = P_Net(states).gather(1, actions)
		loss = P_criterion(P_pred, T_pred + old_p_vals)
		P_opt.zero_grad()
		loss.backward()
		P_opt.step()
		loss_u += loss.item()
	return loss_u/u_steps


@torch.no_grad()
def revalue():
	"""Retrospectively revalue replay states using the combined PT value."""
	for start in range(0, exp_replay.size(), args.batch_size):
		batch = list(itertools.islice(
			exp_replay.memory, start, start + args.batch_size))
		states = torch.stack([transition[0] for transition in batch]).to(device)
		permanent_q = P_Net(states)
		online_q = permanent_q + T_Net(states)
		online_top = online_q.topk(2, dim=1)
		actions = online_top.indices[:, :1]
		margins = online_top.values[:, :1] - online_top.values[:, 1:2]
		keep = torch.ones(states.shape[0], dtype=torch.bool, device=device)
		if args.align:
			target_q = permanent_q + Target_net(states)
			target_top = target_q.topk(2, dim=1)
			keep = actions.squeeze(1) == target_top.indices[:, 0]
			target_margins = target_top.values[:, :1] - target_top.values[:, 1:2]
			margins = torch.minimum(margins, target_margins)
		for state, action, margin in zip(states[keep], actions[keep], margins[keep]):
			retrospective_revaluation.store(state, action, margin)


def get_action(c_obs):
	c_obs = np.moveaxis(c_obs, 2, 0)
	c_obs = torch.tensor(c_obs, dtype=torch.float).to(device)
	with torch.no_grad():
		curr_P_vals = P_Net(c_obs.unsqueeze(0))
		curr_Q_vals = curr_P_vals + T_Net(c_obs.unsqueeze(0))
	if np.random.random() <= epsilon:
		action = env.action_space.sample()
	else:
		action = curr_Q_vals.max(1)[1].item()
	return curr_P_vals[0][action], action


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

filename = "PT_DQN_GAC_DoneRight_0.5x_rho_"+str(args.rho)+"_align_"+str(args.align)+\
		"_env_name_"+str(args.env_name)+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+\
		"_update_"+str(args.update)+"_decay_"+str(args.decay)+\
		"_revaluation_"+str(args.revaluation_size)+"_lr1_"+str(args.lr1)+\
		"_lr2_"+str(args.lr2)+"_batch_"+str(args.batch_size)+"_seed_"+str(args.seed)

return_smoothing_window = 10000
algorithm_name = "PT-DQN w/ GAC"
wandb.init(
	project="MinAtar_testbed",
	name=filename,
	group=algorithm_name,
	config={**vars(args), "algorithm": algorithm_name,
			"task_sequence": list(TASK_SEQUENCE),
			"gamma": gamma, "epsilon": epsilon,
			"return_smoothing_window": return_smoothing_window}
)

task_index = 0
env = CL_envs_func(args.env_name, args.seed, task_index)
in_channels = env.observation_space.shape[2]
num_actions = env.action_space.n

T_Net = CNN_half(in_channels, num_actions).to(device)
T_opt = optim.Adam(T_Net.parameters(), lr=args.lr2)
T_criterion = torch.nn.MSELoss()

P_Net = CNN_half(in_channels, num_actions).to(device)
P_opt = optim.SGD(P_Net.parameters(), lr=args.lr1)
P_criterion = torch.nn.MSELoss()

Target_net = CNN_half(in_channels, num_actions).to(device)
Target_net.load_state_dict(T_Net.state_dict())
Net = CombinedQNetwork(P_Net, T_Net)

exp_replay = expReplay(batch_size=args.batch_size, device=device)
exp_replay_PM = expReplay_PM(max_size=args.update,
		batch_size=args.batch_size, device=device)
retrospective_revaluation = RetrospectiveRevaluation(
		args.revaluation_size, args.batch_size, device)
diagnostics = ContinualDiagnostics(args.seed, gamma, args.eval_episodes,
		args.eval_max_steps, args.eval_epsilon)

returns_array = np.zeros(args.t_steps)
avg_return = 0
epi_return = 0
done = False
cs = env.reset()
return_window_sum = 0.0

for step in range(args.t_steps):
	wandb_metrics = {}

	if step > 0 and step%args.switch == 0:
		task_index += 1
		env = CL_envs_func(args.env_name, args.seed, task_index)
		cs = env.reset()
		avg_return = 0
		epi_return = 0
		diagnostics.start_training_phase()

	val_p, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	epi_return += rew
	exp_replay.store(cs, c_action, ns, rew, done)
	exp_replay_PM.store(cs, c_action, val_p)

	if exp_replay.size() >= args.batch_size:
		loss, td_loss, gac_loss = train_T_Net()

	cs = ns

	if (step+1)%args.switch == 0 and step+1 < args.t_steps:
		revalue()

	if (step+1)%args.update == 0:
		p_loss = train_P_Net()
		for params in T_Net.parameters():
			params.data *= args.decay

	if (step+1)%1000 == 0:
		Target_net.load_state_dict(T_Net.state_dict())
		wandb_metrics.update(diagnostics.action_gap_metrics(
			Net, exp_replay, device, env.game_name
		))

	if done:
		cs = env.reset()
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

wandb.finish()
