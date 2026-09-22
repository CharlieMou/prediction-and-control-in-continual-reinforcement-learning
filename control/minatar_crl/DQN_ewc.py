import numpy as np
import pickle
import itertools
import torch
import torch.optim as optim
import copy
import wandb

from model import *
from replay import *
from CL_envs_fixed import *
from diagnostics import ContinualDiagnostics

from argparse import ArgumentParser
from configparser import ConfigParser


parser = ArgumentParser(description="Parameters for the code - DQN-EWC")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=NUM_TASK_STAGES * 500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for DQN")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--ewc-lambda', type=float, default=400.0,
				help="coefficient of the task-boundary EWC penalty")
parser.add_argument('--fisher-batches', type=int, default=100,
				help="replay minibatches used for each diagonal Fisher estimate")
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
if args.ewc_lambda < 0:
	raise ValueError("ewc-lambda must be non-negative")
if args.fisher_batches <= 0:
	raise ValueError("fisher-batches must be positive")
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


def ewc_penalty():
	penalty = torch.zeros((), device=device)
	parameters = dict(Net.named_parameters())
	for means, fisher in ewc_constraints:
		for name, parameter in parameters.items():
			penalty = penalty + (fisher[name] * (parameter - means[name]) ** 2).sum()
	return 0.5 * penalty


def train_Net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states).max(1)[0]
	pred = Net(states).gather(1, actions)
	targets = rewards + (1 - done) * gamma * next_pred.reshape(-1, 1)
	td_loss = criterion(pred, targets)
	regularization_loss = ewc_penalty()
	loss = td_loss + args.ewc_lambda * regularization_loss
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item(), td_loss.item(), regularization_loss.item()


def consolidate_task():
	"""Estimate the diagonal Fisher from replayed DQN losses."""
	parameters = [(name, parameter) for name, parameter in Net.named_parameters()
			if parameter.requires_grad]
	fisher = {name: torch.zeros_like(parameter) for name, parameter in parameters}
	was_training = Net.training
	Net.eval()
	for _ in range(args.fisher_batches):
		states, actions, next_states, rewards, done = exp_replay.sample()
		with torch.no_grad():
			next_pred = Target_net(next_states).max(1)[0]
			targets = rewards + (1 - done) * gamma * next_pred.reshape(-1, 1)
		pred = Net(states).gather(1, actions)
		fisher_loss = criterion(pred, targets)
		gradients = torch.autograd.grad(fisher_loss,
				[parameter for _, parameter in parameters])
		for (name, _), gradient in zip(parameters, gradients):
			fisher[name].add_(gradient.detach() ** 2)
	for name in fisher:
		fisher[name].div_(args.fisher_batches)
	means = {name: parameter.detach().clone() for name, parameter in parameters}
	ewc_constraints.append((means, fisher))
	if was_training:
		Net.train()
	return sum(value.mean().item() for value in fisher.values()) / len(fisher)


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
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

filename = "DQN_EWC_lambda_"+str(args.ewc_lambda)+\
		"_env_name_"+args.env_name+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+\
		"_fisher_batches_"+str(args.fisher_batches)+\
		"_batch_"+str(args.batch_size)+"_lr1_"+str(args.lr1)+\
		"_seed_"+str(args.seed)

return_smoothing_window = 10000
algorithm_name = "DQN-EWC"
wandb.init(project="MinAtar_testbed", name=filename, group=algorithm_name,
		config={**vars(args), "algorithm": algorithm_name,
			"task_sequence": list(TASK_SEQUENCE), "gamma": gamma,
			"epsilon": epsilon, "return_smoothing_window": return_smoothing_window})

task_index = 0
env = CL_envs_func(args.env_name, args.seed, task_index)
in_channels = env.observation_space.shape[2]
num_actions = env.action_space.n
Net = CNN(in_channels, num_actions).to(device)
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()
Target_net = CNN(in_channels, num_actions).to(device)
Target_net.load_state_dict(Net.state_dict())
exp_replay = expReplay(batch_size=args.batch_size, device=device)
ewc_constraints = []
diagnostics = ContinualDiagnostics(args.seed, gamma, args.eval_episodes,
		args.eval_max_steps, args.eval_epsilon)

returns_array = np.zeros(args.t_steps)
avg_return = 0
epi_return = 0
done = False
cs = env.reset()
return_window_sum = 0.0
last_fisher_mean = None

for step in range(args.t_steps):
	wandb_metrics = {}
	if step > 0 and step%args.switch == 0:
		task_index += 1
		env = CL_envs_func(args.env_name, args.seed, task_index)
		cs = env.reset()
		avg_return = 0
		epi_return = 0
		diagnostics.start_training_phase()
	_, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	epi_return += rew
	exp_replay.store(cs, c_action, ns, rew, done)
	if exp_replay.size() >= args.batch_size:
		loss, td_loss, regularization_loss = train_Net()
	cs = ns
	if (step+1)%args.switch == 0 and step+1 < args.t_steps:
		last_fisher_mean = consolidate_task()
	if (step+1)%1000 == 0:
		Target_net.load_state_dict(Net.state_dict())
		wandb_metrics.update(diagnostics.action_gap_metrics(
			Net, exp_replay, device, env.game_name))
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
			Net, device, completed_stages))
	wandb_metrics.update({"episodic_return_10000": smoothed_return,
			"task_index": task_index,
			"ewc/num_constraints": len(ewc_constraints)})
	if last_fisher_mean is not None:
		wandb_metrics["ewc/fisher_mean"] = last_fisher_mean
	if exp_replay.size() >= args.batch_size:
		wandb_metrics.update({"loss/total": loss, "loss/td": td_loss,
				"loss/ewc": regularization_loss})
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
