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


class MeanExpansionLayer(torch.nn.Module):
	"""Parameter-free mean-expansion layer: q = z + k * mean(z)."""

	def __init__(self, k):
		super().__init__()
		if k < 0:
			raise ValueError("k must be non-negative")
		self.k = k

	def forward(self, inputs):
		return inputs + self.k * inputs.mean(dim=-1, keepdim=True)


parser = ArgumentParser(description="Parameters for the code - IB-DQN(k)")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=NUM_TASK_STAGES * 500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for DQN")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--k', type=float, default=6.0,
				help="mean-expansion coefficient (default: number of MinAtar actions)")
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
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item()


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

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

filename = "IB_DQN_k_"+str(args.k)+"_env_name_"+args.env_name+\
		"_gamma_"+misc_param['gamma']+"_steps_"+str(args.t_steps)+\
		"_switch_"+str(args.switch)+"_batch_"+str(args.batch_size)+\
		"_lr1_"+str(args.lr1)+"_seed_"+str(args.seed)

return_smoothing_window = 10000
algorithm_name = "IB-DQN"
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

Net = torch.nn.Sequential(
	CNN(in_channels, num_actions),
	MeanExpansionLayer(args.k)
).to(device)
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()

Target_net = torch.nn.Sequential(
	CNN(in_channels, num_actions),
	MeanExpansionLayer(args.k)
).to(device)
Target_net.load_state_dict(Net.state_dict())

exp_replay = expReplay(batch_size=args.batch_size, device=device)
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

	_, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	epi_return += rew
	exp_replay.store(cs, c_action, ns, rew, done)

	if exp_replay.size() >= args.batch_size:
		loss = train_Net()

	cs = ns

	if (step+1)%1000 == 0:
		Target_net.load_state_dict(Net.state_dict())
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
			"task_index": task_index})
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
