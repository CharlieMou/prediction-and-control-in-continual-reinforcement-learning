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


class ImplicitPlasticityLayer(torch.nn.Module):
	"""Mean expansion with reduced plasticity in historical action-gap directions."""

	def __init__(self, num_actions, k, plasticity, rank=1):
		super().__init__()
		if k < 0:
			raise ValueError("k must be non-negative")
		if not 0 < plasticity <= 1:
			raise ValueError("lambda must be in (0, 1]")
		if not 0 <= rank <= num_actions - 1:
			raise ValueError("rank must be between 0 and num_actions - 1")
		self.k = k
		self.plasticity = plasticity
		self.rank = rank
		self.register_buffer('gap_covariance', torch.zeros(num_actions, num_actions))
		self.register_buffer('protected_projector', torch.zeros(num_actions, num_actions))
		self.register_buffer('num_gap_samples', torch.zeros(()))

	def forward(self, inputs):
		mean_component = inputs.mean(dim=-1, keepdim=True)
		protected_component = inputs @ self.protected_projector
		return inputs + self.k * mean_component + \
				(self.plasticity - 1) * protected_component

	@torch.no_grad()
	def update_statistics(self, q_values):
		"""Accumulate top-1 versus top-2 action-gap directions."""
		top_actions = q_values.topk(2, dim=-1).indices
		gap_directions = torch.zeros_like(q_values)
		gap_directions.scatter_(1, top_actions[:, :1], 1 / np.sqrt(2))
		gap_directions.scatter_(1, top_actions[:, 1:2], -1 / np.sqrt(2))
		batch_size = q_values.shape[0]
		batch_covariance = gap_directions.T @ gap_directions / batch_size
		total = self.num_gap_samples + batch_size
		self.gap_covariance.mul_(self.num_gap_samples / total)
		self.gap_covariance.add_(batch_covariance * (batch_size / total))
		self.num_gap_samples.copy_(total)

	@torch.no_grad()
	def consolidate(self):
		"""Refresh the protected top-r historical action-gap subspace."""
		old_transform = self.transform_matrix()
		if self.rank == 0 or self.num_gap_samples.item() == 0:
			self.protected_projector.zero_()
		else:
			_, eigenvectors = torch.linalg.eigh(self.gap_covariance)
			basis = eigenvectors[:, -self.rank:]
			self.protected_projector.copy_(basis @ basis.T)
		return old_transform, self.transform_matrix()

	def transform_matrix(self):
		num_actions = self.protected_projector.shape[0]
		identity = torch.eye(num_actions, device=self.protected_projector.device)
		mean_projector = torch.ones_like(self.protected_projector) / num_actions
		return identity + self.k * mean_projector + \
				(self.plasticity - 1) * self.protected_projector


parser = ArgumentParser(description="Parameters for the code - IP-DQN(k, lambda)")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-name', type=str, default="all", help="Environment Name")
parser.add_argument('--t-steps', type=int, default=4500000, help="total number of steps")
parser.add_argument('--switch', type=int, default=500000, help="switch env steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for DQN")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--k', type=float, default=6.0,
				help="mean-expansion coefficient (default: number of MinAtar actions)")
parser.add_argument('--plasticity', '--lambda', dest='plasticity', type=float, default=0.9,
				help="scale for protected action-gap directions")
parser.add_argument('--rank', type=int, default=1,
				help="number of historical action-gap directions to protect")
parser.add_argument('--eval-frequency', type=int, default=50000, help="diagnostic evaluation frequency")
parser.add_argument('--eval-episodes', type=int, default=5, help="episodes per evaluation task")
parser.add_argument('--eval-max-steps', type=int, default=10000, help="maximum evaluation episode length")
parser.add_argument('--eval-epsilon', type=float, default=0.001, help="near-greedy evaluation epsilon")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")

args = parser.parse_args()
if args.switch <= 0:
	raise ValueError("switch must be positive")
if args.t_steps != NUM_TASK_STAGES * args.switch:
	raise ValueError("t-steps must equal switch * 9 for the fixed three-cycle curriculum")
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
		Net[1].update_statistics(Target_net(states))
	pred = Net(states)
	pred = pred.gather(1, actions)
	targets = rewards + (1 - done) * gamma * next_pred.reshape(-1, 1)
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item()


@torch.no_grad()
def consolidate_policy_subspace():
	"""Change geometry without changing the Q-function at consolidation time."""
	old_transform, new_transform = Net[1].consolidate()
	remapping = torch.linalg.solve(new_transform, old_transform)
	output_layer = Net[0].output
	output_layer.weight.copy_(remapping @ output_layer.weight)
	output_layer.bias.copy_(remapping @ output_layer.bias)
	opt.state.pop(output_layer.weight, None)
	opt.state.pop(output_layer.bias, None)


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

filename = "IP_DQN_k_"+str(args.k)+"_lambda_"+str(args.plasticity)+\
		"_env_name_"+args.env_name+"_gamma_"+misc_param['gamma']+\
		"_steps_"+str(args.t_steps)+"_switch_"+str(args.switch)+\
		"_rank_"+str(args.rank)+"_batch_"+str(args.batch_size)+\
		"_lr1_"+str(args.lr1)+"_seed_"+str(args.seed)

return_smoothing_window = 10000
algorithm_name = "IP-DQN"
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
	ImplicitPlasticityLayer(num_actions, args.k, args.plasticity, args.rank)
).to(device)
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()

Target_net = torch.nn.Sequential(
	CNN(in_channels, num_actions),
	ImplicitPlasticityLayer(num_actions, args.k, args.plasticity, args.rank)
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
		consolidate_policy_subspace()
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
