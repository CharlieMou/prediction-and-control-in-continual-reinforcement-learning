import numpy as np
import pickle
import itertools
import torch
import torch.optim as optim
import wandb

from wrappers import *
from model import *
from replay import *
from env_params import *

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


parser = ArgumentParser(description="Parameters for the code - IP-DQN(k, lambda) on gym envs")
parser.add_argument('--seed', type=int, default=0, help="Random seed")
parser.add_argument('--env-id', type=int, default=0, help="Environment ID")
parser.add_argument('--t-steps', type=int, default=500000, help="number of steps")
parser.add_argument('--lr1', type=float, default=0.1, help="learning rate for weights")
parser.add_argument('--batch-size', type=int, default=64, help="Number of samples per batch")
parser.add_argument('--ftype', type=str, default="obj", help="Type of features")
parser.add_argument('--k', type=float, default=4.0,
				help="mean-expansion coefficient (default: number of JBW actions)")
parser.add_argument('--plasticity', '--lambda', dest='plasticity', type=float, default=0.5,
				help="scale for protected action-gap directions")
parser.add_argument('--rank', type=int, default=1,
				help="number of historical action-gap directions to protect")
parser.add_argument('--consolidate', type=int, default=150000,
				help="steps between protected-subspace updates")
parser.add_argument('--save', action="store_true")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--save-model', action="store_true")

args = parser.parse_args()
if args.consolidate <= 0:
	raise ValueError("consolidate must be positive")
config = ConfigParser()
config.read('misc_params.cfg')
misc_param = config['JBW']
gamma = float(misc_param['gamma'])
epsilon = float(misc_param['epsilon'])


def train_Net():
	states, actions, next_states, rewards = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states)
		next_pred = next_pred.max(1)[0]
		Net[1].update_statistics(Target_net(states))
	pred = Net(states)
	pred = pred.gather(1, actions)
	targets = rewards + gamma * next_pred.reshape(-1, 1)
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
	output_layer = Net[0].critic[0]
	output_layer.weight.copy_(remapping @ output_layer.weight)
	output_layer.bias.copy_(remapping @ output_layer.bias)
	# Adam moments from the old output geometry should not leak across remapping.
	opt.state.pop(output_layer.weight, None)
	opt.state.pop(output_layer.bias, None)


def get_action(c_obs):
	c_obs = c_obs.__array__().flatten()
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

sim_config = make_config()
reward_fn = make_reward()
env = JBWEnv(sim_config, reward_fn, render=False, f_type=args.ftype)
env = FrameStack(env, 4)

rewards_array = np.zeros(args.t_steps)

filename = "IP_DQN_k_"+str(args.k)+"_lambda_"+str(args.plasticity)+\
		"_gamma_"+misc_param['gamma']+"_steps_"+str(args.t_steps)+\
		"_rank_"+str(args.rank)+"_consolidate_"+str(args.consolidate)+\
		"_batch_"+str(args.batch_size)+\
		"_lr1_"+str(args.lr1)+"_seed_"+str(args.seed)+"_ftype_"+args.ftype

reward_smoothing_window = 10000
algorithm_name = "IP-DQN"
wandb.init(
	project="JBW_testbed",
	name=filename,
	group=algorithm_name,
	config={**vars(args), "algorithm": algorithm_name,
			"gamma": gamma, "epsilon": epsilon,
			"reward_smoothing_window": reward_smoothing_window}
)

Net = torch.nn.Sequential(
	NN_FA(env.feature_space.shape[0]*env.feature_space.shape[1], env.action_space.n),
	ImplicitPlasticityLayer(env.action_space.n, args.k, args.plasticity, args.rank)
).to(device)
opt = optim.Adam(Net.parameters(), lr=args.lr1)
criterion = torch.nn.MSELoss()

Target_net = torch.nn.Sequential(
	NN_FA(env.feature_space.shape[0]*env.feature_space.shape[1], env.action_space.n),
	ImplicitPlasticityLayer(env.action_space.n, args.k, args.plasticity, args.rank)
).to(device)
Target_net.load_state_dict(Net.state_dict())

exp_replay = expReplay_NN(batch_size=args.batch_size, device=device)

env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
env.feature_space.seed(args.seed)

cs = env.reset()
loss_list = []
reward_window_sum = 0.0
for step in range(args.t_steps):
	#env.render()
	_, c_action = get_action(cs)
	ns, rew, done, _ = env.step(c_action)
	rewards_array[step] = rew
	reward_window_sum += rew
	if step >= reward_smoothing_window:
		reward_window_sum -= rewards_array[step-reward_smoothing_window]
	smoothed_reward = rew / reward_smoothing_window if step < reward_smoothing_window - 1 \
			else reward_window_sum / reward_smoothing_window
	wandb.log({
		"reward_per_step_10000": smoothed_reward
	}, step=step)
	exp_replay.store(cs, c_action, ns, rew)

	if exp_replay.size() >= args.batch_size:
		loss = train_Net()
		loss_list.append(loss)

	cs = ns

	if (step+1)%args.consolidate == 0:
		consolidate_policy_subspace()

	if (step+1)%200 == 0:
		Target_net.load_state_dict(Net.state_dict())

if args.save_model:
	torch.save(Net.state_dict(), "models/"+filename+"_Net"+".pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(rewards_array, 'b')
	plt.show()

if args.save:
	with open("results/"+filename+"_returns.pkl", "wb") as f:
		pickle.dump(rewards_array, f)

wandb.finish()
