from argparse import ArgumentParser
from pathlib import Path
import itertools
import random

import numpy as np
import pickle
import torch
import torch.optim as optim
import wandb

from diagnostics import ContinualDiagnostics
from forager_env import (
	add_forager_arguments,
	make_forager_env,
	validate_forager_arguments,
)
from model import CNN_half
from replay import expReplay, expReplay_PM, observation_tensor


class CombinedQNetwork(torch.nn.Module):
	"""Behavior Q-function Q = P + T used by PT-DQN."""

	def __init__(self, permanent_net, transient_net):
		super().__init__()
		self.permanent_net = permanent_net
		self.transient_net = transient_net

	def forward(self, inputs):
		return self.permanent_net(inputs) + self.transient_net(inputs)


parser = ArgumentParser(description="PT-DQN on the Figure 5 Forager task")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t-steps", type=int, default=10_000_000)
parser.add_argument("--lr1", type=float, default=1e-8,
		help="permanent-network SGD learning rate")
parser.add_argument("--lr2", type=float, default=1e-3,
		help="transient-network Adam learning rate")
parser.add_argument("--update", type=int, default=10_000,
		help="permanent update/ transient decay interval in environment steps")
parser.add_argument("--decay", type=float, default=0.95)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--adam-beta1", type=float, default=0.9)
parser.add_argument("--adam-beta2", type=float, default=0.999)
parser.add_argument("--adam-eps", type=float, default=1e-5)
parser.add_argument("--save", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save-model", action="store_true")
add_forager_arguments(parser)
args = parser.parse_args()
validate_forager_arguments(args)
if args.update <= args.batch_size:
	raise ValueError("update must be greater than batch-size")
if not 0.0 <= args.decay <= 1.0:
	raise ValueError("decay must be in [0, 1]")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


env = make_forager_env(args)
input_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

T_Net = CNN_half(input_dim, num_actions).to(device)
P_Net = CNN_half(input_dim, num_actions).to(device)
Target_net = CNN_half(input_dim, num_actions).to(device)
Target_net.load_state_dict(T_Net.state_dict())
Net = CombinedQNetwork(P_Net, T_Net)

T_opt = optim.Adam(
	T_Net.parameters(),
	lr=args.lr2,
	betas=(args.adam_beta1, args.adam_beta2),
	eps=args.adam_eps,
)
P_opt = optim.SGD(P_Net.parameters(), lr=args.lr1)
T_criterion = torch.nn.MSELoss()
P_criterion = torch.nn.MSELoss()

exp_replay = expReplay(args.batch_size, device, max_size=args.replay_size)
exp_replay_PM = expReplay_PM(args.update, args.batch_size, device)
diagnostics = ContinualDiagnostics(args.seed)


def train_t_net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		t_next = Target_net(next_states)
		p_next = P_Net(next_states)
		p_pred = P_Net(states).gather(1, actions)
	t_pred = T_Net(states).gather(1, actions)
	targets = rewards + (1 - done) * args.gamma * (
		(p_next + t_next).max(1)[0]
	).reshape(-1, 1)
	loss = T_criterion(t_pred + p_pred, targets)
	T_opt.zero_grad()
	loss.backward()
	T_opt.step()
	return loss.item()


def train_p_net():
	u_steps = (exp_replay_PM.size() // args.batch_size) - 1
	if u_steps <= 0:
		return 0.0
	loss_sum = 0.0
	for p_update in range(u_steps):
		batch = list(itertools.islice(
			exp_replay_PM.memory,
			p_update * args.batch_size,
			(p_update + 1) * args.batch_size,
		))
		states, actions, old_p_vals = map(torch.stack, zip(*batch))
		states = states.to(device)
		actions = actions.to(device)
		old_p_vals = old_p_vals.to(device)
		with torch.no_grad():
			t_pred = T_Net(states).gather(1, actions)
		p_pred = P_Net(states).gather(1, actions)
		loss = P_criterion(p_pred, t_pred + old_p_vals)
		P_opt.zero_grad()
		loss.backward()
		P_opt.step()
		loss_sum += loss.item()
	return loss_sum / u_steps


def get_action(obs):
	state = observation_tensor(obs).to(device)
	with torch.no_grad():
		p_values = P_Net(state.unsqueeze(0))
		q_values = p_values + T_Net(state.unsqueeze(0))
	if np.random.random() <= args.epsilon:
		action = env.action_space.sample()
	else:
		action = q_values.max(1)[1].item()
	return p_values[0][action], action


algorithm_name = "PT-DQN tuned"
filename = (
	f"PT_DQN_half_Forager_steps_{args.t_steps}_period_{args.reward_period}"
	f"_fov_{args.aperture_size}_update_{args.update}_decay_{args.decay}"
	f"_lrP_{args.lr1}_lrT_{args.lr2}_seed_{args.seed}"
)
wandb.init(
	project="Forager_testbed",
	name=filename,
	group=algorithm_name,
	mode=args.wandb_mode,
	config={**vars(args), "algorithm": algorithm_name,
			"environment": env.game_name},
)


returns_array = np.zeros(args.t_steps, dtype=np.float32)
return_window_sum = 0.0
gradient_updates = 0
last_t_loss = None
last_p_loss = None
state = env.reset()

for step in range(args.t_steps):
	p_value, action = get_action(state)
	next_state, reward, done, info = env.step(action)
	exp_replay.store(state, action, next_state, reward, done)
	exp_replay_PM.store(state, action, p_value)
	state = next_state

	if (
		exp_replay.size() >= args.minimum_replay_history
		and (step + 1) % args.update_frequency == 0
	):
		last_t_loss = train_t_net()
		gradient_updates += 1
		if gradient_updates % args.target_update_frequency == 0:
			Target_net.load_state_dict(T_Net.state_dict())

	if (step + 1) % args.update == 0:
		last_p_loss = train_p_net()
		with torch.no_grad():
			for parameter in T_Net.parameters():
				parameter.mul_(args.decay)

	returns_array[step] = reward
	return_window_sum += reward
	if step >= args.return_smoothing_window:
		return_window_sum -= returns_array[step - args.return_smoothing_window]
	window_size = min(step + 1, args.return_smoothing_window)

	if (step + 1) % args.log_frequency == 0:
		metrics = diagnostics.action_gap_metrics(Net, exp_replay, device)
		metrics.update({
			"average_reward": return_window_sum / window_size,
			"task_phase": info["task_phase"],
			"gradient_updates": gradient_updates,
			"replay_size": exp_replay.size(),
			"pt/pm_replay_size": exp_replay_PM.size(),
		})
		if last_t_loss is not None:
			metrics["loss/transient_td"] = last_t_loss
		if last_p_loss is not None:
			metrics["loss/permanent"] = last_p_loss
		wandb.log(metrics, step=step + 1)


if args.save_model:
	Path("models").mkdir(exist_ok=True)
	torch.save(Net.state_dict(), Path("models") / f"{filename}_Net.pt")

if args.plot:
	import matplotlib.pyplot as plt
	plt.plot(returns_array)
	plt.show()

if args.save:
	Path("results").mkdir(exist_ok=True)
	with (Path("results") / f"{filename}_returns.pkl").open("wb") as f:
		pickle.dump(returns_array, f)

env.close()
wandb.finish()
