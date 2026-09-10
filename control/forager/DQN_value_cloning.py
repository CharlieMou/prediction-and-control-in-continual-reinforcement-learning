"""Boundary-free DQN with value cloning on the Figure 5 Forager task."""

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
from model import CNN
from replay import expReplay, observation_tensor


class ValueCloningMemory:
	"""Fixed-size reservoir of states and Q-vector cloning targets."""

	def __init__(self, max_size, batch_size, device):
		self.memory = []
		self.max_size = max_size
		self.batch_size = batch_size
		self.device = device
		self.num_seen = 0

	def store(self, state, q_values):
		cloning_example = (state.cpu(), q_values.cpu())
		self.num_seen += 1
		if len(self.memory) < self.max_size:
			self.memory.append(cloning_example)
		else:
			index = random.randrange(self.num_seen)
			if index < self.max_size:
				self.memory[index] = cloning_example

	def sample(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		states, q_values = map(torch.stack, zip(*batch))
		return states.to(self.device), q_values.to(self.device)

	def size(self):
		return len(self.memory)


parser = ArgumentParser(
	description="Boundary-free DQN with Value Cloning on the Figure 5 Forager task"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t-steps", type=int, default=10_000_000)
parser.add_argument("--lr1", type=float, default=3e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--adam-beta1", type=float, default=0.9)
parser.add_argument("--adam-beta2", type=float, default=0.999)
parser.add_argument("--adam-eps", type=float, default=1e-5)
parser.add_argument("--lambda-vc", type=float, default=1.0,
	help="value-cloning loss coefficient")
parser.add_argument("--clone-period", "--clone_period", type=int, default=50_000,
	help="environment steps between cloning events; 0 disables value cloning")
parser.add_argument("--save", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save-model", action="store_true")
add_forager_arguments(parser)
args = parser.parse_args()
validate_forager_arguments(args)

if args.replay_size % 10 != 0:
	raise ValueError("replay-size must be divisible by 10 for the value memory ratio")
args.value_memory_size = args.replay_size // 10
if args.lambda_vc < 0.0:
	raise ValueError("lambda-vc must be non-negative")
if args.clone_period < 0:
	raise ValueError("clone-period must be non-negative")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


env = make_forager_env(args)
input_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

Net = CNN(input_dim, num_actions).to(device)
Target_net = CNN(input_dim, num_actions).to(device)
Target_net.load_state_dict(Net.state_dict())
opt = optim.Adam(
	Net.parameters(),
	lr=args.lr1,
	betas=(args.adam_beta1, args.adam_beta2),
	eps=args.adam_eps,
)
criterion = torch.nn.MSELoss()
exp_replay = expReplay(args.batch_size, device, max_size=args.replay_size)
value_cloning_memory = ValueCloningMemory(
	args.value_memory_size, args.batch_size, device
)
diagnostics = ContinualDiagnostics(args.seed)


def train_net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states).max(1)[0]
	pred = Net(states).gather(1, actions)
	targets = rewards + (1 - done) * args.gamma * next_pred.reshape(-1, 1)
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
	"""Store current replay states and Q-vectors without consulting a boundary."""
	stored = 0
	for start in range(0, exp_replay.size(), args.batch_size):
		batch = list(itertools.islice(
			exp_replay.memory, start, start + args.batch_size
		))
		states = torch.stack([transition[0] for transition in batch]).to(device)
		q_values = Net(states)
		for state, cloning_q_target in zip(states, q_values):
			value_cloning_memory.store(state, cloning_q_target)
			stored += 1
	return stored


def get_action(obs):
	state = observation_tensor(obs).to(device)
	with torch.no_grad():
		q_values = Net(state.unsqueeze(0))
	if np.random.random() <= args.epsilon:
		action = env.action_space.sample()
	else:
		action = q_values.max(1)[1].item()
	return action


algorithm_name = "DQN-valuecloning"
filename = (
	f"DQN_valuecloning_Forager_steps_{args.t_steps}"
	f"_reward_period_{args.reward_period}_clone_period_{args.clone_period}"
	f"_lambda_{args.lambda_vc}_value_memory_{args.value_memory_size}"
	f"_fov_{args.aperture_size}_batch_{args.batch_size}"
	f"_lr_{args.lr1}_seed_{args.seed}"
)
wandb.init(
	project="Forager_testbed",
	name=filename,
	group=algorithm_name,
	mode=args.wandb_mode,
	config={
		**vars(args),
		"algorithm": algorithm_name,
		"environment": env.game_name,
		"boundary_information_used_by_agent": False,
	},
)


returns_array = np.zeros(args.t_steps, dtype=np.float32)
return_window_sum = 0.0
gradient_updates = 0
cloning_events = 0
last_losses = None
last_cloned = 0
state = env.reset()

for step in range(args.t_steps):
	action = get_action(state)
	next_state, reward, done, info = env.step(action)
	exp_replay.store(state, action, next_state, reward, done)
	state = next_state

	if (
		exp_replay.size() >= args.minimum_replay_history
		and (step + 1) % args.update_frequency == 0
	):
		last_losses = train_net()
		gradient_updates += 1
		if gradient_updates % args.target_update_frequency == 0:
			Target_net.load_state_dict(Net.state_dict())

	# The cloning schedule uses only the global interaction count, never the
	# hidden Forager reward period or task phase.
	if (
		args.clone_period > 0
		and (step + 1) % args.clone_period == 0
		and exp_replay.size() >= args.minimum_replay_history
	):
		last_cloned = store_value_cloning_targets()
		cloning_events += 1

	returns_array[step] = reward
	return_window_sum += reward
	if step >= args.return_smoothing_window:
		return_window_sum -= returns_array[step - args.return_smoothing_window]
	window_size = min(step + 1, args.return_smoothing_window)

	if (step + 1) % args.log_frequency == 0:
		metrics = diagnostics.action_gap_metrics(Net, exp_replay, device)
		metrics.update({
			"average_reward": return_window_sum / window_size,
			# Diagnostic only: this value is never passed to the agent or schedule.
			"environment/task_phase": info["task_phase"],
			"gradient_updates": gradient_updates,
			"replay_size": exp_replay.size(),
			"vc/value_memory_size": value_cloning_memory.size(),
			"vc/value_memory_capacity": args.value_memory_size,
			"vc/cloning_events": cloning_events,
			"vc/last_cloned": last_cloned,
		})
		if last_losses is not None:
			metrics.update({
				"loss/total": last_losses[0],
				"loss/td": last_losses[1],
				"loss/value_cloning": last_losses[2],
			})
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
