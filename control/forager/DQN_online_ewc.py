"""Boundary-free DQN with online EWC on the Figure 5 Forager task."""

from argparse import ArgumentParser
from pathlib import Path
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


parser = ArgumentParser(
	description="Boundary-free DQN with online EWC on the Figure 5 Forager task"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t-steps", type=int, default=10_000_000)
parser.add_argument("--lr1", type=float, default=3e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--adam-beta1", type=float, default=0.9)
parser.add_argument("--adam-beta2", type=float, default=0.999)
parser.add_argument("--adam-eps", type=float, default=1e-5)
parser.add_argument("--ewc-lambda", type=float, default=17.5,
	help="coefficient of the online-EWC penalty")
parser.add_argument("--ewc-gamma", type=float, default=0.95,
	help="online-EWC forgetting coefficient")
parser.add_argument("--fisher-batches", type=int, default=100,
	help="replay minibatches used for each diagonal Fisher estimate")
parser.add_argument(
	"--consolidation-period", "--consolidation_period", type=int, default=50_000,
	help="environment steps between consolidations; 0 disables consolidation",
)
parser.add_argument("--save", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save-model", action="store_true")
add_forager_arguments(parser)
args = parser.parse_args()
validate_forager_arguments(args)

if args.ewc_lambda < 0.0:
	raise ValueError("ewc-lambda must be non-negative")
if not 0.0 <= args.ewc_gamma < 1.0:
	raise ValueError("ewc-gamma must be in [0, 1)")
if args.fisher_batches <= 0:
	raise ValueError("fisher-batches must be positive")
if args.consolidation_period < 0:
	raise ValueError("consolidation-period must be non-negative")


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
online_fisher = None
ewc_means = None
diagnostics = ContinualDiagnostics(args.seed)


def ewc_penalty():
	if online_fisher is None:
		return torch.zeros((), device=device)
	penalty = torch.zeros((), device=device)
	parameters = dict(Net.named_parameters())
	for name, parameter in parameters.items():
		penalty = penalty + (
			online_fisher[name] * (parameter - ewc_means[name]) ** 2
		).sum()
	# Equation (8) in Progress & Compress discounts the previous posterior
	# approximation while learning the next segment.
	return 0.5 * args.ewc_gamma * penalty


def train_net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states).max(1)[0]
	pred = Net(states).gather(1, actions)
	targets = rewards + (1 - done) * args.gamma * next_pred.reshape(-1, 1)
	td_loss = criterion(pred, targets)
	regularization_loss = ewc_penalty()
	loss = td_loss + args.ewc_lambda * regularization_loss
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item(), td_loss.item(), regularization_loss.item()


def consolidate():
	"""Update the single normalized Fisher and latest MAP reference point."""
	global online_fisher, ewc_means

	parameters = [
		(name, parameter) for name, parameter in Net.named_parameters()
		if parameter.requires_grad
	]
	current_fisher = {
		name: torch.zeros_like(parameter) for name, parameter in parameters
	}
	was_training = Net.training
	Net.eval()
	for _ in range(args.fisher_batches):
		states, actions, next_states, rewards, done = exp_replay.sample()
		with torch.no_grad():
			next_pred = Target_net(next_states).max(1)[0]
			targets = rewards + (1 - done) * args.gamma * next_pred.reshape(-1, 1)
		pred = Net(states).gather(1, actions)
		fisher_loss = criterion(pred, targets)
		gradients = torch.autograd.grad(
			fisher_loss, [parameter for _, parameter in parameters]
		)
		for (name, _), gradient in zip(parameters, gradients):
			current_fisher[name].add_(gradient.detach() ** 2)

	for name in current_fisher:
		current_fisher[name].div_(args.fisher_batches)
	normalizer = sum(value.sum() for value in current_fisher.values())
	if normalizer.item() > 0.0:
		for name in current_fisher:
			current_fisher[name].div_(normalizer)

	if online_fisher is None:
		online_fisher = current_fisher
	else:
		for name in online_fisher:
			online_fisher[name].mul_(args.ewc_gamma).add_(current_fisher[name])
	ewc_means = {
		name: parameter.detach().clone() for name, parameter in parameters
	}
	if was_training:
		Net.train()
	fisher_mean = sum(
		value.mean().item() for value in current_fisher.values()
	) / len(current_fisher)
	fisher_trace = sum(value.sum().item() for value in online_fisher.values())
	return fisher_mean, fisher_trace


def get_action(obs):
	state = observation_tensor(obs).to(device)
	with torch.no_grad():
		q_values = Net(state.unsqueeze(0))
	if np.random.random() <= args.epsilon:
		action = env.action_space.sample()
	else:
		action = q_values.max(1)[1].item()
	return action


algorithm_name = "DQN w/ online EWC"
filename = (
	f"DQN_online_EWC_Forager_steps_{args.t_steps}_period_{args.reward_period}"
	f"_consolidation_{args.consolidation_period}_lambda_{args.ewc_lambda}"
	f"_ewc_gamma_{args.ewc_gamma}_fisher_batches_{args.fisher_batches}"
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
		"fisher_normalization": "unit_trace",
	},
)


returns_array = np.zeros(args.t_steps, dtype=np.float32)
return_window_sum = 0.0
gradient_updates = 0
consolidation_events = 0
last_losses = None
last_fisher_mean = None
last_fisher_trace = None
last_target_refresh = 0
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

	# The schedule uses only the global interaction count, not the hidden
	# Forager reward period or task phase.
	if (
		args.consolidation_period > 0
		and (step + 1) % args.consolidation_period == 0
		and exp_replay.size() >= args.minimum_replay_history
	):
		last_fisher_mean, last_fisher_trace = consolidate()
		consolidation_events += 1

	# Consolidate first on coincident steps so Fisher estimation uses the same
	# lagged target-network convention as the MinAtar EWC implementation.
	if (
		gradient_updates > 0
		and gradient_updates % args.target_update_frequency == 0
		and gradient_updates != last_target_refresh
	):
		Target_net.load_state_dict(Net.state_dict())
		last_target_refresh = gradient_updates

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
			"ewc/consolidation_events": consolidation_events,
			"ewc/has_constraint": int(online_fisher is not None),
		})
		if last_fisher_mean is not None:
			metrics["ewc/current_fisher_mean"] = last_fisher_mean
			metrics["ewc/online_fisher_trace"] = last_fisher_trace
		if last_losses is not None:
			metrics.update({
				"loss/total": last_losses[0],
				"loss/td": last_losses[1],
				"loss/ewc": last_losses[2],
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
