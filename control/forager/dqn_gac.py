"""Boundary-free DQN with greedy action consolidation on Forager Figure 5."""

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
from lifetime_return.evaluator import (
	HistoricalTaskEvaluator,
	add_lifetime_return_arguments,
	validate_lifetime_return_arguments,
)


def str2bool(value):
	if isinstance(value, bool):
		return value
	if value.lower() in ("true", "1", "yes"):
		return True
	if value.lower() in ("false", "0", "no"):
		return False
	raise ValueError("align must be true or false")


class RetrospectiveRevaluation:
	"""Fixed-size reservoir of historical greedy decisions and action gaps."""

	def __init__(self, max_size, batch_size, device):
		self.memory = []
		self.max_size = max_size
		self.batch_size = batch_size
		self.device = device
		self.num_seen = 0

	def store(self, state, action, margin):
		decision = (state.detach().cpu(), action.detach().cpu(), margin.detach().cpu())
		self.num_seen += 1
		if len(self.memory) < self.max_size:
			self.memory.append(decision)
		else:
			index = random.randrange(self.num_seen)
			if index < self.max_size:
				self.memory[index] = decision

	def sample(self):
		batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
		states, actions, margins = map(torch.stack, zip(*batch))
		return states.to(self.device), actions.to(self.device), margins.to(self.device)

	def size(self):
		return len(self.memory)


parser = ArgumentParser(
	description="Boundary-free DQN-GAC on the Figure 5 Forager task"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t-steps", type=int, default=10_000_000)
parser.add_argument("--lr1", type=float, default=3e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--adam-beta1", type=float, default=0.9)
parser.add_argument("--adam-beta2", type=float, default=0.999)
parser.add_argument("--adam-eps", type=float, default=1e-5)
parser.add_argument("--revalue-period", "--revalue_period", type=int, default=50_000,
	help="environment steps between revaluations; 0 disables GAC revaluation")
parser.add_argument("--revalue-start", "--revalue_start", type=int, default=None,
	help="first revaluation step (default: one revalue period)")
parser.add_argument("--revalue-samples", "--revalue_samples", type=int, default=None,
	help="replay states sampled per event (default: batch-size)")
parser.add_argument("--rho", type=float, default=0.5,
	help="fraction of each historical action gap to preserve")
parser.add_argument("--gac-weight", "--gac_weight", type=float, default=1.0,
	help="coefficient on the GAC hinge loss")
parser.add_argument(
	"--min-consolidation-margin", "--min_consolidation_margin",
	type=float, default=0.0,
	help="discard retrospective decisions with a smaller action gap")
parser.add_argument("--align", type=str2bool, default=True,
	help="require online/target greedy-action agreement")
parser.add_argument("--save", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save-model", action="store_true")
add_forager_arguments(parser)
add_lifetime_return_arguments(parser)
args = parser.parse_args()
validate_forager_arguments(args)
validate_lifetime_return_arguments(args)

if args.replay_size % 10 != 0:
	raise ValueError("replay-size must be divisible by 10 for the GAC memory ratio")
args.revaluation_size = args.replay_size // 10
if args.revalue_period < 0:
	raise ValueError("revalue-period must be non-negative")
if args.revalue_start is None:
	args.revalue_start = args.revalue_period
if args.revalue_start < 0:
	raise ValueError("revalue-start must be non-negative")
if args.revalue_samples is None:
	args.revalue_samples = args.batch_size
if not 0 < args.revalue_samples <= args.replay_size:
	raise ValueError("revalue-samples must be between 1 and replay-size")
if not 0.0 <= args.rho <= 1.0:
	raise ValueError("rho must be in [0, 1]")
if args.gac_weight < 0.0:
	raise ValueError("gac-weight must be non-negative")
if args.min_consolidation_margin < 0.0:
	raise ValueError("min-consolidation-margin must be non-negative")


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
retrospective_revaluation = RetrospectiveRevaluation(
	args.revaluation_size, args.batch_size, device
)
diagnostics = ContinualDiagnostics(args.seed)


def train_net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states).max(1)[0]
	pred = Net(states).gather(1, actions)
	targets = rewards + (1 - done) * args.gamma * next_pred.reshape(-1, 1)
	td_loss = criterion(pred, targets)

	gac_loss = torch.zeros((), device=device)
	if retrospective_revaluation.size() > 0:
		revalued_states, revalued_actions, margins = retrospective_revaluation.sample()
		revalued_q = Net(revalued_states)
		greedy_q = revalued_q.gather(1, revalued_actions)
		competitor_q = revalued_q.clone()
		competitor_q.scatter_(1, revalued_actions, -torch.inf)
		competitor_q = competitor_q.max(dim=1, keepdim=True)[0]
		gac_loss = torch.relu(
			competitor_q - greedy_q + args.rho * margins
		).mean()

	loss = td_loss + args.gac_weight * gac_loss
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item(), td_loss.item(), gac_loss.item()


@torch.no_grad()
def revalue():
	"""Consolidate greedy decisions without consulting a task boundary."""
	sample_size = min(args.revalue_samples, exp_replay.size())
	candidates = random.sample(list(exp_replay.memory), sample_size)

	accepted = 0
	for start in range(0, len(candidates), args.batch_size):
		batch = list(itertools.islice(candidates, start, start + args.batch_size))
		states = torch.stack([transition[0] for transition in batch]).to(device)
		online_top = Net(states).topk(2, dim=1)
		actions = online_top.indices[:, :1]
		margins = online_top.values[:, :1] - online_top.values[:, 1:2]
		keep = margins.squeeze(1) >= args.min_consolidation_margin

		if args.align:
			target_top = Target_net(states).topk(2, dim=1)
			keep &= actions.squeeze(1) == target_top.indices[:, 0]
			target_margins = target_top.values[:, :1] - target_top.values[:, 1:2]
			margins = torch.minimum(margins, target_margins)
			keep &= margins.squeeze(1) >= args.min_consolidation_margin

		for state, action, margin in zip(states[keep], actions[keep], margins[keep]):
			retrospective_revaluation.store(state, action, margin)
			accepted += 1
	return len(candidates), accepted


def get_action(obs):
	state = observation_tensor(obs).to(device)
	with torch.no_grad():
		q_values = Net(state.unsqueeze(0))
	if np.random.random() <= args.epsilon:
		action = env.action_space.sample()
	else:
		action = q_values.max(1)[1].item()
	return action


algorithm_name = "DQN w/ GAC"
filename = (
	f"DQN_GAC_Forager_steps_{args.t_steps}_reward_period_{args.reward_period}"
	f"_revalue_period_{args.revalue_period}_rho_{args.rho}_align_{args.align}"
	f"_gac_memory_{args.revaluation_size}_lr_{args.lr1}_seed_{args.seed}"
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
lifetime_evaluator = HistoricalTaskEvaluator(
	args,
	method=algorithm_name,
	series=(
		f"DQN-GAC (rho={args.rho:g}, weight={args.gac_weight:g}, "
		f"period={args.revalue_period})"
	),
	run_filename=(
		f"{filename}_evalfreq_{args.lifetime_eval_frequency}"
		f"_evalsteps_{args.lifetime_eval_steps}"
		f"_evalrollouts_{args.lifetime_eval_rollouts}"
		f"_evaleps_{args.lifetime_eval_epsilon}"
	),
)


returns_array = np.zeros(args.t_steps, dtype=np.float32)
return_window_sum = 0.0
gradient_updates = 0
revaluation_events = 0
last_losses = None
last_revaluation = (0, 0)
last_target_refresh = 0
state = env.reset()

for step in range(args.t_steps):
	action = get_action(state)
	next_state, reward, done, info = env.step(action)
	lifetime_evaluator.observe_phase(info["task_phase"])
	exp_replay.store(state, action, next_state, reward, done)
	state = next_state

	if (
		exp_replay.size() >= args.minimum_replay_history
		and (step + 1) % args.update_frequency == 0
	):
		last_losses = train_net()
		gradient_updates += 1

	# This schedule is deliberately independent of Forager's reward period.
	if (
		args.revalue_period > 0
		and step + 1 >= args.revalue_start
		and (step + 1 - args.revalue_start) % args.revalue_period == 0
		and exp_replay.size() >= args.minimum_replay_history
	):
		last_revaluation = revalue()
		revaluation_events += 1

	# Revalue first when target refresh and revaluation happen on the same step,
	# preserving the original lagged-target agreement check.
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
			"gac/memory_size": retrospective_revaluation.size(),
			"gac/memory_capacity": args.revaluation_size,
			"gac/revaluation_events": revaluation_events,
			"gac/last_sampled": last_revaluation[0],
			"gac/last_accepted": last_revaluation[1],
		})
		if last_losses is not None:
			metrics.update({
				"loss/total": last_losses[0],
				"loss/td": last_losses[1],
				"loss/gac": last_losses[2],
			})
		metrics.update(lifetime_evaluator.metrics(step + 1, Net, device))
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

lifetime_evaluator.close()
env.close()
wandb.finish()
