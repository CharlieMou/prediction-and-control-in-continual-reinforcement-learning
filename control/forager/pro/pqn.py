"""PQN on the Figure 7 unending Forager benchmark.

The optimization follows Algorithm 1 of "Simplifying Deep Temporal Difference
Learning" and CleanRL's PQN implementation: collect synchronous trajectories,
construct frozen backward Q(lambda) targets, and reuse the current rollout for
multiple shuffled minibatch epochs.  PQN has no replay buffer or target network.
"""

from argparse import ArgumentParser
from pathlib import Path
import random
import sys

import numpy as np
import pickle
import torch
import torch.optim as optim
import wandb


FORAGER_DIR = Path(__file__).resolve().parent.parent
if str(FORAGER_DIR) not in sys.path:
	sys.path.insert(0, str(FORAGER_DIR))

from diagnostics import ContinualDiagnostics
from forager_env import (
	add_unending_environment_arguments,
	make_vectorized_unending_forager_env,
	validate_unending_environment_arguments,
)
from model import UnendingCNN
from replay import observation_tensor


class CurrentRolloutStateBuffer:
	"""Expose the current rollout through the existing diagnostics interface."""

	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.memory = []

	def update(self, states):
		self.memory = [(state,) for state in states.detach().cpu()]

	def size(self):
		return len(self.memory)


parser = ArgumentParser(description="PQN on the Figure 7 Unending Forager task")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t-steps", type=int, default=10_000_000)
parser.add_argument("--lr1", "--learning-rate", dest="lr1", type=float,
		default=1e-3)
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--rollout-steps", type=int, default=32,
		help="transitions collected per environment before each update")
parser.add_argument("--num-minibatches", type=int, default=4)
parser.add_argument("--update-epochs", type=int, default=2)
parser.add_argument("--q-lambda", type=float, default=0.65)
parser.add_argument("--epsilon", type=float, default=0.1,
		help="fixed epsilon-greedy exploration rate, matching DQN")
parser.add_argument("--adam-beta1", type=float, default=0.9)
parser.add_argument("--adam-beta2", type=float, default=0.999)
parser.add_argument("--adam-eps", type=float, default=1e-5)
parser.add_argument("--save", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save-model", action="store_true")
add_unending_environment_arguments(parser)
args = parser.parse_args()
validate_unending_environment_arguments(args)

for name in ("lr1", "num_envs", "rollout_steps", "num_minibatches",
		"update_epochs"):
	if getattr(args, name) <= 0:
		raise ValueError(f"{name.replace('_', '-')} must be positive")
if not 0.0 <= args.q_lambda <= 1.0:
	raise ValueError("q-lambda must be in [0, 1]")
if not 0.0 <= args.epsilon <= 1.0:
	raise ValueError("epsilon must be in [0, 1]")
args.rollout_batch_size = args.num_envs * args.rollout_steps
if args.rollout_batch_size % args.num_minibatches != 0:
	raise ValueError(
		"num-minibatches must divide num-envs * rollout-steps exactly"
	)
args.minibatch_size = args.rollout_batch_size // args.num_minibatches
segments = []
if args.t_steps >= args.log_frequency:
	segments.append(args.log_frequency)
final_segment = args.t_steps % args.log_frequency
if final_segment or not segments:
	segments.append(final_segment or args.t_steps)
for segment in segments:
	partial_rollout_steps = segment % args.rollout_steps
	if (
		partial_rollout_steps > 0
		and partial_rollout_steps * args.num_envs % args.minibatch_size != 0
	):
		raise ValueError(
			"checkpoint-split rollouts must be divisible by the TD minibatch size"
		)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


env = make_vectorized_unending_forager_env(args)
input_dim = env.single_observation_space.shape[0]
num_actions = env.single_action_space.n

Net = UnendingCNN(input_dim, num_actions, env.image_shape).to(device)
opt = optim.Adam(
	Net.parameters(),
	lr=args.lr1,
	betas=(args.adam_beta1, args.adam_beta2),
	eps=args.adam_eps,
)
criterion = torch.nn.MSELoss()
diagnostics = ContinualDiagnostics(args.seed)
diagnostic_states = CurrentRolloutStateBuffer(batch_size=32)


@torch.no_grad()
def get_actions_and_values(obs):
	states = observation_tensor(obs).to(device)
	q_values = Net(states)
	greedy_actions = q_values.argmax(dim=1)
	random_actions = torch.randint(num_actions, (args.num_envs,), device=device)
	explore = torch.rand(args.num_envs, device=device) < args.epsilon
	actions = torch.where(explore, random_actions, greedy_actions)
	return states, actions, q_values.max(dim=1).values


@torch.no_grad()
def q_lambda_returns(rewards, done, values, next_states):
	returns = torch.empty_like(rewards)
	next_values = Net(next_states).max(dim=1).values
	returns[-1] = rewards[-1] + (
		args.gamma * (~done[-1]).float() * next_values
	)
	for rollout_step in range(len(rewards) - 2, -1, -1):
		bootstrap = (
			args.q_lambda * returns[rollout_step + 1]
			+ (1.0 - args.q_lambda) * values[rollout_step + 1]
		)
		returns[rollout_step] = rewards[rollout_step] + (
			args.gamma * (~done[rollout_step]).float() * bootstrap
		)
	return returns


def train_net(states, actions, targets):
	states = states.reshape(-1, input_dim)
	actions = actions.reshape(-1)
	targets = targets.reshape(-1)
	batch_size = len(states)
	minibatch_size = args.minibatch_size
	if batch_size % minibatch_size != 0:
		raise ValueError(
			"each rollout batch must be divisible by the TD minibatch size"
		)
	indices = np.arange(batch_size)
	last_loss = None
	updates = 0

	for _ in range(args.update_epochs):
		np.random.shuffle(indices)
		for start in range(0, batch_size, minibatch_size):
			minibatch = torch.as_tensor(
				indices[start:start + minibatch_size], device=device
			)
			predictions = Net(states[minibatch]).gather(
				1, actions[minibatch, None]
			).squeeze(1)
			loss = criterion(predictions, targets[minibatch])
			opt.zero_grad()
			loss.backward()
			opt.step()
			last_loss = loss.item()
			updates += 1
	return last_loss, updates


algorithm_name = "PQN"
filename = (
	f"PQN_UnendingFourBiomeForager_steps_{args.t_steps}"
	f"_repeat_{args.reward_repeat}_fov_{args.aperture_size}"
	f"_threshold_{args.consumption_threshold}_envs_{args.num_envs}"
	f"_rollout_{args.rollout_steps}_lambda_{args.q_lambda}"
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
		"evaluation_world": 0,
		"environment_semantics": "independent_continual_worlds",
		"training_transitions": args.t_steps * args.num_envs,
		"uses_replay": False,
		"uses_target_network": False,
		"uses_gradient_clipping": False,
		"exploration_decay": False,
	},
)


returns_array = np.zeros(args.t_steps, dtype=np.float32)
return_window_sum = 0.0
figure_7_ewm_decay = 1.0 - 1e-5
figure_7_ewm_weighted_reward = 0.0
figure_7_ewm_weight = 0.0
main_step = 0
gradient_updates = 0
last_loss = None
next_log = args.log_frequency
state = env.reset()

while main_step < args.t_steps:
	remaining_steps = args.t_steps - main_step
	steps_until_log = next_log - main_step
	rollout_length = min(
		args.rollout_steps, remaining_steps, steps_until_log
	)
	states = torch.empty(
		(rollout_length, args.num_envs, input_dim),
		dtype=torch.float32,
		device=device,
	)
	actions = torch.empty(
		(rollout_length, args.num_envs), dtype=torch.long, device=device
	)
	rewards = torch.empty(
		(rollout_length, args.num_envs), dtype=torch.float32, device=device
	)
	done = torch.empty(
		(rollout_length, args.num_envs), dtype=torch.bool, device=device
	)
	values = torch.empty_like(rewards)

	for rollout_step in range(rollout_length):
		state_tensor, selected_actions, state_values = get_actions_and_values(
			state
		)
		next_state, raw_rewards, terminal, infos = env.step(
			selected_actions.cpu().numpy()
		)

		states[rollout_step].copy_(state_tensor)
		actions[rollout_step].copy_(selected_actions)
		rewards[rollout_step].copy_(torch.as_tensor(
			raw_rewards, device=device
		))
		done[rollout_step].copy_(torch.as_tensor(terminal, device=device))
		values[rollout_step].copy_(state_values)
		state = next_state

		main_reward = float(raw_rewards[0])
		returns_array[main_step] = main_reward
		return_window_sum += main_reward
		if main_step >= args.return_smoothing_window:
			return_window_sum -= returns_array[
				main_step - args.return_smoothing_window
			]
		figure_7_ewm_weighted_reward = (
			figure_7_ewm_decay * figure_7_ewm_weighted_reward + main_reward
		)
		figure_7_ewm_weight = (
			figure_7_ewm_decay * figure_7_ewm_weight + 1.0
		)
		main_step += 1

	with torch.no_grad():
		targets = q_lambda_returns(
			rewards,
			done,
			values,
			observation_tensor(state).to(device),
		)
	last_loss, updates = train_net(states, actions, targets)
	gradient_updates += updates

	if main_step == next_log or main_step == args.t_steps:
		diagnostic_states.update(states.reshape(-1, input_dim))
		metrics = diagnostics.action_gap_metrics(
			Net, diagnostic_states, device
		)
		window_size = min(main_step, args.return_smoothing_window)
		metrics.update({
			"average_reward": return_window_sum / window_size,
			"ewm_reward_5": (
				figure_7_ewm_weighted_reward / figure_7_ewm_weight
			),
			"task_phase": infos[0]["task_phase"],
			"gradient_updates": gradient_updates,
			"replay_size": 0,
			"epsilon": args.epsilon,
			"pqn/env_clock": infos[0]["clock"],
			"pqn/rollout_size": rollout_length * args.num_envs,
			"pqn/td_minibatch_size": args.minibatch_size,
			"pqn/effective_num_minibatches": (
				rollout_length * args.num_envs // args.minibatch_size
			),
			"pqn/training_transitions": main_step * args.num_envs,
			"loss/td": last_loss,
		})
		wandb.log(metrics, step=main_step)
		while next_log <= main_step:
			next_log += args.log_frequency


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
