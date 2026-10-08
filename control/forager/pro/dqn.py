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
	add_unending_forager_arguments,
	make_unending_forager_env,
	validate_unending_forager_arguments,
)
from model import UnendingCNN
from replay import expReplay, observation_tensor


parser = ArgumentParser(description="DQN on the Figure 7 Unending Forager task")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t-steps", type=int, default=10_000_000)
parser.add_argument("--lr1", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--adam-beta1", type=float, default=0.9)
parser.add_argument("--adam-beta2", type=float, default=0.999)
parser.add_argument("--adam-eps", type=float, default=1e-5)
parser.add_argument("--save", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save-model", action="store_true")
add_unending_forager_arguments(parser)
args = parser.parse_args()
validate_unending_forager_arguments(args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


env = make_unending_forager_env(args)
input_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

Net = UnendingCNN(input_dim, num_actions, env.image_shape).to(device)
Target_net = UnendingCNN(input_dim, num_actions, env.image_shape).to(device)
Target_net.load_state_dict(Net.state_dict())
opt = optim.Adam(
	Net.parameters(),
	lr=args.lr1,
	betas=(args.adam_beta1, args.adam_beta2),
	eps=args.adam_eps,
)
criterion = torch.nn.MSELoss()
exp_replay = expReplay(args.batch_size, device, max_size=args.replay_size)
diagnostics = ContinualDiagnostics(args.seed)


def train_net():
	states, actions, next_states, rewards, done = exp_replay.sample()
	with torch.no_grad():
		next_pred = Target_net(next_states).max(1)[0]
	pred = Net(states).gather(1, actions)
	targets = rewards + (1 - done) * args.gamma * next_pred.reshape(-1, 1)
	loss = criterion(pred, targets)
	opt.zero_grad()
	loss.backward()
	opt.step()
	return loss.item()


def get_action(obs):
	state = observation_tensor(obs).to(device)
	with torch.no_grad():
		q_values = Net(state.unsqueeze(0))
	if np.random.random() <= args.epsilon:
		action = env.action_space.sample()
	else:
		action = q_values.max(1)[1].item()
	return action


algorithm_name = "DQN"
filename = (
	f"DQN_UnendingFourBiomeForager_steps_{args.t_steps}"
	f"_repeat_{args.reward_repeat}_fov_{args.aperture_size}"
	f"_threshold_{args.consumption_threshold}_batch_{args.batch_size}"
	f"_lr_{args.lr1}_seed_{args.seed}"
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
figure_7_ewm_decay = 1.0 - 1e-5
figure_7_ewm_weighted_reward = 0.0
figure_7_ewm_weight = 0.0
gradient_updates = 0
last_loss = None
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
		last_loss = train_net()
		gradient_updates += 1
		if gradient_updates % args.target_update_frequency == 0:
			Target_net.load_state_dict(Net.state_dict())

	returns_array[step] = reward
	return_window_sum += reward
	if step >= args.return_smoothing_window:
		return_window_sum -= returns_array[step - args.return_smoothing_window]
	window_size = min(step + 1, args.return_smoothing_window)
	figure_7_ewm_weighted_reward = (
		figure_7_ewm_decay * figure_7_ewm_weighted_reward + reward
	)
	figure_7_ewm_weight = figure_7_ewm_decay * figure_7_ewm_weight + 1.0

	if (step + 1) % args.log_frequency == 0:
		metrics = diagnostics.action_gap_metrics(Net, exp_replay, device)
		metrics.update({
			"average_reward": return_window_sum / window_size,
			"ewm_reward_5": (
				figure_7_ewm_weighted_reward / figure_7_ewm_weight
			),
			"task_phase": info["task_phase"],
			"gradient_updates": gradient_updates,
			"replay_size": exp_replay.size(),
		})
		if last_loss is not None:
			metrics["loss/td"] = last_loss
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
