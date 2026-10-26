"""Option-Critic on the Figure 7 Unending Forager task."""

from argparse import ArgumentParser
from collections import deque
from pathlib import Path
from typing import NamedTuple
import copy
import math
import os
import random
import sys
import tempfile

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli, Categorical
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


class OptionCriticOutput(NamedTuple):
	option_values: torch.Tensor
	termination_probs: torch.Tensor
	action_logits: torch.Tensor


class OptionCriticNetwork(nn.Module):
	"""DQN-aligned encoder with the three heads required by Option-Critic."""

	def __init__(
		self, input_dim, num_actions, image_shape, num_options, temperature
	):
		super().__init__()
		self.num_actions = num_actions
		self.num_options = num_options
		self.temperature = temperature

		# Replacing only the output layer preserves the full DQN visual encoder.
		self.encoder = UnendingCNN(input_dim, num_actions, image_shape)
		feature_dim = self.encoder.fc2.out_features
		self.encoder.output = nn.Identity()
		self.option_values = nn.Linear(feature_dim, num_options)
		self.terminations = nn.Linear(feature_dim, num_options)
		self.option_policies = nn.Linear(
			feature_dim, num_options * num_actions
		)
		self._initialize_heads(feature_dim)

	def _initialize_heads(self, feature_dim):
		# Match the DQN value-head initialization. Zero policy logits and
		# termination logits reproduce the neutral OC initialization.
		nn.init.trunc_normal_(
			self.option_values.weight, std=1.0 / math.sqrt(feature_dim)
		)
		nn.init.zeros_(self.option_values.bias)
		nn.init.zeros_(self.terminations.weight)
		nn.init.zeros_(self.terminations.bias)
		nn.init.zeros_(self.option_policies.weight)
		nn.init.zeros_(self.option_policies.bias)

	def forward(self, inputs):
		features = self.encoder(inputs)
		action_logits = self.option_policies(features).reshape(
			-1, self.num_options, self.num_actions
		)
		return OptionCriticOutput(
			option_values=self.option_values(features),
			termination_probs=torch.sigmoid(self.terminations(features)),
			action_logits=action_logits,
		)

	def action_distribution(self, output, options):
		batch = torch.arange(options.shape[0], device=options.device)
		logits = output.action_logits[batch, options] / self.temperature
		return Categorical(logits=logits)


class OptionValueView(nn.Module):
	"""Expose Q_Omega to the existing continual-learning diagnostics."""

	def __init__(self, network):
		super().__init__()
		self.network = network

	def forward(self, inputs):
		return self.network(inputs).option_values


parser = ArgumentParser(
	description="Option-Critic on the Figure 7 Unending Forager task"
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--t-steps", type=int, default=10_000_000)
parser.add_argument("--lr1", type=float, default=1e-3)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--adam-beta1", type=float, default=0.9)
parser.add_argument("--adam-beta2", type=float, default=0.999)
parser.add_argument("--adam-eps", type=float, default=1e-5)
parser.add_argument("--num-options", type=int, default=4)
parser.add_argument("--temperature", type=float, default=1.0,
	help="softmax temperature of each intra-option policy")
parser.add_argument("--termination-reg", type=float, default=0.01,
	help="margin encouraging temporally extended options")
parser.add_argument("--entropy-reg", type=float, default=0.01,
	help="entropy bonus coefficient for intra-option policies")
parser.add_argument("--option-log-frequency", type=int, default=1000,
	help="steps between lightweight option-dynamics W&B records")
parser.add_argument(
	"--option-visualize-interval", "--option_visualize_interval",
	dest="option_visualize_interval", type=int, default=1_000_000,
	help=(
		"steps between W&B visualizations of the learned option policies; "
		"with --save-model, checkpoints are saved at the same steps"
	),
)
parser.add_argument("--save", action="store_true")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save-model", action="store_true")
add_unending_forager_arguments(parser)
args = parser.parse_args()
validate_unending_forager_arguments(args)

if args.num_options < 2:
	raise ValueError("num-options must be at least 2")
if args.temperature <= 0.0:
	raise ValueError("temperature must be positive")
if args.termination_reg < 0.0:
	raise ValueError("termination-reg must be non-negative")
if args.entropy_reg < 0.0:
	raise ValueError("entropy-reg must be non-negative")
if args.option_log_frequency <= 0:
	raise ValueError("option-log-frequency must be positive")
if args.option_visualize_interval <= 0:
	raise ValueError("option-visualize-interval must be positive")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


env = make_unending_forager_env(args)
input_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

Net = OptionCriticNetwork(
	input_dim, num_actions, env.image_shape, args.num_options, args.temperature
).to(device)
Target_net = copy.deepcopy(Net).to(device).eval()
Target_net.requires_grad_(False)
opt = optim.Adam(
	Net.parameters(),
	lr=args.lr1,
	betas=(args.adam_beta1, args.adam_beta2),
	eps=args.adam_eps,
)
# The DQN replay action field stores the behavior option for OC critic updates.
exp_replay = expReplay(args.batch_size, device, max_size=args.replay_size)
diagnostics = ContinualDiagnostics(args.seed)
option_value_view = OptionValueView(Net)


def continuation_value(output, options):
	"""U(o, s') under call-and-return execution and greedy pi_Omega."""
	batch = torch.arange(options.shape[0], device=options.device)
	beta = output.termination_probs[batch, options]
	q_option = output.option_values[batch, options]
	q_greedy = output.option_values.max(dim=1).values
	return (1.0 - beta) * q_option + beta * q_greedy


def critic_loss():
	states, options, next_states, rewards, done = exp_replay.sample()
	options = options.squeeze(1)
	rewards = rewards.squeeze(1)
	done = done.squeeze(1)
	batch = torch.arange(options.shape[0], device=device)

	predictions = Net(states).option_values[batch, options]
	with torch.no_grad():
		next_online = Net(next_states)
		next_target = Target_net(next_states)
		# Beta belongs to the learned behavior options, while target-network
		# option values provide the stable intra-option Q-learning target.
		next_output = OptionCriticOutput(
			option_values=next_target.option_values,
			termination_probs=next_online.termination_probs,
			action_logits=next_target.action_logits,
		)
		targets = rewards + (1.0 - done) * args.gamma * continuation_value(
			next_output, options
		)
	return F.mse_loss(predictions, targets)


def actor_loss(state, option, action, next_state, reward, done):
	"""Intra-option policy and termination-gradient losses for one transition."""
	states = observation_tensor(state).unsqueeze(0).to(device)
	next_states = observation_tensor(next_state).unsqueeze(0).to(device)
	options = torch.tensor([option], dtype=torch.long, device=device)
	actions = torch.tensor([action], dtype=torch.long, device=device)
	rewards = torch.tensor([reward], dtype=torch.float32, device=device)
	done = torch.tensor([float(done)], dtype=torch.float32, device=device)

	output = Net(states)
	action_dist = Net.action_distribution(output, options)
	log_prob = action_dist.log_prob(actions)
	entropy = action_dist.entropy()

	next_output = Net(next_states)
	with torch.no_grad():
		next_target = Target_net(next_states)
		next_for_target = OptionCriticOutput(
			option_values=next_target.option_values,
			termination_probs=next_output.termination_probs.detach(),
			action_logits=next_target.action_logits,
		)
		return_target = rewards + (1.0 - done) * args.gamma * (
			continuation_value(next_for_target, options)
		)
		policy_advantage = return_target - output.option_values[
			0, option
		].detach()
		termination_advantage = (
			next_output.option_values[0, option]
			- next_output.option_values.max(dim=1).values[0]
			+ args.termination_reg
		).detach()

	policy_loss = -(
		log_prob * policy_advantage + args.entropy_reg * entropy
	).mean()
	termination_loss = (
		(1.0 - done)
		* next_output.termination_probs[0, option]
		* termination_advantage
	).mean()
	return policy_loss, termination_loss, entropy.mean()


def train_net(state, option, action, next_state, reward, done, train_critic):
	policy_loss, termination_loss, entropy = actor_loss(
		state, option, action, next_state, reward, done
	)
	td_loss = critic_loss() if train_critic else torch.zeros((), device=device)
	loss = policy_loss + termination_loss + td_loss
	opt.zero_grad()
	loss.backward()
	opt.step()
	return {
		"total": loss.item(),
		"critic": td_loss.item() if train_critic else None,
		"policy": policy_loss.item(),
		"termination": termination_loss.item(),
		"entropy": entropy.item(),
	}


@torch.no_grad()
def get_decision(obs, current_option, option_terminated):
	state = observation_tensor(obs).unsqueeze(0).to(device)
	output = Net(state)
	if option_terminated:
		if np.random.random() <= args.epsilon:
			current_option = random.randrange(args.num_options)
		else:
			current_option = output.option_values.argmax(dim=1).item()
	options = torch.tensor([current_option], dtype=torch.long, device=device)
	distribution = Net.action_distribution(output, options)
	action = distribution.sample().item()
	return current_option, action, distribution.entropy().item()


@torch.no_grad()
def sample_termination(obs, option):
	state = observation_tensor(obs).unsqueeze(0).to(device)
	probabilities = Net(state).termination_probs[0]
	terminated = Bernoulli(probabilities[option]).sample()
	return bool(terminated.item()), probabilities.cpu().numpy()


def option_gap_metrics():
	metrics = diagnostics.action_gap_metrics(
		option_value_view, exp_replay, device
	)
	if not metrics:
		return metrics
	return {
		"option_gap/mean": metrics["action_gap/mean"],
		"average_option_q/recent": metrics["average_q/recent"],
		"relative_option_gap/mean": metrics["relative_action_gap/mean"],
	}


def option_grid_outputs(reference_observation):
	"""Evaluate every option at each traversable cell without stepping env."""
	core = env._env
	width, height = core._size
	center = args.aperture_size // 2
	scalar_context = np.asarray(reference_observation)[
		-env.scalar_observation_size:
	]
	observations = []
	coordinates = []
	valid = np.ones((height, width), dtype=np.bool_)

	for y in range(height):
		for x in range(width):
			idx = x + y * width
			if (
				idx in core._obj_store.idx_to_name
				and core._obj_store.idx_to_name[idx] == "wall"
			):
				valid[y, x] = False
				continue

			raw = core._get_observation((x, y)).copy()
			# A collectable is removed before the agent observes its new cell.
			# Clear the hypothetical agent cell to match that transition order.
			raw[center, center] = False
			observation = env._paper_observation(raw, None, 0.0)
			# Position is the only variable in this diagnostic slice. Retain the
			# current previous-action, previous-reward, and cue context.
			observation[-env.scalar_observation_size:] = scalar_context
			observations.append(observation)
			coordinates.append((x, y))

	was_training = Net.training
	Net.eval()
	with torch.no_grad():
		batch = observation_tensor(np.stack(observations)).to(device)
		output = Net(batch)
		actions = output.action_logits.argmax(dim=2).cpu().numpy()
		terminations = output.termination_probs.cpu().numpy()
		option_values = output.option_values.cpu().numpy()
	if was_training:
		Net.train()

	policy_maps = np.full(
		(args.num_options, height, width), -1, dtype=np.int64
	)
	termination_maps = np.full(
		(args.num_options, height, width), np.nan, dtype=np.float32
	)
	value_maps = np.full(
		(args.num_options, height, width), np.nan, dtype=np.float32
	)
	for index, (x, y) in enumerate(coordinates):
		policy_maps[:, y, x] = actions[index]
		termination_maps[:, y, x] = terminations[index]
		value_maps[:, y, x] = option_values[index]
	return valid, policy_maps, termination_maps, value_maps


def option_visualization(reference_observation, step):
	"""Create a W&B-ready summary of all learned options on the live world."""
	os.environ.setdefault(
		"MPLCONFIGDIR",
		str(Path(tempfile.gettempdir()) / "forager_matplotlib_cache"),
	)
	import matplotlib
	matplotlib.use("Agg", force=True)
	import matplotlib.pyplot as plt
	from matplotlib.collections import LineCollection
	from matplotlib.patches import Patch

	valid, policies, terminations, option_values = option_grid_outputs(
		reference_observation
	)
	width, height = env._env._size
	world = env._env.render(mode="world")
	figure, axes = plt.subplots(
		3, args.num_options,
		figsize=(4.1 * args.num_options, 11.5),
		dpi=220,
		constrained_layout=True,
		squeeze=False,
	)
	action_vectors = ((0.0, 0.34), (0.34, 0.0), (0.0, -0.34), (-0.34, 0.0))
	option_palette = (
		"#0072B2", "#D55E00", "#009E73", "#CC79A7",
		"#E69F00", "#56B4E9", "#F0E442", "#000000",
	)
	y_grid, x_grid = np.mgrid[0:height, 0:width]
	value_min = float(np.nanmin(option_values))
	value_max = float(np.nanmax(option_values))
	if math.isclose(value_min, value_max):
		value_max = value_min + 1e-6

	for option in range(args.num_options):
		policy_axis = axes[0, option]
		policy_axis.imshow(
			world, origin="lower", extent=(0, width, 0, height), alpha=0.55
		)
		for action, (dx, dy) in enumerate(action_vectors):
			mask = valid & (policies[option] == action)
			policy_axis.quiver(
				x_grid[mask] + 0.5, y_grid[mask] + 0.5,
				np.full(mask.sum(), dx), np.full(mask.sum(), dy),
				angles="xy", scale_units="xy", scale=1.0,
				color="black", width=0.008, pivot="middle",
			)
		policy_axis.set_title(f"Option {option}: greedy intra-option action")

		termination_axis = axes[1, option]
		termination_axis.imshow(
			world, origin="lower", extent=(0, width, 0, height), alpha=0.18
		)
		termination_image = termination_axis.imshow(
			terminations[option], origin="lower", extent=(0, width, 0, height),
			cmap="magma", vmin=0.0, vmax=1.0, alpha=0.82,
		)
		termination_axis.set_title(
			f"Option {option}: termination probability"
		)

		value_axis = axes[2, option]
		value_axis.imshow(
			world, origin="lower", extent=(0, width, 0, height), alpha=0.18
		)
		value_image = value_axis.imshow(
			option_values[option], origin="lower",
			extent=(0, width, 0, height), cmap="viridis",
			vmin=value_min, vmax=value_max, alpha=0.82,
		)
		value_axis.set_title(f"Option {option}: Q_Omega(s, option)")

	for axis in axes.flat:
		axis.set_xlim(0, width)
		axis.set_ylim(0, height)
		axis.set_xticks([])
		axis.set_yticks([])
	figure.colorbar(
		termination_image, ax=list(axes[1]), shrink=0.82,
		label="termination probability",
	)
	figure.colorbar(
		value_image, ax=list(axes[2]), shrink=0.82, label="option value",
	)
	cue = np.asarray(reference_observation)[-4:]
	figure.suptitle(
		f"Option-Critic at step {step:,} | seed={args.seed} | "
		f"task phase={env.task_phase} | cue={cue.astype(int).tolist()} | "
		f"biome generations={list(env.biome_generations)}\n"
		"Each map fixes the current previous-action/reward/cue context and "
		"varies only agent position; walls are masked.",
		fontsize=13,
	)
	all_options_image = wandb.Image(
		figure,
		caption=(
			f"All option policies, termination maps, and option values at "
			f"step {step:,} (seed {args.seed})"
		),
	)
	plt.close(figure)

	def draw_executed_policy(axis, executed_options, title):
		axis.imshow(
			world, origin="lower", extent=(0, width, 0, height), alpha=0.58
		)
		# Encode the executed option with a slightly inset cell outline. This
		# leaves the Forager cell contents unobscured and prevents adjacent
		# cells from overwriting a shared edge with different option colors.
		outline_inset = 0.055
		for option in range(args.num_options):
			color = option_palette[option % len(option_palette)]
			option_mask = valid & (executed_options == option)
			segments = []
			for y, x in zip(*np.nonzero(option_mask)):
				x0, x1 = x + outline_inset, x + 1.0 - outline_inset
				y0, y1 = y + outline_inset, y + 1.0 - outline_inset
				segments.extend((
					((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
					((x1, y1), (x0, y1)), ((x0, y1), (x0, y0)),
				))
			if segments:
				axis.add_collection(LineCollection(
					segments, colors=color, linewidths=1.15,
					alpha=0.98, zorder=3,
				))
			for action, (dx, dy) in enumerate(action_vectors):
				mask = (
					option_mask
					& (policies[option] == action)
				)
				axis.quiver(
					x_grid[mask] + 0.5, y_grid[mask] + 0.5,
					np.full(mask.sum(), dx), np.full(mask.sum(), dy),
					angles="xy", scale_units="xy", scale=1.0,
					color="black", edgecolor="white", linewidth=0.20,
					width=0.009, pivot="middle", zorder=4,
				)
		axis.set_xlim(0, width)
		axis.set_ylim(0, height)
		axis.set_xticks([])
		axis.set_yticks([])
		axis.set_title(title)

	option_legend = [
		Patch(
			facecolor="none",
			edgecolor=option_palette[option % len(option_palette)],
			linewidth=2.0,
			label=f"Option {option}",
		)
		for option in range(args.num_options)
	]
	greedy_options = np.argmax(
		np.where(np.isnan(option_values), -np.inf, option_values), axis=0
	)

	# This map composes the two greedy policies at an option boundary and does
	# not condition on the option that was active before reaching the state.
	boundary_figure, boundary_axis = plt.subplots(
		figsize=(9.5, 9.5), dpi=240, constrained_layout=True
	)
	draw_executed_policy(
		boundary_axis, greedy_options,
		f"Greedy executed-option policy at step {step:,} | seed={args.seed}\n"
		"ignoring termination: reselect argmax Q_Omega at every state",
	)
	boundary_axis.legend(
		handles=option_legend, loc="upper center",
		bbox_to_anchor=(0.5, -0.015), ncol=min(args.num_options, 4),
		frameon=False,
	)
	boundary_image = wandb.Image(
		boundary_figure,
		caption=(
			"Ignoring termination: cell-outline color is argmax Q_Omega; the "
			f"black arrow is that option's greedy action at step {step:,}"
		),
	)
	plt.close(boundary_figure)

	# Under call-and-return, behavior is also conditioned on the incoming
	# option. For a deterministic map, display the modal transition: terminate
	# when beta >= 0.5, otherwise continue the incoming option.
	termination_figure, termination_axes = plt.subplots(
		1, args.num_options,
		figsize=(4.4 * args.num_options, 6.0), dpi=240,
		constrained_layout=False, squeeze=False,
	)
	for incoming_option, axis in enumerate(termination_axes[0]):
		executed_options = np.full(
			(height, width), incoming_option, dtype=np.int64
		)
		terminate = terminations[incoming_option] >= 0.5
		executed_options[terminate] = greedy_options[terminate]
		draw_executed_policy(
			axis, executed_options,
			f"Incoming option {incoming_option}\n"
			"continue if beta < 0.5; otherwise reselect",
		)
	termination_figure.legend(
		handles=option_legend, loc="lower center", ncol=min(args.num_options, 4),
		bbox_to_anchor=(0.5, 0.015), frameon=False,
	)
	termination_figure.suptitle(
		f"Modal call-and-return executed policy at step {step:,} | seed={args.seed}\n"
		"cell-outline color: executed option | black arrow: its greedy action",
		fontsize=13, y=0.98,
	)
	termination_figure.subplots_adjust(
		left=0.02, right=0.99, bottom=0.14, top=0.72, wspace=0.16
	)
	termination_image = wandb.Image(
		termination_figure,
		caption=(
			"With termination: one panel per incoming option; beta >= 0.5 "
			f"triggers greedy option reselection at step {step:,}"
		),
	)
	plt.close(termination_figure)
	return {
		"option_visualization/all_options": all_options_image,
		"option_visualization/executed_option": [
			boundary_image, termination_image,
		],
	}


algorithm_name = "Option-Critic"
filename = (
	f"OC_UnendingFourBiomeForager_steps_{args.t_steps}"
	f"_repeat_{args.reward_repeat}_fov_{args.aperture_size}"
	f"_threshold_{args.consumption_threshold}_options_{args.num_options}"
	f"_temperature_{args.temperature}_termination_reg_{args.termination_reg}"
	f"_entropy_reg_{args.entropy_reg}_lr_{args.lr1}_seed_{args.seed}"
)


def save_training_checkpoint(step):
	"""Save model and optimizer state at a visualization boundary."""
	checkpoint_dir = Path("models") / filename
	checkpoint_dir.mkdir(parents=True, exist_ok=True)
	checkpoint_path = checkpoint_dir / f"checkpoint_step_{step:08d}.pt"
	torch.save({
		"step": step,
		"seed": args.seed,
		"environment": env.game_name,
		"config": vars(args),
		"model_state_dict": Net.state_dict(),
		"target_model_state_dict": Target_net.state_dict(),
		"optimizer_state_dict": opt.state_dict(),
		"model_spec": {
			"input_dim": input_dim,
			"num_actions": num_actions,
			"image_shape": env.image_shape,
			"num_options": args.num_options,
			"temperature": args.temperature,
		},
	}, checkpoint_path)
	return checkpoint_path


wandb.init(
	project="Forager_testbed",
	name=filename,
	group=algorithm_name,
	mode=args.wandb_mode,
	config={
		**vars(args),
		"algorithm": algorithm_name,
		"environment": env.game_name,
		"option_execution": "call-and-return",
		"policy_over_options": "epsilon-greedy",
		"actor_updates": "online after replay warmup",
		"option_visualization": (
			"current world and scalar context; exhaustive traversable positions"
		),
	},
)


returns_array = np.zeros(args.t_steps, dtype=np.float32)
return_window_sum = 0.0
figure_7_ewm_decay = 1.0 - 1e-5
figure_7_ewm_weighted_reward = 0.0
figure_7_ewm_weight = 0.0
critic_updates = 0
actor_updates = 0
optimizer_updates = 0
termination_events = 0
option_selection_events = 0
option_step_counts = np.zeros(args.num_options, dtype=np.int64)
recent_option_lengths = deque(maxlen=1000)
last_losses = None
last_critic_loss = None
last_policy_entropy = math.log(num_actions)
last_termination_probability = 1.0
last_termination_probabilities = np.full(args.num_options, 0.5)
current_option = 0
current_option_length = 0
option_terminated = True
state = env.reset()

for step in range(args.t_steps):
	selected_new_option = option_terminated
	previous_option = current_option
	current_option, action, behavior_entropy = get_decision(
		state, current_option, option_terminated
	)
	last_policy_entropy = behavior_entropy
	last_option_changed = int(
		selected_new_option and current_option != previous_option
	)
	if selected_new_option:
		option_selection_events += 1
		current_option_length = 0
	option_step_counts[current_option] += 1
	next_state, reward, done, info = env.step(action)
	exp_replay.store(state, current_option, next_state, reward, done)
	current_option_length += 1

	ready = exp_replay.size() >= args.minimum_replay_history
	update_critic = ready and (step + 1) % args.update_frequency == 0
	if ready:
		last_losses = train_net(
			state, current_option, action, next_state, reward, done,
			update_critic,
		)
		actor_updates += 1
		optimizer_updates += 1
		last_policy_entropy = last_losses["entropy"]
		if update_critic:
			critic_updates += 1
			last_critic_loss = last_losses["critic"]
			if critic_updates % args.target_update_frequency == 0:
				Target_net.load_state_dict(Net.state_dict())

	option_terminated, last_termination_probabilities = sample_termination(
		next_state, current_option
	)
	last_termination_probability = float(
		last_termination_probabilities[current_option]
	)
	if done:
		option_terminated = True
	if option_terminated:
		termination_events += 1
		recent_option_lengths.append(current_option_length)
	state = next_state

	returns_array[step] = reward
	return_window_sum += reward
	if step >= args.return_smoothing_window:
		return_window_sum -= returns_array[step - args.return_smoothing_window]
	window_size = min(step + 1, args.return_smoothing_window)
	figure_7_ewm_weighted_reward = (
		figure_7_ewm_decay * figure_7_ewm_weighted_reward + reward
	)
	figure_7_ewm_weight = figure_7_ewm_decay * figure_7_ewm_weight + 1.0

	log_option_dynamics = (step + 1) % args.option_log_frequency == 0
	log_training_metrics = (step + 1) % args.log_frequency == 0
	visualize_options = (step + 1) % args.option_visualize_interval == 0
	if visualize_options and args.save_model:
		save_training_checkpoint(step + 1)
	if log_option_dynamics or log_training_metrics or visualize_options:
		metrics = {}
		if log_option_dynamics:
			metrics.update({
				# The option that generated the action at this environment step.
				"option_dynamic/active_option": current_option,
				"option_dynamic/terminated": int(option_terminated),
				"option_dynamic/changed_on_selection": last_option_changed,
				# Backward-compatible names retained from the initial OC logger.
				"oc/current_option": current_option,
				"oc/termination_probability": last_termination_probability,
				"oc/option_changed_on_last_selection": last_option_changed,
			})
			for option, probability in enumerate(last_termination_probabilities):
				metrics[f"option_termination/option_{option}"] = float(
					probability
				)

		if log_training_metrics:
			metrics.update(option_gap_metrics())
			metrics.update({
				"average_reward": return_window_sum / window_size,
				"ewm_reward_5": (
					figure_7_ewm_weighted_reward / figure_7_ewm_weight
				),
				"task_phase": info["task_phase"],
				# Retain the DQN meaning of gradient_updates.
				"gradient_updates": critic_updates,
				"replay_size": exp_replay.size(),
				"oc/actor_updates": actor_updates,
				"oc/optimizer_updates": optimizer_updates,
				"oc/option_selection_events": option_selection_events,
				"oc/termination_events": termination_events,
				"oc/termination_rate": termination_events / (step + 1),
				"oc/mean_option_length": (
					float(np.mean(recent_option_lengths))
					if recent_option_lengths else float(current_option_length)
				),
				"oc/intra_option_entropy": last_policy_entropy,
			})
			for option in range(args.num_options):
				metrics[f"oc/option_{option}_step_fraction"] = (
					option_step_counts[option] / (step + 1)
				)
			if last_losses is not None:
				metrics.update({
					"loss/total": last_losses["total"],
					"loss/policy": last_losses["policy"],
					"loss/termination": last_losses["termination"],
				})
			if last_critic_loss is not None:
				metrics["loss/critic"] = last_critic_loss
		if visualize_options:
			metrics.update(option_visualization(state, step + 1))
		wandb.log(metrics, step=step + 1)


if args.save_model:
	Path("models").mkdir(exist_ok=True)
	torch.save(Net.state_dict(), Path("models") / f"{filename}_Net.pt")
	if args.t_steps % args.option_visualize_interval != 0:
		save_training_checkpoint(args.t_steps)

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
