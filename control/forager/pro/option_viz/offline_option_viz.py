"""Render greedy OC executed-option policies from a saved checkpoint."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
PRO_DIR = SCRIPT_DIR.parent
FORAGER_DIR = PRO_DIR.parent
MODELS_DIR = PRO_DIR / "models"
if str(FORAGER_DIR) not in sys.path:
	sys.path.insert(0, str(FORAGER_DIR))

os.environ.setdefault(
	"MPLCONFIGDIR",
	str(Path(tempfile.gettempdir()) / "forager_matplotlib_cache"),
)
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from forager_env import UnendingFourBiomeForagerEnv
from model import UnendingCNN
from replay import observation_tensor


OPTION_PALETTE = (
	"#0072B2", "#D55E00", "#009E73", "#CC79A7",
	"#E69F00", "#56B4E9", "#F0E442", "#000000",
)
ACTION_VECTORS = (
	(0.0, 0.34), (0.34, 0.0), (0.0, -0.34), (-0.34, 0.0),
)


class OptionCriticOutput(NamedTuple):
	option_values: torch.Tensor
	termination_probs: torch.Tensor
	action_logits: torch.Tensor


class OptionCriticNetwork(nn.Module):
	"""Checkpoint-compatible copy of the OC network in ``pro/oc.py``."""

	def __init__(
		self, input_dim, num_actions, image_shape, num_options, temperature
	):
		super().__init__()
		self.num_actions = num_actions
		self.num_options = num_options
		self.temperature = temperature
		self.encoder = UnendingCNN(input_dim, num_actions, image_shape)
		feature_dim = self.encoder.fc2.out_features
		self.encoder.output = nn.Identity()
		self.option_values = nn.Linear(feature_dim, num_options)
		self.terminations = nn.Linear(feature_dim, num_options)
		self.option_policies = nn.Linear(
			feature_dim, num_options * num_actions
		)

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


def resolve_checkpoint(path):
	path = path.expanduser()
	for candidate in (path, PRO_DIR / path, MODELS_DIR / path):
		if candidate.is_file():
			return candidate.resolve()
	raise FileNotFoundError(
		f"checkpoint not found: {path}; also searched under {MODELS_DIR}"
	)


def load_checkpoint(path, device):
	try:
		checkpoint = torch.load(path, map_location=device, weights_only=False)
	except TypeError:
		checkpoint = torch.load(path, map_location=device)
	if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
		raise ValueError(
			"expected checkpoint_step_*.pt with model_state_dict, config, and "
			"model_spec; legacy *_Net.pt files do not contain enough metadata"
		)
	return checkpoint


def make_environment(config, seed_override):
	seed = config["seed"] if seed_override is None else seed_override
	return UnendingFourBiomeForagerEnv(
		seed=seed,
		aperture_size=config.get("aperture_size", 9),
		fourier_terms=config.get("fourier_terms", 10),
		reward_repeat=config.get("reward_repeat", 1000),
		consumption_threshold=config.get("consumption_threshold", 10000),
		cue_period=config.get("cue_period", 100),
		cue_duration=config.get("cue_duration", 10),
	)


@torch.no_grad()
def option_grid_outputs(net, env, reference_observation, device, num_options):
	"""Evaluate all option heads at every traversable agent position."""
	core = env._env
	width, height = core._size
	center = env.aperture_size // 2
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
			raw[center, center] = False
			observation = env._paper_observation(raw, None, 0.0)
			observation[-env.scalar_observation_size:] = scalar_context
			observations.append(observation)
			coordinates.append((x, y))

	output = net(observation_tensor(np.stack(observations)).to(device))
	actions = output.action_logits.argmax(dim=2).cpu().numpy()
	terminations = output.termination_probs.cpu().numpy()
	option_values = output.option_values.cpu().numpy()
	policy_maps = np.full(
		(num_options, height, width), -1, dtype=np.int64
	)
	termination_maps = np.full(
		(num_options, height, width), np.nan, dtype=np.float32
	)
	value_maps = np.full(
		(num_options, height, width), np.nan, dtype=np.float32
	)
	for index, (x, y) in enumerate(coordinates):
		policy_maps[:, y, x] = actions[index]
		termination_maps[:, y, x] = terminations[index]
		value_maps[:, y, x] = option_values[index]
	return valid, policy_maps, termination_maps, value_maps


def draw_executed_policy(
	axis, world, valid, policies, executed_options, option_palette, title,
	outline_linewidth=1.15,
):
	height, width = valid.shape
	y_grid, x_grid = np.mgrid[0:height, 0:width]
	axis.imshow(
		world, origin="lower", extent=(0, width, 0, height), alpha=0.58
	)
	# Use an inset outline rather than a colored fill/arrow so the underlying
	# Forager cell remains visible and neighboring option colors cannot fight
	# over the same grid edge.
	outline_inset = 0.055
	for option in range(policies.shape[0]):
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
				segments, colors=color, linewidths=outline_linewidth,
				alpha=0.98, zorder=3,
			))
		for action, (dx, dy) in enumerate(ACTION_VECTORS):
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


def render_policies(
	env, valid, policies, terminations, option_values,
	checkpoint, output_without, output_with, dpi,
):
	world = env._env.render(mode="world")
	num_options = policies.shape[0]
	greedy_options = np.argmax(
		np.where(np.isnan(option_values), -np.inf, option_values), axis=0
	)
	legend = [
		Patch(
			facecolor="none",
			edgecolor=OPTION_PALETTE[option % len(OPTION_PALETTE)],
			linewidth=2.0,
			label=f"Option {option}",
		)
		for option in range(num_options)
	]
	step = checkpoint.get("step", "unknown")
	seed = checkpoint.get("seed", "unknown")

	boundary_figure, boundary_axis = plt.subplots(
		figsize=(9.5, 9.5), dpi=dpi, constrained_layout=True
	)
	draw_executed_policy(
		boundary_axis, world, valid, policies, greedy_options, OPTION_PALETTE,
		f"Greedy executed-option policy | step {step} | seed={seed}\n"
		"ignoring termination: reselect argmax Q_Omega at every state\n"
		"environment context: fresh reset with checkpoint configuration",
		outline_linewidth=2.25,
	)
	boundary_axis.legend(
		handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.015),
		ncol=min(num_options, 4), frameon=False,
	)
	output_without.parent.mkdir(parents=True, exist_ok=True)
	boundary_figure.savefig(
		output_without, format="png", dpi=dpi, bbox_inches="tight"
	)
	plt.close(boundary_figure)

	termination_figure, axes = plt.subplots(
		1, num_options,
		figsize=(4.4 * num_options, 6.0), dpi=dpi,
		constrained_layout=False, squeeze=False,
	)
	for incoming_option, axis in enumerate(axes[0]):
		executed_options = np.full(
			valid.shape, incoming_option, dtype=np.int64
		)
		terminate = terminations[incoming_option] >= 0.5
		executed_options[terminate] = greedy_options[terminate]
		draw_executed_policy(
			axis, world, valid, policies, executed_options, OPTION_PALETTE,
			f"Incoming option {incoming_option}\n"
			"continue if beta < 0.5; otherwise reselect",
		)
	termination_figure.legend(
		handles=legend, loc="lower center", ncol=min(num_options, 4),
		bbox_to_anchor=(0.5, 0.015), frameon=False,
	)
	termination_figure.suptitle(
		f"Modal call-and-return executed policy | step {step} | seed={seed}\n"
		"cell-outline color: executed option | black arrow: its greedy action\n"
		"environment context: fresh reset with checkpoint configuration",
		y=0.98,
	)
	termination_figure.subplots_adjust(
		left=0.02, right=0.99, bottom=0.14, top=0.72, wspace=0.16
	)
	output_with.parent.mkdir(parents=True, exist_ok=True)
	termination_figure.savefig(
		output_with, format="png", dpi=dpi, bbox_inches="tight"
	)
	plt.close(termination_figure)


def parse_args():
	parser = ArgumentParser(description=__doc__)
	parser.add_argument(
		"--checkpoint", type=Path, required=True,
		help=(
			"checkpoint path, or run-directory/checkpoint filename relative "
			f"to {MODELS_DIR}"
		),
	)
	parser.add_argument(
		"--output", type=Path,
		help=(
			"base .png path; _without_termination and _with_termination are "
			"appended to create the two output files"
		),
	)
	parser.add_argument("--dpi", type=int, default=300)
	parser.add_argument(
		"--device", choices=("auto", "cpu", "cuda"), default="auto"
	)
	parser.add_argument(
		"--environment-seed", type=int,
		help="override the checkpoint seed used for the evaluation world",
	)
	return parser.parse_args()


def output_paths(output, checkpoint_path):
	if output is None:
		base = (
			SCRIPT_DIR / "outputs" / checkpoint_path.parent.name
			/ checkpoint_path.stem
		)
	else:
		if output.suffix.lower() != ".png":
			raise ValueError("output must use the .png extension")
		base = output.with_suffix("")
	return (
		base.with_name(f"{base.name}_without_termination.png"),
		base.with_name(f"{base.name}_with_termination.png"),
	)


def main():
	args = parse_args()
	if args.dpi <= 0:
		raise ValueError("dpi must be positive")
	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(args.device)
	checkpoint_path = resolve_checkpoint(args.checkpoint)
	checkpoint = load_checkpoint(checkpoint_path, device)
	config = checkpoint["config"]
	spec = checkpoint["model_spec"]
	env = make_environment(config, args.environment_seed)
	reference_observation = env.reset()
	net = OptionCriticNetwork(
		spec["input_dim"], spec["num_actions"], spec["image_shape"],
		spec["num_options"], spec["temperature"],
	).to(device)
	net.load_state_dict(checkpoint["model_state_dict"])
	net.eval()
	valid, policies, terminations, option_values = option_grid_outputs(
		net, env, reference_observation, device, spec["num_options"]
	)
	output_without, output_with = output_paths(args.output, checkpoint_path)
	render_policies(
		env, valid, policies, terminations, option_values, checkpoint,
		output_without, output_with, args.dpi,
	)
	env.close()
	print(f"Saved no-termination policy to {output_without.resolve()}")
	print(f"Saved call-and-return policy to {output_with.resolve()}")


if __name__ == "__main__":
	main()
