"""Plot mean historical-task lifetime return with 95% seed confidence bands."""

from argparse import ArgumentParser
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = Path(__file__).resolve().parent


def load_points(input_dir):
	points = defaultdict(list)
	for path in sorted(input_dir.glob("*_lifetime_return.csv")):
		with path.open(newline="", encoding="utf-8") as csv_file:
			for row in csv.DictReader(csv_file):
				if not row.get("lifetime_return/historical"):
					continue
				key = (
					row["series"],
					row.get("configuration", "legacy"),
					int(float(row["lifetime_return/historical_phase"])),
					int(row["step"]),
				)
				points[key].append(float(row["lifetime_return/historical"]))
	return points


def plot(input_dir, output_prefix):
	points = load_points(input_dir)
	if not points:
		raise ValueError(f"no lifetime-return data found in {input_dir}")

	fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True)
	configurations = sorted({key[:2] for key in points})
	series_counts = {
		series: sum(name == series for name, _ in configurations)
		for series, _ in configurations
	}
	colors = plt.get_cmap("tab10")
	for phase, axis in enumerate(axes):
		for series_index, (series, configuration) in enumerate(configurations):
			steps = sorted(
				step for name, config, historical_phase, step in points
				if name == series and config == configuration
				and historical_phase == phase
			)
			if not steps:
				continue
			means, cis = [], []
			for step in steps:
				values = np.asarray(points[(series, configuration, phase, step)])
				means.append(values.mean())
				cis.append(
					0.0 if len(values) == 1
					else 1.96 * values.std(ddof=1) / np.sqrt(len(values))
				)
			x = np.asarray(steps, dtype=np.float64) / 1_000_000.0
			y = np.asarray(means)
			ci = np.asarray(cis)
			color = colors(series_index % 10)
			label = series if series_counts[series] == 1 \
				else f"{series} [{configuration}]"
			axis.plot(x, y, label=label, color=color)
			axis.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)
		axis.set_title(f"Historical reward mapping {phase + 1}")
		axis.set_ylabel("Lifetime return")
		axis.grid(alpha=0.25)
	axes[-1].set_xlabel("Training steps (millions)")
	handles, labels = axes[0].get_legend_handles_labels()
	if not handles:
		handles, labels = axes[1].get_legend_handles_labels()
	fig.legend(
		handles, labels, loc="center left", bbox_to_anchor=(0.79, 0.5),
	)
	fig.suptitle("TwoBiomeForagerEnv historical-task reevaluation")
	fig.tight_layout(rect=(0.0, 0.0, 0.79, 0.96))

	output_prefix.parent.mkdir(parents=True, exist_ok=True)
	for suffix in (".pdf", ".png"):
		fig.savefig(output_prefix.with_suffix(suffix), bbox_inches="tight", dpi=300)
	plt.close(fig)


def main():
	parser = ArgumentParser()
	parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
	parser.add_argument(
		"--output-prefix", type=Path,
		default=DEFAULT_DIR / "figs" / "two_biome_lifetime_return",
	)
	args = parser.parse_args()
	plot(args.input_dir, args.output_prefix)


if __name__ == "__main__":
	main()
