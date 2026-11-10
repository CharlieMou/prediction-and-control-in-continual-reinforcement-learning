#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

gpu=0
seeds=(16 32 48)
t_steps=10000000
batch_size=32
consolidation_period=50000

mkdir -p logs

pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="DQN_online_ewc_seed_${seed}_gpu_${gpu}"
	echo "Launching DQN online EWC, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python DQN_online_ewc.py \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--batch-size "$batch_size" \
		--aperture-size 9 \
		--reward-period 500000 \
		--replay-size 1000 \
		--minimum-replay-history 50 \
		--update-frequency 4 \
		--target-update-frequency 128 \
		--lr1 3e-3 \
		--ewc-lambda 17.5 \
		--ewc-gamma 0.95 \
		--fisher-batches 100 \
		--wandb-mode offline \
		--consolidation-period "$consolidation_period" \
		--save >"logs/${label}.log" 2>&1 &
	pids+=("$!")
	labels+=("$label")
done

status=0
for index in "${!pids[@]}"; do
	if wait "${pids[$index]}"; then
		echo "Finished: ${labels[$index]}"
	else
		echo "Failed: ${labels[$index]} (see logs/${labels[$index]}.log)" >&2
		status=1
	fi
done

exit "$status"
