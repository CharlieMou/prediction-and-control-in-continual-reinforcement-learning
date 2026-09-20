#!/usr/bin/env bash
# Usage: ./launch_ptdqn.sh [gpu=0]

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

seeds=(16 32 48)
gpu="${1:-0}"
t_steps=$(python -c 'from CL_envs_fixed import NUM_TASK_STAGES; print(NUM_TASK_STAGES * 1_000_000)')
switch_steps=1000000
batch_size=64

mkdir -p logs

pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="PT_DQN_half_seed_${seed}_gpu_${gpu}"
	echo "Launching PT_DQN_half.py, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python PT_DQN_half.py \
		--env-name all \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--switch "$switch_steps" \
		--batch-size "$batch_size" \
		--lr1 1e-8 \
		--lr2 1e-4 \
		--update 50000 \
		--decay 0.75 \
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
