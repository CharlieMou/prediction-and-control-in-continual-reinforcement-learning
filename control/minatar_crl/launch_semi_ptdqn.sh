#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

seeds=(16 32 48)
gpu="${GPU_ID:-0}"
t_steps=$(python -c 'from CL_envs_fixed import NUM_TASK_STAGES; print(NUM_TASK_STAGES * 1_000_000)')
switch_steps=1000000
batch_size=64
pm_buffer_size=50000
permanent_lr=3e-6
transient_lr=1e-4

mkdir -p logs

pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="semi_ptdqn_seed_${seed}_gpu_${gpu}"
	echo "Launching semi-continual PT-DQN, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python Semi_PT_DQN_half.py \
		--env-name all \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--switch "$switch_steps" \
		--batch-size "$batch_size" \
		--lr1 "$permanent_lr" \
		--lr2 "$transient_lr" \
		--update "$pm_buffer_size" \
		--decay 0 \
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
