#!/usr/bin/env bash

# Version 1: launch all three seeds on one specified GPU (disabled).
#
# set -uo pipefail
#
# script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$script_dir"
#
# gpu=0
# seeds=(16 32 48)
# t_steps=$(python -c 'from CL_envs_fixed import NUM_TASK_STAGES; print(NUM_TASK_STAGES * 1_000_000)')
# switch_steps=1000000
# batch_size=64
# lambda_vc=1.0
# value_memory_size=10000
#
# mkdir -p logs
#
# pids=()
# labels=()
# for seed in "${seeds[@]}"; do
# 	label="DQN_value_cloning_seed_${seed}_gpu_${gpu}"
# 	echo "Launching DQN Value Cloning, seed=${seed}, GPU=${gpu}"
# 	CUDA_VISIBLE_DEVICES="$gpu" python DQN_value_cloning.py \
# 		--env-name all \
# 		--seed "$seed" \
# 		--t-steps "$t_steps" \
# 		--switch "$switch_steps" \
# 		--batch-size "$batch_size" \
# 		--lr1 1e-5 \
# 		--lambda-vc "$lambda_vc" \
# 		--value-memory-size "$value_memory_size" \
# 		--save >"logs/${label}.log" 2>&1 &
# 	pids+=("$!")
# 	labels+=("$label")
# done
#
# status=0
# for index in "${!pids[@]}"; do
# 	if wait "${pids[$index]}"; then
# 		echo "Finished: ${labels[$index]}"
# 	else
# 		echo "Failed: ${labels[$index]} (see logs/${labels[$index]}.log)" >&2
# 		status=1
# 	fi
# done
#
# exit "$status"

# Version 2: launch all three seeds on the specified GPU (enabled).

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

gpu=2
seeds=(16 32 48)
t_steps=$(python -c 'from CL_envs_fixed import NUM_TASK_STAGES; print(NUM_TASK_STAGES * 1_000_000)')
switch_steps=1000000
batch_size=64
lambda_vc=1
value_memory_size=10000

mkdir -p logs

pids=()
labels=()
for index in "${!seeds[@]}"; do
	seed="${seeds[$index]}"
	label="DQN_value_cloning_seed_${seed}_gpu_${gpu}"
	echo "Launching DQN Value Cloning, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python DQN_value_cloning.py \
		--env-name all \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--switch "$switch_steps" \
		--batch-size "$batch_size" \
		--lr1 1e-5 \
		--lambda-vc "$lambda_vc" \
		--value-memory-size "$value_memory_size" \
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
