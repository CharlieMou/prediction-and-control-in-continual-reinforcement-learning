#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

gpu=3
seeds=(16 32 48)
t_steps=$(python -c 'from CL_envs_fixed import NUM_TASK_STAGES; print(NUM_TASK_STAGES * 1_000_000)')
switch_steps=1000000
batch_size=64
lambda_gac=1

mkdir -p logs

pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="dqn_gac_ablation_seed_${seed}_gpu_${gpu}"
	echo "Launching DQN+GAC w/o hist. action gaps, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python dqn_gac_ablation.py \
		--env-name all \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--switch "$switch_steps" \
		--batch-size "$batch_size" \
		--lr1 1e-5 \
		--lambda-gac "$lambda_gac" \
		--rho 0.5 \
		--align True \
		--revaluation-size 10000 \
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
