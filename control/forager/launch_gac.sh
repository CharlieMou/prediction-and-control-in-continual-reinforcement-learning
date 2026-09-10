#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

seeds=(20 36 52)
gpu=3
t_steps=10000000
batch_size=32
revalue_period=50000

mkdir -p logs

pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="dqn_gac_seed_${seed}_gpu_${gpu}"
	echo "Launching DQN-GAC, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python dqn_gac.py \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--batch-size "$batch_size" \
		--lr1 3e-3 \
		--aperture-size 9 \
		--reward-period 500000 \
		--replay-size 1000 \
		--minimum-replay-history 50 \
		--update-frequency 4 \
		--target-update-frequency 128 \
		--revalue-period "$revalue_period" \
		--revalue-samples "$batch_size" \
		--gac-weight 10 \
		--rho 0.5 \
		--align true \
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
