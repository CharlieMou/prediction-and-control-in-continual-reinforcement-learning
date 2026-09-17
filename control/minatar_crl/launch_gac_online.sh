#!/usr/bin/env bash
# Usage: ./launch_gac_online.sh [gpu=0] [streaming_online=false]

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

seeds=(16 32 48)
gpu="${1:-0}"
streaming_online="${2:-false}"
t_steps=9000000
switch_steps=1000000
refresh_period=50000
refresh_size=25000
batch_size=64
lambda_gac=0.01

mkdir -p logs

pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="dqn_gac_online_streaming_${streaming_online}_seed_${seed}_gpu_${gpu}"
	echo "Launching online DQN-GAC, streaming=${streaming_online}, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python dqn_gac_online.py \
		--env-name all \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--switch "$switch_steps" \
		--streaming_online "$streaming_online" \
		--refresh_period "$refresh_period" \
		--refresh_size "$refresh_size" \
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
