#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# 原配置：所有实验均在 GPU 3 上运行
gpu=3
# gpus=(0 1 2)
seeds=(20 36 52)
t_steps=10000000
batch_size=32
revalue_period=50000

mkdir -p logs

pids=()
labels=()
# 原配置：所有 seed 共用同一块 GPU
for seed in "${seeds[@]}"; do
# for launch_index in "${!seeds[@]}"; do
	# seed="${seeds[$launch_index]}"
	# gpu="${gpus[$launch_index]}"
	label="ptdqn_gac_seed_${seed}_gpu_${gpu}"
	echo "Launching PT-DQN-GAC, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python ptdqn_gac.py \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--batch-size "$batch_size" \
		--aperture-size 9 \
		--reward-period 500000 \
		--replay-size 1000 \
		--minimum-replay-history 50 \
		--update-frequency 4 \
		--target-update-frequency 128 \
		--lr1 1e-8 \
		--lr2 1e-3 \
		--update 10000 \
		--decay 0.95 \
		--revalue-period "$revalue_period" \
		--revalue-samples "$batch_size" \
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
