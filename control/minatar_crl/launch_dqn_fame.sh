#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

seeds=(16 32 48)
t_steps=9000000
switch_steps=1000000
batch_size=64

mkdir -p logs

pids=()
labels=()
for gpu in "${!seeds[@]}"; do
	seed="${seeds[$gpu]}"
	label="dqn_fame_seed_${seed}_gpu_${gpu}"
	echo "Launching FAME, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python dqn_fame.py \
		--env-name all \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--switch "$switch_steps" \
		--batch-size "$batch_size" \
		--lr1 1e-3 \
		--lr2 1e-5 \
		--size-fast2meta 12000 \
		--detection-step 1200 \
		--epoch-meta 200 \
		--warmstep 50000 \
		--lambda-reg 1.0 \
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
