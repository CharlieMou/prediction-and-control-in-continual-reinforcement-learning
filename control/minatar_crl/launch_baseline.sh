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

launch() {
	local gpu="$1"
	local program="$2"
	local seed="$3"
	shift 3
	local label="${program%.py}_seed_${seed}_gpu_${gpu}"

	echo "Launching ${program}, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python "$program" \
		--env-name all \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--switch "$switch_steps" \
		--batch-size "$batch_size" \
		--lr1 1e-5 \
		--save \
		"$@" >"logs/${label}.log" 2>&1 &
	pids+=("$!")
	labels+=("$label")
}

# Each GPU runs both baselines for the same seed.
for gpu in "${!seeds[@]}"; do
	seed="${seeds[$gpu]}"
	launch "$gpu" DQN_value_cloning.py "$seed" \
		--lambda-vc 1.0 --value-memory-size 10000
	launch "$gpu" DQN_ewc.py "$seed" \
		--ewc-lambda 400 --fisher-batches 100
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
