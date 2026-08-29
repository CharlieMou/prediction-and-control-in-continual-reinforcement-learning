#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

seeds=(0 7)
t_steps=2100000
learning_rate=1e-4
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
		--seed "$seed" \
		--t-steps "$t_steps" \
		--lr1 "$learning_rate" \
		--batch-size "$batch_size" \
		--save \
		"$@" >"logs/${label}.log" 2>&1 &
	pids+=("$!")
	labels+=("$label")
}

# GPU 0: two DQN seeds.
for seed in "${seeds[@]}"; do
	launch 0 DQN.py "$seed"
done

# GPU 1: two IB-DQN(k) seeds.
for seed in "${seeds[@]}"; do
	launch 1 ib_dqn.py "$seed" --k 4
done

# GPU 2: two IP-DQN(k, lambda) seeds.
for seed in "${seeds[@]}"; do
	launch 2 ip_dqn.py "$seed" --k 4 --lambda 0.1 --rank 1 --consolidate 150000
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
