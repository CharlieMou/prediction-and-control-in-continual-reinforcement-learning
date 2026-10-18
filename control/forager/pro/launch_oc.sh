#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

gpu=0
seeds=(16 32 48)
t_steps=10000000
batch_size=32
num_options=4
temperature=1.0
termination_reg=0.015
entropy_reg=0.01
option_log_frequency=1000

mkdir -p logs

pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="OC_UnendingFourBiome_seed_${seed}_gpu_${gpu}"
	echo "Launching oc.py, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python oc.py \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--batch-size "$batch_size" \
		--aperture-size 9 \
		--fourier-terms 10 \
		--reward-repeat 1000 \
		--consumption-threshold 10000 \
		--cue-period 100 \
		--cue-duration 10 \
		--gamma 0.99 \
		--epsilon 0.1 \
		--replay-size 1000 \
		--minimum-replay-history 50 \
		--update-frequency 4 \
		--target-update-frequency 128 \
		--lr1 3e-4 \
		--num-options "$num_options" \
		--temperature "$temperature" \
		--termination-reg "$termination_reg" \
		--entropy-reg "$entropy_reg" \
		--option-log-frequency "$option_log_frequency" \
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
