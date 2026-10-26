#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

gpu=1
seeds=(17 33 49)
t_steps=10000000

mkdir -p logs

# Four independent continual worlds keep PQN vectorized without the cost of the
# paper's 128 Atari environments.  A 4x32 rollout split into four minibatches
# gives TD minibatches of 32, matching DQN/PT-DQN; two epochs also give eight
# updates per 32 main-world steps, matching DQN's one update every four steps.
# Optimizer and exploration settings match DQN; t_steps counts world-zero steps.
pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="PQN_UnendingFourBiome_seed_${seed}_gpu_${gpu}"
	echo "Launching pqn.py, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python pqn.py \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--aperture-size 9 \
		--fourier-terms 10 \
		--reward-repeat 1000 \
		--consumption-threshold 10000 \
		--cue-period 100 \
		--cue-duration 10 \
		--num-envs 4 \
		--rollout-steps 32 \
		--num-minibatches 4 \
		--update-epochs 2 \
		--q-lambda 0.65 \
		--epsilon 0.1 \
		--lr1 3e-4 \
		--adam-beta1 0.9 \
		--adam-beta2 0.999 \
		--adam-eps 1e-5 \
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
