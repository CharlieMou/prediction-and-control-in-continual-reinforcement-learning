#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

gpu=3
seeds=(19 35 51)
t_steps=10000000
batch_size=32
revalue_period=50000
revalue_start=50000
revalue_samples="$batch_size"
rho=0.5
gac_weight=1.0
min_consolidation_margin=0.0
align=true

mkdir -p logs

pids=()
labels=()
for seed in "${seeds[@]}"; do
	label="PT_DQN_GAC_UnendingFourBiome_seed_${seed}_gpu_${gpu}"
	echo "Launching PT-DQN-GAC, seed=${seed}, GPU=${gpu}"
	CUDA_VISIBLE_DEVICES="$gpu" python pt_dqn_gac.py \
		--seed "$seed" \
		--t-steps "$t_steps" \
		--batch-size "$batch_size" \
		--aperture-size 9 \
		--fourier-terms 10 \
		--reward-repeat 1000 \
		--consumption-threshold 10000 \
		--cue-period 100 \
		--cue-duration 10 \
		--replay-size 1000 \
		--minimum-replay-history 50 \
		--update-frequency 4 \
		--target-update-frequency 128 \
		--lr1 3e-8 \
		--lr2 3e-4 \
		--update 10000 \
		--decay 0.95 \
		--revalue-period "$revalue_period" \
		--revalue-start "$revalue_start" \
		--revalue-samples "$revalue_samples" \
		--rho "$rho" \
		--gac-weight "$gac_weight" \
		--min-consolidation-margin "$min_consolidation_margin" \
		--align "$align" \
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
