#!/usr/bin/env bash

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# seeds=(16 32 48)
seeds=(20 36 52)
t_steps=10000000
batch_size=32

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
		--batch-size "$batch_size" \
		--aperture-size 9 \
		--reward-period 500000 \
		--replay-size 1000 \
		--minimum-replay-history 50 \
		--update-frequency 4 \
		--target-update-frequency 128 \
		--save \
		"$@" >"logs/${label}.log" 2>&1 &
	pids+=("$!")
	labels+=("$label")
}

# Each GPU runs the three Forager agents for one seed.
for gpu in "${!seeds[@]}"; do
	seed="${seeds[$gpu]}"
	launch "$gpu" DQN.py "$seed" --lr1 3e-3
	launch "$gpu" dqn_gac.py "$seed" \
		--lr1 3e-3 --revalue-period 50000 --revalue-samples "$batch_size" \
		--rho 0.5 --align true
	launch "$gpu" PT_DQN_half.py "$seed" \
		--lr1 1e-8 --lr2 1e-3 --update 10000 --decay 0.95
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
