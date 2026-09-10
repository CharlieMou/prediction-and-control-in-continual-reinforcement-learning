#!/usr/bin/env bash

# Previous launch configuration (disabled).
#
# #!/usr/bin/env bash
#
# set -uo pipefail
#
# script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$script_dir"
#
# seeds=(16 32 48)
# t_steps=9000000
# switch_steps=1000000
# batch_size=64
#
# mkdir -p logs
#
# pids=()
# labels=()
#
# launch() {
# 	local gpu="$1"
# 	local program="$2"
# 	local seed="$3"
# 	shift 3
# 	local label="${program%.py}_seed_${seed}_gpu_${gpu}"
#
# 	echo "Launching ${program}, seed=${seed}, GPU=${gpu}"
# 	CUDA_VISIBLE_DEVICES="$gpu" python "$program" \
# 		--env-name all \
# 		--seed "$seed" \
# 		--t-steps "$t_steps" \
# 		--switch "$switch_steps" \
# 		--batch-size "$batch_size" \
# 		--save \
# 		"$@" >"logs/${label}.log" 2>&1 &
# 	pids+=("$!")
# 	labels+=("$label")
# }
#
# # Each GPU runs all four algorithms for one seed.
# for gpu in "${!seeds[@]}"; do
# 	seed="${seeds[$gpu]}"
# 	launch "$gpu" DQN.py "$seed" --lr1 1e-5
# 	launch "$gpu" ib_dqn.py "$seed" --lr1 1e-5 --k 6
# 	launch "$gpu" ip_dqn.py "$seed" --lr1 1e-5 --k 6 --lambda 0.1 --rank 1
# 	launch "$gpu" PT_DQN_half.py "$seed" \
# 		--lr1 1e-8 --lr2 1e-4 --update 50000 --decay 0.75
# done
#
# status=0
# for index in "${!pids[@]}"; do
# 	if wait "${pids[$index]}"; then
# 		echo "Finished: ${labels[$index]}"
# 	else
# 		echo "Failed: ${labels[$index]} (see logs/${labels[$index]}.log)" >&2
# 		status=1
# 	fi
# done
#
# exit "$status"

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
		--save \
		"$@" >"logs/${label}.log" 2>&1 &
	pids+=("$!")
	labels+=("$label")
}

# Each GPU runs the four methods for the corresponding seed.
for gpu in "${!seeds[@]}"; do
	seed="${seeds[$gpu]}"
	launch "$gpu" DQN.py "$seed" --lr1 1e-5
	launch "$gpu" dqn_gac.py "$seed" \
		--lr1 1e-5 --rho 0.5 --align true --revaluation-size 10000
	launch "$gpu" PT_DQN_half.py "$seed" \
		--lr1 1e-8 --lr2 1e-4 --update 50000 --decay 0.75
	launch "$gpu" DQN_value_cloning.py "$seed" \
		--lr1 1e-5 --lambda-vc 1.0 --value-memory-size 10000
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
