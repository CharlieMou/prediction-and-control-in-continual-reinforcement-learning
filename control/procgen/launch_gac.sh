#!/usr/bin/env bash
# Usage: ./launch_gac.sh [gpu=0] [easy|hard]

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

gpu="${1:-0}"
mode="${2:-easy}"
seeds_string="${SEEDS:-17 33}"
envs="${ENVS:-coinrun,starpilot,caveflyer,dodgeball,fruitbot,miner}"
IFS=',' read -ra selected_envs <<< "$envs"
num_tasks="${NUM_TASKS:-${#selected_envs[@]}}"
cycles="${CYCLES:-3}"
switch_steps="${SWITCH_STEPS:-25000000}"
replay_size="${REPLAY_SIZE:-1000000}"
# Defaults mirror launch_value_cloning.sh for a controlled comparison. The
# Value Cloning environment-variable names are accepted as shared overrides.
revaluation_size="${REVALUATION_SIZE:-${VALUE_MEMORY_SIZE:-100000}}"
lambda_gac="${LAMBDA_GAC:-${LAMBDA_VC:-1.0}}"
rho="${RHO:-0.5}"
align="${ALIGN:-true}"
wandb_mode="${WANDB_MODE:-online}"

if [[ "$mode" != "easy" && "$mode" != "hard" ]]; then
	echo "difficulty must be easy or hard" >&2
	exit 2
fi

mkdir -p logs
status=0
# Sequential seeds avoid concurrently allocating multiple 1M replay buffers.
for seed in $seeds_string; do
	label="DQN_GAC_${mode}_seed_${seed}_gpu_${gpu}"
	echo "Launching dqn_gac.py: mode=${mode}, seed=${seed}, GPU=${gpu}"
	if ! CUDA_VISIBLE_DEVICES="$gpu" python -u dqn_gac.py \
		--seed "$seed" \
		--distribution-mode "$mode" \
		--envs "$envs" \
		--num-tasks "$num_tasks" \
		--cycles "$cycles" \
		--switch-steps "$switch_steps" \
		--replay-size "$replay_size" \
		--update-frequency 64 \
		--epsilon-reset-on-switch \
		--lr1 2.5e-4 \
		--revaluation-size "$revaluation_size" \
		--lambda-gac "$lambda_gac" \
		--rho "$rho" \
		--align "$align" \
		--wandb-mode "$wandb_mode" \
		--save >"logs/${label}.log" 2>&1; then
		echo "Failed: ${label} (see logs/${label}.log)" >&2
		status=1
	else
		echo "Finished: ${label}"
	fi
done

exit "$status"
