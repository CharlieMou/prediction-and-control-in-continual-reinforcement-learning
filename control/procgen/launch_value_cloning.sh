#!/usr/bin/env bash
# Usage: ./launch_value_cloning.sh [gpu=0] [easy|hard]

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
value_memory_size="${VALUE_MEMORY_SIZE:-100000}"
lambda_vc="${LAMBDA_VC:-1.0}"
wandb_mode="${WANDB_MODE:-online}"

if [[ "$mode" != "easy" && "$mode" != "hard" ]]; then
	echo "difficulty must be easy or hard" >&2
	exit 2
fi

mkdir -p logs
status=0
# Sequential seeds avoid concurrently allocating multiple 1M replay buffers.
for seed in $seeds_string; do
	label="DQN_ValueCloning_${mode}_seed_${seed}_gpu_${gpu}"
	echo "Launching dqn_value_cloning.py: mode=${mode}, seed=${seed}, GPU=${gpu}"
	if ! CUDA_VISIBLE_DEVICES="$gpu" python -u dqn_value_cloning.py \
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
		--value-memory-size "$value_memory_size" \
		--lambda-vc "$lambda_vc" \
		--wandb-mode "$wandb_mode" \
		--save >"logs/${label}.log" 2>&1; then
		echo "Failed: ${label} (see logs/${label}.log)" >&2
		status=1
	else
		echo "Finished: ${label}"
	fi
done

exit "$status"
