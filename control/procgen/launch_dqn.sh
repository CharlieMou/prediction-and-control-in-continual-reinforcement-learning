#!/usr/bin/env bash
# Usage: ./launch_dqn.sh [gpu=0] [easy|hard]

set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

gpu="${1:-0}"
mode="${2:-easy}"
seeds_string="${SEEDS:-17 33}"
envs="${ENVS:-coinrun,starpilot,caveflyer,fruitbot,jumper}"
IFS=',' read -ra selected_envs <<< "$envs"
num_tasks="${NUM_TASKS:-${#selected_envs[@]}}"
cycles="${CYCLES:-3}"
switch_steps="${SWITCH_STEPS:-4000000}"
replay_size="${REPLAY_SIZE:-1000000}"
wandb_mode="${WANDB_MODE:-online}"

if [[ "$mode" != "easy" && "$mode" != "hard" ]]; then
	echo "difficulty must be easy or hard" >&2
	exit 2
fi

mkdir -p logs
status=0
for seed in $seeds_string; do
	label="DQN_${mode}_seed_${seed}_gpu_${gpu}"
	echo "Launching dqn.py: mode=${mode}, seed=${seed}, GPU=${gpu}"
	if ! CUDA_VISIBLE_DEVICES="$gpu" python -u dqn.py \
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
		--wandb-mode "$wandb_mode" \
		--save >"logs/${label}.log" 2>&1; then
		echo "Failed: ${label} (see logs/${label}.log)" >&2
		status=1
	else
		echo "Finished: ${label}"
	fi
done

exit "$status"
