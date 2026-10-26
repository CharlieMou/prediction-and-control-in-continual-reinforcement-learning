#!/usr/bin/env bash
# Usage: ./launch_pqn_gac.sh [gpu=0] [easy|hard]

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
rollout_steps="${ROLLOUT_STEPS:-32}"
num_minibatches="${NUM_MINIBATCHES:-16}"
update_epochs="${UPDATE_EPOCHS:-2}"
q_lambda="${Q_LAMBDA:-0.8}"
revaluation_size="${REVALUATION_SIZE:-${VALUE_MEMORY_SIZE:-100000}}"
stage_sampling_tail_fraction="${STAGE_SAMPLING_TAIL_FRACTION:-0.25}"
lambda_gac="${LAMBDA_GAC:-${LAMBDA_VC:-1.0}}"
rho="${RHO:-0.5}"
wandb_mode="${WANDB_MODE:-online}"
final_epsilon="${FINAL_EPSILON:-0.1}"

if [[ "$mode" != "easy" && "$mode" != "hard" ]]; then
	echo "difficulty must be easy or hard" >&2
	exit 2
fi

mkdir -p logs
status=0
for seed in $seeds_string; do
	label="PQN_GAC_${mode}_seed_${seed}_gpu_${gpu}"
	echo "Launching pqn_gac.py: mode=${mode}, seed=${seed}, GPU=${gpu}"
	if ! CUDA_VISIBLE_DEVICES="$gpu" python -u pqn_gac.py \
		--final-epsilon "$final_epsilon" \
		--seed "$seed" \
		--distribution-mode "$mode" \
		--envs "$envs" \
		--num-tasks "$num_tasks" \
		--cycles "$cycles" \
		--switch-steps "$switch_steps" \
		--rollout-steps "$rollout_steps" \
		--num-minibatches "$num_minibatches" \
		--update-epochs "$update_epochs" \
		--q-lambda "$q_lambda" \
		--learning-rate 2.5e-4 \
		--gradient-clip 10 \
		--epsilon-reset-on-switch \
		--revaluation-size "$revaluation_size" \
		--stage-sampling-tail-fraction "$stage_sampling_tail_fraction" \
		--lambda-gac "$lambda_gac" \
		--rho "$rho" \
		--wandb-mode "$wandb_mode" \
		--save >"logs/${label}.log" 2>&1; then
		echo "Failed: ${label} (see logs/${label}.log)" >&2
		status=1
	else
		echo "Finished: ${label}"
	fi
done

exit "$status"
