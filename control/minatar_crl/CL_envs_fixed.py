# TASK_SEQUENCE = ("breakout", "space_invaders", "freeway")
# TASK_SEQUENCE = ("freeway", "breakout", "space_invaders")
# TASK_SEQUENCE = ("space_invaders","freeway", "breakout")
TASK_SEQUENCE = ("seaquest", "asterix")
NUM_CYCLES = 4 if TASK_SEQUENCE == ("seaquest", "asterix") else 3
NUM_TASK_STAGES = len(TASK_SEQUENCE) * NUM_CYCLES


def CL_envs_func(env_name, seed, task_index):
	"""Build one stage of the selected fixed continual-learning curriculum."""
	from gym_wrapper import BaseEnv

	if env_name != "all":
		raise NotImplementedError
	if not 0 <= task_index < NUM_TASK_STAGES:
		raise ValueError(f"task_index must be in [0, {NUM_TASK_STAGES})")
	game = TASK_SEQUENCE[task_index % len(TASK_SEQUENCE)]
	stage_seed = seed + task_index
	env = BaseEnv(game, seed=stage_seed,
				use_minimal_action_set=False, use_minimal_observation=False)
	env.action_space.seed(stage_seed)
	return env
