from gym_wrapper import BaseEnv


# TASK_SEQUENCE = ("breakout", "space_invaders", "freeway")
# TASK_SEQUENCE = ("freeway", "breakout", "space_invaders")
TASK_SEQUENCE = ("breakout", "freeway", "space_invaders", "breakout",
		"space_invaders", "freeway", "freeway", "breakout", "space_invaders")
NUM_CYCLES = 1
NUM_TASK_STAGES = len(TASK_SEQUENCE) * NUM_CYCLES


def CL_envs_func(env_name, seed, task_index):
	"""Build one stage of the fixed three-game continual-learning curriculum."""
	if env_name != "all":
		raise NotImplementedError
	if not 0 <= task_index < NUM_TASK_STAGES:
		raise ValueError("task_index must identify one of the nine task stages")
	game = TASK_SEQUENCE[task_index % len(TASK_SEQUENCE)]
	stage_seed = seed + task_index
	env = BaseEnv(game, seed=stage_seed,
				use_minimal_action_set=False, use_minimal_observation=False)
	env.action_space.seed(stage_seed)
	return env
