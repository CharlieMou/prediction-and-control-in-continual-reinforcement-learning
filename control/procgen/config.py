"""Shared experiment configuration for continual Procgen control."""

from pathlib import Path


PROCGEN_ENVS = (
	"coinrun",
	"starpilot",
	"caveflyer",
	"dodgeball",
	"fruitbot",
	"miner",
	"jumper",
	"leaper",
	"maze",
	"bigfish",
	"heist",
	"climber",
	"plunder",
	"ninja",
	"bossfight",
	"chaser",
)

# The off-policy Procgen study uses the standard IMPALA-CNN widths and a
# 512-unit MLP.  PT widths keep the same block topology while making each
# component as close as possible to half of the DQN parameter budget.
DQN_CHANNELS = (16, 32, 32)
DQN_HIDDEN_SIZE = 512
PT_CHANNELS = (11, 20, 23)

DEFAULT_LEVEL_COUNTS = {
	"easy": 200,
	"hard": 500,
}


def network_parameter_count(channels, hidden_size, num_actions=15):
	"""Parameter count of the IMPALA Q-network without constructing PyTorch."""
	inputs = (3, channels[0], channels[1])
	conv_parameters = sum(
		(9 * in_channels * out_channels + out_channels)
		+ 4 * (9 * out_channels * out_channels + out_channels)
		for in_channels, out_channels in zip(inputs, channels)
	)
	feature_size = 8 * 8 * channels[-1]
	return (
		conv_parameters
		+ (feature_size + 1) * hidden_size
		+ (hidden_size + 1) * num_actions
	)


def pt_hidden_size(num_actions=15):
	"""Choose the integer PT hidden width closest to half a DQN."""
	target = network_parameter_count(
		DQN_CHANNELS, DQN_HIDDEN_SIZE, num_actions
	) / 2
	return min(
		range(1, DQN_HIDDEN_SIZE + 1),
		key=lambda width: abs(
			network_parameter_count(PT_CHANNELS, width, num_actions) - target
		),
	)


def add_common_arguments(parser):
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument(
		"--envs",
		type=str,
		default=",".join(PROCGEN_ENVS[:6]),
		help="comma-separated ordered Procgen curriculum",
	)
	parser.add_argument(
		"--num-tasks",
		type=int,
		default=None,
		help="use the first N entries from --envs (default: all listed environments)",
	)
	parser.add_argument("--cycles", type=int, default=3)
	parser.add_argument(
		"--switch-steps", "--switch", dest="switch_steps", type=int,
		default=4_000_000, help="environment transitions per task stage",
	)
	parser.add_argument(
		"--t-steps", type=int, default=0,
		help="total transitions; 0 derives num_tasks * cycles * switch_steps",
	)

	parser.add_argument(
		"--distribution-mode", choices=("easy", "hard"), default="easy"
	)
	parser.add_argument("--num-train-levels", type=int, default=None)
	parser.add_argument("--num-test-levels", type=int, default=None)
	parser.add_argument("--train-start-level", type=int, default=0)
	parser.add_argument(
		"--test-start-level", type=int, default=None,
		help="default: first level immediately after the training interval",
	)
	parser.add_argument(
		"--level-seed", type=int, default=0,
		help="controls level sampling order independently of the agent seed",
	)
	parser.add_argument("--num-envs", type=int, default=64)
	parser.add_argument("--num-threads", type=int, default=4)

	parser.add_argument("--gamma", type=float, default=0.99)
	parser.add_argument("--batch-size", type=int, default=512)
	parser.add_argument("--replay-size", type=int, default=1_000_000)
	parser.add_argument("--warmup-steps", type=int, default=128_000)
	parser.add_argument(
		"--update-frequency", type=int, default=64,
		help="environment transitions per gradient update",
	)
	parser.add_argument("--target-update-frequency", type=int, default=32_000)
	parser.add_argument("--adam-eps", type=float, default=1.5e-4)
	parser.add_argument("--gradient-clip", type=float, default=10.0)
	parser.add_argument("--initial-epsilon", type=float, default=1.0)
	parser.add_argument("--final-epsilon", type=float, default=0.1)
	parser.add_argument(
		"--epsilon-decay-steps", type=int, default=None,
		help="exploration decay duration; default: 10%% of each task stage",
	)
	epsilon_group = parser.add_mutually_exclusive_group()
	epsilon_group.add_argument(
		"--epsilon-reset-on-switch", dest="epsilon_reset_on_switch",
		action="store_true",
		help="restart exploration annealing at each known task boundary (default)",
	)
	epsilon_group.add_argument(
		"--no-epsilon-reset-on-switch", dest="epsilon_reset_on_switch",
		action="store_false",
		help="use one lifetime-wide exploration schedule",
	)
	parser.set_defaults(epsilon_reset_on_switch=True)
	parser.add_argument(
		"--reward-clip", type=float, default=0.0,
		help="symmetric clipping threshold; 0 preserves raw Procgen rewards",
	)
	parser.add_argument("--clear-replay-on-switch", action="store_true")

	parser.add_argument("--eval-frequency", type=int, default=200_000)
	parser.add_argument("--eval-episodes", type=int, default=10)
	parser.add_argument("--eval-num-envs", type=int, default=4)
	parser.add_argument(
		"--eval-max-steps", type=int, default=10_000,
		help="maximum vector steps in one task evaluation",
	)
	parser.add_argument("--eval-epsilon", type=float, default=0.05)
	parser.add_argument("--probe-size", type=int, default=512)
	parser.add_argument("--log-frequency", type=int, default=10_000)

	parser.add_argument("--wandb-project", default="Procgen_testbed")
	parser.add_argument(
		"--wandb-mode", choices=("online", "offline", "disabled"),
		default="online",
	)
	parser.add_argument("--save", action="store_true")
	parser.add_argument("--plot", action="store_true")
	parser.add_argument("--save-model", action="store_true")
	parser.add_argument("--output-dir", type=Path, default=Path("."))


def resolve_and_validate(args):
	envs = tuple(name.strip().lower() for name in args.envs.split(",") if name.strip())
	if not envs:
		raise ValueError("--envs must contain at least one Procgen environment")
	unknown = sorted(set(envs) - set(PROCGEN_ENVS))
	if unknown:
		raise ValueError(f"unknown Procgen environments: {unknown}")
	if len(set(envs)) != len(envs):
		raise ValueError("--envs must not contain duplicates")
	if args.num_tasks is None:
		args.num_tasks = len(envs)
	if not 1 <= args.num_tasks <= len(envs):
		raise ValueError("--num-tasks must be between 1 and len(--envs)")
	args.task_sequence = envs[:args.num_tasks]

	if args.cycles <= 0 or args.switch_steps <= 0:
		raise ValueError("--cycles and --switch-steps must be positive")
	if args.num_envs <= 0:
		raise ValueError("--num-envs must be positive")
	if args.epsilon_decay_steps is None:
		decay_steps = max(args.num_envs, int(0.1 * args.switch_steps))
		args.epsilon_decay_steps = (
			decay_steps // args.num_envs * args.num_envs
		)
	args.num_stages = args.num_tasks * args.cycles
	expected_steps = args.num_stages * args.switch_steps
	if args.t_steps == 0:
		args.t_steps = expected_steps
	elif args.t_steps != expected_steps:
		raise ValueError(
			"--t-steps must equal num_tasks * cycles * switch_steps "
			f"({expected_steps})"
		)

	default_levels = DEFAULT_LEVEL_COUNTS[args.distribution_mode]
	if args.num_train_levels is None:
		args.num_train_levels = default_levels
	if args.num_test_levels is None:
		args.num_test_levels = default_levels
	if args.num_train_levels <= 0 or args.num_test_levels <= 0:
		raise ValueError("training and test level counts must be positive")
	if args.train_start_level < 0:
		raise ValueError("--train-start-level must be non-negative")
	if args.test_start_level is None:
		args.test_start_level = args.train_start_level + args.num_train_levels
	if args.test_start_level < 0 or args.level_seed < 0:
		raise ValueError("--test-start-level and --level-seed must be non-negative")
	train_levels = range(
		args.train_start_level,
		args.train_start_level + args.num_train_levels,
	)
	test_levels = range(
		args.test_start_level,
		args.test_start_level + args.num_test_levels,
	)
	if max(train_levels.start, test_levels.start) < min(train_levels.stop, test_levels.stop):
		raise ValueError("training and test level intervals must not overlap")

	positive = (
		"num_envs", "num_threads", "batch_size", "replay_size",
		"warmup_steps", "update_frequency", "target_update_frequency",
		"epsilon_decay_steps", "eval_frequency", "eval_episodes",
		"eval_num_envs", "eval_max_steps", "probe_size", "log_frequency",
	)
	for name in positive:
		if getattr(args, name) <= 0:
			raise ValueError(f"--{name.replace('_', '-')} must be positive")
	if args.batch_size > args.replay_size:
		raise ValueError("--batch-size cannot exceed --replay-size")
	if args.switch_steps % args.num_envs != 0:
		raise ValueError("--num-envs must divide --switch-steps exactly")
	if args.eval_frequency % args.num_envs != 0:
		raise ValueError("--num-envs must divide --eval-frequency exactly")
	if args.switch_steps % args.eval_frequency != 0:
		raise ValueError("--eval-frequency must divide --switch-steps")
	if not 0 <= args.final_epsilon <= args.initial_epsilon <= 1:
		raise ValueError("epsilon values must satisfy 0 <= final <= initial <= 1")
	if not 0 <= args.eval_epsilon <= 1:
		raise ValueError("--eval-epsilon must be in [0, 1]")
	if args.reward_clip < 0:
		raise ValueError("--reward-clip cannot be negative")
	return args


def epsilon_at(args, lifetime_steps, stage_steps):
	"""Paper-style linear epsilon schedule, optionally restarted per task."""
	steps = stage_steps if args.epsilon_reset_on_switch else lifetime_steps
	fraction = min(max(steps, 0) / args.epsilon_decay_steps, 1.0)
	return args.initial_epsilon + fraction * (
		args.final_epsilon - args.initial_epsilon
	)
