# Forager continual-control experiments

This directory runs `DQN.py`, `dqn_gac.py`, and `PT_DQN_half.py` on the
two-biome square-wave task used in Figure 5 of the Forager paper.  The adapter
in `forager_env.py` uses the vendored Numba Forager source in `forager/` and
recreates the official `ForagaxSquareWaveTwoBiome-v11` configuration.  When
`numba` is unavailable, the vendored API falls back to the same pure-Python
operations so lightweight installations remain runnable.

The same adapter also provides `UnendingFourBiomeForagerEnv`, the Numba-backed
port of Figure 7's `ForagaxUnendingTasks-v1` (`ForagaxBig-v5`).  DQN and PT-DQN
entry points for this benchmark live in `pro/`.

## Figure 5 environment

| Setting | Value |
| --- | --- |
| World / field of view | `24 x 15` / `9 x 9` |
| Biomes | left `x=[4,8)`, right `x=[16,20)` |
| Food per biome | 15 purple, 15 yellow |
| Walls | 24 background walls, plus 4 per food biome (32 total) |
| Observation | color one-hot view + previous action + previous reward (248 values) |
| Reward cycle | 500,000 steps; hidden task changes every 250,000 steps |
| Respawn / expiry | uniform 9--11 steps in the same biome / 500 steps |
| DQN | gamma .99, epsilon .1, replay 1,000, warm-up 50, batch 32 |
| Update schedule | one update per 4 environment steps, target every 128 updates |
| Network | two 64-unit LayerNorm/ReLU hidden layers |

During the first half-cycle, left purple/yellow rewards are `+4/-2` and right
purple/yellow rewards are `-8/-14`.  During the second half-cycle they become
`-14/-8` and `-2/+4`, respectively.  The environment is continuing: `done` is
always false and the observation does not reveal the square-wave phase.

## Figure 7 unending-tasks environment

| Setting | Value |
| --- | --- |
| World / field of view | `28 x 28` / `9 x 9` RGB |
| Food biomes | four `9 x 9` regions separated by five-cell gaps |
| Food density | `0.2` (17 mushrooms per biome with deterministic spawning) |
| Walls | `0.1` background density and `0.6` in each central `3 x 3` patch |
| Reward process | 10-term normalized Fourier series, one value per 1,000 steps |
| Species lifetime | replaced after 10,000 consumptions in its biome |
| Respawn | uniform integer delay from the upstream `(9, 11)` half-open range |
| Cue | best biome as a 4-vector for 10 of every 100 steps |
| Agent input | RGB view + previous action + previous reward + cue (252 values) |
| Duration | 10 million continuing interaction steps |

The Fourier coefficients have standard deviation `1/n`, the integer period is
sampled from `[10, 1000]`, and each series is min-max normalized to `[-1, 1]`.
Rewards are centered by the current global mushroom mean.  A species replacement
resamples both its RGB color and reward process and immediately restores its
biome's configured population.

Supplement 14.5 prints the period range as `[1, 1000]`, while the released
`FourierObject` used by `ForagaxUnendingTasks-v1` samples `[10, 1000]`.  This
adapter follows the executable reference implementation (`10` inclusive).

The `pro/` agents preserve the original DQN and PT-DQN update rules and wandb
metrics.  They add only the paper's pointwise `1 x 1` / spatial `3 x 3`
convolutional vision encoder and the new environment plumbing.  DQN uses the
Unending Tasks supplement's selected Adam step size (`1e-3`); PT-DQN retains
the two-biome permanent/transient settings because no Unending Tasks PT-DQN
configuration was reported.

## Running

Run from this directory.  The defaults reproduce the Figure 5 duration and
DQN hyperparameters:

```bash
python DQN.py --seed 0 --wandb-mode disabled
python PT_DQN_half.py --seed 0 --wandb-mode disabled
python dqn_gac.py --seed 0 --wandb-mode disabled \
  --revalue-period 50000 --rho 0.5 --align true
```

For the Figure 7 environment, either run one seed directly or use the launchers:

```bash
python pro/dqn.py --seed 0 --wandb-mode disabled
python pro/pt_dqn.py --seed 0 --wandb-mode disabled
bash pro/launch_dqn.sh
bash pro/launch_ptdqn.sh
```

`launch.sh` starts all three methods for seeds 16, 32, and 48; `launch_gac.sh`
starts only GAC.

## Boundary-free GAC

GAC no longer consolidates at a supplied task boundary.  Its periodic
retrospective revaluation uses only the global interaction count and replayed
observations.  In particular, `--reward-period` configures the environment but
is never used to trigger GAC.  Useful controls are:

- `--revalue-period` / `--revalue_period`: revaluation interval; `0` disables it.
- `--revalue-start`: first revaluation step; defaults to one interval.
- `--revalue-samples`: replay states randomly sampled per event; defaults to one minibatch.
- GAC memory capacity is fixed to one tenth of `--replay-size` (100 for the
  Figure 5 default replay size of 1,000).
- `--rho`: fraction of the saved greedy-action margin to preserve.
- `--gac-weight`: multiplier on the GAC hinge loss.
- `--min-consolidation-margin`: ignore low-confidence greedy decisions.
- `--align`: require agreement between online and lagged target greedy actions.

The logged `environment/task_phase` is an analysis label only.  It is not read
by action selection, optimization, or the revaluation schedule.

## Sources

- Forager paper: `../libraries/Forager_a lightweight testbed for continual learning with partial observability in RL.pdf`
- Original Numba environment: <https://github.com/steventango/forager>
- JAX environment and `ForagaxUnendingTasks-v1`: <https://github.com/steventango/continual-foragax>
- Paper agent implementations: <https://github.com/steventango/forager-agents>
- Final JAX experiment configurations: <https://github.com/steventango/continual-foragax-agents>
