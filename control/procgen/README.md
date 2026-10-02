# Continual Procgen：DQN、PT-DQN、FAME、Value Cloning 与 GAC

这个目录实现跨 Procgen 游戏的 continual RL 测试框架。训练期间，网络只从当前任务的固定训练 level 集合收集 transition；每次评估都在所有任务的固定 held-out level 集合上运行，并以 lifetime environment transitions 作为 W&B 横轴。

## 实验协议与设计校验

每个任务由 `(env_name, distribution_mode)` 确定。对所有任务，level seed 区间在一次实验开始前确定且全程不变：

- easy 默认 `D_i = [0, 200)`，`D_hat_i = [200, 400)`；
- hard 默认 `D_i = [0, 500)`，`D_hat_i = [500, 1000)`；
- `--train-start-level`、`--test-start-level`、`--num-train-levels` 和 `--num-test-levels` 可覆盖默认值；程序会拒绝任何相交区间；
- `--level-seed` 只控制固定 level 集合内的采样顺序，与 agent 的 `--seed` 分离，便于不同算法和随机种子之间进行公平比较；
- 每次评估都以相同 `rand_seed` 重建测试环境，因此测试 level 集合和采样流可复现。

这与 Procgen 论文常见的“有限训练集、无限测试分布”略有不同。无限测试分布不能严格保证与训练集合不相交，也不满足本项目要求的固定 `D_hat_i`，因此这里采用有限、固定、显式不相交的测试区间。easy 的 200 个训练 levels 和 hard 的 500 个训练 levels 分别沿用 [off-policy DQN 研究](https://openreview.net/forum?id=rtWOANa41W5)与 [Procgen 原论文](https://arxiv.org/abs/1912.01588)的 generalization 协议。

默认 curriculum 使用以下 6 个游戏；传入 `--envs` 后默认使用列表中的全部环境，并严格保持给定顺序：

```text
coinrun -> starpilot -> caveflyer -> dodgeball -> fruitbot -> miner
```

每阶段 25M transitions，循环 3 次。25M 与 off-policy Procgen 研究的单环境训练预算对齐，3 cycles 来自 [SANE](https://proceedings.mlr.press/v199/powers22a.html) 的循环测试思路。可以用 `--envs`、`--num-tasks`、`--switch-steps` 和 `--cycles` 完整控制序列；`--num-tasks` 仅在需要截取所列环境的前 N 个时使用。

所有 Python 入口和启动脚本默认在每次任务切换时重启 epsilon 退火，以便每个 Procgen 环境都有完整的探索阶段；可用 `--no-epsilon-reset-on-switch` 恢复 lifetime-wide schedule。replay 仍然不会在任务边界清空。

## DQN 配置

DQN 使用 off-policy Procgen 论文及作者参考实现中的设置：

| 项目 | 默认值 |
|---|---:|
| observation | RGB `64 x 64 x 3`，除以 255，无 frame stack |
| actors | 64 |
| network | IMPALA-CNN `[16, 32, 32]`，每 stage 两个 residual blocks，MLP 512 |
| optimizer | Adam，learning rate `2.5e-4`，epsilon `1.5e-4` |
| replay | uniform，1M transitions |
| warm-up | 128k transitions |
| batch | 512 |
| update | 每 64 transitions 一次 |
| target copy | 每 32k transitions |
| discount | 0.99 |
| loss | Huber |
| epsilon | 1.0 到 0.1，在每个任务阶段的前 512k transitions 线性退火 |
| evaluation epsilon | 0.05 |
| reward clipping | 关闭 |

算法主体保持项目 MinAtar DQN 的 one-step target、uniform replay、hard target copy 逻辑。论文作者公开代码将网络和环境细节与若干扩展混在一起；这里没有把 Double DQN、multi-step、PER、distributional RL 或 augmentation 偷带进 baseline。

## PT-DQN 配置

PT-DQN 的行为值为 `Q = P + T`，训练和 consolidation 逻辑沿用 `control/minatar_crl/PT_DQN_half.py`：

- transient 使用与 DQN 相同的 Adam rate、replay、target frequency、batch、Huber TD loss 和 epsilon schedule；
- permanent 使用 SGD，默认 learning rate `1e-5`；
- 每 100k transitions 将最近窗口中的 transient value 合并进 permanent，随后令 transient 参数乘以 `0.95`；
- consolidation 后立即同步 transient target，避免 bootstrap target 仍引用 decay 前的 transient；
- 两个 component 结构完全相同，均使用 IMPALA residual topology，宽度为 `[11, 20, 23]`、MLP 358。

标准 15-action DQN 有 1,154,383 个 behavior 参数，是奇数；两个参数量相等的 component 总参数必为偶数，所以逐参数完全相等在数学上不可能。当前配置中 P 和 T 各 577,206 个参数，总计 1,154,412，与 DQN 相差 29（0.002512%）。程序启动时会打印并自动校验这一预算。

`lrP=1e-5`、100k consolidation interval 和 decay 0.95 是 Procgen 上尚无先例时采用的保守初值，不应被当作已调优结果。正式结论前建议至少对以下组合做独立 validation sweep，并保证 DQN 不参与 PT-specific 调参：

```text
lrP:    {3e-6, 1e-5, 3e-5}
update: {50k, 100k, 200k}
decay:  {0.75, 0.90, 0.95, 1.0}
```

## DQN w/ FAME 配置

FAME 严格沿用 `control/minatar_crl/DQN_mle.py` 的 dual-learner 协议：Fast learner 使用 DQN TD learning；每个任务阶段末尾保留 Fast 最后的状态–动作对，通过 softmax-MLE 更新 Meta learner；下一阶段开始时用相同训练环境分别评估 Fast、Meta 和 Random，选择 Fast/Random 初始化或启用 Meta KL warm-up。Fast replay 在每个阶段结束时清空。

- Fast 的网络、Adam、replay、batch、target copy、Huber loss、discount 和 epsilon 设置与 DQN 完全一致；
- Fast 与 Meta 都使用完整的 DQN IMPALA-CNN；
- `size_fast2meta` 默认是阶段预算的 2%，即 2M 阶段中的 40k transitions；
- Meta replay 默认与 DQN replay 同为 1M entries；
- 每个候选默认评估 600 vector steps，即 64 actors 时的 38,400 transitions；
- KL warm-up 默认覆盖阶段前 10%，`lambda=1`、`temperature=1`；
- Meta 默认使用 Adam `2.5e-4`、200 epochs，每 2 epochs 将 learning rate 乘以 0.95。该学习率采用论文的 pixel-based Atari 设置，其他比例来自论文 MinAtar 设置并按 Procgen 阶段长度和并行 actors 换算。

FAME 的 Meta forgetting 严格使用 Fast learner 在任务结束时保存的 probe states、greedy actions 和 Q-values 作为 reference，而不是使用 Meta 自身作为 reference。

## DQN w/ Value Cloning 配置

Value Cloning 沿用 `control/minatar_crl/DQN_value_cloning.py` 的任务边界协议：每个非最终任务阶段完成后，用当前 DQN 为 replay 中的状态生成冻结的完整 Q-vector target，并通过 reservoir sampling 保留固定容量的跨任务记忆；后续更新最小化 `L_TD + lambda_vc * L_VC`。

- DQN 网络、Adam、replay、batch、target copy、Huber loss、discount、epsilon schedule 及其余共享超参数均与 `dqn.py` 一致；
- `value_memory_size=10000` 和 `lambda_vc=1.0` 沿用 MinAtar 实验，作为未经 Procgen 专门调优的保守起点；
- cloning memory 中的图像状态以 `uint8` 保存，10k 条状态约占 117 MiB，采样后才转换成归一化 tensor；
- 边界快照先完成 reservoir 决策，再仅为最终入选状态计算 Q target。任务边界处网络固定，因此其语义等价于 MinAtar 的逐状态实现，同时避免对最多 1M replay 状态进行无用的前向传播；
- Value Cloning 使用普通单网络 DQN，因此 `performance/*`、`training_performance/*`、`policy_forgetting/*` 与 `q_forgetting/*` 直接复用现有 baseline 的同名 W&B panels。

由于 Procgen 上没有可直接照搬的 Value Cloning 调参结果，正式比较前建议只用独立 validation seeds 搜索：

```text
lambda_vc:         {0.1, 0.3, 1.0}
value_memory_size: {5000, 10000, 20000}
```

## DQN w/ GAC 配置

Greedy Action Consolidation 严格沿用 `control/minatar_crl/dqn_gac.py` 的 semi-continual 协议。在每个非最终任务阶段结束时，先用 online network 找到 replay state 的 greedy action 和 action gap；默认只保留 online/target greedy action 一致的样本，并以两者 action gap 的较小值作为历史 margin。后续通过 hinge loss 保证历史 greedy action 相对其他 action 至少保留 `rho` 比例的 margin。

- DQN 网络、Adam、replay、batch、Huber TD loss、target copy、discount 和 epsilon schedule 与 `dqn.py` 一致；
- Python 默认 `revaluation_size=10000`、`lambda_gac=1.0`，分别与 Value Cloning 的 `value_memory_size=10000`、`lambda_vc=1.0` 对齐；
- 当前 `launch_gac.sh` 与当前 `launch_value_cloning.sh` 同样使用 100k auxiliary-memory entries 和系数 1.0，以便直接进行受控比较；
- GAC-specific 默认值 `rho=0.5`、`align=true` 来自 MinAtar 实验；
- retrospective revaluation 发生在边界处可能恰好触发的 target update 之前，确保 alignment 使用真正滞后的 target policy；
- GAC 使用同一组无后缀 W&B 指标 panel，额外记录 `loss/gac` 和 `gac/revaluation_size`。

当 `align=true` 时，GAC 必须在任务边界对 replay 中的状态分别运行 online 和 target network，默认 1M replay 会造成一次明显的边界计算停顿。这是算法的 retrospective revaluation，而不是训练进程卡死。

## W&B 指标

- `performance/<game>`：固定 held-out levels 上的 raw episodic return；
- `training_performance/<game>`：固定训练 levels 上的 raw episodic return，用于分离任务内学习表现与 held-out level 泛化；
- `performance_normalized/<game>`：Procgen 原论文的 difficulty-specific min-max score；
- `policy_forgetting/<game>`：最近一次完成该任务时保存的 probe states 上，greedy action 改变的比例；
- `q_forgetting/<game>`：同一 probe 上旧 greedy action 的 Q-value MSE；
- `overestimation/<game>`：预测 Q 与 discounted empirical return-to-go 的平均差；
- `action_gap/<game>`、`average_q/<game>`、`relative_action_gap/<game>`；
- `train_return`：当前训练任务 return 沿全局 step 连续记录，构成完整 lifetime 训练曲线；
- `loss/*`、`task_index`、`task_stage`、`cycle_index`。

所有 `performance/*` 和 `training_performance/*` 指标在每次 evaluation 时覆盖全部任务，包括尚未训练、当前训练和已经训练的任务；`policy_forgetting/*` 和 `q_forgetting/*` 只对已经完成过至少一个阶段且已有 reference probe 的任务定义。

FAME 额外记录 `performance_{fast,meta}/*`、`training_performance_{fast,meta}/*`、`policy_forgetting_{fast,meta}/*` 和 `q_forgetting_{fast,meta}/*`。无后缀的 `performance/*`、`training_performance/*` 和 `policy_forgetting/*` 对应 Meta learner，因此会与 DQN/PT-DQN 出现在现有同名 W&B panel；无后缀的 `q_forgetting/*` 和 `overestimation/*` 对应 TD-trained Fast learner。

## 安装与运行

Procgen 0.10.7 的预编译 wheel 支持 Python 3.7–3.10，推荐单独创建 Python 3.10 环境：

```bash
cd control/procgen
python -m pip install -r requirements.txt
```

单次实验示例：

```bash
# easy，默认 6 tasks x 3 cycles
CUDA_VISIBLE_DEVICES=0 python dqn.py --seed 16 --distribution-mode easy

# hard，自定义游戏序列、任务数和切换周期
CUDA_VISIBLE_DEVICES=0 python pt_dqn.py \
  --seed 16 \
  --distribution-mode hard \
  --envs coinrun,starpilot,fruitbot,miner \
  --num-tasks 4 \
  --switch-steps 4000000 \
  --cycles 3

# DQN w/ FAME；未显式给出的 FAME 参数按阶段长度自动推导
CUDA_VISIBLE_DEVICES=0 python dqn_fame.py \
  --seed 16 --distribution-mode easy

# DQN w/ Value Cloning
CUDA_VISIBLE_DEVICES=0 python dqn_value_cloning.py \
  --seed 16 --distribution-mode easy

# DQN w/ GAC
CUDA_VISIBLE_DEVICES=0 python dqn_gac.py \
  --seed 16 --distribution-mode easy
```

三随机种子启动脚本按顺序运行，以免三个 1M replay 同时占用大量内存：

```bash
./launch_dqn.sh 0 easy
./launch_dqn.sh 0 hard
./launch_ptdqn.sh 0 easy
./launch_ptdqn.sh 0 hard
./launch_fame.sh 0 easy
./launch_fame.sh 0 hard
./launch_value_cloning.sh 0 easy
./launch_value_cloning.sh 0 hard
./launch_gac.sh 0 easy
./launch_gac.sh 0 hard
```

可用环境变量覆盖脚本设置，例如：

```bash
ENVS="coinrun,fruitbot,miner" SEEDS="1 2" CYCLES=2 SWITCH_STEPS=1000000 \
  WANDB_MODE=offline ./launch_dqn.sh 0 easy
```

快速 smoke run：

```bash
python dqn.py \
  --num-tasks 1 --cycles 1 --switch-steps 256 --num-envs 4 --num-threads 1 \
  --num-train-levels 8 --num-test-levels 8 \
  --warmup-steps 64 --batch-size 4 --replay-size 128 \
  --update-frequency 64 --target-update-frequency 64 \
  --eval-frequency 256 --eval-episodes 1 --eval-num-envs 1 \
  --wandb-mode disabled
```

## 资源说明

论文默认 1M replay 仅 observation/next-observation 的 uint8 存储就预留约 22.9 GiB；PT-DQN 的 100k permanent window 还需要约 1.15 GiB。FAME 默认再使用一个 1M state-only Meta replay 和 40k recent state buffer，总 replay observation 容量约 34.8 GiB。Value Cloning 或 GAC 每 10k 条 auxiliary states 额外使用约 117 MiB；当前两个 launch 脚本的 100k 配置各自约需 1.14 GiB auxiliary-state storage。显存占用远小于主存占用。资源有限时可以降低 `--replay-size`、FAME 的 `--size-meta`、Value Cloning 的 `--value-memory-size` 或 GAC 的 `--revaluation-size`，但这会改变 baseline，应在报告中明确记录。
