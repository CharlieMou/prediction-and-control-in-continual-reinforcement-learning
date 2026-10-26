# Continual Procgen：DQN、PQN、PT-DQN、FAME、Value Cloning 与 GAC

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

每阶段 4M transitions，循环 3 次；默认 6-task curriculum 因而总计 72M lifetime transitions。4M 是在保证每个任务具有实质学习窗口的同时降低完整 continual 实验成本所采用的预算，3 cycles 来自 [SANE](https://proceedings.mlr.press/v199/powers22a.html) 的循环测试思路。可以用 `--envs`、`--num-tasks`、`--switch-steps` 和 `--cycles` 完整控制序列；`--num-tasks` 仅在需要截取所列环境的前 N 个时使用。

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
| epsilon | 1.0 到 0.1，在每个任务阶段的前 10%（默认 400k transitions）线性退火 |
| evaluation epsilon | 0.05 |
| reward clipping | 关闭 |

算法主体保持项目 MinAtar DQN 的 one-step target、uniform replay、hard target copy 逻辑。论文作者公开代码将网络和环境细节与若干扩展混在一起；这里没有把 Double DQN、multi-step、PER、distributional RL 或 augmentation 偷带进 baseline。

## PQN 配置

PQN 按照 [ICLR 2025 论文](https://openreview.net/forum?id=7IzeL0kflu)与 [CleanRL 参考实现](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/pqn_atari_envpool.py)实现。它不使用 replay buffer 和 target network：64 个同步 actors 每次收集短 rollout，用 rollout 结束处的当前 Q 网络 bootstrap，从后往前构造冻结的 Q(lambda) targets，再随机打乱并复用当前 rollout 进行多个 epoch 的 MSE 更新。

| 项目 | 默认值 |
|---|---:|
| actors | 64（与其他 Procgen baselines 一致） |
| network | DQN 的 IMPALA `[16, 32, 32]` / MLP 512 拓扑，每个卷积层与 hidden layer 加 LayerNorm |
| optimizer | RAdam，learning rate `2.5e-4`，weight decay `0` |
| rollout | 每 actor 32 steps，共 2048 transitions |
| minibatches / epochs | 16 / 2，minibatch 128 |
| effective update rate | 常规 rollout 为每 64 transitions 一次梯度更新 |
| loss / gradient clip | MSE / 10 |
| discount / lambda | 0.99 / 0.65（当前 launch 显式设为 0.8） |
| epsilon | 每阶段前 10%（默认 400k transitions）从 1.0 线性退火到 0.01（当前 launch 显式设为 0.1） |
| reward clipping | 关闭，与其他 Procgen baselines 一致 |

论文 Atari 配置使用 128 actors、32 minibatches、2 epochs 和 lambda=0.65。这里把 actors 减为共享协议的 64，同时把 minibatches 减为 16，因此仍保留 128 的 minibatch 和 `1 update / 64 transitions`。Python 入口保留 lambda=0.65、final epsilon=0.01 的直接运行默认值；当前 `launch_pqn*.sh` 为正在比较的实验显式使用 lambda=0.8、final epsilon=0.1，并默认采用 `coinrun,starpilot,caveflyer,fruitbot,jumper` 五任务序列。论文和 CleanRL 的关键 LayerNorm、orthogonal initialization、RAdam、冻结 lambda-return targets、多 epoch shuffle 与 MSE 逻辑均予以保留。

所有 baseline 的 epsilon decay 都会根据 `0.1 * switch_steps` 自动推导，因此任务时长从 25M 改为 4M 后由约 2.5M 相应缩短为 400k；可用 `--epsilon-decay-steps` 显式覆盖。PQN 的 rollout 32、lambda=0.65、minibatch 128、2 epochs 和 `1 update / 64 transitions` 描述的是局部 target 与优化协议，不应随任务总时长缩放，因而保持不变。若 task stage 不能被完整 rollout 整除，最后一个 rollout 会自动缩短，保证 lambda-return 永远不跨越已知任务边界。`--batch-size`、`--replay-size`、`--warmup-steps`、`--update-frequency`、`--adam-eps`、`--target-update-frequency` 与 `--clear-replay-on-switch` 是共享 CLI 中的 DQN 参数；PQN 不用 replay、warm-up、Adam epsilon 或 target copy，其中 `--batch-size` 只控制 action-gap probe 的最大采样数，`--target-update-frequency` 只复用为该 diagnostics 的记录周期。

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
- `size_fast2meta` 默认是阶段预算的 2%，即当前 4M 阶段中的 80k transitions；
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

## PQN w/ Value Cloning 与 PQN w/ GAC

两个 PQN 扩展不维护用于学习的 replay，也不在任务边界增加额外 rollout。正常 PQN training rollout 中会同步维护一个当前阶段的 `uint8` 状态均匀样本；它不参与动作、Q(lambda) target 或梯度更新。`--stage-sampling-tail-fraction` 控制哪些阶段位置有资格进入该样本：`1.0` 保持全阶段均匀采样，`0.25` 则只在最后 25% transitions（4M 阶段的最后 1M）中均匀采样，launch 默认使用 `0.25`。完成该阶段最后一次 PQN 更新后，才用 outgoing-task 的最终网络生成 VC 的完整 Q-vector，或 GAC 的 greedy action/action gap，并写入 lifetime auxiliary memory。evaluation trajectory 不会进入样本，真正用于 loss 的 auxiliary memory 仍然只在已知任务边界改变，因此保持 semi-continual 设定。

这个方案相比“为 PQN 额外维护 1M replay”显著降低内存，也不会像额外 boundary rollout 那样改变环境交互预算。每次提交后会清空阶段候选缓存；最终 lifetime stage 后没有未来训练，因而无需再提交。VC和GAC都使用基于各阶段 eligible state population 的hypergeometric reservoir merge，在分布上等价于对所有已完成任务的 eligible training states 做固定容量 reservoir sampling，并且都只为最终入选的新状态生成冻结标签。固定 tail fraction 和固定 stage duration 下，每个已完成阶段在 lifetime memory 中仍具有相同的期望权重。二者的长期记忆类统一命名为`AuxiliaryMemory`；区别仅在于VC保存完整Q-vector，而GAC使用outgoing-task最终online network保存greedy action和action gap。

- PQN 的 RAdam、MSE Q(lambda) TD loss、rollout、epoch、minibatch、epsilon 以及梯度裁剪均不变；VC/GAC loss 加到同一次 minibatch backward 中，不引入额外 optimizer step；
- PQN-VC 使用 `MSE(Q(s), Q_old(s))`，默认 `value_memory_size=10000`、`lambda_vc=1.0`；launch 使用 100k/1.0；
- PQN-GAC 使用与 DQN-GAC 相同的 margin hinge loss，默认 `revaluation_size=10000`、`lambda_gac=1.0`、`rho=0.5`；launch 使用 100k memory；
- PQN-GAC 只保留 online network，不额外维护第二套冻结网络；边界 revaluation 只对hypergeometric merge最终选中的新状态运行online network；
- GAC和VC的`AuxiliaryMemory`采用相同的`size / commit / sample`协议、原始transition计数及边界合并规则；
- 额外指标为 `loss/value_cloning`、`vc/value_memory_size` 或 `loss/gac`、`gac/revaluation_size`；performance、training performance 与 forgetting 继续写入现有无后缀 panels。

### Full-online PQN-VC / PQN-GAC

`pqn_online_value_cloning.py` 与 `pqn_online_gac.py` 不再在任务切换时触发 auxiliary-memory 更新，而是从 lifetime 起点开始按固定的 `--update-interval` 切分与任务无关的窗口。每个窗口内的正常 PQN rollout states 先进入均匀 reservoir；窗口结束时，候选 states 由当时的 online Q-network 生成 VC full-Q targets 或 GAC greedy-action/action-gap targets，再通过 hypergeometric merge 合入 lifetime auxiliary memory。窗口在任务切换时不会被清空，因此除非 interval 恰好与任务阶段对齐，memory protocol 不观察任务边界。PQN 本身仍不使用 replay 或 target network，rollout 长度和梯度更新频率也不受 auxiliary-memory 更新频率影响。

- 默认 `update_interval=2048`，即一个 64-actor × 32-step PQN learner batch。这对应 CLEAR 在每次 learner update 混入新经验与历史经验的 online 思路；这里没有照搬 CLEAR 的 actor-critic replay loss；
- `update_interval=1` 表示每次同步 vector-environment interaction 都更新。默认 64 actors 下，一次可观测的最细粒度是 64 transitions；由于 PQN 只在完整 rollout 后更新网络，1 与 2048 在常规完整 rollout 内使用的是同一组网络参数，前者主要用于严格验证 streaming ingestion，而不是额外增加 PQN 梯度频率；
- `update_interval=switch_steps` 会使窗口与已知任务边界对齐；配合 semi 脚本的 `stage_sampling_tail_fraction=0.25` 可得到相同的末段状态采样范围；
- online launch 默认 `stage_sampling_tail_fraction=1.0`，所以 tail 筛选不会周期性丢弃 stream。当前实验显式使用 `--epsilon-reset-on-switch`，让每个任务获得相同的探索退火阶段，以暂时排除跨任务探索不足的影响；这意味着探索协议使用任务边界信息，但 auxiliary-memory 更新协议仍不读取任务边界；
- auxiliary memory 达到一个 PQN minibatch（当前 128）后，VC/GAC loss 才启用。默认 2048 interval 在首次窗口提交后即可启用；
- 要检验 VC 对早期、不成熟 Q 尺度的敏感性，建议用默认 `UPDATE_INTERVAL=2048` 对比 `UPDATE_INTERVAL=SWITCH_STEPS`：前者在每个 learner batch 持续引入早期 full-Q target，后者只在任务末端创建 target。GAC 只冻结 greedy action 与部分 action gap，理论上对全局 Q offset/scale 漂移更不敏感。

### Streaming PQN-VC / PQN-GAC

`pqn_streaming_value_cloning.py` 与 `pqn_streaming_gac.py` 使用更严格的生成时存储协议。每次 vector-environment interaction 中，动作选择所用的同一个 Q-network forward 同时返回完整 Q-vector；当前的 `(state, Q)` 立即经过 Algorithm R reservoir sampling。VC 对入选 state 保存当时的完整 Q-vector，GAC 则立即从该 Q-vector 保存 greedy action 和 top-two action gap。旧 state 不会在后续窗口或任务边界使用最新网络重新 revalue，因此这里没有 `update_interval`、阶段候选 memory 或 hypergeometric window merge。当前launch同样启用task-switch epsilon reset；该边界信息只作用于探索率，不作用于streaming memory。

容量为 `M` 的 memory 在见过第 `n` 个 state 时，以 `min(1, M/n)` 的概率保留它，所以任意时刻的 memory 都是整个 lifetime state stream 的均匀样本。默认 64 actors 下，每次 vector step 会按固定 actor 顺序处理 64 个 state，但 reservoir law 与逐 state 更新相同。memory 达到一个 PQN minibatch 后开始计算 VC/GAC regularization；PQN 的 TD rollout、Q(lambda) target 和 optimizer update schedule 均保持不变。这更接近 [CLEAR](https://proceedings.neurips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html) 的持续写入有限 reservoir 思路，但仍然是 PQN-specific adaptation，而不是复现 CLEAR 的 IMPALA/V-trace replay learner。

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

# PQN
CUDA_VISIBLE_DEVICES=0 python pqn.py \
  --seed 16 --distribution-mode easy

# PQN w/ Value Cloning / GAC
CUDA_VISIBLE_DEVICES=0 python pqn_value_cloning.py \
  --seed 16 --distribution-mode easy
CUDA_VISIBLE_DEVICES=0 python pqn_gac.py \
  --seed 16 --distribution-mode easy
```

启动脚本中的随机种子按顺序运行，避免多个大容量 memory 同时占用主存；当前脚本默认 seeds 为 `17 33`：

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
./launch_pqn.sh 0 easy
./launch_pqn.sh 0 hard
./launch_pqn_value_cloning.sh 0 easy
./launch_pqn_value_cloning.sh 0 hard
./launch_pqn_gac.sh 0 easy
./launch_pqn_gac.sh 0 hard
./launch_pqn_online_vc.sh 0 easy
./launch_pqn_online_gac.sh 0 easy
./launch_pqn_streaming_vc.sh 0 easy
./launch_pqn_streaming_gac.sh 0 easy
```

可用环境变量覆盖脚本设置，例如：

```bash
ENVS="coinrun,fruitbot,miner" SEEDS="1 2" CYCLES=2 SWITCH_STEPS=1000000 \
  WANDB_MODE=offline ./launch_dqn.sh 0 easy

# 严格 streaming auxiliary-memory ingestion
UPDATE_INTERVAL=1 ./launch_pqn_online_vc.sh 0 easy
UPDATE_INTERVAL=1 ./launch_pqn_online_gac.sh 0 easy
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

论文默认 1M replay 仅 observation/next-observation 的 uint8 存储就预留约 22.9 GiB；PT-DQN 的 100k permanent window 还需要约 1.15 GiB。FAME 默认再使用一个 1M state-only Meta replay 和当前 80k recent state buffer，总 replay observation 容量约 35.2 GiB。Value Cloning 或 GAC 每 10k 条 auxiliary states 额外使用约 117 MiB；当前 launch 的 100k 配置约需 1.14 GiB auxiliary-state storage。普通 PQN 只保留约 24 MiB 的当前 rollout；semi-continuous PQN-VC/GAC 还各自同时持有 100k 阶段候选与 100k lifetime memory，state storage 合计约 2.29 GiB。online 默认 2048-transition 窗口只需约 24 MiB 候选空间，因此 auxiliary state storage 约为 1.17 GiB；streaming 版本没有候选窗口，只保留约 1.14 GiB lifetime states。VC Q targets 另约 5.7 MiB，GAC action/margin 的额外开销很小。它们仍远低于额外维护 1M transition replay。资源有限时可以降低相应 memory size，但这会改变 baseline，应在报告中明确记录。
