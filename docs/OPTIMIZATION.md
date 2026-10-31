# 两条路线的优化空间与成功率提升方案

> 当前状态：P0 的 padding mask、四元数规范化、DPPO 前缀 log-prob、
> 残差头 base action，以及 P1 的图像增强、critic 独立学习率/γ=0.995
> 已在 `startcode/` 实现（见文末“已实现”表）。2 帧观测历史和评测
> temporal ensembling 仍未实现，是下一步的首选。

本文只讨论当前 `startcode/` 里的实现，不改变课程约束：`stack_blocks`、16 维
`ee` 动作、只用 `cam_head` + 本体、外部得分 0/15/100 不改、不用积木位姿做策略
输入。所有“提升成功率”的建议都按“离线基座 → 在线微调 → 评测”的顺序排列。

## 1. 结论：优先级最高的 7 件事

| 优先级 | 改动 | 影响 | 成本 |
| --- | --- | --- | --- |
| P0 | 动作块 padding mask（当前把补齐动作也计入 loss） | 中高：直接影响 episode 末尾“收回双臂” | 低 |
| P0 | 四元数规范化/连续旋转表示（q 与 -q 等价） | 中：消除旋转双覆盖造成的平均错误 | 低 |
| P0 | 评测加 temporal ensembling（重叠动作块加权平均） | 中高：ACT/DP 都适用，通常直接涨点 | 中 |
| P0 | DPPO 的 log-prob 只算“实际执行的动作前缀” | 中高：当前执行第一步却优化整个块 | 低 |
| P0 | 残差头输入加入 base action（和 chunk 内位置） | 中高：残差需要知道在修正什么 | 低 |
| P1 | 观测历史 2 帧 + 图像随机裁剪/颜色抖动 | 高：小样本视觉 BC 的最大杠杆 | 中 |
| P1 | 在线 reward shaping（只用训练期特权信息） | 高：0/15/100 太稀疏 | 中 |

如果只能做三件事：**padding mask + 2 帧历史/图像增强 + 评测 temporal ensembling**。
它们不改变方法，只修实现和推理，风险最低。

## 2. 现状体检

两条路线共享同一套数据/环境/评测骨架，当前实现的结构是对的：

- 离线：按 episode 划分训练/验证；图像用 XPolicyLab 解码；动作块预测；
  EMA；checkpoint 带归一化统计量。
- DPPO：DDIM 采样，最后 K 步高斯化；链式 log-prob；GAE；按去噪顺序折扣；
  EMA；评测加载 EMA。
- 残差 PPO：ACT 基座冻结；残差有界；基座 SHA-256 校验；PPO 只对残差求概率。

但和论文/成熟实现相比，有几处“实现级”缺口，会限制成功率：

1. 动作块在 episode 末尾用最后一个动作补齐，**loss 没有 mask**
   （`dp_core.py:651`、`act_core.py:475`；loss 在 `dp_core.py:309`、
   `act_core.py:225`）。模型会把“重复最后一帧动作”当成监督信号。
2. 四元数没有做符号规范化，q 和 -q 表示同一旋转，L1/MSE 会互相抵消；
   输出时才归一化，训练目标本身不连续。
3. 只有单帧观测；DP 论文默认 observation horizon 2，单帧无法判断速度/接触。
4. 没有任何图像增强；小样本视觉 BC 容易过拟合背景/纹理。
5. ACT 评测是开环执行整块，没有 temporal ensembling；ACT 论文明确用它
   平滑重叠动作块。
6. DPPO 每个环境步只执行动作块第一步（`dp_agent.py:167`），但 PPO 的
   log-prob 在整块 `H×16` 上计算（`dp_core.py:400`、`dp_agent.py:376`）。
   未被执行的维度也在参与似然，信用分配被稀释。
7. 残差头只看观测特征，不看 base action（`act_core.py:246/276`）；
   而残差学习的核心是“在基座动作上修多少”。
8. 在线 reward 默认没有 shaping（`intrinsic_weight=0`），而 0/15/100 中
   只有 15 和 100 两个稀疏信号；PPO 在 4 个场景上很难等到 100。
9. 策略和 critic 共用一个 AdamW 和同一个 lr（`dp_agent.py:122`、
   `act_agent.py:123`），critic 通常需要更大 lr。
10. `gamma=0.99` 对 550 步长任务偏短（有效视野约 100 步）。
11. 视觉主干很小（4 层 CNN + MLP），DP 论文用 ResNet18/Transformer；
    小模型可能欠拟合三块积木的精细纹理。
12. 没有 15 分率、平均步数等中间指标；只看 success 容易误判“15 分能力”。

## 3. 离线 BC：决定成功率上限

在线阶段只能把“已经会一点”的策略推过 100；如果离线一次都堆不成，PPO 基本
看不到 100（课程 README 也这么说）。因此优先投资离线。

### 3.1 动作块 padding mask（P0，低风险）

给 `EpisodeDataset` 增加 `valid`（长度 H 的 0/1），`chunk` 仍用最后动作补齐；
loss 改成：

```python
loss = (per_step_loss * valid).sum() / valid.sum().clamp(min=1)
```

DP 的噪声 MSE 和 ACT 的 L1 都要乘 mask。ACT 论文在动作块训练中就是用
mask 排除 padding。

### 3.2 旋转表示（P0/P1）

两条路都预测 `[x,y,z,qw,qx,qy,qz]`。四元数是 SO(3) 的双覆盖，且
Zhou et al. 证明 3D 旋转在 ≤4 维欧氏表示中不连续，5/6 维才连续
（[arXiv:1812.07035](https://arxiv.org/abs/1812.07035)）。

最低成本：

- 数据加载时把每个四元数规范到 `qw >= 0`（q 与 -q 选一个）；
- 训练 loss 前再规范化一次；输出仍归一化。

更彻底：内部预测 6D 旋转（每臂 3+6+1=10 维，共 20 维），送环境前转回
四元数 16 维。这会改 checkpoint 和归一化，属于中等改动，但对精度/稳定性
更好。

### 3.3 观测历史（P1，高影响）

单帧无法区分“正在靠近”还是“正在离开”，也无法判断夹爪是否刚闭合。
Diffusion Policy 论文使用 observation horizon 2，并配合 receding horizon
（[arXiv:2303.04137](https://arxiv.org/abs/2303.04137)）。

建议：

- `obs_horizon: 2`，输入通道 6（当前帧 + 上一帧）；
- 训练时以 10% 概率把上一帧替换成当前帧（frame dropout），
  避免策略过度依赖历史；
- 评测时维护同样的 2 帧缓冲。

ACT 也建议至少 1 帧历史 + 本体速度（如果有）。

### 3.4 图像增强（P1，高影响）

DP 论文使用随机裁剪；DrQ/RAD 也证明随机平移+颜色抖动对小样本视觉控制很有效
（[DrQ, arXiv:2004.13637](https://arxiv.org/abs/2004.13637)）。建议：

- 随机裁剪 84–92% 区域再缩放回 96×96；
- 亮度/对比度/饱和度轻微抖动；
- 训练增强，评测关闭；
- 不要做水平翻转（左右臂/纹理语义会变）。

### 3.5 视觉主干与容量（P1/P2）

当前 DP 约 1.3M、ACT 约 2.7M 参数（默认配置），相对 DP/ACT 论文的
ResNet18/Transformer 仍偏小。可做 A/B：

- ResNet18（ImageNet 预训练或从零）替换 CNN；
- 或把 CNN 宽度加倍、加一层；
- DP 可把 MLP 去噪器换成 1D U-Net/Transformer（DP 论文的 time-series
  diffusion transformer 对长动作块更稳）。

预训练特征在小样本下通常有帮助，但合成纹理场景不一定；先做 1 seed A/B，
不要直接上大模型。

### 3.6 动作块长度与 temporal ensembling（P0 评测 / P1 训练）

ACT 论文的 chunk size 约 90–100（50Hz 下约 1.8–2 秒），并用 temporal ensembling 对重叠块做
指数加权平均，减少块边界抖动（[arXiv:2304.13705](https://arxiv.org/abs/2304.13705)、
[项目页](https://tonyzhaozh.github.io/aloha/)）。当前 ACT 用 16、DP 用 8，
且评测是开环执行。

建议：

- 离线扫 `chunk ∈ {16, 32, 64}`（ACT）/ `{8, 16}`（DP）；
- 评测加 temporal ensembling：每个新块与旧块重叠部分按
  `w_i = exp(-m * age)` 加权平均，`m` 从 0.1 扫到 1.0；
- DPPO 在线不要做 ensembling（会破坏 PPO 的 on-policy 假设），只在评测用。

### 3.7 Diffusion 训练细节（P2）

- **min-SNR 加权**：按信噪比给不同 t 的 loss 加权，论文报告收敛快 3.4×
  （[arXiv:2303.09556](https://arxiv.org/abs/2303.09556)）；
- **cosine β schedule** 替代线性，通常更稳；
- **DDIM 评测步数** 10 → 16/20，动作质量更好但更慢；
- **EMA decay** 0.999 → 0.9995（数据少时更平滑）；
- 训练更久/更大 batch，配合 early stopping。

### 3.8 数据与标签（P0 检查项）

- 确认 16 维动作与 16 维状态时间对齐，夹爪下标 7/15 正确；
- 确认 `decode_image_bit` 输出的 RGB 没有被二次 BGR 转换；
- 过滤失败/截断示范（如果数据里有）；
- 训练集统计量只用训练 episode；
- 检查三块纹理是否在 96×96 下可分；必要时 128×128。

## 4. DPPO 的优化

DPPO 论文的核心是“两层 MDP + PPO + 去噪调度最佳实践”，并强调结构化、
on-manifold 探索，稀疏 reward 下在 Square/Transport 上 >90%
（[arXiv:2409.00588](https://arxiv.org/abs/2409.00588)、
[项目页](https://diffusion-ppo.github.io/)）。当前实现方向对，但有细节差距。

### 4.1 log-prob 与实际执行动作对齐（P0）

当前执行块的第一步，却对整块 128 维算似然。两种修法：

- **前缀 mask**：只对实际执行的 `prefix_dims=16` 计算链式 log-prob；
  采样仍生成整块，`x_t/x_prev` 只取前 16 维；
- **执行整块**：每个决策执行 H 步再重新规划（更像 DP 论文的 action horizon），
  但 PPO 的决策频率会下降。

推荐前缀 mask，改动小，信用分配最干净。

### 4.2 critic 的学习率与价值目标（P1）

`dp_agent.py:122` 把策略和 critic 放进同一个 AdamW、同一个 lr（5e-5）。
建议：

- 策略 lr 5e-5，critic lr 3e-4；
- value loss 用 Huber 或 value clip；
- 对 returns 做 running scale（或保持 `reward_scale=0.01` 也可以）；
- `gamma` 0.99 → 0.995；GAE λ 0.95 保持。

### 4.3 探索（P1）

`std_min=0.02` 在归一化动作空间里很小，且 `stochastic_steps=4/8`。
建议扫：

- `stochastic_steps ∈ {4, 5, 6}`；
- `std_min ∈ {0.02, 0.05, 0.1}`；
- 可加 entropy 项或可学习 log-std（注意 DPPO 原论文用固定方差也能工作，
  先扫固定值）。

### 4.4 reward shaping（P1，高影响）

0/15/100 太稀疏。用训练期特权积木位姿做 **potential-based shaping**
（Ng et al. 1999），不改变最优策略：

```text
Phi(s) = -w1 * min_ee_block_distance
         - w2 * distance_between_stacked_blocks（堆叠进展）
         - w3 * both_ee_distance_to_origin（收回双臂）
r_int = gamma * Phi(s') - Phi(s)
```

要点：

- 只用 `get_privileged_state()`，不进入策略输入；
- 权重小（例如 0.1–1.0），训练后期 anneal 到 0；
- 不把 15 改成 100，也不改环境 reward；
- 当前 `_intrinsic`（`dp_agent.py:185`）只有“到最近积木的距离”，
  可扩展成上面的三项。

### 4.5 采样吞吐（P1）

PPO 是 on-policy：`num_envs=4`、每次 4 条 episode 偏少。若显存允许：

- `num_envs=8/16`，`finetune_episodes` 同步放大；
- 每轮更新更多 transitions；
- 但这会线性增加仿真时间，先测单轮耗时。

### 4.6 不要做的事

DPPO 项目页明确说 **action penalty/平滑惩罚会妨碍微调**；不要为了“动作平滑”
加 L2 惩罚。也不要改外部得分、终止条件，或把积木位姿喂给策略。

## 5. ACT 残差 PPO 的优化

残差 RL 的经典做法是“基座控制器 + RL 残差叠加”
（[arXiv:1812.03201](https://arxiv.org/abs/1812.03201)）。当前实现正确，
但残差头的信息和训练/评测一致性可以加强。

### 5.1 残差头输入 base action（P0）

`ResidualHead.forward(features)` 只看观测特征。残差需要知道“基座打算怎么动”，
否则同一观测下基座动作变化时残差无法适配。建议：

```python
inp = torch.cat([features, base_action, chunk_pos_embedding], dim=-1)
mean = head(inp)
```

其中 `base_action` 是当前步的基座动作（16 维），`chunk_pos` 是块内位置
（0..H-1 的 embedding）。改动小，通常明显提升残差学习效率。

### 5.2 训练/评测的 chunk 一致性（P1）

训练时基座每 H 步刷新一次块（`act_agent.py`），评测时 `eval_chunk_size=1`
（`act_core.py:589/616`）每步重新规划基座。两者分布不同。建议二选一：

- 评测也用 chunk=H（开环执行 + temporal ensembling）；
- 或训练也每步重规划基座（receding horizon），与评测一致。

推荐前者，和 ACT 论文一致，且评测更省推理。

### 5.3 残差幅度课程（P1）

当前 `residual_scales` 平移 2 cm、四元数 0.005、夹爪 0.05。若基座已经不错，
这个幅度合理；若要探索，建议：

- 训练前 20% 更新用 0.5× 幅度，逐步升到 1×；
- 或按 success/15 分率自动调整；
- 监控残差均值/方差，避免一开始就盖住基座。

### 5.4 critic / entropy / gamma（P1）

与 DPPO 类似：critic 单独 lr（3e-4）、`gamma=0.995`、entropy 系数
1e-3–1e-2、`std_min` 0.05 可再扫。当前 `act_agent.py:109/123` 共用一个
optimizer。

### 5.5 更样本高效的替代（P2，研究向）

如果 PPO 在稀疏 reward 下长期无提升，可做对照（不改课程结论）：

- **RLPD**：critic ensemble + LayerNorm + 高 UTD + 离线数据对称采样，
  论文报告 2.5× 提升（[arXiv:2302.02948](https://arxiv.org/abs/2302.02948)）；
- **HIL-SERL**：示范 + 人工干预 + RLPD，1–2.5 小时接近满成功率
  （[arXiv:2410.21845](https://arxiv.org/abs/2410.21845)）；
- **Cal-QL**：calibrated conservative value，适合离线→在线
  （[arXiv:2303.05479](https://arxiv.org/abs/2303.05479)）；
- **FQL**：把 flow/diffusion 蒸馏成一步策略再做 Q-learning，避免递归反传
  （[arXiv:2502.02538](https://arxiv.org/abs/2502.02538)）；
- **DSRL**：在扩散潜空间做 RL 微调
  （[arXiv:2506.15799](https://arxiv.org/abs/2506.15799)）。

这些会偏离“残差 PPO”的课程要求，适合作为报告里的对比/未来工作，
不建议作为主实现。

## 6. 评测侧：几乎免费的涨点

1. **Temporal ensembling**（ACT/DP）：重叠块指数加权平均，减少抖动；
2. **多步采样取 critic 最高**（DPPO）：评测时采样 N 个块，用 critic 选 V 最大，
   成本低；但 critic 不准时可能反效果，先 A/B；
3. **确定性评测**：ACT z=0、残差取均值；DP 用 DDIM 确定性采样；
4. **15 分率/平均步数/平均得分**都记录，不只看 success；
5. **3 seeds**，报告均值±标准差；不要只报最好 seed；
6. 离线/在线检查点分开评，`eval_num: native`。

## 7. 建议的实验顺序

### 第 1 轮：离线消融（1 seed，先不跑在线）

| 配置 | 目的 |
| --- | --- |
| baseline | 当前指标 |
| + padding mask | 验证末尾动作是否改善 |
| + quat canonicalization | 旋转稳定性 |
| + obs horizon 2 + frame dropout | 最大预期收益 |
| + random crop/color jitter | 小样本泛化 |
| + chunk 32 / temporal ensembling | 长动作块与评测平滑 |

每项只在验证集 loss + 离线 eval 上比较；离线 15 分率明显提升后再进在线。

### 第 2 轮：DPPO

- prefix logp mask；
- critic lr 3e-4 + gamma 0.995；
- `stochastic_steps/std_min` 网格（各 2–3 个值）；
- reward shaping on/off。

### 第 3 轮：残差 PPO

- base action 输入；
- 训练/评测 chunk 对齐；
- 残差幅度课程；
- critic lr/gamma。

### 第 4 轮：3 seeds 正式评测

- 离线/在线各 3 seeds；
- success、≥15 率、平均得分、平均步数；
- 保留曲线、配置、`_result.json`、代表性视频。

## 8. 预期收益（定性）

| 改动 | 对离线 | 对在线 | 风险 |
| --- | --- | --- | --- |
| padding mask | 中 | 中 | 低 |
| 四元数规范化/6D | 中 | 中 | 低/中 |
| 2 帧历史 | 高 | 高 | 中 |
| 图像增强 | 中高 | 中高 | 低 |
| 更大主干/预训练 | 中高 | 中 | 中 |
| chunk + temporal ensembling | 中高 | 中（仅评测） | 低 |
| DPPO prefix logp | — | 中高 | 低 |
| reward shaping | — | 高 | 中 |
| 残差 base action | — | 中高 | 低 |
| 更多 envs | — | 高 | 成本高 |
| RLPD/FQL 替代 | — | 高 | 方法变更 |

## 9. 论文与资料

- Diffusion Policy: [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- DPPO: [arXiv:2409.00588](https://arxiv.org/abs/2409.00588)、
  [项目页](https://diffusion-ppo.github.io/)
- ACT: [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)、
  [项目页](https://tonyzhaozh.github.io/aloha/)
- Residual RL: [arXiv:1812.03201](https://arxiv.org/abs/1812.03201)
- RLPD: [arXiv:2302.02948](https://arxiv.org/abs/2302.02948)
- HIL-SERL: [arXiv:2410.21845](https://arxiv.org/abs/2410.21845)
- Cal-QL: [arXiv:2303.05479](https://arxiv.org/abs/2303.05479)
- FQL: [arXiv:2502.02538](https://arxiv.org/abs/2502.02538)
- DSRL: [arXiv:2506.15799](https://arxiv.org/abs/2506.15799)
- 6D rotation: [arXiv:1812.07035](https://arxiv.org/abs/1812.07035)
- min-SNR: [arXiv:2303.09556](https://arxiv.org/abs/2303.09556)
- DrQ: [arXiv:2004.13637](https://arxiv.org/abs/2004.13637)
- Potential-based shaping: Ng, Harada, Russell, ICML 1999


## 10. 本轮已实现（代码状态）

| 项目 | 位置 | 说明 |
| --- | --- | --- |
| 动作块 padding mask | `dp_core.EpisodeDataset/training_loss`、`act_core.EpisodeDataset/training_loss` | 数据集返回 `valid`，loss 只在有效步平均 |
| 四元数规范化 | `dp_core.canonicalize_quaternions`、`act_core.canonicalize_quaternions` | 单位化 + `qw>=0`，加载示范时应用 |
| 图像增强 | `augment_image` + `augment: true` | 随机裁剪 85–95% + 亮度/对比度抖动；验证/评测关闭 |
| DPPO 前缀 log-prob | `dp_core.sample/chain_logp`、`dp_agent.logp_prefix_dims` | 只统计执行的前 16 维动作 |
| critic 独立学习率 | `dp_agent`、`act_agent` 的 AdamW param groups | `critic_learning_rate: 3e-4` |
| γ=0.995 | `configs/dppo.yaml`、`configs/act_residual.yaml` | 550 步任务的有效视野更长 |
| 残差头 base action | `act_core.ResidualHead`、`act_agent`、`ActRuntime` | `residual_use_base_action: true`；旧 checkpoint 默认 False 兼容 |

仍未实现（按优先级）：

1. 2 帧观测历史 + frame dropout（需要改输入通道、数据集、在线/评测缓冲）；
2. 评测 temporal ensembling（需要改评测服务的 chunk 缓冲/执行方式）；
3. 6D 旋转表示、min-SNR、cosine schedule；
4. potential-based reward shaping 的完整三项（积木距离/堆叠/回原点）；
5. 更大视觉主干或预训练 ResNet。
