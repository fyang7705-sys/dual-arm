# branch_w：双臂积木 ACT / Diffusion 训练与仿真完整说明

## 1. 实现范围与验收边界

本分支包含可运行的算法、数据、通信、检查点和启动流程，不包含训练好的积木策略。

| 路线 | 离线 | 在线 | 部署推理 |
| --- | --- | --- | --- |
| ACT | CNN 视觉特征、Transformer CVAE、动作块 L1 + KL | 冻结整个 ACT 基座，小幅有界残差 + PPO | 潜变量取零，残差取均值 |
| Diffusion | 图像条件动作去噪、cosine DDPM schedule、噪声 MSE、EMA | 冻结前段去噪模型，对最后 K 步高斯转移做 PPO | 保存的冻结前段 + 更新后的后段；确定性去噪，从随机初始噪声开始 |

这些是本项目独立实现的紧凑基线，不是原论文所有实验设置的逐项复现。没有使用预训练视觉大模型，不会自动下载骨干网络权重。默认 96×96 图像、10 步动作块只是起始超参，不意味着足够取得高堆叠成功率。

选题接口：只有 `cam_head` RGB 和本体状态进入 actor/critic；本实现选择 16 维末端状态。每侧状态/动作是位置 xyz、四元数 wxyz、夹爪各 7+1 维，两侧合计 16。没有使用积木位置作为策略输入。奖励直接使用官方 `get_reward()` 的 0/1 成功信号，`get_score()` 的 0/15/100 仅写入日志。三块堆稳、收回机械臂等成功判定仍由上游任务维护，不修改任务外部奖励定义。

离线示范来自标准版 `stack_blocks`；在线与正式评测默认 `stack_blocks_random`，包含不同积木集合和杂物。训练时根据 seed 新生成场景，不把正式评测布局用于训练。若课程最新通知另有规定，明确修改配置并在报告记录。

本地验证使用小型合成 HDF5 和明确标记的假环境，全部限制 CPU。真实 Isaac Sim / PhysX / RTX、真实演示文件、服务器驱动与显存预算没有在本机验证。假环境最后一步总给 1，目的是测试训练与通信，不能把其成功率当作积木结果。真实评测结果必须由服务器生成 `_result.json` 与 `episode_*.mp4`。

没有实现或调用 SHA256 校验。冻结基座检查用逐张量 `torch.equal`；检查点使用格式、维度与 `load_state_dict` 检查，写盘使用临时文件替换。Git 版本锁定仅用于选择上游源码版本。

## 2. 文件逐项说明

下列路径除根 README、本文外，均相对于 `startcode/`。

| 文件 | 作用 |
| --- | --- |
| `requirements-policy.txt` | 策略进程的数据、PyTorch、WebSocket、测试、下载依赖；不安装 Isaac |
| `configs/env.yaml` | 两种策略共用的离线/在线模型、优化器、GPU、环境、PPO 参数 |
| `configs/eval.yaml` | 独立正式评测配置：checkpoint、策略种类、random 任务、种子、评测条数 |
| `configs/sim_image.env` | 课程发布镜像下载地址和标签参考；实际 launcher 使用配置里的 `sim_image` |
| `third_party/pins.lock` | 固定 RoboDojo 与 XPolicyLab 源码提交，避免随上游 HEAD 漂移 |
| `src/starter/interfaces.py` | 动作与状态字段、16 维打包/拆包、相机名、成功/部分得分常数 |
| `src/starter/config.py` | 本项目简单 YAML 配置读取，支持标量、嵌套 map、标量列表、行内注释 |
| `src/starter/paths.py` | 保留原 starter 的路径工具；新训练器由 `runtime.run_directory` 管理分阶段输出 |
| `src/starter/cli.py` | 统一 `offline/online --config ... --set key=value` 参数解析与覆盖 |
| `src/starter/pipelines/act.py` | ACT 的直接 Python 入口 |
| `src/starter/pipelines/diffusion.py` | Diffusion 的直接 Python 入口 |
| `src/starter/policies/act.py` | ACT 对外训练与加载函数，委托公共离线/在线引擎 |
| `src/starter/policies/diffusion.py` | Diffusion 对外训练与加载函数 |
| `src/starter/policies/models.py` | `Vision`、`Condition`、ACT CVAE、条件去噪器、DDPM、高斯残差 actor、视觉 critic |
| `src/starter/policies/offline.py` | episode 切分、归一化、DataLoader、BC、验证、EMA、梯度裁剪、保存恢复 |
| `src/starter/policies/online.py` | 环境采样、动作块执行、GAE、ACT residual PPO、去噪链 PPO、KL 停止、冻结检查、保存恢复 |
| `src/starter/policies/runtime.py` | 随机种子/RNG、模型构建、检查点格式、预处理、动作投影、统一推理、JSONL 日志 |
| `src/starter/data/demos.py` | HDF5 字段读取、图像解码、训练集统计、按需读取的 `ChunkDataset`；没有缺数据自动假训练 |
| `src/starter/envs/stack_blocks.py` | 官方任务到训练 reset/step 的适配、成功/超时/无效状态、保留原始 reward 与 score |
| `src/starter/envs/isaac_batch.py` | 为普通任务绑定并行观测与末端 IK 控制、插值、物理步进；保留显式诊断用评测布局绑定函数 |
| `src/starter/envs/remote.py` | TCP JSON/JPEG 服务器与客户端，reset/step/close、超时、错误信息、默认不发送特权位置 |
| `src/starter/envs/parallel.py` | 并行采样数量约束工具，要求每轮 episode 数为场景数的正整数倍 |
| `src/starter/envs/fake.py` | 原接口测试桩及 `ToyBatchEnv`，仅在明确选择 fake 时使用 |
| `src/starter/rewards/intrinsic.py` | 保留内在奖励扩展工具；默认零，本分支训练器不添加额外奖励 |
| `src/starter/launch.py` | 激活 GPU 可见性、提前验证 checkpoint、启动/清理 Docker 或 native 仿真进程并运行学习器 |
| `src/starter/sim_launch.py` | 单独启动环境服务器，供另一个终端的 external learner 使用 |
| `src/starter/evaluation.py` | 基于真实 checkpoint 的 XPolicyLab 模型对象、单/批量预测、官方 WS PolicyServer |
| `src/starter/eval_deploy.py` | 无 GPU 依赖的官方评测循环桥，处理 chunk 中途结束与批量 active 环境变化 |
| `src/starter/eval_launch.py` | 启动策略服务、等待就绪、启动官方 RoboDojo client、清理服务/容器 |
| `scripts/bootstrap_third_party.sh` | 下载固定版本源码，建立 import/目录链接，安装 StudentACT/StudentDiffusion 评测桥 |
| `scripts/download_stack_blocks.py` | 只下载所需 HDF5，可指定少量 episode，不下载全数据仓库 |
| `scripts/download_stack_blocks.sh` | 原课程数据下载脚本，默认完整标准版数据；首次试跑优先用 Python 下载器 |
| `scripts/train_act.sh`、`scripts/train_dp.sh` | 自动定位项目根目录并运行两种离线训练 |
| `scripts/train_act_residual.sh`、`scripts/train_dppo.sh` | 同样传递配置和覆盖参数，运行两种在线训练，按配置启动仿真 |
| `scripts/run_online_env.sh` | 单独运行仿真服务器入口 |
| `scripts/isaac_env_server.py` | 在 Isaac Python 内启动 AppLauncher、构建任务与相机、随机场景、绑定控制器、服务 TCP |
| `scripts/eval_policy.sh` | 正式评测 Python launcher 的 shell 包装 |
| `scripts/install_eval_adapters.py` | 在忽略的依赖目录安装 StudentACT/StudentDiffusion 的 model/deploy、YAML、服务启动与原生评测脚本 |
| `scripts/configure_sim.py` | 只修改依赖目录中的相机捕获开关、并行场景数，不修改任务、奖励或评测布局 |
| `scripts/sim_image.sh` | 原课程镜像是否存在的辅助函数，保留作为手动诊断工具 |
| `scripts/plot_metrics.py` | 把一次运行的 JSONL 转为曲线 PNG；需 matplotlib |
| `policies/act/setup_eval_policy_server.sh` | 按课程参数顺序在指定 conda 环境启动 ACT 策略服务 |
| `policies/diffusion/setup_eval_policy_server.sh` | 同上，启动 Diffusion 策略服务 |
| `tests/test_env_data.py` | 观测、原始奖励、动作打包、配置、接口、TCP 超时测试 |
| `tests/test_training.py` | 合成数据训练/加载/恢复、冻结检查、PPO 概率回放、GAE、真实 TCP 和可选官方 WS 联调 |
| 各层 `__init__.py` | Python 包入口，通常不执行运行逻辑 |
| `.gitignore` | 排除运行输出、大数据、镜像和依赖；注意使用 `/data/`，不会再误忽略源码 `src/starter/data/` |

## 3. 服务器首次准备

### 3.1 获取本分支并建立策略环境

```bash
git clone --branch branch_w https://github.com/fyang7705-sys/dual-arm.git
cd dual-arm/startcode
conda create -n dual_arm_policy python=3.11 -y
conda activate dual_arm_policy
# 以下 cu128 是 CUDA 12.8 环境示例；以服务器驱动和 PyTorch 支持矩阵为准。
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-policy.txt
bash scripts/bootstrap_third_party.sh
```

策略环境独立于 Isaac 环境，不要把所有策略依赖装入腾讯预装 Isaac 的 Python。
bootstrap 不初始化/下载 Isaac 子模块，也不下载资产。它把 `third_party/RoboDojo/XPolicyLab` 链到统一的 `third_party/XPolicyLab`，并为上游辅助函数建立 `third_party/env_cfg` 链接。

首次只取 5 条真实示范检查链路；这仍是有体积的真实文件，在服务器执行，不在个人电脑执行：

```bash
python scripts/download_stack_blocks.py --episodes 5
# 完整基线准备好后：python scripts/download_stack_blocks.py --episodes 0
```

如果 Hugging Face 不可达，可以从课程给出的 ModelScope 镜像取相同 HDF5，并放到：

```text
startcode/data/RoboDojo/stack_blocks/arx_x5/data/episode_0000000.hdf5
```

代码不会从网页 URL 直接训练，也不会在找不到数据时悄悄切到假示范。只做离线训练时不需要启动 Isaac，不需要 Assets。

### 3.2 仿真环境：Docker 或 native 二选一

**Docker 路线**：使用课程发布的 `stack-blocks-sim:isaacsim5.1`。地址在 `configs/sim_image.env`；在服务器下载对应 tar 并执行 `docker load -i ...tar`。不把 tar 放进 Git。宿主机须有可用的 Docker 与 NVIDIA Container Toolkit。

```bash
bash third_party/RoboDojo/scripts/init_assets.sh
docker image inspect stack-blocks-sim:isaacsim5.1
nvidia-smi
df -h
```

资产从上游脚本下载，可能较大；下载范围随锁定上游脚本而定。容器会挂载本项目、源码和资产，使用主机网络。镜像不包含你的训练权重，也不能替代 RoboDojo Assets。

**native 路线**：服务器已经有匹配的 RoboDojo、定制 IsaacLab、cuRobo 时可复用：

```bash
export SIM_PYTHON=/absolute/path/to/simulator-env/bin/python
# 可选：export ROBODOJO_ROOT=/absolute/path/to/matching/RoboDojo
```

在线命令加 `--set env_launch=native`；正式评测也同样覆盖。`SIM_PYTHON` 必须能够 import 所需 isaaclab、RoboDojo 依赖与相机扩展，不能只是一个空 Python 环境。腾讯预装的通用 IsaacLab 即使版本号接近，也不代表包含 RoboDojo 的定制修改。若无法确认，使用课程镜像路线。

默认策略卡 0、仿真卡 1。只做离线训练只需一张卡。共享单卡需要显式 `--set policy_gpu=0 --set env_gpu=0 --set allow_shared_gpu=true`，并从一个场景开始实测显存；不保证 24 GB 一定够。默认仅采集头部相机，评测输出也只要求课程所需的视频，不能据此要求上游三相机汇总脚本全部通过。

## 4. 离线训练：完整数据流与核心代码

### 4.1 示例命令

从 `startcode/`、激活策略环境后执行：

```bash
# 首次真实数据短训练，不是有效性能基线
bash scripts/train_act.sh --set epochs=1 --set max_episodes=5 --set batch_size=2
bash scripts/train_dp.sh --set epochs=1 --set max_episodes=5 --set batch_size=2
# 正式训练；按服务器显存调整 batch_size
bash scripts/train_act.sh --set epochs=100 --set seed=0
bash scripts/train_dp.sh --set epochs=100 --set seed=0
```

所有入口支持 `--config configs/env.yaml` 和重复的 `--set key=value`。`max_episodes=0` 读取目录中全部 episode；分割前按 seed 打乱文件。

### 4.2 按轨迹划分、按需读取

调用链：shell → `launch.main` → `offline.train` → `split_episodes` → `normalization_stats` → `ChunkDataset` → DataLoader。

`split_episodes` 按完整文件分训练与验证，因此同一条轨迹不会同时出现在两个集合。均值、标准差只用训练文件；统计时每次读取最多 4096 行的 16 维数据，不解码图像。

`ChunkDataset.__getitem__` 的关键逻辑：

```python
state = _pack_group(f["state"], start)
action = _pack_group(f["action"], slice(start, start + self.chunk))
image = resize_rgb(f["vision/cam_head/colors"][start], self.image_size)
valid = np.arange(self.chunk) < len(action)
```

动作块接近 episode 末尾时重复最后一个动作补齐，但用 `valid` mask 排除补齐项的训练损失。不会从下一条 episode 拼动作。每次取样只打开一个文件读取一个图像/状态/动作块，不把数十 GB 图像一次放进内存。默认 `workers=0`，服务器可逐步增加；没有跨 worker 共享 h5py 文件句柄。

原始 RGB 数组可直接缩放；正式编码图像使用 XPolicyLab 的 `decode_image_bit`，兼容旧格式及带颜色标记的新格式，不自行猜测 BGR/RGB。模型固定输入 `[B,3,S,S]` 的 0..1 RGB 和 `[B,16]` 归一化本体状态。

### 4.3 ACT：CVAE + 动作块 Transformer

`models.ACT.forward` 训练时将 CLS、本体 token、示范动作 token 加入 posterior Transformer，padding mask 排除无效动作，然后得到潜变量分布：

```python
mu, logvar = self.posterior_out(encoded).chunk(2, -1)
latent = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).sum(-1).mean()
```

CNN 的 4×4 空间 token、本体 token、潜变量 token 作为 decoder memory；H 个可学习 query 预测 `[B,H,16]` 动作。离线损失为有效动作的 L1 均值加 `kl_weight * KL`，优化器 AdamW，随后梯度裁剪。验证及部署时潜变量置零，不使用未来示范动作。没有巨大时间聚合缓存。

### 4.4 Diffusion：条件噪声预测与 EMA

`Diffusion.training_loss` 随机选择 t，把标准化动作块加噪：

```python
noisy = abar.sqrt() * action + (1 - abar).sqrt() * noise
prediction = self.denoiser(noisy, time, image, state)
```

去噪器输入动作块、视觉/本体条件及正弦时间编码，使用残差 MLP 预测同形状噪声；只在有效动作位置计算噪声 MSE。cosine schedule 支持小步数测试，正式默认 20 步，可增加。

离线更新后同步 EMA 参数。验证及离线部署加载 EMA；在线阶段从 EMA 初始化待训练模型，但不继续部署陈旧的离线 EMA。checkpoint 的 `ema=None` 明确表示部署更新后的模型。

## 5. 在线训练：先有基座，再与仿真交互

离线训练输出目录由程序打印。把路径设为自己的真实文件，以下 `/absolute/...` 都是需要替换的示例：

```bash
bash scripts/train_act_residual.sh \
  --set offline_checkpoint=/absolute/path/to/act/offline/run/policy_last.ckpt \
  --set finetune_updates=1 --set finetune_episodes=1 --set num_envs=1

bash scripts/train_dppo.sh \
  --set offline_checkpoint=/absolute/path/to/diffusion/offline/run/policy_last.ckpt \
  --set finetune_updates=1 --set finetune_episodes=1 --set num_envs=1
```

以上先做一轮真实仿真验证，确认日志、显存及动作后再增大 `finetune_updates`。不提供 checkpoint 会在启动昂贵仿真前报错，不会随机初始化伪装成微调。模型种类、stage 不匹配也报错。

### 5.1 仿真进程与策略进程如何连接

```text
策略 Python                         仿真 Python（native 或容器）
online.train                        isaac_env_server.py
  Learner.sample                      AppLauncher → RoboDojo task
  RemoteEnv.reset_batch ──TCP──────→   ParallelAdapter.reset_batch
  RemoteEnv.step_batch  ──TCP──────→   末端 IK → 关节插值 → PhysX
       ← RGB JPEG + 本体 + reward ──   相机同步 → 观测 → 官方奖励
  GAE → PPO update                    成功 / 超时 / 无效 IK
```

自动模式由 `launch.py` 启动服务器，日志在 `outputs/simulator/`，退出时清理子进程和容器。native 模式明确设置独立 `SIM_PYTHON`。容器通过 NVIDIA Toolkit 暴露目标卡，不再依赖原 starter 中针对某台 Podman 主机手工挂载驱动库的代码。

`isaac_env_server.py` 在导入 Isaac 相关任务前启动 AppLauncher，构建任务/机器人/相机配置，关闭腕部捕获。训练不调用 `bind_eval_layouts`，而按 seed 调用任务 reset 重新生成场景。`--layout-source eval` 仅为显式诊断选项，正式训练启动器固定 random，不拿评测布局训练。

可开启 `--set domain_randomization=true`，接通上游桌面、地面、背景随机化。不会自动改变质量、摩擦、相机标定等所有物理/视觉参数；这不等于完整 sim-to-real 域随机化。

`isaac_batch._control_sequence` 将每侧 xyz+wxyz 交给官方 IK，再将得到的关节目标和夹爪目标插值为一个观测周期的控制序列，统一步进物理。IK 失败记为无效终止而非成功。`StackBlocksAdapter` 返回原始 binary reward，`info` 单独含 score、success 与原因。超时保留为 truncated，训练 critic 会 bootstrap；成功或无效 terminal 不 bootstrap。

TCP 图像是临时传输 JPEG，由 `remote.py` 的成对编解码处理；它不写入示范数据。默认不传输积木位置。官方示范的图像编码规则仍全部由 XPolicyLab 解码函数负责。

### 5.2 ACT 冻结基座 + 残差 PPO

`Learner` 加载离线 ACT 后执行 `requires_grad_(False)`，预测基座动作块。残差 actor 使用图像、本体和基座动作，输出高斯潜变量分布。实际标准化动作是：

```python
latent = dist.sample()
action = base + self.scale * latent.tanh()
```

因此残差每维限制在 `residual_scale` 内（乘动作标准差后才是实际物理尺度）。PPO 的概率针对未投影的 latent 计算，不错误地对四元数归一化后或夹爪裁剪后的动作计算高斯概率。部署使用 latent 的均值，保持同一动作组合方式。

每轮更新后 `assert_same_state` 对 ACT state_dict 每个 tensor 与冻结快照执行 `torch.equal`。优化器只持有 residual actor 参数；checkpoint 同时包含原基座与残差，其他机器无需猜测/寻找基座权重。

### 5.3 Diffusion 的去噪链 PPO

离线 EMA 复制为两份：冻结的前段模型，以及可训练后段模型。默认最后 4 个去噪转移保留非零高斯方差（包括 t=0 的方差下限），记录每一步的 `x_t`、`x_(t-1)`、t 和旧 log probability。早期去噪使用冻结模型的均值，不参与本轮 PPO。

更新时在相同观测与保存的链状态上重算：

```python
dist = self.model.distribution(stored_x, stored_t, image, state, min_std)
logprob = dist.log_prob(stored_next_x).flatten(1).sum(-1)
```

把每个高斯转移视为内层 MDP 的一个动作，计算新旧概率比、PPO clip，并用 `denoising_gamma ** t` 对较早去噪步骤的 advantage 降权。概率来自扩散去噪转移，而不是把最终动作硬套普通高斯 PPO。固定前段模型也随在线 checkpoint 保存，部署时使用相同分段模型。

这里实现 DPPO 思路中的截断去噪链 PPO，不包含原论文所有调度、训练技巧或任务配方；应在报告中如实说明。

### 5.4 动作块、回报与 GAE

每次预测 H 步，默认只执行前 `execute_steps=4` 步再根据新相机画面重规划。动作执行前统一归一化两侧四元数并裁剪夹爪，训练和评测共用这个投影。

一个策略决策可能跨 n 个控制步，累计回报是 `r0 + gamma*r1 + ...`。GAE 使用 `gamma**n` 的时间折扣，trace 使用 `(gamma*lambda)**n`。成功/无效终止的 next value 为 0；超时/人为采样上限使用最后观测的 V，但切断跨 reset 的 GAE。

每轮 `finetune_episodes` 必须是 `num_envs` 的正整数倍。采样以完整 episode wave 为单位，已结束的 slot 从 active 列表移除，不继续发送动作。`max_decisions_per_episode` 是调试用采样上限；正式设置应至少覆盖任务 550 控制步。日志中的 collector_cutoff 表示人为截断，不能算环境失败或任务成功。

PPO 使用 minibatch、多轮更新、优势标准化、策略 clip、value clip、梯度裁剪及近似 KL 提前停止。默认无内在奖励/奖励塑形。稀疏奖励可能使在线学习无提升，这需要更好的离线基座、超参或探索方法解决，不能通过改成功条件解决。

### 5.5 手动分开运行服务器与学习器

```bash
# 终端 A，策略环境内：启动器会再启动正确的仿真解释器/容器
bash scripts/run_online_env.sh --set env_launch=docker
# 终端 B：连接已有仿真，避免重复启动
bash scripts/train_act_residual.sh --set env_launch=external \
  --set offline_checkpoint=/absolute/path/to/offline/policy_last.ckpt
```

两终端 task、场景数、端口必须一致；先用同一份 config。这里 TCP 环境协议与正式评测的 WebSocket 协议是两套独立用途的通信。

## 6. 保存、恢复与正式评测

每次运行独立写入：

```text
outputs/<act|diffusion>/<offline|online>/seed<seed>/<timestamp>/
    config.json
    metrics.jsonl
    policy_last.ckpt
    policy_best.ckpt       # 离线验证最好的一次
    split.json             # 离线 episode 划分
    episodes.jsonl         # 在线逐 episode 信息
    policy_epoch_*.ckpt / policy_update_*.ckpt
```

checkpoint 自带 format、kind、stage、action_dim、模型结构、训练集归一化统计、权重、优化器、训练位置及 RNG 状态。在线还带 critic、残差/冻结前段模型。重新训练使用新目录，不覆盖此前运行。恢复在 epoch / PPO update 边界进行，不恢复一个正在执行的物理场景，也不保证跨 GPU/驱动版本逐位一致。

```bash
# epochs / finetune_updates 是目标总数，必须大于已完成数量
bash scripts/train_act.sh --set resume=/absolute/offline/policy_last.ckpt --set epochs=200
bash scripts/train_act_residual.sh --set resume=/absolute/online/policy_last.ckpt --set finetune_updates=200
```

Diffusion 对应替换脚本名。缺少文件、类型不符或结构不符会报错，不绕过。只加载本项目/可信同学给出的 checkpoint；PyTorch 格式包含优化器和 RNG 对象。

正式评测离线和在线 checkpoint 各运行一次，参数保持一致：

```bash
bash scripts/eval_policy.sh --set kind=act \
  --set ckpt=/absolute/act/offline/run/policy_last.ckpt --set eval_num=1
bash scripts/eval_policy.sh --set kind=act \
  --set ckpt=/absolute/act/online/run/policy_last.ckpt --set eval_num=1
bash scripts/eval_policy.sh --set kind=diffusion \
  --set ckpt=/absolute/diffusion/online/run/policy_last.ckpt --set eval_num=native
```

流程为 `eval_launch` → `evaluation.Model/PolicyServer` → 上游 `robodojo.sh client` → `StudentACT/StudentDiffusion.deploy` → 真实策略预测。使用官方客户端生成 JSON/MP4，不用假环境日志替代。输出在 `third_party/RoboDojo/eval_result/`；外部 ROBODOJO_ROOT 则在对应仓库下。

`install_eval_adapters.py` 向忽略的 XPolicyLab 安装无 GPU 依赖的 deploy bridge、指向本项目 Model 的导出及启动脚本。`configure_sim.py` 只调整相机和并行数，不改任务。单独策略启动脚本用于课程约定的十参数服务接口；本项目最直接的正式评测命令是上面统一 wrapper，不将 starter 的 `policies/act` 路径误当成上游可导入策略包名。

如需 PDF 中的原生官方命令口径，确保策略/仿真两个 conda 环境都已配置后，可在 `third_party/RoboDojo` 执行：

```bash
bash scripts/robodojo.sh eval --policy-dir XPolicyLab/policy/StudentACT \
  --task stack_blocks_random --ckpt /absolute/path/to/policy_last.ckpt \
  --policy-env dual_arm_policy --eval-env RoboDojo --env-cfg arx_x5 \
  --action-type ee --policy-gpu 0 --env-gpu 1 --seed 0 --eval-num native
```

这条路径由上游 `scripts/internal/run_policy_eval.sh` 启动本项目生成的 server/client setup 脚本，分别选择策略和仿真 conda 环境。Diffusion 改为 `StudentDiffusion`。原生客户端默认一个场景、头部捕获，可用 `SIM_NUM_ENVS` 调整。生成的 `eval.sh` 还提供单独调用的兼容入口；它会走统一 evaluator。单卡使用前须自行实测显存。

需方法对照时至少重复 seed 0、1、2；离线预训练、在线微调及评测种子均记录。`eval_num=1` 仅为连通检查，最终按 `native` 或课程指定次数。曲线：

```bash
python -m pip install matplotlib
python scripts/plot_metrics.py /absolute/path/to/run_directory
```

## 7. 本地验证与已知限制

```bash
cd startcode
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=src:third_party:third_party/XPolicyLab python -m pytest -q tests
```

有锁定的 XPolicyLab 时会额外执行官方 WebSocket 单/批量通信测试；没下载依赖时这一个测试显式 skip，其余数据/训练/TCP 测试不依赖 Isaac。合成数据只在测试临时目录生成，不进入 Git。默认没有大型 GPU 内存分配，没有运行本地 Isaac 或下载真实大数据。

测试覆盖：跨 episode padding、训练/验证隔离与统计、真实离线梯度更新、两种在线更新、ACT 基座冻结、Diffusion 前段冻结、概率回放一致性、PPO clip、timeout bootstrap、动作有限值与四元数、检查点保存/加载/恢复、TCP JPEG 传输、官方 WS 接口。

本次本地验证记录：Python 3.10、PyTorch 2.13.0、CUDA_VISIBLE_DEVICES 为空，CPU 测试 29 项通过；官方 RGB 编码解码和 WS 测试使用 pins.lock 中固定的 XPolicyLab aa46382 代码，额外使用独立临时目录中的少量依赖源码，未下载真实 Assets/演示/镜像。也只读核对了固定的 RoboDojo f57e163 的任务、IK、相机、控制队列和评测入口 API。服务器策略环境仍推荐 Python 3.11，与课程运行环境保持一致。

服务器仍需验证：真实 HDF5 格式与图像解码、课程镜像环境、固定版本的 Isaac API、真实随机场景构建/重置、IK 与物理步进、显存、长训练稳定性及成功率。启动前检查源码、资产、镜像、GPU；先一条真实评测，再扩展训练。不要把纯 CPU 测试通过写成机器人任务已完成。

Git 只提交源码、配置与说明；大数据、第三方仓库、镜像、训练 checkpoint、日志、视频留在服务器或另行备份。关闭 SSH 窗口不等于停止 GPU 进程；长任务用 tmux，结束后在平台确认资源计费状态。

## 8. 方法来源

- [ACT 作者项目页](https://tonyzhaozh.github.io/aloha/)：动作块与 CVAE/Transformer 思路。
- [Diffusion Policy 作者项目页](https://diffusion-policy.cs.columbia.edu/)：条件动作去噪学习。
- [DPPO 作者项目页](https://diffusion-ppo.github.io/)：扩散去噪内层 MDP 与 PPO。
- 固定版本的 RoboDojo / XPolicyLab 负责实际任务、奖励与评测协议；本分支没有把上游大模型权重作为实现的一部分。
