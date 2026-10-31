# 双臂积木：完整代码与仿真说明

本文档对应当前 `startcode/` 的实现，替换了原 `branch_w` 的
`models.py / offline.py / online.py / runtime.py` 方案。两条路线均可用：

| 路线 | 离线 | 在线 | 主要配置 |
| --- | --- | --- | --- |
| A | Diffusion Policy 行为克隆 | 去噪链 DPPO | `configs/dp.yaml`、`configs/dppo.yaml` |
| B | ACT-CVAE 行为克隆 | 冻结基座残差 PPO | `configs/act.yaml`、`configs/act_residual.yaml` |

## 1. 范围与验收边界

- 仿真任务是 `stack_blocks`，环境配置 `arx_x5`，动作类型 `ee`。**不是**
  `stack_blocks_random`；离线示范、在线训练、评测全部使用 `stack_blocks`。
- 策略输入只有头部相机 `cam_head` 和 16 维末端状态；腕部图像和积木位姿
  不进入 actor/critic。积木位姿只在 `get_privileged_state()`，本实现默认
  不加内在奖励，也不把它写进观测。
- 动作 16 维：`[left_ee_pose(7), left_gripper(1), right_ee_pose(7),
  right_gripper(1)]`；位姿 `[x,y,z,qw,qx,qy,qz]`；夹爪下标 7、15。
  输出时统一归一化四元数、裁剪夹爪到 `[0,1]`。
- 环境得分保持 0/15/100。训练日志原样记录 `score`；PPO 目标乘
  `reward_scale=0.01` 只是为了数值稳定，不改环境得分，也不把 15 分当成功。
- 回合上限 550 步；成功由环境 `terminated` 决定，超时为 `truncated`。
- 本仓库不包含数据、镜像、Assets、训练好的权重。CPU 测试使用合成张量与
  假环境，只验证代码正确性，不代表真实堆叠成功率。

## 2. 文件逐项说明

以下路径均相对 `startcode/`。

### 2.1 策略实现

| 文件 | 作用 |
| --- | --- |
| `src/starter/policies/dp_core.py` | 图像/本体编码器、ε 网络、DDPM/DDIM、示范数据集、归一化、checkpoint、离线/在线共用的 `DiffusionRuntime` |
| `src/starter/policies/dp_agent.py` | DPPO：采样、去噪链 log-prob、GAE、PPO 更新、EMA、critic |
| `src/starter/policies/diffusion.py` | `train_offline` / `train_online` 入口、checkpoint 发现与日志 |
| `src/starter/policies/act_core.py` | ACT-CVAE + transformer decoder、数据集、冻结基座、`ActRuntime` |
| `src/starter/policies/act_agent.py` | 冻结 ACT 上的残差 PPO、GAE、critic、基座指纹 |
| `src/starter/policies/act.py` | `train_offline` / `train_online` 入口 |

### 2.2 环境、数据、奖励（沿用官方起点）

| 文件 | 作用 |
| --- | --- |
| `src/starter/interfaces.py` | 16 维动作打包/拆包、字段名、相机名、得分常数 |
| `src/starter/config.py` | 简化 YAML 读取 |
| `src/starter/paths.py` | `outputs/dp`、`outputs/act` 运行目录 |
| `src/starter/data/demos.py` | `episode_files` / `read_episode`：状态、动作、`cam_head` 编码缓冲 |
| `src/starter/envs/stack_blocks.py` | 训练用单环境/并行适配，返回 0/15/100 原始 score；积木位姿只在特权状态 |
| `src/starter/envs/remote.py` | 在线训练 TCP JSON/JPEG 环境客户端 `RemoteEnv` |
| `src/starter/envs/parallel.py` | `finetune_episodes` 必须是 `num_envs` 的整数倍 |
| `src/starter/envs/isaac_batch.py` | 在 Isaac 内绑定并行观测、末端 IK 与物理步进 |
| `src/starter/rewards/intrinsic.py` | 内在奖励扩展点，默认 0 |

### 2.3 入口、脚本、评测

| 文件 | 作用 |
| --- | --- |
| `src/starter/pipelines/diffusion.py` | `python -m starter.pipelines.diffusion {offline,online} --config ...` |
| `src/starter/pipelines/act.py` | 同上，ACT |
| `scripts/train_dp.sh` | 离线 Diffusion |
| `scripts/train_dppo.sh` | 在线 DPPO，自己启动 Isaac 镜像 |
| `scripts/train_act.sh` | 离线 ACT |
| `scripts/train_act_residual.sh` | 在线残差 PPO，自己启动 Isaac 镜像 |
| `scripts/run_online_env.sh` | 循环启动仿真容器，直到停止文件出现 |
| `scripts/isaac_env_server.py` | 容器内启动 Isaac、构建任务、服务 `RemoteEnv` |
| `scripts/eval_policy.sh` | 策略服务 + RoboDojo 客户端的分体评测 |
| `scripts/bootstrap_third_party.sh` | 固定 RoboDojo/XPolicyLab，并调用评测适配器链接 |
| `scripts/install_eval_adapters.sh` | 把 `policies/<name>` 链到两个 XPolicyLab checkout 的 `policy/<name>` |
| `scripts/download_stack_blocks.sh` | 下载标准 `stack_blocks` 示范 |
| `scripts/sim_image.sh` | 解析仿真镜像标签 |
| `policies/diffusion/serve_policy.py` | Diffusion WebSocket 策略服务 |
| `policies/diffusion/deploy.py` / `deploy.yml` | XPolicyLab 客户端适配与部署元数据 |
| `policies/diffusion/setup_eval_policy_server.sh` | 课程十参数评测入口 |
| `policies/act/...` | ACT 版同结构 |
| `tests/test_env_data.py` | 观测、动作、配置、TCP 超时测试 |
| `tests/test_dp_smoke.py` | Diffusion/DPPO CPU 冒烟测试 |
| `tests/test_act_smoke.py` | ACT/残差 PPO CPU 冒烟测试 |

## 3. 服务器首次准备

### 3.1 策略环境

```bash
cd startcode
conda create -n dual_arm_policy python=3.10 -y
conda activate dual_arm_policy
# 按服务器驱动选择 CUDA 版本，例如：
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-policy.txt
```

`requirements-policy.txt` 只包含策略进程依赖（torch、h5py、OpenCV、
WebSocket、msgpack、pydantic 等），不安装 Isaac。

### 3.2 上游源码与评测适配器

```bash
bash scripts/bootstrap_third_party.sh
```

该脚本：

1. 按 `third_party/pins.lock` 拉取 RoboDojo `f57e163...` 与 XPolicyLab
   `aa46382...`；
2. 建立 `third_party/data` 链接；
3. 调用 `scripts/install_eval_adapters.sh`，把 `policies/diffusion`、
   `policies/act` 链接到 `third_party/XPolicyLab/policy/` 和
   `third_party/RoboDojo/XPolicyLab/policy/`。

评测客户端按目录名导入 `XPolicyLab.policy.<name>.deploy`，所以第 3 步不能
省略。若 RoboDojo 的 `XPolicyLab` 子目录为空，脚本会把它链到
`third_party/XPolicyLab`。如果之前已经执行过 bootstrap、但没有运行过该链接
脚本，单独执行 `bash scripts/install_eval_adapters.sh` 即可。

### 3.3 资产、数据、镜像

```bash
bash third_party/RoboDojo/scripts/init_assets.sh
bash scripts/download_stack_blocks.sh
docker load -i stack-blocks-sim_isaacsim5.1.tar
docker image inspect stack-blocks-sim:isaacsim5.1 >/dev/null && echo sim-ok
```

示范位于 `data/RoboDojo/stack_blocks/arx_x5/data/episode_*.hdf5`。下载失败时
可换 Hugging Face 镜像或 ModelScope，目录结构保持不变。

### 3.4 GPU

```yaml
# configs/dppo.yaml / configs/act_residual.yaml / configs/eval_*.yaml
policy_gpu: 0
env_gpu: 1
```

策略进程与 Isaac 必须用不同物理卡。离线训练只用 `policy_gpu`，可以只开一张卡。

## 4. 数据与观测

`starter.data.demos.read_episode` 返回 16 维 `state`、16 维 `action` 和原始
`vision/cam_head/colors`。图像必须用 `XPolicyLab.utils.process_data.decode_image_bit`
解码：

```python
frames = decode_image_bit(episode["cam_head"])   # (T,H,W,3) RGB
```

旧 JPEG 的通道顺序与 PIL 的假设相反，不要用 PIL 直接解。解码后缩放到 96×96，
归一化到 `[-1,1]`；本体状态与动作按训练集均值/方差做 z-score（std 下限 1e-3）。

动作块：当前步 `t` 取 `action[t:t+H]`，不足 H 时用最后一步补齐。离线训练
同时预测未来 H 步；在线阶段按配置执行。

## 5. 路线 A：Diffusion Policy + DPPO

### 5.1 离线

```bash
bash scripts/train_dp.sh
# 或：python -m starter.pipelines.diffusion offline --config configs/dp.yaml
```

模型：

- `dp_core.ImageEncoder`：4 层 stride-2 卷积 + GroupNorm + 全局池化；
- `dp_core.ObsEncoder`：图像特征 + 16 维本体 MLP，拼成 256 维上下文；
- `dp_core.EpsilonNet`：对展平的 `H×16` 动作块预测噪声，条件为上下文和
  正弦时间编码；
- DDPM 100 步线性 β；训练随机采 t，加噪后做 MSE；
- AdamW 1e-4、200 epochs、EMA 0.999、warmup+cosine。

关键训练代码（`dp_core.DiffusionPolicy.training_loss`）：

```python
flat = action.reshape(batch, -1)
noise = torch.randn_like(flat)
t = torch.randint(0, self.diffusion_steps, (batch,), device=flat.device)
x_t = self.q_sample(flat, t, noise)
pred = self.predict_eps(x_t, self.encode_obs(image, proprio), t)
loss = F.mse_loss(pred, noise)
```

产物：`outputs/dp/<run_name>/<stamp>/checkpoints/offline_last.ckpt`
（另有 `last.ckpt`、`best.ckpt`、`epoch_*.ckpt`）。checkpoint 内含模型配置、
归一化统计量、EMA 权重。

### 5.2 在线 DPPO

```bash
bash scripts/train_dppo.sh
# 或：python -m starter.pipelines.diffusion online --config configs/dppo.yaml
```

脚本先读 `configs/dppo.yaml`，用 `run_online_env.sh` 启动 Isaac 容器，再在
当前环境启动 DPPO。在线阶段从最新离线 checkpoint 的 EMA 权重继续。

采样：8 步 DDIM；最后 `stochastic_steps=4` 个转移保留高斯噪声
（σ 由调度给出并夹到 `[0.02,0.4]`）。每个环境步执行动作块的第一步
（receding horizon），但整条去噪链的概率都会参与 PPO。

核心代码（`dp_core.DiffusionPolicy.sample` + `chain_logp`）：

```python
mean = sqrt(a_prev) * x0_hat + sqrt(1 - a_prev - sigma**2) * eps
x_prev = mean + sigma * noise
logp = gaussian_logp(x_prev, mean, sigma)

# PPO 更新时用保存的 x_t、x_prev、t、t_prev 重算同一个 mean
new_logp = self.chain_logp(x_t, x_prev, context, t, t_prev, sigma)
```

`dp_agent.DPPOAgent.update`：优势沿去噪链乘
`gamma_diff ** (K-1-j)`，再走 PPO clip 0.2、GAE(0.99, 0.95)、4 epochs。
critic 是独立 CNN+MLP。每次优化后更新 EMA，checkpoint 保存 EMA。

产物：`.../checkpoints/online_last.ckpt`、`online_best.ckpt`、`online_*.ckpt`，
日志 `online_log.jsonl`。

## 6. 路线 B：ACT + 残差 PPO

### 6.1 离线 ACT

```bash
bash scripts/train_act.sh
# 或：python -m starter.pipelines.act offline --config configs/act.yaml
```

模型：

- CNN 图像特征 + 本体 MLP；
- VAE 编码器把 `(观测特征, H×16 动作块)` 映射到 32 维对角高斯；
- transformer decoder 用 H 个可学习 query 交叉注意力到
  `[图像 token, 本体 token, latent token]`，输出 H 步动作；
- 训练时重参数化采样 z，损失 = L1 + 10×KL；推理时 z=0。

关键代码（`act_core.ActPolicy`）：

```python
mu, logvar = self.vae(features, action)
latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # 训练
prediction = self._decode(memory, latent)
loss = F.l1_loss(prediction, action) + kl_weight * kl
```

产物：`outputs/act/<run_name>/<stamp>/policy_offline.ckpt`
（另有 `policy_last.ckpt`、`policy_best.ckpt`、`policy_epoch_*.ckpt`）。

### 6.2 在线残差 PPO

```bash
bash scripts/train_act_residual.sh
# 或：python -m starter.pipelines.act online --config configs/act_residual.yaml
```

基座 `eval()` + `requires_grad_(False)`，每 H 步刷新一次动作块；残差头每步读
冻结的 512 维观测特征，输出 `u ~ N(mu, sigma^2)`，动作：

```python
action = base_action + residual_scale * clamp(u, -1, 1)
```

残差幅度默认：平移 2 cm、四元数 0.005、夹爪 0.05。PPO 只对 `u` 计算
对数概率；四元数归一化和夹爪裁剪发生在动作送入环境之前，不参与概率。
优势用标准 GAE；critic 读同一份冻结特征。

每次更新后重新计算基座 `state_dict` 的 SHA-256 指纹，与训练前不一致立即报错。
在线产物：`residual_last.ckpt`、`residual_best.ckpt`、`residual_*.ckpt`，
内含 `base_state`、残差头、critic、归一化统计量和指纹。

## 7. 仿真连接

```text
策略 Python                          Isaac 容器
train_online / DPPO agent            isaac_env_server.py
  RemoteEnv.reset_batch  ──TCP──→    ParallelAdapter.reset_batch
  RemoteEnv.step_batch   ──TCP──→    末端 IK → 关节插值 → PhysX
       ← RGB JPEG + 16 维本体 + score
  PPO / GAE 更新                     550 步成功/超时判定
```

- `scripts/train_dppo.sh` / `train_act_residual.sh` 读取配置里的
  `policy_gpu / env_gpu / env_port / rollout_task / env_cfg / num_envs`，
  先启动 `run_online_env.sh`，再启动学习器。
- `run_online_env.sh` 只挂载 `env_gpu` 一张卡，容器内 `CUDA_VISIBLE_DEVICES=0`。
- `RemoteEnv` 一次连接包含 `num_envs` 个场景；`finetune_episodes` 必须是
  `num_envs` 的正整数倍。
- 观测经 JPEG 传输（不落盘），图像与本体在策略侧还原；积木位姿不传。
- `train_dppo.sh` 退出时 touch 停止文件、删除容器；异常退出也走同一清理。

## 8. 评测

先把 `configs/eval_*.yaml` 的 `policy_env` 改成你的 conda 环境名，
`policy_gpu/env_gpu` 改成当前空闲的两张卡，然后：

```bash
# 离线
bash scripts/eval_policy.sh configs/eval_act.yaml
bash scripts/eval_policy.sh configs/eval_dp.yaml

# 在线
bash scripts/eval_policy.sh configs/eval_act_residual.yaml
bash scripts/eval_policy.sh configs/eval_dppo.yaml
```

`ckpt` 支持：

- `latest_offline`：最新 `offline_last.ckpt` / `policy_offline.ckpt`；
- `latest_online` / `latest_residual`：最新在线检查点；
- 显式 `.ckpt` 路径。

`eval_policy.sh` 的流程：

1. 读 `configs/eval_*.yaml`，校验 GPU 分卡；
2. 执行 `policies/<name>/setup_eval_policy_server.sh`，在 `policy_env` 中启动
   `serve_policy.py`，监听 127.0.0.1 随机端口；
3. 在仿真镜像里执行 RoboDojo `scripts/robodojo.sh client`；
4. 客户端按 `policy_dir` 目录名找到 `deploy.py`，通过 WebSocket 请求动作；
5. 结果写到 `third_party/RoboDojo/eval_result/...`，至少保留
   `_result.json` 和 `episode_*.mp4`。

`setup_eval_policy_server.sh` 的参数顺序是：
`bench task ckpt env_cfg action_type seed policy_gpu conda_env port host`。
服务端实现 `reset / update_obs(_batch) / get_action(_batch)`，动作以
`unpack_ee_action` 的字典返回，夹爪已裁剪、四元数已归一化。

正式对比至少跑 seed 0/1/2，离线与在线检查点分开评，`eval_num: native`；
保留曲线、配置和每个 run 的评测结果。

## 9. 产物与目录

```text
outputs/dp/<run_name>/<stamp>/
├── train_config.json
├── train_log.csv
├── online_log.jsonl          # 在线阶段
└── checkpoints/
    ├── offline_last.ckpt / last.ckpt / best.ckpt / epoch_*.ckpt
    └── online_last.ckpt / online_best.ckpt / online_*.ckpt

outputs/act/<run_name>/<stamp>/
├── train_config.json
├── train_log.csv
├── online_log.jsonl
├── policy_offline.ckpt / policy_last.ckpt / policy_best.ckpt
└── residual_last.ckpt / residual_best.ckpt / residual_*.ckpt
```

`run_name` 由 `bench/data_task/env_cfg/action_type/seed` 组成；在线阶段按
`configs/dppo.yaml` / `configs/act_residual.yaml` 的 `seed` 查找离线检查点，
也可在配置里写 `offline_checkpoint: /absolute/path.ckpt`。

本实现没有实现「断点续训」；重新训练会创建新的时间戳目录，评测通过
`latest_*` 或显式路径加载已有检查点。

## 10. 本地验证

```bash
cd startcode
PYTHONPATH=src python -m unittest tests.test_env_data
PYTHONPATH=src python tests/test_dp_smoke.py
PYTHONPATH=src python tests/test_act_smoke.py
```

当前机器（Python 3.8、CPU 版 torch 2.4）结果：`test_env_data` 12 项、
DP 冒烟 5 项、ACT 冒烟 5 项全部通过。冒烟测试覆盖：

- 前向/反向、DDIM 采样；
- 采样时与更新时链式 log-prob 一致（DPPO 正确性）；
- PPO / 残差 PPO 更新、GAE、checkpoint 存取；
- ACT 基座指纹不变；
- 离线/在线 runtime、RoboDojo 原始观测转换、四元数/夹爪投影。

**未在本机验证**：真实 HDF5 图像解码、Isaac/PhysX、Docker、WebSocket
评测全链路。这些依赖服务器镜像、驱动、资产和 GPU，必须在云平台按本文档与项目路线图的阶段检查点执行。

## 11. 方法来源与改动声明

- Diffusion Policy：条件动作去噪与动作块推理；
- DPPO：把最后若干去噪转移视为高斯策略步，PPO 优化去噪链；
- ACT：CVAE + transformer 动作块行为克隆；
- RoboDojo / XPolicyLab（固定 commit）：任务、奖励、评测协议。

本仓库的模型、训练循环、PPO、数据处理和评测服务为独立实现；没有把上游
训练脚本输出当作结果，也没有使用预训练视觉大模型。原 `branch_w` 的实现
（binary reward、`stack_blocks_random`、`models/offline/online/runtime`
架构）已被本版本替换。
