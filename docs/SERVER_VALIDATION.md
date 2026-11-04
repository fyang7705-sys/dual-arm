# ACT / DDPM 修复分支与服务器验证

本分支合并 `origin/master` 的评测部署和实验日志，以及 `branch_l` 的 ACT、
Diffusion Policy、残差 PPO、DPPO 实现。原分支保留，修复位于
`fix/act-ddpm-server`。

## 修复范围

- **DDPM 日程**：新训练默认 cosine schedule，终点累计信号接近零；原来的
  100 步线性 `beta=0.0001..0.02` 仍留下 `alpha_bar≈0.364`。支持真正的
  DDPM 全链祖先采样，离线默认 `eval_sampler: ddpm`；可用 `ddim` 加速。
- **旧扩散检查点**：加载保留权重中保存的 betas/alpha_bars，不在评测时擅自
  换日程。终点噪声不足会提示重新训练。旧模型不能靠改推理参数修好。
- **共同预处理**：离线、在线、评测使用同一四元数单位化/符号约定，包含
  `qw=0` 的正负号歧义；退化零四元数在控制边界回退到单位四元数。
  在线图像先缩到模型训练尺寸；修复暗 uint8 图像被误当作 0..1 浮点图像。
- **ACT**：padding 不再影响 CVAE 后验；验证/最佳模型选择使用零 latent 的
  重建误差。残差 PPO 每步重新规划基座动作，与评测执行一致。
- **DPPO**：单样本 advantage 标准化不产生 NaN；安全步数上限处的 critic
  bootstrap 使用策略设备；在线保存沿用离线模型的实际架构，评测保持 DDIM、
  相同采样步数和每次执行第一步。
- **仿真通信**：并行控制等所有活动队列完成；reset 清理观测缓存；记录在线
  `info.ik_status`；TCP 的特权信息按全局环境索引更新，服务器异常返回错误。
  普通失败终止不再被 PPO 日志计为成功。
- **数据**：接受任务根目录、直接包含 HDF5 的目录或单文件；拒绝维度不符、
  非有限值、长度不齐和训练/验证共用一个 episode。真实训练不回退到假示范。
  修复 XPolicyLab 解码器导入路径。
- **部署**：旧 `policies/eval_server.py` 转发到新的服务；保留宿主机评测入口。
  多个 run 匹配检查点别名时要求显式路径，缺失的指定模型不再回退到其他 run。
  服务记录实际权重绝对路径、SHA-256、epoch、配置、采样器和动作执行长度，
  两种评测脚本把记录复制到结果目录。新训练目录使用微秒时间戳。

这轮保持队友的模型结构；没有加入更大视觉网络、观测历史、temporal
ensembling 或奖励塑形，也不宣称已提高真实堆叠成功率。

## 检查点兼容性

`branch_w` / 原主分支的 `dual-arm.policy.v1` 与 `branch_l` 模型结构不同。
本分支会给出明确错误，不能直接加载那份 88 epoch 的旧 ACT 权重。
需要在本分支重训 ACT 和 DDPM；旧模型对照请使用独立的 `origin/master`
checkout。`branch_l` 同结构权重仍可加载，但日程和预处理修复后的正式结果
应使用重新训练的模型。

## 服务器步骤

在队友现有 checkout 有未提交修改时，先保留那些修改，再切换分支。

```bash
git fetch origin
git switch --track origin/fix/act-ddpm-server
cd startcode
conda activate <策略环境>
```

使用与服务器 CUDA 匹配的 PyTorch；其余依赖安装自
`requirements-policy.txt`。首次部署运行下面两条；已有依赖与数据时可跳过下载。
bootstrap 遇到依赖仓库内有已跟踪文件修改会停止，以免覆盖服务器修补。

```bash
python -m pip install -r requirements-policy.txt
bash scripts/bootstrap_third_party.sh
bash scripts/download_stack_blocks.sh
```

先验证模型和接口，再分别训练两条离线路线：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0 bash scripts/train_act.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/train_dp.sh
```

`configs/act.yaml` 与 `configs/dp.yaml` 的 `data_path`、`device` 等需匹配服务器。
初次可复制配置并缩短 `epochs`，用
`PYTHONPATH=src python -m starter.pipelines.act offline --config <配置路径>` 或
`starter.pipelines.diffusion` 运行。短训练只验证链路，不代表任务效果。

先跑单个布局；将下面的环境名、GPU 和**完整检查点文件路径**换为实际值。
宿主机方式要求该机器已有可用的 Isaac conda 环境。

```bash
bash scripts/eval_host.sh configs/eval_act.yaml \
  --policy-env <ACT环境> --sim-env <Isaac环境> \
  --ckpt /绝对路径/policy_best.ckpt --eval-num 1

bash scripts/eval_host.sh configs/eval_dp.yaml \
  --policy-env <DP环境> --sim-env <Isaac环境> \
  --ckpt /绝对路径/checkpoints/best.ckpt --eval-num 1
```

客户端与策略 GPU 由各自的 `configs/eval_*.yaml` 指定，必须不同。
Docker 方式先在对应 YAML 写入检查点完整路径、环境名和显卡编号，然后运行
`bash scripts/eval_policy.sh configs/eval_act.yaml` 或 `configs/eval_dp.yaml`。
启动脚本会打印 `*_checkpoint.json` 的保存位置，里面必须能对应本次训练。

单布局链路正常后，去掉 `--eval-num 1`，使用配置中的 `eval_num: native`
进行完整评测。两条离线策略都应分别保留结果。

在线微调前，在 `configs/act_residual.yaml` / `configs/dppo.yaml` 中填写各自
的 `offline_checkpoint` 完整文件路径、`policy_gpu`、`env_gpu`。两条在线脚本
默认使用同一端口，依次运行：

```bash
bash scripts/train_act_residual.sh
bash scripts/train_dppo.sh
```

在线产物分别用 `configs/eval_act_residual.yaml` / `configs/eval_dppo.yaml`
评测，仍显式指定 `residual_best.ckpt` / `online_best.ckpt`。DPPO 的评测使用
训练过的 DDIM 链，不改成 DDPM 采样。

## 验证边界与需要保留的结果

2026-11-04 本地验证：Python 3.10、PyTorch 2.13、OpenCV 4.14，43 项测试全部通过（包含 CUDA bootstrap 回归）；全部 Python/Shell 语法检查与宿主机评测 dry-run 通过。

本地测试覆盖前向/反向、DDPM oracle 反向链、两条离线 HDF5 训练、两条 PPO
更新及其检查点部署加载、预处理、并行队列、远程状态索引与检查点来源记录。
HDF5 测试用合成 RGB 图像和假环境，不能代替真实 JPEG 示范解码、Isaac/PhysX、
Docker、WebSocket 全链路与正式任务评测。

服务器每次请保留 `train_config.json`、`train_log.csv`、`online_log.jsonl`、
`*_checkpoint.json`、`_result.json` 和失败视频。若仍为 0 分，先区分机械臂
预测目标不正确、IK 失败、末端跟踪误差、夹爪时序及闭环漂移；不要仅依据平均
标准化 loss 判断抓取精度。正式对比记录成功率、平均分与至少 15 分的布局比例。
