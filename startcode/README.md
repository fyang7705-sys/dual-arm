# 方舟 X5 双臂堆积木：完整实现

本目录实现了课程要求的两种「先离线、再在线」路线：

- **Diffusion Policy + DPPO**：`configs/dp.yaml` / `configs/dppo.yaml`
- **ACT + 残差 PPO**：`configs/act.yaml` / `configs/act_residual.yaml`

策略输入只有 `cam_head` 和 16 维末端状态；输出 16 维 `ee` 动作。任务、
外部得分（0/15/100）和 550 步限制均按课程要求保留，不改成随机任务，也不
用积木位姿做策略输入。

完整文件说明、核心代码、服务器部署、命令与限制见
[训练与仿真完整说明](../docs/TRAINING_AND_SIMULATION.md)。

## 快速命令

```bash
# 一次性准备
conda create -n dual_arm_policy python=3.10 -y
conda activate dual_arm_policy
pip install -r requirements-policy.txt          # torch 按服务器 CUDA 单独装
bash scripts/bootstrap_third_party.sh           # 固定上游 + 链接评测适配器
bash third_party/RoboDojo/scripts/init_assets.sh
bash scripts/download_stack_blocks.sh
docker load -i stack-blocks-sim_isaacsim5.1.tar

# 主路线（ACT）
bash scripts/train_act.sh
bash scripts/train_act_residual.sh
bash scripts/eval_policy.sh configs/eval_act.yaml
bash scripts/eval_policy.sh configs/eval_act_residual.yaml

# 备选路线（Diffusion）
bash scripts/train_dp.sh
bash scripts/train_dppo.sh
bash scripts/eval_policy.sh configs/eval_dp.yaml
bash scripts/eval_policy.sh configs/eval_dppo.yaml
```

在线训练与评测默认要求策略 GPU 与 Isaac GPU 不同（`policy_gpu: 0`、
`env_gpu: 1`）；离线训练只需一张卡。

## 测试

```bash
PYTHONPATH=src python -m unittest tests.test_env_data
PYTHONPATH=src python tests/test_dp_smoke.py
PYTHONPATH=src python tests/test_act_smoke.py
```

CPU 测试使用合成数据与假环境，不证明真实任务成功；真实仿真、显存与
成功率必须在服务器上验证。数据、镜像、Assets、checkpoint 与输出不提交
到 Git。
