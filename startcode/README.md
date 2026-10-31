# 方舟 X5 双臂堆积木：ACT / Diffusion 完整训练入口

本目录现已实现 ACT-CVAE / Diffusion 的离线训练，以及 ACT residual PPO /
去噪链 PPO 的在线训练。使用头部 RGB + 16 维末端状态，输出 16 维末端动作。

完整文件说明、核心代码解释、服务器部署、所有命令、输出与验证边界见
[训练与仿真完整说明](../docs/TRAINING_AND_SIMULATION.md)。

```bash
# 激活已经装好 requirements-policy.txt 的策略环境后
bash scripts/bootstrap_third_party.sh
python scripts/download_stack_blocks.py --episodes 5  # 服务器下载真实数据
bash scripts/train_act.sh --set epochs=1 --set max_episodes=5
bash scripts/train_dp.sh --set epochs=1 --set max_episodes=5
# 将打印的真实离线 checkpoint 路径填入，再运行在线训练
bash scripts/train_act_residual.sh --set offline_checkpoint=/absolute/path/policy_last.ckpt
bash scripts/train_dppo.sh --set offline_checkpoint=/absolute/path/policy_last.ckpt
bash scripts/eval_policy.sh --set kind=act --set ckpt=/absolute/path/policy_last.ckpt --set eval_num=1
```

在线和评测默认需要课程仿真镜像与 RoboDojo Assets；也支持已配置好的 native
Isaac Python。正式任务为 `stack_blocks_random`，离线演示仍取标准 `stack_blocks`。
奖励与 benchmark score 分开记录；不修改官方任务奖励；不使用 SHA256 校验。

CPU 测试使用合成轨迹和显式假环境，不证明真实任务成功。真实仿真、显存与正式
成功率请在服务器上验证。权重/数据/镜像/输出不提交到 Git。
