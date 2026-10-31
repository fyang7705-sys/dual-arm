# 双臂积木：Diffusion + DPPO / ACT + 残差 PPO

`startcode/` 是可运行项目主体。本仓库当前实现已替换为两条完整路线：

| 路线 | 离线 | 在线 | 入口 |
| --- | --- | --- | --- |
| A | Diffusion Policy 行为克隆 | 去噪链 DPPO | `bash scripts/train_dp.sh` / `train_dppo.sh` |
| B | ACT-CVAE 行为克隆 | 冻结基座残差 PPO | `bash scripts/train_act.sh` / `train_act_residual.sh` |

两条路线都使用头部相机 `cam_head` + 16 维末端本体，输出 16 维 `ee` 动作；
任务固定为 `stack_blocks`，不修改环境奖励，不使用积木位姿作为策略输入。

完整文件说明、核心代码、服务器部署、全部命令、评测与已知限制见
[训练与仿真完整说明](docs/TRAINING_AND_SIMULATION.md) 与
[优化空间与成功率提升方案](docs/OPTIMIZATION.md)。

```bash
cd startcode
conda activate <policy-env>
bash scripts/bootstrap_third_party.sh
bash scripts/download_stack_blocks.sh
bash scripts/train_act.sh                 # 主路线离线
bash scripts/train_act_residual.sh        # 主路线在线
bash scripts/eval_policy.sh configs/eval_act.yaml
bash scripts/eval_policy.sh configs/eval_act_residual.yaml
```

本地 CPU 测试使用合成数据与假环境，只验证模型、PPO、checkpoint 和协议代码，
不代表真实堆叠成功率。真实 Isaac、显存与任务指标必须在服务器上验证。
