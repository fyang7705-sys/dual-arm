# 项目起点：方舟 X5 双臂堆积木

命令在 `startcode/` 下执行。这里保留了仿真环境、示范数据读取，以及 [Diffusion](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/DP/README.md) 和 [ACT](https://github.com/XPolicyLab/XPolicyLab/blob/main/policy/ACT/README.md) 两条链路的启动方式。模型结构、训练过程、评测服务和策略超参需要自己实现。完成一种实现即可。

不启动仿真时，可以先检查环境和数据接口：

```bash
PYTHONPATH=src python3 -m unittest tests.test_env_data
```

`NotImplementedError` 来自 `src/starter/policies/`。这是留给你的入口，不是环境故障。

## 和选题的对应关系

这版起点只覆盖期中的仿真阶段。真机部署、4 块积木和域随机化不在代码里，期末由课程组另行提供机器人与部署接口。

选题要求三块纹理不同的积木从随机位置堆稳，并在规定步数内把双臂收回。策略输入是头部相机和本体状态，输出是末端位姿，因此动作类型用 `ee`。不要把积木位姿送进策略。外部奖励不能改。

建议的路线是离线预训练再在线强化学习，基线可参考 XPolicyLab。本目录给出 Diffusion Policy 与 ACT 两条这样的链路，完成其中一种即可。选题也允许用 PPO、SAC 或其他算法，配上内在奖励、探索或课程学习，直接在线训练。那条路线没有单独的启动脚本；若采用，仍使用这里的环境和下面的评测命令。不带这些设计的随机高斯策略，看不到成功样本，不能当作完成。

使用公开库时，报告里写明来源、改了什么，以及关键训练参数。保留配置、随机种子、学习曲线和检查点，不必保存全部交互数据。

## 目录里没有的大文件

起始代码不包含镜像、资产、示范数据和上游仓库。磁盘请预留数百 GB。训练得到的 checkpoint 也不在里面，由你自己保存。


| 组件                                  | 是否必须      | 获得方式                                                                               |
| ----------------------------------- | --------- | ---------------------------------------------------------------------------------- |
| 仿真镜像 `stack-blocks-sim:isaacsim5.1` | 在线训练和评测必须 | 从下面的数据集下载 `stack-blocks-sim_isaacsim5.1.tar`，再 `docker load`                       |
| 策略镜像 `stack-blocks-policy:dp-act`   | 不必须，只供参考  | 同一数据集里的 `stack-blocks-policy_dp-act.tar`，再 `docker load`                           |
| RoboDojo、XPolicyLab 源码              | 必须        | `bash scripts/bootstrap_third_party.sh`                                            |
| Isaac 子模块                           | 不必须       | 只有自己重新构建仿真镜像时才执行 `git -C third_party/RoboDojo submodule update --init --recursive` |
| 场景资产 `Assets/`                      | 必须，不在镜像里  | `bash third_party/RoboDojo/scripts/init_assets.sh`                                 |
| 离线示范                                | 必须        | `bash scripts/download_stack_blocks.sh`                                            |


锁定版本写在 `third_party/pins.lock`：


| 仓库         | commit                                     |
| ---------- | ------------------------------------------ |
| RoboDojo   | `f57e1633ffed59761bc8e568cfe7530c4b33381c` |
| XPolicyLab | `aa46382aa12dc46b360e2fdc63d0d3d39215b3e9` |


```bash
bash scripts/bootstrap_third_party.sh
bash third_party/RoboDojo/scripts/init_assets.sh
bash scripts/download_stack_blocks.sh
```

`init_assets.sh` 用 `git` 和 `git lfs` 做稀疏检出，从 Hugging Face 数据集 `RoboDojo-Benchmark/RoboDojo` 只取本任务用到的机器人、房间、方块、杂物、材质和 `arx_x5` 布局，然后链到 `third_party/RoboDojo/Assets`。它不使用 `huggingface-cli`。

示范脚本只下载 `data/RoboDojo/stack_blocks/arx_x5`，并链到 `data/RoboDojo/stack_blocks/arx_x5`。命令是新版 `hf download`，没有 `hf` 时用旧版 `huggingface-cli download`。选题也允许从 ModelScope 取同一份离线数据；若从那边下载，仍放到这个目录结构里。在线训练和评测都用这个任务，不要改成 `stack_blocks_random`。

已发布的仿真镜像里已经有 Isaac Sim。使用这个镜像时不必初始化 Isaac 子模块；那一步只在自己重新构建仿真镜像时才需要。

上游仓库里有别人的策略实现。要交的是你自己在对应入口里写的模型，而不是把上游训练脚本的输出直接当作结果。

## 两个镜像

两个 tar 包都在 [NantianHe/stack-blocks-images](https://huggingface.co/datasets/NantianHe/stack-blocks-images)。页面上的 Dataset Viewer 会报错，因为这两个文件是容器镜像包，不是可预览的数据集。直接下载文件即可。


| 文件                                 | 导入后的标签                         |
| ---------------------------------- | ------------------------------ |
| `stack-blocks-sim_isaacsim5.1.tar` | `stack-blocks-sim:isaacsim5.1` |
| `stack-blocks-policy_dp-act.tar`   | `stack-blocks-policy:dp-act`   |


直接下载地址写在 `configs/sim_image.env`。仿真包大约 37 GB，策略包大约 800 MB。有 `hf` 时：

```bash
hf download NantianHe/stack-blocks-images \
  stack-blocks-sim_isaacsim5.1.tar \
  stack-blocks-policy_dp-act.tar \
  --repo-type dataset \
  --local-dir .
```

没有 `hf` 时，把命令里的 `hf` 换成 `huggingface-cli`。下载完成后导入。这是 tar 包，不要对这两个链接执行 `docker pull`。

```bash
docker load -i stack-blocks-sim_isaacsim5.1.tar
docker image inspect stack-blocks-sim:isaacsim5.1 >/dev/null && echo "sim ok"

docker load -i stack-blocks-policy_dp-act.tar
docker image inspect stack-blocks-policy:dp-act >/dev/null && echo "policy ok"
```

本机的 `docker` 命令若实际是 Podman，同样使用这些命令。Podman 常常忽略 `--gpus`，仿真脚本因此改为只挂载 `env_gpu` 那一张卡。

仿真镜像里是 Isaac Sim。它不包含你的策略代码，也不包含 `Assets/` 和示范数据，这两样在运行时挂载。在线训练和评测脚本会自己 `docker run` 这个镜像。

策略镜像是参考环境，不是必须项。里面是 Ubuntu 22.04、Miniconda，以及两个空的 Python 3.10 环境 `dp` 和 `act`。没有 Isaac，没有训练好的权重，也还没有 PyTorch 或模型库。训练脚本不会启动它。可以不使用这个镜像，在宿主机用 conda 或 venv 自己装依赖，再激活后运行下面的脚本。

如果用它作参考，挂上本目录后进入容器，在 `dp` 或 `act` 里安装你的实现所需的包。只想对照上游 DP / ACT 的依赖时，可以在容器里对克隆下来的 XPolicyLab 执行 `policy/DP/install.sh` 和 `policy/ACT/install.sh`。离线训练可以在这个容器里跑。在线训练要能再启动仿真容器：把宿主机的容器套接字挂进策略容器，或者更直接地在宿主机环境里执行 `train_dppo.sh` / `train_act_residual.sh`，让这两个脚本去启动仿真镜像。

```bash
docker run --rm -it \
  --network host \
  -v "$PWD:/stack-blocks" \
  -w /stack-blocks \
  stack-blocks-policy:dp-act
```

容器内：

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate dp    # 或 act
```



## 镜像在什么时候被调用

在线训练时，策略进程留在当前环境，Isaac 由镜像启动。

- `bash scripts/train_dppo.sh` 读取 `configs/env.yaml`，用 `scripts/run_online_env.sh` 执行 `docker run`，容器内运行 `scripts/isaac_env_server.py`。然后在当前环境调用 `starter.pipelines.diffusion` 的 online 阶段。
- `bash scripts/train_act_residual.sh` 同样启动镜像，再调用 `starter.pipelines.act` 的 online 阶段。

离线阶段不启动镜像：

```bash
bash scripts/train_dp.sh
bash scripts/train_act.sh
```

评测时，策略服务在你的环境里监听端口，Isaac 客户端在镜像里运行。

```bash
bash scripts/eval_policy.sh configs/eval.yaml
```

`eval_policy.sh` 是在这台机器上拆开跑选题规定的评测：策略服务留在你的环境里，Isaac 客户端进仿真镜像。它先执行 `policy_dir` 指向的 `setup_eval_policy_server.sh`，再在容器里执行 RoboDojo 的 `scripts/robodojo.sh client`。容器内仿真卡写成 `cuda:0`，因为只挂载了 `env_gpu` 那一张卡。结果写到 RoboDojo 的 `eval_result/`。期中提交至少要有其中的 `_result.json` 和 `episode_*.mp4`。

选题原文的入口是：

```bash
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task stack_blocks \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --env-cfg arx_x5 \
  --action-type ee \
  --seed 0 \
  --eval-num native
```

策略要包成 XPolicyLab 能加载的目录，里面有 `setup_eval_policy_server.sh`。客户端按目录名到 `third_party/RoboDojo/XPolicyLab/policy/<名字>/deploy.py` 找适配。两阶段训练必须分别评离线检查点和在线检查点，在线相对离线的变化是评分的一项。若报告里写了相对基线的改进，对照至少用 3 个随机种子，并保留训练曲线和评测指标。

换 ACT 时，把 `policy_dir` 改成 `policies/act`，并换成该链路的 checkpoint 名。

## 任务约定

仿真任务是 `stack_blocks`，环境配置 `arx_x5`，动作类型 `ee`。离线示范、在线训练和评测都是这个任务：同一张背景、同一间房间、同一张桌子，没有杂物。各条评测布局里变化的是积木型号和桌面位置。

动作 16 维，世界基座坐标系：

```text
[left_ee_pose(7), left_gripper(1), right_ee_pose(7), right_gripper(1)]
```

位姿顺序是 `[x, y, z, qw, qx, qy, qz]`。夹爪下标是 7 和 15，送进环境前裁到 `[0, 1]`。关节角是 14 维，不要把它当成 `ee` 的宽度。官方数据脚本对 `arx_x5` 报的动作维度是关节的 14，这里要使用 16。

策略观测只有头部相机 `cam_head` 和这 16 维本体。`src/starter/envs/stack_blocks.py` 已经把腕部图像和积木位姿从策略观测里拿掉。积木位姿在 `get_privileged_state()`，只能用于训练时的内在奖励。`src/starter/rewards/intrinsic.py` 现在加的是 0。奖励可以改，但不要写回环境的 `reward_manager`，也不要改外部得分。

HDF5 在 `data/RoboDojo/stack_blocks/arx_x5/data/episode_*.hdf5`。`starter.data.demos.read_episode` 给出 16 维状态、16 维动作，以及 `vision/cam_head/colors` 里原样存放的图像缓冲。解码请使用克隆下来的 `XPolicyLab.utils.process_data.decode_image_bit`。这批数据里有旧的 JPEG 标记，用 PIL 直接解会把通道读反。文件里还有腕部相机，不要送进策略。图像如何缩放、是否堆历史帧，由你决定。

选题表里的奖励写成成功为 1、其余为 0。环境实际给出的外部得分不能改，数值是：


| 得分  | 含义             |
| --- | -------------- |
| 0   | 还没有两块积木堆在一起    |
| 15  | 任意两块堆在一起，且夹爪张开 |
| 100 | 三块堆稳，且双臂回到原点   |


100 就是选题里的成功，不要把它改写成 1，也不要把 15 改写成 100。回合最长 550 步。成功时 `terminated` 为真；走到 550 步仍未成功时 `truncated` 为真。环境异常或无效结束不是成功。不要在尚未成功时把回合立刻截断。15 分可以当作回报的一部分。

`env.reset` 返回 `(observation, info)`。`env.step` 返回 `(observation, score, terminated, truncated, info)`。在线采样用 `RemoteEnv`，一次连接里有 `configs/env.yaml` 的 `num_envs` 个场景。

## 需要实现的部分

两条链路都是先离线行为克隆，再在同一个 `stack_blocks` 任务里在线训练。超参写在你自己的配置里。起始代码没有学习率、扩散步数、clip、PPO 轮数或残差方差。

1. `src/starter/policies/diffusion.py`
  `train_offline` 读取 `stack_blocks` 示范。`train_online` 从这份权重继续，并用 `RemoteEnv` 与已经启动的仿真通信。`bash scripts/train_dp.sh` 和 `bash scripts/train_dppo.sh` 分别进入这两个函数。
2. `src/starter/policies/act.py`
  同样分离线和在线。入口是 `bash scripts/train_act.sh` 和 `bash scripts/train_act_residual.sh`。
3. `policies/diffusion/setup_eval_policy_server.sh` 和 `policies/act/setup_eval_policy_server.sh`
  参数顺序是：bench、task、ckpt、env_cfg、action_type、seed、policy_gpu、conda 环境名、端口、主机。在你的环境里加载对应 checkpoint，按 RoboDojo 评测客户端期待的协议应答。目录名要和 `deploy.py` 的查找对上，网络结构和 checkpoint 格式可以自己定。离线检查点和在线检查点按上一节分开评。



## 两条 pipeline 的原理

两条都是先在 `stack_blocks` 的示范上做行为克隆，再在同一个任务里用环境得分做在线强化学习。观测是头部相机和 16 维末端状态。一次预测可以是未来一小段动作，执行其中的若干步再重新规划。网络用卷积、Transformer 还是别的结构，历史帧取几帧，损失和学习率取多少，都由你定。

### [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)，然后做 [DPPO](https://github.com/irom-princeton/dppo)

离线阶段把动作序列的生成看成条件去噪。给一个观测和某个噪声水平，网络预测这一水平上的噪声；对示范动作加噪后再预测，就是行为克隆。推理时从高斯噪声出发，按调度逐步去掉噪声，得到动作。

在线阶段把这条去噪链当成策略。可以只让最后几个去噪步保留高斯随机性，更早的步用后验均值，不再抽样。PPO 只用这些随机步的对数概率，并沿去噪顺序打折，使越靠近最终动作的步权重越大。回报来自环境得分，可以再加只在训练时使用的内在奖励。若评测加载的是 EMA 权重，参数更新后也要写回 EMA。步数、折扣、标准差下限和 clip 自己选。

### [ACT](https://github.com/tonyzhaozh/act)，然后做残差 PPO

离线阶段训练一个确定性策略，由当前观测直接预测一段未来动作，用示范做行为克隆，然后冻结。原始 ACT 用条件变分结构把动作块和图像对齐，测试时潜变量取零；你也可以用别的确定性序列模型，只要能冻结、并能在同样的观测上重复给出基座动作。

在线阶段不改基座。另学一个小的高斯残差，执行的动作是基座动作加残差。PPO 的对数概率只对残差计算。残差幅度应明显小于动作本身，否则一开始就会盖住已经克隆到的行为。基座权重在更新之后必须和冻结时相同。官方 ACT 常用三台相机；本作业只使用 `cam_head`。

不利用示范、也不设计内在奖励或课程的随机高斯策略，很难在 550 步内碰到 100 分。选题允许的直接在线学习，指的是 PPO、SAC 这一类，并配上合理的内在奖励、探索或课程，而且仍然不改外部得分。离线策略如果一次都堆不成功，在线阶段也看不到 100 分。15 分是中途信号，可以进入回报，不能改成成功。