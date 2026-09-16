# Project 3 · 王者荣耀 1v1 强化学习（hok_env）

基于腾讯开源的 [Honor of Kings Arena](https://github.com/tencent-ailab/hok_env)（NeurIPS 2022 Datasets and Benchmarks）训练 1v1 智能体，使用框架内置的分布式 PPO 实现（支持 PyTorch / TensorFlow 后端）。

## 整体架构

三台"机器"各司其职：

```
┌─────────────────────────┐   ZMQ/HTTP   ┌──────────────────────────────┐
│ Windows（本地）          │ ◄──────────► │ WSL2 中的 Docker 容器         │
│ gamecore-server.exe     │              │ /hok_env   hok_env SDK        │
│ D:\Code\Project 3(Hok)\ │              │ /aiarena   1v1 训练代码        │
│   hok_env_gamecore_*/   │              │ /rl_framework 分布式训练框架   │
└─────────────────────────┘              │   Actor(采样) → Mem_pool      │
                                         │   → Learner(PPO) → Model_pool │
                                         └──────────────────────────────┘
```

- **Gamecore**（游戏引擎）运行在 Windows，需在 https://aiarena.tencent.com/aiarena/en/open-gamecore 申请授权后下载，`license.dat` 放在 `gamecore/core_assets/` 下。该目录**仅存本地，不进 git**（已由仓库根目录 `.gitignore` 排除）。
- **训练代码**（`tencent-ailab/hok_env` 上游仓库）克隆在 WSL：`~/hok/hok_env`，本地已按需修改奖励配置。
- **训练框架**运行在 Docker 容器（镜像 `tencentailab/hok_env`）中，通过 `host.docker.internal` 访问 Windows 侧的 gamecore。

## 环境搭建与训练

```bash
# 1. Windows 侧启动 gamecore
cd "D:\Code\Project 3(Hok)\hok_env_gamecore_20260420\gamecore"
gamecore-server.exe server --server-address :23432

# 2. WSL 侧拉取镜像并启动容器（有 NVIDIA 卡用 gpu 标签，否则用 cpu）
docker run -it -p 35000-35400:35000-35400 tencentailab/hok_env:gpu_v2.0.4 bash

# 3. 容器内连通性测试（通过即完成一局随机对战）
export GAMECORE_SERVER_ADDR="host.docker.internal:23432"
cd /hok_env/hok/hok1v1/unit_test && python3 test_env.py

# 4. 启动训练（单机：先 learner 后 actor）
cd /aiarena/scripts/learner && sh start_learner.sh
cd /aiarena/scripts/actor
export NO_MODEL_POOL=1        # 与 learner 同容器时设为 1
sh start_actor.sh

# 5. 观察训练
tail -f /aiarena/logs/learner/train.log     # 训练日志
tail -f /aiarena/logs/learner/loss.txt      # loss 曲线数据
tail -f /aiarena/logs/actor/actor_0.log     # 对局统计（win/kill/death）
# 模型保存在容器内 /aiarena/checkpoints，建议挂载宿主机目录持久化：
# docker run ... -v ~/hok/output/checkpoints:/aiarena/checkpoints -v ~/hok/output/logs:/aiarena/logs
```

## 训练建议（RL 入门路线）

- **对手选择**：`export ENEMY_TYPE=1` 先打内置 common_ai 训练初步模型（curriculum 起点，loss 下降明显），熟练后切 `ENEMY_TYPE=2` 自博弈（默认）。
- **奖励调参**：[config/actor_reward_config.json](config/actor_reward_config.json) 是当前使用的奖励权重参考（金钱/经验/血量/推塔/击杀/死亡/补刀），修改后需重启 actor 生效。
- **看对局**：gamecore 生成的 `.abs` 回放文件放入 `replay_tool/Replays/`，运行 `ABSTool.exe` 可视化整局比赛。

## 目录说明

| 路径 | 说明 |
|---|---|
| `hok_env_gamecore_20260420/` | gamecore 程序 + 回放工具 + license.dat（本地专有，不提交） |
| `config/actor_reward_config.json` | 奖励权重配置参考（与 WSL 中 `~/hok/hok_env` 的修改对应） |

## 参考

- 上游仓库：https://github.com/tencent-ailab/hok_env
- 环境文档：https://aiarena.tencent.com/hok/doc/quickstart/index.html
- 论文：Honor of Kings Arena: an Environment for Generalization in Competitive Reinforcement Learning（NeurIPS 2022）
