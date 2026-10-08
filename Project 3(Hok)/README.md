# Project 3 · 王者荣耀 1v1 强化学习（hok_env）

基于腾讯开源的 [Honor of Kings Arena](https://github.com/tencent-ailab/hok_env)（NeurIPS 2022 Datasets and Benchmarks）训练 1v1 MOBA 智能体。

**本目录的 `hok1v1_rl/` 是一套自研的单进程 PPO 实现**，不依赖官方 `rl_framework` 分布式框架，
直接通过 zmq 与 Windows 侧的 gamecore 通信。选择自研而非直接用官方框架的原因见
[「为什么不用官方 rl_framework」](#为什么不用官方-rl_framework)。

---

## 整体架构

```
┌──────────────────────────┐    HTTP(开局/关局)  ┌────────────────────────────┐
│ Windows（游戏引擎）       │ ◄─────────────────► │ WSL2（训练进程）            │
│  gamecore-server.exe     │                     │  python train_ppo.py       │
│  listen :23432           │    zmq(逐帧动作)     │   ├ EnvRunner 采样          │
│  sgame_simulator_*.exe   │ ◄─────────────────► │   ├ PPOLearner 更新         │
│  （按需拉起，随即退出）    │   tcp://127.0.0.1   │   └ metrics.jsonl / ckpt   │
└──────────────────────────┘      :35150+        └────────────────────────────┘
```

- **gamecore**（游戏引擎，闭源二进制）跑在 Windows。需在
  https://aiarena.tencent.com/aiarena/en/open-gamecore 申请授权，`license.dat` 放在
  `gamecore/core_assets/` 下。**该目录仅存本地，不进 git**（已由仓库根 `.gitignore` 排除）。
- **训练进程**跑在 WSL2，通过 `predict_frequency=3` 每 3 帧决策一次，动作经 zmq 回传。
- 一个训练进程 = 一个环境。多环境并行靠**多进程**而非向量化（见「并行训练」）。

---

## 快速开始

### 1. 启动 gamecore（Windows）

```powershell
cd "D:\Code\Project 3(Hok)\hok_env_gamecore_20260420\gamecore"
.\gamecore-server.exe server --server-address :23432
```

### 2. 安装依赖（WSL）

```bash
# Python 3.9（hok_env 原生扩展只支持 3.6–3.9）
pip install -e ~/hok/hok_env/hok_env      # 官方 SDK
pip install -r hok1v1_rl/requirements.txt # torch 等
```

### 3. 自检（不需要 gamecore）

```bash
cd hok1v1_rl
python scripts/_smoke_test.py                    # 动作空间 / 奖励 / GAE / 网络形状
python scripts/check_config_consistency.py       # 配置一致性（23 个字段）
```

### 4. 训练

```bash
# 短测（2 步，约 3 分钟）
python scripts/train_ppo.py --total-steps 2

# 正式训练
python scripts/train_ppo.py --total-steps 1000
```

> `--total-steps` 是 **LR/熵退火的基准**，不是"跑够就停"。当前配置**不退火**，
> 所以传小值做短测是安全的。

### 5. 看板

```bash
python scripts/dashboard.py --port 8600     # 浏览器打开 http://localhost:8600
```

### 6. 并行训练（多组对照）

```bash
# 两个实例共享同一个 gamecore-server，靠 runtime_id / zmq 端口 / 日志目录隔离
python scripts/train_ppo.py --runtime-id hok1v1rl-a --zmq-port 35150 --log-dir logs_A &
python scripts/train_ppo.py --runtime-id hok1v1rl-b --zmq-port 35152 --log-dir logs_B &
```

| 隔离维度 | 参数 | 说明 |
|---|---|---|
| 对局标识 | `--runtime-id` | **必须各不相同**（不能含下划线） |
| zmq 端口段 | `--zmq-port` | 每个实例占 **2 个**连续端口 |
| 指标/存档 | `--log-dir` | 必须隔离，否则指标会混成一条乱曲线 |
| 配置 | `--config <path>` | 换实验配置 |
| gamecore | `--gamecore-addr` | 跑多个 gamecore 实例时用 |

**性能实测**：单实例 114 决策/秒；两实例并行时各约 79，合计约 160
（**总吞吐 +40%，单实例慢 31%**）。⇒ 只跑一个实验就串行；跑两组对照才并行。

---

## 代码结构

```
hok1v1_rl/
├── hok1v1_rl/                 # 训练包
│   ├── env_runner.py          # 环境封装、观测拼装、6 头分层动作采样、整局轨迹收集
│   ├── network.py             # LSTM(512) + 分层动作头 + Multi-head Value
│   ├── learner.py             # PPO：逐头 ratio/clip、head 屏蔽、KL 双阈值、GradClip
│   ├── reward.py              # 奖励（req_pb 状态差分）与 GAE
│   ├── action_mask.py         # 分层动作 mask、172→84 legal_action 压缩
│   ├── config.py              # 配置（含每个取值的依据）
│   └── logger.py / metrics_store.py / vram_guard.py
├── scripts/
│   ├── train_ppo.py           # 训练入口（含并行所需参数）
│   ├── dashboard.py           # 自研看板（整数步轴 / 当前值常驻 / 平滑 / 多运行隔离）
│   ├── _smoke_test.py         # 离线自检
│   ├── check_config_consistency.py
│   └── probe_*.py             # 诊断探针（奖励量级 / 胜负判定 / 观测语义）
└── configs/default.yaml       # 唯一配置真源
```

### 几个关键设计（都是踩过坑之后定下来的）

| 设计 | 原因 |
|---|---|
| `use_time_attention=false`、`attn_dropout=0`、`amp=false` | 采样侧 T=1、训练侧 T=16。若两侧不是**同一个前向**，PPO 的 `ratio`/`KL` 全部失真 —— 表现为"loss 在动但胜率不动" |
| 尾部 chunk 用 0 补齐 + `valid_mask` | 原实现丢弃不足一个 chunk 的尾巴，而 `n % T` 几乎不可能为 0 ⇒ **每局都丢掉含 `done=1` 的终局步**。learner 按 `valid_mask` 归一化，避免补齐的 0 稀释梯度 |
| 逐头 ratio + `sub_action_mask` 屏蔽 | 官方是**逐头** clip 再按 head 权重求和（不是 6 头乘积 ratio）；且只有与该 button 相关的头才该参与 loss |
| `feature_dim=725`（705 + 20 维英雄身份 one-hot） | 本项目每局轮换英雄，缺这 20 维策略"不知道自己是谁" |
| 胜负判定用 req_pb 的**水晶血量** | 官方 `actor.py` 的"我方英雄 hp>0 即胜"在双方都活着的推塔局里恒判为胜 |

---

## 当前进展与已知不足

### 已跑通并验证

| 项目 | 结果 |
|---|---|
| 离线自检 / 配置一致性 | ✅ 全部通过 |
| 真实训练（连接 gamecore） | ✅ 可跑，`old_new_logp_gap ≈ 1e-5`（两侧前向一致） |
| 终局步进 batch | ✅ `env/terminal_in_batch` 恒为 1.0（实测每局需补齐 1~3 步，旧实现会丢） |
| 胜负判定对账 | ✅ 5 局实测全部自洽（`lose` ⇔ 我方水晶归零、推塔差分同向） |
| 并行训练 | ✅ 两实例并行互不干扰 |
| PPO 更新幅度 | ✅ `train/kl` 已进入 1e-3~1e-2 健康区间（调整 lr 前仅 8e-5） |

### 不足与未来改进方向

**① 训练量严重不足 —— 这是最根本的问题**
累计只有约 **60 次 update（约 50 万决策帧）**，而参考量级是千万帧起步。
目前 `env/win_rate` 仍接近 0。**"训练效果好不好"这个问题现在还没有有效证据。**

**② 对手固定为内置 `common_ai`，能力上限被写死**
只打脚本 AI，缺少 self-play 与对手池，学不到超越脚本的策略。
改进：实现模型池（最新 + 每 N 步快照，"80% 最新 / 20% 历史"采样），评估仍固定打
`common_ai` 以保证基准不漂移。

**③ 价值函数没学好**
`train/explained_variance` 长期在 0 附近甚至为负，说明 value 头未能解释回报方差，
advantage 噪声偏大。已尝试关闭在线归一化（官方本就没有），仍需继续定位。

**④ 单环境吞吐是瓶颈，并行只是权宜**
单环境 114 决策/秒，GPU 利用率仅 6% —— 说明瓶颈在 CPU 侧的模拟器与串行采样。
真正的解法是做 **actor-learner 分离 + 多进程采样**（官方即此架构），
本项目目前只是"多开几个进程"。

**⑤ 奖励的信用分配仍有结构性问题**
`env/reward/tower` 几乎恒为 −10（每局都丢基地）。**常数项在 advantage 中会被减掉**，
等于权重最大的那一项没在指导策略。改进方向：把推塔改成**每步差分**而非局末结算。

**⑥ 训练规模与评估口径**
- `batch_size` 只有 4~5 局且连续对局高度相关（官方是 512 条独立序列）；
- 评估每 50 步 10 局，样本量仍偏小，缺少 Elo 或模型池式的长期能力追踪。

**⑦ 工程细节**
- `checkpoint` 无版本标记：改过 `feature_dim`（705→725）后旧权重全部失效，
  且没有"断点续训"的兼容性保障，新实验一律从头训；
- `metrics.jsonl` 是追加写的，多次运行混在一起（看板已做隔离，但文件本身没分段）；
- 随机种子固定为 42，单一 seed 的实验结论不足以判断稳定性。

**⑧ 尚未实现的功能**
`self_play` 与 `bc`（行为克隆预训练）在配置里有占位，但代码未实现。
多环境向量化（`num_envs > 1`）也未实现 —— `interface.Interface` 是进程内单例，
必须走多进程。

### 为什么不用官方 rl_framework

官方提供了完整的 actor-learner 分布式框架（Docker 容器 + mem_pool + model_pool +
horovod），本目录早期也按那条路线搭过环境。最终改为自研单进程实现，原因是：

- 官方框架的**采样吞吐依赖分布式部署**（一 CPU 核一个 actor、2000 并发环境），
  单机单卡上跑不出它的设计收益，反而很难调试；
- 自研实现能把**每一次前向、每一步奖励**都摊开看（这也是本项目发现
  "hp 占位值导致幽灵伤害""终局步被丢弃""LR 退火到 0"这些问题的前提）；
- 代价是**没有现成的 self-play / 模型池 / 多机扩展**——这正是上面 ② ④ 的缺口，
  也是后续要补的。

> 官方框架的搭建步骤仍保留在本仓库历史提交里，需要时可以参考。

---

## 其他

### 奖励配置参考

[config/actor_reward_config.json](config/actor_reward_config.json) 是 gamecore 侧的原始奖励权重
（金钱/经验/血量/推塔/击杀/死亡/补刀）。注意：**训练实际使用的是
`hok1v1_rl/reward.py` 里基于 `req_pb` 状态差分的奖励**，不走这份配置 ——
因为实测 gamecore 的 reward 向量里 `dead` 分量会把小兵死亡也算进去。

### 看回放

gamecore 生成的 `.abs` 回放文件放在 `replay_tool/Replays/`，运行 `ABSTool.exe` 可视化整局比赛。

### 参考

- 上游仓库：https://github.com/tencent-ailab/hok_env
- 环境文档：https://aiarena.tencent.com/hok/doc/quickstart/index.html
- 论文：Honor of Kings Arena: an Environment for Generalization in Competitive Reinforcement Learning（NeurIPS 2022）
