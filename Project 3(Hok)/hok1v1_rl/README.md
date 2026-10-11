# HoK 1v1 PPO 训练

自研 PyTorch PPO 策略，20 英雄随机对战 `common_ai`。训练目标是固定评估口径下胜率超过 90%；截至 2026-10-11 尚未达到。这里保存训练代码、配置、测试和可复现实验入口。模型、对局日志和详细学习笔记只保存在本地。

## 项目边界

`hok1v1_rl/` 是策略、采样、奖励和 PPO 实现；`configs/` 保存基线与单因素实验配置；`scripts/` 提供训练、评估、诊断和对照入口；`dashboard/` 是本地只读看板；`tests/` 是离线回归测试。

游戏引擎、授权文件、素材、原生 SDK 和运行产物不属于本仓库，也不随代码分发。使用者须自行取得并按授权条件配置外部运行环境。仓库提供不含授权信息的 `config.example.json`；本地 `config.json` 不提交。

## 环境与关键参数

已验证的本地组合是 Python 3.9、兼容当前 GPU 的 PyTorch 与外部 `hok_env` SDK。安装项目依赖后，在本目录执行离线检查；真实训练还需要已授权的本地游戏服务。

```bash
pip install -r requirements.txt
pip install -e /path/to/authorized/hok_env/hok_env  # 替换为你自己的 SDK 路径
cp config.example.json config.json
python scripts/check_env.py --no-gamecore
python scripts/_smoke_test.py
python scripts/check_config_consistency.py
```

| 项目 | 当前基线 |
|---|---|
| 英雄池 | 双方各 20 英雄，见 `configs/default.yaml` |
| PPO | `gamma=0.999`，`lamda=0.95`，`clip=0.2`，`value_coef=0.5` |
| 优化 | 恒定 `lr=1e-4`，`entropy_beta=0.01`，每批 2 轮，最少 8192 决策 |
| 序列 | LSTM 反传 16 步；当前短测用片段 512、burn-in 64 |
| 快评 | 当前对照在更新 4/8 各评 80 局；最终验收至少 800 局 |
| 动作 | 基线为 `targeted_attack_v1`；塔前保护候选半径 15000 |

参数真值以运行目录的 `config.yaml` 和 `manifest.json` 为准。`--resume` 恢复同一实验的模型、Adam 与计数；改变 PPO、奖励或动作规则时，以 `--init-from` 开新目录做对照。所有并行运行须使用独立的输出目录、`runtime_id` 和连续 ZMQ 端口段。

## 训练与验证

从零开始的新实验示例：

```bash
python scripts/train_ppo.py --config configs/reward_events_stage8_control.yaml \
  --total-steps 8 --log-dir runs/new-control-v1
```

当前下一步是从本地更新 24 的已审计权重，做塔前动作保护的单因素短测；脚本会检查来源 SHA256、配置差异、输出目录与端口。该权重不上传，须在本机保留于脚本指定路径。

```bash
bash scripts/run_tower_guard_ab.sh both
```

对照与候选各训练 8 次更新，在更新 4/8 各固定评估 80 局。此前 `value_coef=0.05`、死亡惩罚加大、熵系数降低及经济差奖励 `2.0` 等候选未通过整体评估；当前冻结权重虽能用塔前保护消除诊断中的越塔路径，尚未证明提高胜率。先核对胜率、未知局、待机率、经济、建筑伤害和英雄 154 的红蓝双方表现，再决定是否扩大样本；目前不进入长训。

```bash
python -m unittest discover -s tests -q
python scripts/evaluate.py --checkpoint runs/<run>/best_ckpt.pt --episodes 800
python scripts/dashboard.py --port 8600
```

`runs/`、`logs*/`、`学习笔记/`、模型与原生运行文件均由 `.gitignore` 排除。历史审计、调参理由、逐局诊断及交接命令保存在本地学习笔记，不作为公开仓库的稳定操作说明。
