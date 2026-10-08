# Chunmo · 个人 AI 项目集

本仓库包含三个方向的项目：量化投资 AI、经典机器学习入门、MOBA 游戏强化学习。

| 项目 | 内容 | 技术栈 |
|---|---|---|
| [Project 1(量化AI)](<Project 1(量化AI)/>) | 量化研报与多模型预测一体化平台 | Python · baostock/akshare · XGBoost/LightGBM · LLM |
| [Project 2(ML泰坦尼克)](<Project 2(ML泰坦尼克)/>) | Kaggle Titanic 机器学习入门项目 | Python · pandas · scikit-learn |
| [Project 3(Hok)](<Project 3(Hok)/>) | 王者荣耀 1v1 强化学习训练 | hok_env · 自研 PPO · PyTorch |

---

## Project 1 · 量化AI（QuantAI）

「智能可视化分析报表 + 多模型机器学习预测与智能决策」一体化平台：

输入股票代码 → 自动下载行情 → 技术指标计算 → 交互式 K 线图 → AI 股评 → 分组对比 → 多模型预测 → 风控建议。

- **研报生成**：K 线 + MA/BOLL/RSI/MACD 四联图，AI 四段式股评，HTML/PDF 导出
- **分组对比**：两组股票归一化收益率曲线 + 对比表 + AI 对比研报
- **多模型预测**：逻辑回归 / XGBoost / LightGBM 时序交叉验证赛跑，冠军模型输出上涨概率 + AI 风控建议
- **工程化**：双数据源容灾、数据缓存、LLM 失败降级兜底、一键打包 exe

目录：[project](<Project 1(量化AI)/project>)（代码，含详细 README）

快速开始：

```bash
cd "Project 1(量化AI)/project"
pip install -r requirements.txt
python run.py        # Windows 也可直接双击 run.bat
```

## Project 2 · ML泰坦尼克

Kaggle 经典入门赛 [Titanic](https://www.kaggle.com/c/titanic) 的完整机器学习流程实践：数据加载 → 特征工程 → 多模型训练 → 并行调参 → 生成提交文件。

目录：[titanic](<Project 2(ML泰坦尼克)/titanic>)

快速开始：

```bash
cd "Project 2(ML泰坦尼克)/titanic"
pip install -r requirements.txt
python run.py        # 训练并生成 submission.csv
```

## Project 3 · Hok（王者荣耀 1v1 强化学习）

基于腾讯开源的 [Honor of Kings Arena（hok_env）](https://github.com/tencent-ailab/hok_env)（NeurIPS 2022 Datasets and Benchmarks）训练 1v1 MOBA 智能体。

**本项目没有沿用官方的分布式训练框架（rl_framework / Docker / Actor-Learner 集群），而是自研了一套单进程 PPO 实现**，直接通过 zmq 与 Windows 侧的 gamecore 通信。取舍理由与代价写在项目 README 里。

大纲：

- **环境与动作**：观测 725 维（705 + 20 维英雄身份 one-hot，每局轮换英雄）；动作是 6 头分层空间 `[12,16,16,16,16,8]`，推理用 172 维 mask、训练用 84 维压缩 mask，并按 `sub_action_mask` 屏蔽与该按钮无关的动作头
- **网络与算法**：LSTM(512) + 分层动作头；PPO 逐头 ratio/clip、GAE(γ=0.995, λ=0.95)、KL 双阈值、梯度裁剪；采样侧与训练侧强制走同一个前向（关时间维 attention / dropout / AMP），否则 `ratio` 与 `KL` 全部失真
- **轨迹处理**：尾部不足一个 chunk 的决策用 0 补齐并给出 `valid_mask`，保证含 `done=1` 的终局步一定进入 batch（此项此前被整段丢弃）
- **工程配套**：自研训练看板（整数步轴、当前值常驻、可调平滑、多次运行隔离）、离线自检、配置一致性校验，以及诊断探针（奖励量级 / 胜负判定对账 / 观测语义）
- **验证情况**：离线自检与配置一致性通过；真实训练可跑通，`old_new_logp_gap ≈ 1e-5`、终局步覆盖率 1.0；胜负判定经 5 局实测对账自洽；支持多实例并行对照（实测两实例互不干扰）

目录：[hok1v1_rl](<Project 3(Hok)/hok1v1_rl/>)

快速开始：

```bash
# 1. Windows 侧先启动 gamecore-server（需自行申请授权，见下方注意）
cd "Project 3(Hok)"
.\hok_env_gamecore_20260420\gamecore\gamecore-server.exe server --server-address :23432

# 2. WSL 侧（Python 3.9）
cd hok1v1_rl
pip install -r requirements.txt
python scripts/_smoke_test.py            # 离线自检，不需要 gamecore
python scripts/train_ppo.py --total-steps 1000
```

> ⚠️ **注意**：`gamecore`（游戏引擎二进制）与授权文件 `license.dat` 为腾讯专有，**仅保存在本地**，已由仓库根目录 `.gitignore` 排除。本仓库只包含自研训练代码；游戏引擎需在 [aiarena 开放平台](https://aiarena.tencent.com/aiarena/en/open-gamecore) 申请授权后自行获取。
>
> 📌 **当前不足与后续方向**（详见项目 README）：训练量仍很小（约 60 次 update，尚未看到胜率提升证据）；对手固定为内置 `common_ai`、缺少 self-play；value 头尚未学好；单环境吞吐是瓶颈，多进程采样待做。

架构与完整训练指南见 [Project 3(Hok)/README.md](<Project 3(Hok)/README.md>)。
