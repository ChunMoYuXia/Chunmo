# Chunmo · 个人 AI 项目集

本仓库包含三个方向的项目：量化投资 AI、经典机器学习入门、MOBA 游戏强化学习。

| 项目 | 内容 | 技术栈 |
|---|---|---|
| [Project 1(量化AI)](<Project 1(量化AI)/>) | 量化研报与多模型预测一体化平台 | Python · baostock/akshare · XGBoost/LightGBM · LLM |
| [Project 2(ML泰坦尼克)](<Project 2(ML泰坦尼克)/>) | Kaggle Titanic 机器学习入门项目 | Python · pandas · scikit-learn |
| [Project 3(Hok)](<Project 3(Hok)/>) | 王者荣耀 1v1 强化学习训练 | hok_env · 自研 PPO · PyTorch |
| [QuantAI 初版归档](QuantAI/) | 本地选股、组合配置、回测与模拟交易客户端 | FastAPI · SQLAlchemy · Svelte · TypeScript · Tauri |

---

## QuantAI · 股票组合研究客户端初版

[QuantAI/README.md](QuantAI/README.md) 保存 2026-10-15 已验收的 M0 / v0.1.0 核心源码：CSV 与交易日历导入、历史行情、条件选股、组合优化、回测、模拟交易及可追溯报告。包含运行步骤、依赖锁、回归测试和维护脚本，不含开发环境、缓存、日志、个人数据、密钥、模型权重或安装包。

这是后续统一 QuantAI 工作台的初版归档；在线下载、RAG、多模型中心和 LangChain / LangGraph 的合并开发仍在本地进行，待用户验收，不作为本归档已实现功能。本次不发布新的安装包或 Release。

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

[HoK 1v1 PPO 项目](<Project 3(Hok)/hok1v1_rl/README.md>)包含 20 英雄训练、固定对阵评估、动作诊断和单因素对照脚本。当前目标为 90% 以上胜率，尚未达标；下一步先验证塔前动作保护，再决定是否扩大训练。

仓库只保存自研代码与配置；外部 SDK、受授权约束的游戏引擎、模型、日志和本地学习笔记均不公开。
