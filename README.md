# Chunmo · 个人 AI 项目集

本仓库包含三个方向的项目：量化投资 AI、经典机器学习入门、MOBA 游戏强化学习。

| 项目 | 内容 | 技术栈 |
|---|---|---|
| [Project 1(量化AI)](<Project 1(量化AI)/>) | 量化研报与多模型预测一体化平台 | Python · baostock/akshare · XGBoost/LightGBM · LLM |
| [Project 2(ML泰坦尼克)](<Project 2(ML泰坦尼克)/>) | Kaggle Titanic 机器学习入门项目 | Python · pandas · scikit-learn |
| [Project 3(Hok)](<Project 3(Hok)/>) | 王者荣耀 1v1 强化学习训练 | hok_env · PPO · Docker |

---

## Project 1 · 量化AI（QuantAI）

「智能可视化分析报表 + 多模型机器学习预测与智能决策」一体化平台：

输入股票代码 → 自动下载行情 → 技术指标计算 → 交互式 K 线图 → AI 股评 → 分组对比 → 多模型预测 → 风控建议。

- **研报生成**：K 线 + MA/BOLL/RSI/MACD 四联图，AI 四段式股评，HTML/PDF 导出
- **分组对比**：两组股票归一化收益率曲线 + 对比表 + AI 对比研报
- **多模型预测**：逻辑回归 / XGBoost / LightGBM 时序交叉验证赛跑，冠军模型输出上涨概率 + AI 风控建议
- **工程化**：双数据源容灾、数据缓存、LLM 失败降级兜底、一键打包 exe

目录：[doc-量化AI项目](<Project 1(量化AI)/doc-量化AI项目>)（项目文档）· [project](<Project 1(量化AI)/project>)（代码，含详细 README）

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

基于腾讯开源的 [Honor of Kings Arena（hok_env）](https://github.com/tencent-ailab/hok_env)（NeurIPS 2022 Datasets and Benchmarks）进行 1v1 智能体强化学习训练，使用框架自带的分布式 PPO 实现。

> ⚠️ **注意**：gamecore（游戏核心程序）与 `license.dat` 为腾讯专有授权文件，仅保存在本地，已被 `.gitignore` 排除。训练代码（上游开源仓库 `tencent-ailab/hok_env`，Apache-2.0）以本地克隆的形式运行于 WSL 中，不包含在本仓库内。

架构与完整训练指南见 [Project 3(Hok)/README.md](<Project 3(Hok)/README.md>)。
