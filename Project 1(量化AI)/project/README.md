# QuantAI · 量化研报与多模型预测一体化平台

> 将「智能可视化分析报表」与「多模型机器学习预测与智能决策」整合为一个工程化项目。
> 输入股票代码 → 自动下载行情 → 技术指标 → 交互式 K 线图 → AI 股评 → 分组对比 → 多模型预测 → 风控建议。

---

## ✨ 功能特性

| 模块 | 功能 |
|------|------|
| **研报生成** | 批量多只股票，K 线 + MA/BOLL + 成交量 + RSI + MACD 四联图，AI 四段式股评，支持 HTML/PDF 导出 |
| **分组对比** | 两组股票归一化收益率曲线 + 对比表 + AI 对比研报 |
| **多模型预测** | 逻辑回归 / XGBoost / LightGBM 时序交叉验证赛跑，冠军模型预测上涨概率 + AI 风控建议 |
| **设置中心** | 提示词模板编辑、模型注册表增删、API 钥匙管理、数据缓存管理 |

**工程化增强：**
- 双数据源容灾（baostock 主 / akshare 备）
- 数据缓存（CSV 本地缓存，重复查询秒级返回）
- 技术指标扩展（MA5/10/20、RSI、MACD、BOLL、KDJ）
- AI 降级兜底（LLM 失败自动切换模板股评）
- 日志系统（控制台 + 文件）
- 一键打包 exe

---

## 📁 目录结构

```
project/
├── run.py                     # 启动入口（开发模式）
├── run.bat                    # Windows 一键启动
├── build_exe.bat              # exe 打包脚本
├── quantai.spec               # PyInstaller 打包配置
├── requirements.txt           # 依赖清单
├── test_pipeline.py           # 流程测试脚本
├── config/                    # 配置文件
│   ├── providers.json         # 大模型注册表
│   ├── prompt_template.txt    # 股评提示词模板
│   ├── compare_template.txt   # 对比研报模板
│   └── keys.env.example       # 钥匙模板（复制为 keys.env）
├── data/                      # 行情缓存（运行时生成）
├── exports/                   # 导出报告（运行时生成）
├── logs/                      # 日志（运行时生成）
└── src/quantai/               # 核心包
    ├── __init__.py
    ├── paths.py               # 路径管理
    ├── config.py              # 配置读写
    ├── logger.py              # 日志
    ├── data/                  # 数据层
    │   ├── fetcher.py         # 双源下载+清洗
    │   ├── cache.py           # CSV 缓存
    │   └── indicators.py      # 技术指标
    ├── ml/                    # 机器学习层
    │   ├── features.py        # 特征工程
    │   ├── models.py          # 三模型赛跑
    │   └── predictor.py       # 预测主入口
    ├── ai/                    # AI 层
    │   ├── client.py          # LLM 调用+降级
    │   └── prompts.py         # 提示词组装
    ├── report/                # 报表层
    │   ├── chart.py           # Plotly 图表
    │   ├── generator.py       # 单股全流程
    │   ├── batch.py           # 批量管道
    │   ├── compare.py         # 分组对比
    │   └── export.py          # PDF/HTML 导出
    └── web/
        └── app.py             # Streamlit 界面
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 如果使用 conda 环境：
> ```bash
> conda activate Xm_env
> pip install -r requirements.txt
> ```

### 2. 配置 API 钥匙（可选，不配则 AI 自动降级为模板股评）

复制 `config/keys.env.example` 为 `config/keys.env`，填入你的大模型 API Key：

```
DEEPSEEK_API_KEY=sk-你的钥匙
# ZHIPU_API_KEY=
# DASHSCOPE_API_KEY=
# MOONSHOT_API_KEY=
```

或在网页「设置中心 → API 钥匙」中填写。

### 3. 启动

**方式一：双击 `run.bat`**

**方式二：命令行**

```bash
python run.py
```

浏览器会自动打开 `http://localhost:8501`。

---

## 📦 打包为 exe

```bash
build_exe.bat
```

打包完成后，exe 位于 `dist/QuantAI/QuantAI.exe`，双击即可运行（无需安装 Python）。

> 打包需要 5-10 分钟，产物约 500MB-1GB（含 pandas/sklearn 等）。

---

## 🔧 配置说明

| 文件 | 说明 |
|------|------|
| `config/providers.json` | 大模型注册表（网址/模型名/钥匙环境变量名） |
| `config/prompt_template.txt` | 股评提示词模板，占位符 `{code}` `{date}` `{facts}` `{style_req}` |
| `config/compare_template.txt` | 对比研报模板，占位符 `{start}` `{end}` `{stats}` |
| `config/keys.env` | API 钥匙（.gitignore 忽略，不进版本库） |

**钥匙优先级：** 环境变量 > keys.env > 报错降级

**容灾演练开关（环境变量）：**
```bash
set FAIL_BAOSTOCK=1   # 掐断主数据源，看 akshare 接管
set FAIL_LLM=1        # 掐断 AI 通道，看模板兜底接管
```

---

## ⚠️ 已知边界

- 仅 A 股日线、前复权数据
- 分类预测（涨/跌）而非价格回归
- AI 输出基于技术指标，仅供参考，不构成投资建议
