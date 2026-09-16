"""全局配置：路径、随机种子、交叉验证参数。

集中管理所有可调参数，便于复现与调优。
"""
from pathlib import Path

# ── 路径配置 ──────────────────────────────────────────────
# 项目根目录：本文件位于 src/config.py，上溯一级到 titanic/
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR          # train.csv / test.csv 与代码同目录
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SUBMISSION_CSV = DATA_DIR / "submission.csv"   # 最终输出，格式同 gender_submission.csv

# ── 随机性与复现 ──────────────────────────────────────────
# 统一随机种子：所有含随机性的模型共用，保证多次运行结果一致
RANDOM_STATE = 42

# ── 交叉验证 ──────────────────────────────────────────────
# 用 StratifiedKFold（分层 K 折）：保持每折正负样本比例一致，
# 适合泰坦尼克这种略微不均衡（生存率 38%）的二分类任务
CV_FOLDS = 5

# ── 并行 ──────────────────────────────────────────────────
# n_jobs=-1 表示使用全部 CPU 核心并行训练各算法
N_JOBS = -1
