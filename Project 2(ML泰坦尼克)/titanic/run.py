"""泰坦尼克号生存预测 - 主入口。

一键跑完完整流程：数据加载 → 特征工程 → 多算法并行训练 → 排行榜 → 生成提交文件。

运行方式：
    cd D:\\Code\\Project 2\\titanic
    python run.py

输出：
    - 控制台打印数据概览、多算法排行榜、冠军模型、提交概览。
    - 在同目录生成 submission.csv（格式同 gender_submission.csv，可提交 Kaggle）。

模块化设计：
    data_loader → features → models → parallel → evaluate → submit
    每个模块职责单一，可独立测试与替换。
"""
import sys
from pathlib import Path

# 把 src 目录加入 sys.path，使 `from src import ...` 可解析
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_train_test, print_overview
from src.features import build_features
from src.models import get_models, SKIPPED
from src.parallel import run_parallel
from src.evaluate import make_leaderboard, print_leaderboard, select_best
from src.submit import create_submission, print_submission_summary
import pandas as pd


def main():
    # ── 1. 数据加载 ──
    print("步骤 1/5: 加载数据...")
    train_df, test_df = load_train_test()
    print_overview(train_df, test_df)

    # ── 2. 特征工程 ──
    print("步骤 2/5: 特征工程...")
    X_train, y, X_test, passenger_ids = build_features(train_df, test_df)
    print(f"  特征矩阵: {X_train.shape[0]} 训练行 × {X_train.shape[1]} 特征")
    print(f"  测试集: {X_test.shape[0]} 行")
    print(f"  特征列: {list(X_train.columns)}")

    # ── 3. 多算法并行训练 ──
    print("步骤 3/5: 多算法并行训练（分层 5 折交叉验证）...")
    models = get_models()
    print(f"  参赛算法: {list(models.keys())}")
    if SKIPPED:
        print(f"  跳过(依赖缺失): {SKIPPED}")
    results = run_parallel(X_train, y, models)
    print(f"  完成: {len(results)} 个算法评估完毕")

    # ── 4. 排行榜与最优模型 ──
    print("步骤 4/5: 生成排行榜...")
    leaderboard = make_leaderboard(results)
    best_name = select_best(results)
    print_leaderboard(leaderboard, best_name)

    if best_name is None:
        print("所有算法均训练失败，无法生成提交文件。")
        return

    # ── 5. 生成提交文件 ──
    print("步骤 5/5: 用冠军模型生成提交文件...")
    path = create_submission(best_name, X_train, y, X_test, passenger_ids)
    submission = pd.read_csv(path)
    print_submission_summary(path, submission)

    print("\n全流程完成。")


if __name__ == "__main__":
    main()
