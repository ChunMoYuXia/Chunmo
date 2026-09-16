"""提交文件生成：用最优模型在全量训练集上重训 → 预测测试集 → 写 submission.csv。

输出格式与 gender_submission.csv 完全一致：两列 PassengerId, Survived。
可直接提交 Kaggle 泰坦尼克竞赛。
"""
import pandas as pd

from .config import SUBMISSION_CSV
from .models import get_models


def create_submission(best_name: str, X_train, y, X_test, passenger_ids) -> str:
    """用冠军模型生成提交文件并返回路径。

    参数:
        best_name: 冠军算法名（来自 evaluate.select_best）。
        X_train, y: 训练集特征与标签。
        X_test: 测试集特征矩阵（列已与 X_train 对齐）。
        passenger_ids: 测试集 PassengerId Series，作为提交文件的行标识。

    返回:
        submission.csv 的绝对路径。

    流程:
        1. 从 get_models() 取冠军模型的新实例（避免用交叉验证时已 fit 的旧对象）。
        2. 在全量训练集上 fit（用最多数据训练，泛化更好）。
        3. predict 测试集，输出 0/1 标签（非概率，Kaggle 要求类别标签）。
        4. 拼成 DataFrame 写 CSV，index=False。
    """
    model = get_models()[best_name]
    model.fit(X_train, y)
    preds = model.predict(X_test)
    # 确保是整数 0/1，部分模型（如 Pipeline）可能返回 numpy 数组
    preds = preds.astype(int)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids.values,
        "Survived": preds,
    })
    submission.to_csv(SUBMISSION_CSV, index=False)
    return str(SUBMISSION_CSV)


def print_submission_summary(path: str, submission: pd.DataFrame) -> None:
    """打印提交文件概览。"""
    print("\n" + "=" * 50)
    print(f"提交文件已生成: {path}")
    print(f"共 {len(submission)} 条预测")
    print("生存/遇难分布:")
    dist = submission["Survived"].value_counts().sort_index()
    for label, n in dist.items():
        tag = "生存" if label == 1 else "遇难"
        print(f"  {tag}: {n} ({n / len(submission):.1%})")
    print("=" * 50)
