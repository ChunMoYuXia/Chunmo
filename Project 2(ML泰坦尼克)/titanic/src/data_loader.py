"""数据加载：读取原始 CSV，返回训练集与测试集 DataFrame。

不做特征工程，只负责把磁盘上的 CSV 装进内存，并给出数据概览。
特征工程统一在 features.py 完成，保证职责单一。
"""
import pandas as pd

from .config import TEST_CSV, TRAIN_CSV


def load_train_test():
    """读取训练集与测试集 CSV。

    返回:
        (train_df, test_df): 两个 DataFrame。
        - train_df 含 12 列（含 Survived 标签），891 行。
        - test_df 含 11 列（无 Survived），418 行。

    说明:
        - 不在此处做任何清洗或填充，保持原始数据原貌，便于特征工程统一处理。
        - PassengerId 保留：提交文件需要它作为行标识。
    """
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, test_df


def print_overview(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """打印数据概览：形状、缺失值、生存率，供运行时快速诊断。"""
    print("=" * 50)
    print(f"训练集: {train_df.shape}  测试集: {test_df.shape}")
    print("-" * 50)
    # 只展示有缺失的列，避免输出冗长
    train_nulls = train_df.isnull().sum()
    test_nulls = test_df.isnull().sum()
    print("训练集缺失:")
    for col, n in train_nulls[train_nulls > 0].items():
        print(f"  {col}: {n} ({n / len(train_df):.1%})")
    print("测试集缺失:")
    for col, n in test_nulls[test_nulls > 0].items():
        print(f"  {col}: {n} ({n / len(test_df):.1%})")
    print("-" * 50)
    if "Survived" in train_df.columns:
        rate = train_df["Survived"].mean()
        print(f"训练集生存率: {rate:.2%}（轻微不均衡，采用分层交叉验证）")
    print("=" * 50)
