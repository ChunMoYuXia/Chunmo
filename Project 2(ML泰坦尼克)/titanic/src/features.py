"""特征工程：原始数据 → 特征矩阵 X 与标签 y。

泰坦尼克特征工程的核心思路（业界常用且有效）：
1. Title 提取：从 Name 的称谓（Mr/Mrs/Miss/Master）推断身份与年龄段，
   再用 Title 分组的中位数填充 Age 缺失，比全局中位数更准。
2. 家庭规模：SibSp + Parch + 1，并衍生 is_alone（独自登船者生存率更高）。
3. 票价分箱与人均票价：Fare 按分位数分箱平滑噪声，家庭/团体共票时算人均。
4. 类别编码：Sex / Embarked / Title 用独热编码，避免引入虚假大小关系。
5. 丢弃高缺失与无信息列：Cabin（77% 缺失）、Name（已提取 Title）、Ticket、PassengerId。

防泄露：所有填充统计量（中位数、众数）只从训练集计算，再同时变换训练/测试集，
避免测试集信息反向污染训练集统计量。
"""
import numpy as np
import pandas as pd
import re

from .config import RANDOM_STATE


# ── 缺失值填充所需的统计量（从训练集算，变换两集）─────────────
# 这些属性在 fit_transform 阶段赋值，transform 阶段复用
class FeatureBuilder:
    """特征工程器：fit（从训练集学统计量）→ transform（应用到任意集）。

    采用 fit/transform 两阶段设计，与 sklearn Transformer 协议一致：
    - fit 只在训练集上调用，学习 Age 中位数、Fare 中位数、Embarked 众数；
    - transform 对训练集和测试集用同一套统计量变换，杜绝数据泄露。
    """

    def __init__(self):
        self.age_by_title: dict = {}      # {Title: Age 中位数}
        self.fare_median: float = 0.0
        self.embarked_mode: str = "S"
        self.feature_columns: list = []   # 最终特征列顺序

    # ── 第一步：从 Name 提取称谓 Title ───────────────────
    @staticmethod
    def _extract_title(name: str) -> str:
        """从姓名字符串提取称谓，如 'Braund, Mr. Owen Harris' → 'Mr'。

        把罕见称谓归并为常见四类，减少稀疏类别：
        - Mlle/Ms → Miss（法语小姐/现代缩写，等同 Miss）
        - Mme → Mrs（法语夫人）
        - 其余罕见称谓（Dr/Rev/Col/Major/Lady/Sir/Capt/Don/Jonkheer/Countess...）→ Rare
        """
        # 称谓夹在逗号与点号之间：", Title." 的正则匹配
        match = re.search(r",\s*([^.]*)\.", name)
        if not match:
            return "Rare"
        title = match.group(1).strip()
        # 归并同义称谓
        if title in ("Mlle", "Ms"):
            return "Miss"
        if title == "Mme":
            return "Mrs"
        if title in ("Mr", "Mrs", "Miss", "Master"):
            return title
        return "Rare"

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        """从训练集学习填充统计量。

        参数:
            df: 原始训练集 DataFrame（含 Name/Age/Embarked/Fare 等原始列）。
        返回:
            self，便于链式调用 fit().transform()。
        """
        # 先临时提取 Title，用于按组算 Age 中位数
        tmp_title = df["Name"].apply(self._extract_title)
        self.age_by_title = (
            df.assign(Title=tmp_title)
            .groupby("Title")["Age"].median().to_dict()
        )
        # Fare 用全局中位数（测试集只有 1 个缺失，无需分组）
        self.fare_median = df["Fare"].median()
        # Embarked 众数（缺失仅 2 条，众数最稳）
        self.embarked_mode = df["Embarked"].mode()[0]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """把原始 DataFrame 变换为特征 DataFrame。

        参数:
            df: 原始训练集或测试集 DataFrame。
        返回:
            仅含数值特征列的 DataFrame（已填充缺失、已编码类别）。
        """
        df = df.copy()

        # 1) Title 提取
        df["Title"] = df["Name"].apply(self._extract_title)

        # 2) Age 缺失填充：用对应 Title 的中位数，比全局中位数更贴近个体
        df["Age"] = df.apply(
            lambda r: r["Age"] if pd.notna(r["Age"])
            else self.age_by_title.get(r["Title"], 30),
            axis=1,
        )

        # 3) Embarked 缺失填充
        df["Embarked"] = df["Embarked"].fillna(self.embarked_mode)

        # 4) Fare 缺失填充（测试集 1 条）
        df["Fare"] = df["Fare"].fillna(self.fare_median)

        # 5) 家庭规模与是否独自登船
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

        # 6) 票价分箱：按分位数切成 4 档，平滑长尾分布（少数高价票）
        df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False, duplicates="drop")

        # 7) 年龄分箱：等宽切成 5 段，给树模型额外的分段信号
        df["AgeBin"] = pd.cut(df["Age"], bins=5, labels=False)

        # 8) 类别独热编码：Sex / Embarked / Title
        # drop_first=False 保留全部类别，树模型不受影响，线性模型用正则抑制多重共线性
        cat_cols = ["Sex", "Embarked", "Title"]
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

        # 9) 丢弃无信息列
        # Cabin 77% 缺失 → 弃；Name/Ticket 信息已提取或太杂 → 弃；
        # PassengerId 仅作行标识，非特征 → 弃；Survived 是标签，由调用方单独取
        drop_cols = ["Cabin", "Name", "Ticket", "PassengerId"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # 统一为 float，保证所有模型都能吃（包括要求 float 的 sklearn 接口）
        df = df.astype(float)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """便捷方法：fit 后立即 transform 训练集自身。"""
        return self.fit(df).transform(df)


def build_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """一键完成特征工程：训练集学统计量 → 变换两集 → 返回 (X_train, y, X_test)。

    参数:
        train_df: 原始训练集（含 Survived）。
        test_df: 原始测试集（无 Survived）。

    返回:
        X_train: 训练集特征矩阵 DataFrame。
        y: 训练集标签 Series（0=遇难, 1=生存）。
        X_test: 测试集特征矩阵 DataFrame（列与 X_train 完全对齐）。
        passenger_ids: 测试集 PassengerId，生成提交文件时用。

    列对齐说明:
        训练集与测试集的 Title/Embarked 类别可能不同（如测试集多/少某 Title），
        get_dummies 后列数会不一致。用 reindex 对齐到训练集的列集合，
        缺失的列补 0，保证模型 predict 时特征顺序与训练完全一致。
    """
    builder = FeatureBuilder()
    y = train_df["Survived"].copy()
    X_train = builder.fit_transform(train_df.drop(columns=["Survived"]))
    X_test = builder.transform(test_df)

    # 列对齐：测试集补齐训练集没有出现过的列（补 0），多余列丢弃
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0.0
    X_test = X_test[X_train.columns]

    return X_train, y, X_test, test_df["PassengerId"].copy()
