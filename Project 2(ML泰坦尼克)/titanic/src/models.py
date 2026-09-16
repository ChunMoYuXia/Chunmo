"""多算法定义：逻辑回归 / SVM / KNN / 决策树 / 随机森林 / GBDT / XGBoost / LightGBM。

每个算法包在 sklearn Pipeline 里：
- 距离/梯度类模型（LR/SVM/KNN）前面接 StandardScaler，消除量纲差异；
- 树模型对单调变换不敏感，无需缩放（但保留 Pipeline 统一接口）。
统一 Pipeline 的好处：交叉验证时 scaler 只在训练折上拟合，杜绝泄露。

可选依赖（xgboost/lightgbm）缺失时自动跳过对应模型，保证至少跑通 sklearn 内置算法。
"""
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

# 记录被跳过的可选模型，供运行时提示
SKIPPED: list[str] = []


def _scaled(estimator):
    """构造「StandardScaler + 模型」Pipeline，给对尺度敏感的模型用。"""
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def get_models() -> dict:
    """返回 {算法名: Pipeline} 字典，包含全部可用算法。

    算法清单（8 个，覆盖线性/距离/树/提升四大流派）：
      1. 逻辑回归   LR        — 线性基准，可解释
      2. 支持向量机  SVM       — 最大化间隔，小样本强
      3. K 近邻      KNN       — 距离投票，简单稳健
      4. 决策树      DT        — 规则可解释，非线性
      5. 随机森林    RF        — 装袋降方差，抗过拟合
      6. 梯度提升树  GBDT      — 串行提升，sklearn 原生
      7. XGBoost     XGB       — 工业级提升树（可选）
      8. LightGBM    LGB       — 微软提升树，叶子优先生长（可选）

    降级逻辑：
      xgboost / lightgbm 是第三方可选依赖，缺失时跳过并记入 SKIPPED，
      剩余 6 个 sklearn 内置算法仍能完整跑完。
    """
    models = {
        # ── 线性流派：对尺度敏感，套 StandardScaler ──
        "逻辑回归": _scaled(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        # ── 距离流派：KNN 依赖欧氏距离，必须缩放 ──
        "KNN": _scaled(KNeighborsClassifier(n_neighbors=7)),
        # SVM：RBF 核对尺度极敏感；roc_auc 评分用 decision_function 即可，
        # 无需 probability=True（sklearn 1.9 起已弃用，且会拖慢训练）
        "SVM": _scaled(SVC(random_state=RANDOM_STATE)),

        # ── 树流派：对单调变换不敏感，无需缩放 ──
        "决策树": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "随机森林": RandomForestClassifier(
            n_estimators=200, max_depth=7, random_state=RANDOM_STATE, n_jobs=1,
        ),
        "GBDT": GradientBoostingClassifier(random_state=RANDOM_STATE),
        # AdaBoost：弱学习器提升，与 GBDT 互补
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
            n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE,
        ),
    }

    # ── 可选：XGBoost ──
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE,
            n_jobs=1,
        )
    except Exception as exc:  # noqa: BLE001
        SKIPPED.append(f"XGBoost({exc})")

    # ── 可选：LightGBM ──
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=RANDOM_STATE, n_jobs=1,
        )
    except Exception as exc:  # noqa: BLE001
        SKIPPED.append(f"LightGBM({exc})")

    return models
