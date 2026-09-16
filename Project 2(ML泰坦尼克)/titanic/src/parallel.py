"""多进程并行训练：用 joblib 把各算法的交叉验证并行跑。

核心思路：
- 每个算法是一个独立任务（_train_one），内部用 cross_validate 做分层 5 折交叉验证；
- joblib.Parallel 把这些任务分发到多个进程并行执行（n_jobs=-1 用满 CPU 核）；
- 进程间无共享状态，天然线程安全，比 threading 更适合 CPU 密集的 sklearn 训练。

为什么用进程而非线程：
  sklearn 底层（如 SVM、RF）释放 GIL，多线程也能并行；但 XGBoost/LightGBM
  自带多线程会与外层线程池竞争。用进程级并行（loky 后端）更稳，避免线程爆炸。
  注意：每个模型内部 n_jobs 设为 1，把并行度让给外层 joblib，避免嵌套并行。
"""
from time import perf_counter

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import CV_FOLDS, N_JOBS, RANDOM_STATE


def _train_one(name, model, X, y, cv):
    """训练单个算法并返回交叉验证指标。

    参数:
        name: 算法名（用于结果标识）。
        model: sklearn Pipeline 或估计器。
        X, y: 特征与标签。
        cv: 交叉验证分割器。

    返回:
        dict: {name, accuracy, precision, recall, f1, roc_auc, fit_time, error}。
        error 非 None 时表示该算法训练失败，其他指标为 0。

    指标说明:
        - accuracy:   整体准确率（Kaggle 评分指标）。
        - precision:  预测生存中真生存的比例（误判代价高时关注）。
        - recall:     真生存中被找出来的比例（漏救代价高时关注）。
        - f1:         precision 与 recall 的调和平均，综合指标。
        - roc_auc:    概率排序能力，不依赖阈值。
    """
    t0 = perf_counter()
    try:
        # scoring 列出全部需要的指标，cross_validate 一次性算完
        scores = cross_validate(
            model, X, y, cv=cv, scoring=scoring(),
            return_train_score=False, n_jobs=1,  # 内部不并行，让外层 joblib 调度
        )
        elapsed = perf_counter() - t0
        return {
            "name": name,
            "accuracy": float(np.mean(scores["test_accuracy"])),
            "precision": float(np.mean(scores["test_precision"])),
            "recall": float(np.mean(scores["test_recall"])),
            "f1": float(np.mean(scores["test_f1"])),
            "roc_auc": float(np.mean(scores["test_roc_auc"])),
            "fit_time": elapsed,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        # 单算法失败不影响其他算法，记录错误后继续
        return {
            "name": name, "accuracy": 0, "precision": 0, "recall": 0,
            "f1": 0, "roc_auc": 0, "fit_time": perf_counter() - t0,
            "error": str(exc),
        }


def scoring():
    """返回 cross_validate 的 scoring 字典（名称 → sklearn 指标字符串）。"""
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }


def run_parallel(X, y, models: dict) -> list[dict]:
    """并行训练所有算法，返回结果列表。

    参数:
        X, y: 训练特征与标签。
        models: {算法名: 模型} 字典（来自 models.get_models()）。

    返回:
        list[dict]: 每个算法一个结果字典，按完成顺序返回（已排序见 evaluate.py）。

    并行调度:
        - delayed(_train_one)(name, model, X, y, cv) 为每个算法构造一个任务；
        - Parallel(n_jobs=N_JOBS) 用 loky 进程池并行执行这些任务；
        - CV 分割器在主进程构造后传入各子进程，保证每折划分一致。
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    # 用固定 cv 对象而非整数，保证各子进程的折划分完全一致（可复现）
    tasks = [delayed(_train_one)(name, model, X, y, cv) for name, model in models.items()]
    results = Parallel(n_jobs=N_JOBS, prefer="processes")(tasks)
    return results
