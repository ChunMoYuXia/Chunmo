import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.portfolio.models import OptimizationMethod, WeightSolution


def _capped_proportions(scores: NDArray[np.float64], cap: float) -> NDArray[np.float64]:
    """Redistribute the remaining budget among uncapped assets proportionally."""
    weights = np.zeros(len(scores))
    free = np.ones(len(scores), dtype=bool)
    while np.any(free):
        allocation = (1 - weights.sum()) * scores[free] / scores[free].sum()
        breached = allocation > cap
        if not np.any(breached):
            weights[free] = allocation
            break
        indices = np.flatnonzero(free)[breached]
        weights[indices] = cap
        free[indices] = False
    return weights


def optimize_weights(
    returns: NDArray[np.float64], method: OptimizationMethod, max_weight: float
) -> WeightSolution:
    if returns.ndim != 2 or returns.shape[1] == 0 or not np.all(np.isfinite(returns)):
        raise DomainError("INVALID_RETURNS", "收益矩阵必须非空且全部为有限数")
    observations, assets = returns.shape
    if observations < 2:
        raise DomainError("INSUFFICIENT_HISTORY", "至少需要两个完整日收益观测")
    if not np.isfinite(max_weight) or max_weight <= 0 or max_weight > 1:
        raise DomainError("INFEASIBLE_CONSTRAINTS", "个股上限必须在(0,1]内")
    if assets * max_weight < 1 - 1e-10:
        raise DomainError("INFEASIBLE_CONSTRAINTS", "个股上限不足以分配全部权重")
    shrinkage = None
    if method == "equal":
        weights = np.full(assets, 1 / assets)
    else:
        covariance = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
        variances = np.diag(covariance)
        if not np.all(np.isfinite(covariance)) or np.any(variances <= 0):
            raise DomainError("ZERO_VOLATILITY", "零波动或数值异常的资产不能用于风险优化")
        if method == "inverse_volatility":
            weights = _capped_proportions(1 / np.sqrt(variances), max_weight)
        elif method == "minimum_variance":
            shrinkage = 0.1
            covariance = 0.9 * covariance + 0.1 * np.diag(variances)
            # Scaling prevents tiny daily variances from satisfying SLSQP's tolerance at the initial point.
            scaled = covariance / float(np.max(variances))
            result = minimize(
                lambda w: float(w @ scaled @ w),
                np.full(assets, 1 / assets),
                jac=lambda w: 2 * scaled @ w,
                method="SLSQP",
                bounds=[(0, max_weight)] * assets,
                constraints=[
                    {"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(assets)}
                ],
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if not result.success:
                raise DomainError(
                    "OPTIMIZATION_FAILED", "最小方差求解未收敛", {"reason": str(result.message)}
                )
            weights = np.asarray(result.x, dtype=np.float64)
        else:
            raise DomainError("INVALID_OPTIMIZER", "未知优化方法")
    if (
        not np.all(np.isfinite(weights))
        or abs(weights.sum() - 1) > 1e-8
        or np.any(weights < -1e-8)
        or np.any(weights > max_weight + 1e-8)
    ):
        raise DomainError("OPTIMIZATION_FAILED", "优化结果未通过权重和与上限校验")
    weights = np.clip(weights, 0, max_weight)
    weights /= weights.sum()
    return WeightSolution(tuple(float(value) for value in weights), method, observations, shrinkage)
