import numpy as np
import pytest

from portfolio_client.domain.errors import DomainError


def optimize(returns, method, cap=1.0):
    from portfolio_client.domain.portfolio.optimizers import optimize_weights

    return optimize_weights(np.array(returns, dtype=float), method, cap)


def test_impossible_cap_has_specific_error():
    with pytest.raises(DomainError) as error:
        optimize([[0.01, 0.02], [0.02, 0.01], [0.03, 0.02]], "equal", 0.4)
    assert error.value.code == "INFEASIBLE_CONSTRAINTS"


def test_minimum_variance_matches_independent_diagonal_solution():
    # Centered orthogonal columns have variances in the ratio 1:4.
    solution = optimize(
        [[0.01, 0.02], [-0.01, 0.02], [0.01, -0.02], [-0.01, -0.02]], "minimum_variance"
    )
    assert solution.weights == pytest.approx((0.8, 0.2), abs=1e-6)
    assert solution.covariance_shrinkage == 0.1


def test_inverse_volatility_respects_cap_and_redistributes_remaining_budget():
    solution = optimize(
        [[0.01, 0.02], [-0.01, 0.02], [0.01, -0.02], [-0.01, -0.02]], "inverse_volatility", 0.6
    )
    assert solution.weights == pytest.approx((0.6, 0.4))


def test_equal_weights_do_not_require_nonzero_volatility():
    assert optimize([[0, 0], [0, 0]], "equal").weights == (0.5, 0.5)


@pytest.mark.parametrize(
    "matrix,method,code",
    [
        ([[0, 0], [0, 0]], "inverse_volatility", "ZERO_VOLATILITY"),
        ([[0, 0], [0, 0]], "minimum_variance", "ZERO_VOLATILITY"),
        ([[0.01, float("nan")], [0.02, 0.01]], "minimum_variance", "INVALID_RETURNS"),
        ([[0.01, 0.02]], "inverse_volatility", "INSUFFICIENT_HISTORY"),
    ],
)
def test_invalid_risk_inputs_have_actionable_errors(matrix, method, code):
    with pytest.raises(DomainError) as error:
        optimize(matrix, method)
    assert error.value.code == code
