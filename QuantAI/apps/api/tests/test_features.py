from decimal import Decimal

import pytest

from tests.helpers import bars_for_prices


def test_short_history_never_becomes_zero_risk_or_zero_momentum() -> None:
    from portfolio_client.domain.research.features import compute_features

    assert compute_features([]).momentum_20d is None
    assert compute_features(bars_for_prices([Decimal("100")])).volatility_20d is None


def test_twenty_session_momentum_uses_twenty_one_prices() -> None:
    from portfolio_client.domain.research.features import compute_features

    bars = bars_for_prices([Decimal(100 + index) for index in range(21)])
    result = compute_features(list(reversed(bars)))
    assert result.momentum_20d == pytest.approx(0.2)
    assert result.close == Decimal("120")
    assert result.average_amount_20d == Decimal("100000")


def test_sample_volatility_matches_alternating_returns_reference() -> None:
    from portfolio_client.domain.research.features import compute_features

    prices = [Decimal("100")]
    for index in range(20):
        prices.append(prices[-1] * (Decimal("1.01") if index % 2 == 0 else Decimal(".99")))
    result = compute_features(bars_for_prices(prices))
    # Twenty alternating +/-1% returns: sample variance uses 19, annualization uses 244.
    assert result.volatility_20d == pytest.approx(0.16026294183720638, rel=1e-12)
