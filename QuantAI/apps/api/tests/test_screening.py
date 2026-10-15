from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal

from portfolio_client.domain.market.models import Security
from portfolio_client.domain.research.features import FeatureSnapshot, compute_features
from tests.helpers import bars_for_prices

AS_OF = datetime(2026, 2, 10, 7, tzinfo=UTC)
IDS = ["XSHE:000001", "XSHE:000002", "XSHE:000003"]


def candidate(security_id, price, momentum=None):
    security = Security(security_id, security_id.split(":")[1], "XSHE", security_id, "科技", "kj")
    return FeatureSnapshot(
        security_id=security_id,
        close=Decimal(price) if price is not None else None,
        momentum_20d=momentum,
        volatility_20d=0.1,
        average_amount_20d=Decimal("100000"),
        sample_count=21,
        last_session=date(2026, 2, 10),
        as_of=AS_OF,
        security=security,
    )


def specification(filters=(), universe=IDS, sort_by="close", limit=20):
    from portfolio_client.domain.screening.dsl import ScreeningSpec

    return ScreeningSpec(
        version=1,
        dataset_id="ds_test",
        as_of=AS_OF,
        universe=universe,
        filters=filters,
        sort_by=sort_by,
        descending=True,
        limit=limit,
    )


def test_funnel_counts_missing_and_below_threshold_separately() -> None:
    from portfolio_client.domain.screening.engine import screen

    snapshot = [candidate(IDS[0], "8"), candidate(IDS[1], None), candidate(IDS[2], "12")]
    result = screen(specification([{"field": "close", "operator": "gte", "value": "10"}]), snapshot)
    assert result.steps[0].input_count == 3
    assert result.steps[0].remaining_count == 1
    assert result.steps[0].exclusions == {"below_threshold": 1, "missing_value": 1}
    assert result.rows[0].security_id == IDS[2]


def test_missing_sort_values_are_last_and_top_n_is_stable() -> None:
    from portfolio_client.domain.screening.engine import screen

    snapshot = [
        candidate(IDS[0], "10", 0.05),
        candidate(IDS[1], "10"),
        candidate(IDS[2], "10", 0.10),
    ]
    result = screen(specification(sort_by="momentum_20d"), snapshot)
    assert [row.security_id for row in result.rows] == [IDS[2], IDS[0], IDS[1]]
    assert screen(specification(limit=1), snapshot).rows[0].security_id == IDS[0]


def test_twenty_percent_boundary_includes_exact_twenty_percent_price_return() -> None:
    from portfolio_client.domain.screening.engine import screen

    computed = compute_features(bars_for_prices([Decimal(100 + i) for i in range(21)]))
    enriched = replace(computed, security=candidate(IDS[0], "120").security)
    result = screen(
        specification(
            [{"field": "momentum_20d", "operator": "gte", "value": ".2"}], universe=[IDS[0]]
        ),
        [enriched],
    )
    assert len(result.rows) == 1


def test_empty_pool_is_an_empty_result_not_the_whole_market() -> None:
    from portfolio_client.domain.screening.engine import screen

    assert screen(specification(universe=[]), [candidate(IDS[0], "10")]).rows == ()
