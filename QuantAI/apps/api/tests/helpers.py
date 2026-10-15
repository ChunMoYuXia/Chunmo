from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from portfolio_client.domain.accounting.models import FeeSchedule
from portfolio_client.domain.market.models import DailyBar


def schedule() -> FeeSchedule:
    return FeeSchedule(
        "cn-demo",
        date(2023, 8, 28),
        None,
        Decimal("2.5"),
        Decimal("5"),
        Decimal(".1"),
        Decimal("5"),
    )


def bars_for_prices(prices: list[Decimal]) -> list[DailyBar]:
    from portfolio_client.domain.market.models import DailyBar

    return [
        DailyBar(
            security_id="XSHE:000001",
            trade_date=date(2026, 1, 1) + timedelta(days=index),
            open=price,
            high=price,
            low=price,
            close=price,
            volume=1000,
            amount=Decimal("100000"),
            available_at=datetime(2026, 1, 1, 7, tzinfo=UTC) + timedelta(days=index),
        )
        for index, price in enumerate(prices)
    ]
