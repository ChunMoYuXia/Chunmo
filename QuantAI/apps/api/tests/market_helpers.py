from decimal import Decimal as D

from portfolio_client.domain.market.calendar import TradingCalendar
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.market.models import DailyBar, Security


class MemoryMarket:
    def __init__(self, sessions, prices=None):
        self.sessions = tuple(sessions)
        prices = prices or [D("10")] * len(sessions)
        self.history = [
            DailyBar(
                "XSHE:000001",
                day,
                D("10"),
                max(D("10"), price),
                min(D("10"), price),
                price,
                1_000_000,
                D("10000000"),
                at_close(day),
                limit_up=D("20"),
                limit_down=D("1"),
            )
            for day, price in zip(sessions, prices, strict=True)
        ]

    def calendar(self, dataset_id):
        return TradingCalendar(self.sessions)

    def bars(self, dataset_id, security_ids, as_of, *, start=None, lookback=None):
        values = [
            bar
            for bar in self.history
            if bar.security_id in security_ids
            and bar.available_at <= as_of
            and (start is None or bar.trade_date >= start)
        ]
        return values[-lookback:] if lookback else values

    def securities(self, dataset_id, as_of):
        return [Security("XSHE:000001", "000001", "XSHE", "测试", "科技", "cs")]

    def actions(self, dataset_id, as_of):
        return []
