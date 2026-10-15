from bisect import bisect_right
from datetime import date

from portfolio_client.domain.market.calendar import TradingCalendar
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.market.models import DailyBar


class MarketWindow:
    """Indexes immutable bars, then checks disclosure time for every historical window."""

    def __init__(self, bars: list[DailyBar], calendar: TradingCalendar) -> None:
        self.calendar = calendar
        self.by_key = {(bar.security_id, bar.trade_date): bar for bar in bars}

    def bar(self, security_id: str, session: date) -> DailyBar | None:
        value = self.by_key.get((security_id, session))
        return value if value is not None and value.available_at <= at_close(session) else None

    def window(
        self, security_ids: list[str], session: date, length: int = 21
    ) -> list[DailyBar] | None:
        end = bisect_right(self.calendar.sessions, session)
        if end < length:
            return None
        selected: list[DailyBar] = []
        cutoff = at_close(session)
        for day in self.calendar.sessions[end - length : end]:
            for security_id in security_ids:
                bar = self.by_key.get((security_id, day))
                if bar is None or bar.available_at > cutoff:
                    return None
                selected.append(bar)
        return selected
