from bisect import bisect_right
from dataclasses import dataclass
from datetime import date

from portfolio_client.domain.errors import DomainError


@dataclass(frozen=True)
class TradingCalendar:
    sessions: tuple[date, ...]

    def __post_init__(self) -> None:
        normalized = tuple(self.sessions)
        if not normalized or tuple(sorted(set(normalized))) != normalized:
            raise DomainError("INVALID_CALENDAR", "交易日历必须为非空且无重复的有序日期")
        object.__setattr__(self, "sessions", normalized)

    def next_session(self, day: date) -> date:
        index = bisect_right(self.sessions, day)
        if index == len(self.sessions):
            raise DomainError("CALENDAR_EXHAUSTED", "日历中没有下一交易日")
        return self.sessions[index]
