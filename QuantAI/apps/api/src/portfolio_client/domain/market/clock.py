from datetime import UTC, date, datetime, time, timedelta, timezone

from portfolio_client.domain.errors import DomainError

CN_TIMEZONE = timezone(timedelta(hours=8))


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DomainError("INVALID_AS_OF", "历史时点必须包含时区")
    return value.astimezone(UTC)


def at_close(session: date) -> datetime:
    return datetime.combine(session, time(15), CN_TIMEZONE).astimezone(UTC)
