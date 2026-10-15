from datetime import date
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from portfolio_client.domain.accounting.models import FeeBreakdown, FeeSchedule, Side
from portfolio_client.domain.errors import DomainError

BPS_DIVISOR = Decimal("10000")
CENT = Decimal(".01")


def money(value: Decimal) -> Decimal:
    try:
        if not value.is_finite():
            raise InvalidOperation
        return value.quantize(CENT, rounding=ROUND_HALF_UP)
    except InvalidOperation as error:
        raise DomainError("INVALID_MONEY_PRECISION", "金额超出支持的分位精度范围") from error


def calculate_fees(notional: Decimal, side: Side, schedule: FeeSchedule) -> FeeBreakdown:
    if not notional.is_finite() or notional <= 0:
        raise DomainError("INVALID_NOTIONAL", "成交金额必须为正有限数")
    if not schedule.commission_includes_regulatory:
        raise DomainError("UNSUPPORTED_FEE_COMPONENTS", "M0需使用已包含规费的佣金配置")
    commission = max(schedule.minimum_commission, notional * schedule.commission_bps / BPS_DIVISOR)
    transfer = notional * schedule.transfer_bps / BPS_DIVISOR
    stamp = notional * schedule.stamp_bps / BPS_DIVISOR if side is Side.SELL else Decimal("0")
    return FeeBreakdown(money(commission), money(transfer), money(stamp), schedule.version)


def default_fee_schedules() -> tuple[FeeSchedule, ...]:
    """Illustrative broker commission; statutory transfer/stamp periods are explicit.

    See docs/engineering-supplement.md for the official sources. Earlier periods require a user schedule.
    """
    return (
        FeeSchedule(
            "cn-stock-20220429",
            date(2022, 4, 29),
            date(2023, 8, 27),
            Decimal("2.5"),
            Decimal("5"),
            Decimal(".1"),
            Decimal("10"),
        ),
        FeeSchedule(
            "cn-stock-20230828",
            date(2023, 8, 28),
            None,
            Decimal("2.5"),
            Decimal("5"),
            Decimal(".1"),
            Decimal("5"),
        ),
    )


def schedule_for(schedules: tuple[FeeSchedule, ...], session: date) -> FeeSchedule:
    matching = [
        schedule
        for schedule in schedules
        if schedule.valid_from <= session
        and (schedule.valid_to is None or session <= schedule.valid_to)
    ]
    if len(matching) != 1:
        raise DomainError(
            "FEE_PERIOD_UNDEFINED", "该交易日缺少唯一有效费率", {"session": session.isoformat()}
        )
    return matching[0]
