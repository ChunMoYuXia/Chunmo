from dataclasses import dataclass
from decimal import Decimal

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import Security


@dataclass(frozen=True)
class ExecutionRules:
    lot_size: int = 100
    slippage_bps: Decimal = Decimal("8")
    participation_rate: Decimal = Decimal(".05")
    allow_odd_lot_sell: bool = False
    require_price_limits: bool = True

    def __post_init__(self) -> None:
        if (
            self.lot_size <= 0
            or not self.slippage_bps.is_finite()
            or not 0 <= self.slippage_bps <= 1000
            or not self.participation_rate.is_finite()
            or not 0 < self.participation_rate <= 1
        ):
            raise DomainError("INVALID_EXECUTION_RULES", "申报单位、滑点或参与率无效")


def lot_size_for(security: Security) -> int:
    if security.exchange not in {"XSHG", "XSHE"} or security.code.startswith(("688", "689", "920")):
        raise DomainError("UNSUPPORTED_MARKET_RULES", "M0执行仅支持沪深100股整数手规则")
    return 100
