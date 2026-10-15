from dataclasses import dataclass
from typing import Literal

import numpy as np

from portfolio_client.application.market import MarketContext, MarketService
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.portfolio.ports import PortfolioRepositoryPort
from portfolio_client.domain.research.exposures import (
    ExposureResult,
    exposures,
    exposures_from_values,
)
from portfolio_client.domain.research.metric_models import Metric, TailRisk
from portfolio_client.domain.research.metrics import historical_tail_risk, metric
from portfolio_client.domain.research.returns import aligned_returns


@dataclass(frozen=True)
class PortfolioRisk:
    exposure: ExposureResult
    tail_risk: TailRisk
    annualized_volatility: Metric
    source: str
    price_basis: str | None
    warnings: tuple[str, ...]
    context: MarketContext


class RiskService:
    def __init__(self, market: MarketService, portfolios: PortfolioRepositoryPort) -> None:
        self.market = market
        self.portfolios = portfolios

    def portfolio(
        self,
        portfolio_id: str,
        mode: Literal["holdings", "target"],
        confidence: float,
        lookback: int,
    ) -> PortfolioRisk:
        spec = self.portfolios.get(portfolio_id).spec
        context = self.market.context(spec.dataset_id, spec.as_of, spec.knowledge_mode)
        features = self.market.features(
            context, sorted(set(spec.holdings) | set(spec.target_weights))
        )
        prices = {
            item.security_id: item.close
            for item in features
            if item.close is not None and not item.stale
        }
        securities = [item.security for item in features if item.security is not None]
        actual = exposures(spec.holdings, prices, securities, cash=spec.cash)
        source = (
            "fixed_current_weights_historical_returns"
            if mode == "holdings"
            else "target_weight_historical_estimate"
        )
        exposure = (
            actual
            if mode == "holdings"
            else exposures_from_values(
                {key: actual.total_value * weight for key, weight in spec.target_weights.items()},
                securities,
                cash=actual.total_value * (1 - sum(spec.target_weights.values())),
            )
        )
        keys = sorted(exposure.asset_weights)
        if not keys:
            tail = historical_tail_risk(
                np.array([], dtype=np.float64),
                confidence,
                source=source,
                as_of=context.as_of,
                empty_reason="all_cash_no_market_returns",
            )
            return PortfolioRisk(
                exposure,
                tail,
                metric(
                    None, "annualized_ratio", source, context.as_of, "all_cash_no_market_returns"
                ),
                source,
                None,
                exposure.warnings,
                context,
            )
        calendar = self.market.market.calendar(spec.dataset_id)
        sessions = [day for day in calendar.sessions if at_close(day) <= context.as_of][
            -lookback - 1 :
        ]
        if len(sessions) != lookback + 1:
            raise DomainError("INSUFFICIENT_HISTORY", "指定风险窗口缺少完整交易日")
        bars = self.market.market.bars(spec.dataset_id, keys, context.as_of, lookback=lookback + 1)
        matrix = aligned_returns(bars, keys, sessions)
        returns = matrix.values @ np.array([exposure.asset_weights[key] for key in keys])
        tail = historical_tail_risk(returns, confidence, source=source, as_of=context.as_of)
        volatility = metric(
            float(np.std(returns, ddof=1) * np.sqrt(244)), "annualized_ratio", source, context.as_of
        )
        return PortfolioRisk(
            exposure,
            tail,
            volatility,
            source,
            matrix.price_basis,
            (
                *exposure.warnings,
                *matrix.warnings,
                "按当前/目标固定权重估计历史风险，非实际组合收益。",
            ),
            context,
        )
