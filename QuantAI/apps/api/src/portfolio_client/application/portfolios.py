from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_DOWN, Decimal

from portfolio_client.application.market import MarketContext, MarketService
from portfolio_client.domain.accounting.costs import default_fee_schedules, schedule_for
from portfolio_client.domain.accounting.models import FeeSchedule
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.execution.rules import lot_size_for
from portfolio_client.domain.market.clock import CN_TIMEZONE, at_close
from portfolio_client.domain.market.models import KnowledgeMode
from portfolio_client.domain.portfolio.models import (
    OptimizationMethod,
    PortfolioSpec,
    SavedPortfolio,
    WeightSolution,
)
from portfolio_client.domain.portfolio.optimizers import optimize_weights
from portfolio_client.domain.portfolio.ports import PortfolioRepositoryPort
from portfolio_client.domain.portfolio.rebalancing import RebalancePlan, plan_rebalance
from portfolio_client.domain.research.returns import aligned_returns


@dataclass(frozen=True)
class OptimizationRun:
    target_weights: dict[str, Decimal]
    solution: WeightSolution
    context: MarketContext
    price_basis: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RebalanceRun:
    plan: RebalancePlan
    fee_schedule: FeeSchedule
    context: MarketContext


class PortfolioService:
    def __init__(self, market: MarketService, repository: PortfolioRepositoryPort) -> None:
        self.market = market
        self.repository = repository

    def save(self, spec: PortfolioSpec) -> SavedPortfolio:
        context = self.market.context(spec.dataset_id, spec.as_of, spec.knowledge_mode)
        self.market.features(context, sorted(set(spec.target_weights) | set(spec.holdings)))
        return self.repository.save(spec)

    def rebalance(
        self, portfolio_id: str, sellable: dict[str, int] | None, fees: FeeSchedule | None
    ) -> RebalanceRun:
        spec = self.repository.get(portfolio_id).spec
        context = self.market.context(spec.dataset_id, spec.as_of, spec.knowledge_mode)
        features = self.market.features(
            context, sorted(set(spec.target_weights) | set(spec.holdings))
        )
        prices = {
            item.security_id: item.close
            for item in features
            if item.close is not None and not item.stale
        }
        lots = {}
        for item in features:
            security = item.security
            if security is None:
                raise DomainError("SECURITY_NOT_FOUND", "标的资料缺失")
            lots[item.security_id] = lot_size_for(security)
        schedule = schedule_for(
            (fees,) if fees is not None else default_fee_schedules(),
            spec.as_of.astimezone(CN_TIMEZONE).date(),
        )
        plan = plan_rebalance(
            spec.target_weights, spec.holdings, prices, spec.cash, lots, schedule, sellable=sellable
        )
        return RebalanceRun(plan, schedule, context)

    def optimize(
        self,
        dataset_id: str,
        as_of: datetime,
        universe: list[str],
        method: OptimizationMethod,
        max_weight: Decimal,
        lookback: int,
        knowledge_mode: KnowledgeMode,
    ) -> OptimizationRun:
        context = self.market.context(dataset_id, as_of, knowledge_mode)
        self.market.features(context, universe)  # Validate time-aware identities first.
        calendar = self.market.market.calendar(dataset_id)
        sessions = [day for day in calendar.sessions if at_close(day) <= context.as_of][
            -lookback - 1 :
        ]
        if len(sessions) != lookback + 1:
            raise DomainError(
                "INSUFFICIENT_HISTORY",
                "指定窗口需要更多完整交易日",
                {"required_prices": lookback + 1, "available_prices": len(sessions)},
            )
        bars = (
            self.market.market.bars(dataset_id, universe, context.as_of, lookback=lookback + 1)
            if context.visible
            else []
        )
        matrix = aligned_returns(bars, universe, sessions)
        solution = optimize_weights(matrix.values, method, float(max_weight))
        # Floor to 12 places so rounding never creates leverage or violates a cap.
        weights = {
            security_id: min(max_weight, Decimal(str(weight)))
            .quantize(Decimal(".000000000001"), rounding=ROUND_DOWN)
            .normalize()
            for security_id, weight in zip(universe, solution.weights, strict=True)
        }
        return OptimizationRun(weights, solution, context, matrix.price_basis, matrix.warnings)
