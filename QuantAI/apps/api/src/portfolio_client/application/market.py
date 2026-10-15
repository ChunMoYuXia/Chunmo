from dataclasses import dataclass, replace
from datetime import datetime

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.clock import as_utc, at_close
from portfolio_client.domain.market.models import DailyBar, DatasetSnapshot, KnowledgeMode, Security
from portfolio_client.domain.market.ports import DatasetRepositoryPort, MarketDataPort
from portfolio_client.domain.research.features import FeatureSnapshot, compute_features


@dataclass(frozen=True)
class MarketContext:
    snapshot: DatasetSnapshot
    as_of: datetime
    knowledge_mode: KnowledgeMode

    @property
    def visible(self) -> bool:
        return self.knowledge_mode == "public_pit" or self.snapshot.ingested_at <= self.as_of


class MarketService:
    def __init__(self, datasets: DatasetRepositoryPort, market: MarketDataPort) -> None:
        self.datasets = datasets
        self.market = market

    def context(
        self,
        dataset_id: str,
        as_of: datetime | None = None,
        knowledge_mode: KnowledgeMode = "public_pit",
    ) -> MarketContext:
        snapshot = self.datasets.get_dataset(dataset_id)
        cutoff = as_utc(as_of) if as_of is not None else at_close(snapshot.last_session)
        return MarketContext(snapshot, cutoff, knowledge_mode)

    def securities(self, context: MarketContext) -> list[Security]:
        return (
            self.market.securities(context.snapshot.dataset_id, context.as_of)
            if context.visible
            else []
        )

    def search_securities(
        self, context: MarketContext, query: str, limit: int = 20
    ) -> list[Security]:
        normalized = query.strip().casefold()
        ranked: list[tuple[int, str, Security]] = []
        for security in self.securities(context):
            if not normalized or normalized in {
                security.code.casefold(),
                security.security_id.casefold(),
            }:
                rank = 0
            elif security.code.casefold().startswith(normalized):
                rank = 1
            elif security.name.casefold() == normalized:
                rank = 2
            elif normalized in security.name.casefold():
                rank = 3
            elif security.initials.casefold().startswith(normalized):
                rank = 4
            elif security.industry is not None and normalized in security.industry.casefold():
                rank = 5
            else:
                continue
            ranked.append((rank, security.security_id, security))
        return [item[2] for item in sorted(ranked, key=lambda item: (item[0], item[1]))[:limit]]

    def bars(self, context: MarketContext, security_id: str, limit: int = 1000) -> list[DailyBar]:
        if not context.visible:
            return []
        if not any(item.security_id == security_id for item in self.securities(context)):
            raise DomainError("SECURITY_NOT_FOUND", "该时点没有对应标的资料")
        return self.market.bars(
            context.snapshot.dataset_id, [security_id], context.as_of, lookback=limit
        )

    def features(
        self, context: MarketContext, security_ids: list[str] | None = None
    ) -> list[FeatureSnapshot]:
        identities = self.securities(context)
        if security_ids is not None:
            selected = set(security_ids)
            unknown = selected - {item.security_id for item in identities}
            if unknown:
                raise DomainError(
                    "SECURITY_NOT_FOUND",
                    "候选池含该时点不可见的标的",
                    {"security_ids": sorted(unknown)},
                )
            identities = [item for item in identities if item.security_id in selected]
        if not identities:
            return []
        bars = self.market.bars(
            context.snapshot.dataset_id,
            [item.security_id for item in identities],
            context.as_of,
            lookback=21,
        )
        histories: dict[str, list[DailyBar]] = {item.security_id: [] for item in identities}
        for bar in bars:
            histories[bar.security_id].append(bar)
        calendar = self.market.calendar(context.snapshot.dataset_id)
        expected = [day for day in calendar.sessions if at_close(day) <= context.as_of]
        results: list[FeatureSnapshot] = []
        for security in identities:
            history = histories[security.security_id]
            computed = compute_features(history)
            observed = {bar.trade_date for bar in history}
            missing_prices = set(expected[-21:]) - observed
            missing_amounts = set(expected[-20:]) - observed
            warnings = list(computed.warnings)
            if missing_prices:
                warnings.append("历史窗口缺少交易日行情，相关收益特征不可用。")
            results.append(
                replace(
                    computed,
                    security_id=security.security_id,
                    security=security,
                    data_version=context.snapshot.version,
                    as_of=context.as_of,
                    momentum_20d=None if missing_prices else computed.momentum_20d,
                    volatility_20d=None if missing_prices else computed.volatility_20d,
                    average_amount_20d=None if missing_amounts else computed.average_amount_20d,
                    missing_sessions=tuple(sorted(missing_prices)),
                    warnings=tuple(warnings),
                    stale=bool(expected and computed.last_session != expected[-1]),
                )
            )
        return results
