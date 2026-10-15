from collections.abc import Callable
from dataclasses import dataclass

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.market.models import DatasetSnapshot
from portfolio_client.domain.market.ports import DatasetRepositoryPort, MarketDataPort
from portfolio_client.domain.research.backtest import run_backtest
from portfolio_client.domain.research.backtest_models import (
    BacktestRequest,
    BacktestResult,
    Progress,
)


@dataclass(frozen=True)
class ResearchRun:
    result: BacktestResult
    snapshot: DatasetSnapshot


class ResearchService:
    def __init__(self, datasets: DatasetRepositoryPort, market: MarketDataPort) -> None:
        self.datasets = datasets
        self.market = market

    def validate(self, request: BacktestRequest) -> DatasetSnapshot:
        snapshot = self.datasets.get_dataset(request.dataset_id)
        if snapshot.calendar_id != request.calendar_id:
            raise DomainError("CALENDAR_MISMATCH", "回测日历与固定数据版本不匹配")
        if request.knowledge_mode == "system_replay" and snapshot.ingested_at > at_close(
            request.start
        ):
            raise DomainError(
                "REPLAY_DATA_UNAVAILABLE",
                "系统回放起点尚未入库该数据版本，请使用公开历史模式或后移起点",
            )
        return snapshot

    def run(self, request: BacktestRequest, progress: Callable[[Progress], None]) -> ResearchRun:
        snapshot = self.validate(request)
        return ResearchRun(run_backtest(request, self.market, progress), snapshot)
