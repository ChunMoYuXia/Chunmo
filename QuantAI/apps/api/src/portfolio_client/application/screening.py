from dataclasses import dataclass

from portfolio_client.application.market import MarketContext, MarketService
from portfolio_client.domain.screening.dsl import ScreeningSpec
from portfolio_client.domain.screening.engine import ScreeningResult, screen
from portfolio_client.domain.screening.models import SavedScreening
from portfolio_client.domain.screening.ports import ScreeningPresetPort


@dataclass(frozen=True)
class ScreeningRun:
    result: ScreeningResult
    context: MarketContext


class ScreeningService:
    def __init__(self, market: MarketService, presets: ScreeningPresetPort) -> None:
        self.market = market
        self.presets = presets

    def run(self, spec: ScreeningSpec) -> ScreeningRun:
        context = self.market.context(spec.dataset_id, spec.as_of, spec.knowledge_mode)
        snapshot = self.market.features(context, list(spec.universe))
        return ScreeningRun(screen(spec, snapshot), context)

    def save(self, name: str, spec: ScreeningSpec) -> SavedScreening:
        self.market.context(spec.dataset_id, spec.as_of, spec.knowledge_mode)
        return self.presets.save(name, spec)

    def get(self, preset_id: str) -> SavedScreening:
        return self.presets.get(preset_id)

    def list(self) -> list[SavedScreening]:
        return self.presets.list()
