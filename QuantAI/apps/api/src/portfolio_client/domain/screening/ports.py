from typing import Protocol

from portfolio_client.domain.screening.dsl import ScreeningSpec
from portfolio_client.domain.screening.models import SavedScreening


class ScreeningPresetPort(Protocol):
    def save(self, name: str, spec: ScreeningSpec) -> SavedScreening: ...
    def get(self, preset_id: str) -> SavedScreening: ...
    def list(self) -> list[SavedScreening]: ...
