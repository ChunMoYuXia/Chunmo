from dataclasses import dataclass
from datetime import datetime

from portfolio_client.domain.screening.dsl import ScreeningSpec


@dataclass(frozen=True)
class SavedScreening:
    id: str
    name: str
    spec: ScreeningSpec
    created_at: datetime
