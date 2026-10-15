from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from portfolio_client.infrastructure.ai.compatible import ProviderSettings


class Settings(BaseModel):
    """Validated per-application configuration; secrets never enter API DTOs."""

    model_config = ConfigDict(frozen=True)

    database_url: str
    data_dir: Path
    session_token: SecretStr
    ai: ProviderSettings | None = None
    static_dir: Path | None = None
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:5173"])
    allowed_hosts: list[str] = Field(default_factory=lambda: ["127.0.0.1", "localhost"])
