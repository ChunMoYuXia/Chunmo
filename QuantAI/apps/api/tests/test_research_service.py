from datetime import date
from decimal import Decimal as D

import pytest

from portfolio_client.domain.errors import DomainError
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.market_queries import SQLiteMarketData
from portfolio_client.infrastructure.db.repositories import SQLiteDatasetRepository


def test_research_checks_calendar_identity_and_system_replay_visibility(
    client, auth, settings, market_dataset
):
    from portfolio_client.application.research import ResearchService
    from portfolio_client.domain.research.backtest import BacktestRequest

    database = Database(settings.database_url)
    try:
        service = ResearchService(SQLiteDatasetRepository(database), SQLiteMarketData(database))
        request = BacktestRequest(
            dataset_id=market_dataset["dataset_id"],
            calendar_id=market_dataset["calendar_id"],
            security_ids=["XSHG:600001"],
            weights={"XSHG:600001": D("1")},
            frequency="daily",
            start=date.fromisoformat(market_dataset["first_session"]),
            end=date.fromisoformat(market_dataset["last_session"]),
        )
        run = service.run(request, lambda _: None)
        assert run.snapshot.version == market_dataset["version"]
        assert run.result.trades and run.result.final_account.cash >= 0
        with pytest.raises(DomainError) as error:
            service.run(request.model_copy(update={"calendar_id": "different"}), lambda _: None)
        assert error.value.code == "CALENDAR_MISMATCH"
        with pytest.raises(DomainError) as error:
            service.run(
                request.model_copy(update={"knowledge_mode": "system_replay"}), lambda _: None
            )
        assert error.value.code == "REPLAY_DATA_UNAVAILABLE"
    finally:
        database.close()
