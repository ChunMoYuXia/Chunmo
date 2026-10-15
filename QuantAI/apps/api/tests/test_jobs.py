import time

import pytest
from fastapi.testclient import TestClient

from portfolio_client.main import create_app


def spec(dataset):
    return {
        "dataset_id": dataset["dataset_id"],
        "calendar_id": dataset["calendar_id"],
        "security_ids": ["XSHG:600001"],
        "weights": {"XSHG:600001": "1"},
        "frequency": "daily",
        "start": dataset["first_session"],
        "end": dataset["last_session"],
        "capital": "100000",
        "seed": 42,
    }


def wait_job(client, auth, job_id):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        record = client.get(f"/v1/jobs/{job_id}", headers=auth).json()["data"]
        if record["state"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    raise AssertionError("research job did not finish within bounded test timeout")


def test_cancel_unknown_job_is_explicit(client, auth):
    response = client.post("/v1/jobs/does-not-exist/cancel", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "JOB_NOT_FOUND"


def test_job_submission_is_idempotent_and_result_survives_restart(
    client, auth, settings, market_dataset
):
    headers = {**auth, "Idempotency-Key": "research-1"}
    created = client.post("/v1/backtests", json=spec(market_dataset), headers=headers)
    assert created.status_code == 202
    job_id = created.json()["data"]["task_id"]
    assert (
        client.post("/v1/backtests", json=spec(market_dataset), headers=headers).json()["data"][
            "task_id"
        ]
        == job_id
    )
    assert (
        client.post(
            "/v1/backtests", json={**spec(market_dataset), "capital": "200000"}, headers=headers
        ).status_code
        == 409
    )
    completed = wait_job(client, auth, job_id)
    assert completed["state"] == "succeeded"
    assert completed["progress"]["completed"] == completed["progress"]["total"] == 30
    result = client.get(f"/v1/jobs/{job_id}/result", headers=auth)
    assert result.status_code == 200
    assert result.json()["data"]["snapshot"]["version"] == market_dataset["version"]
    assert result.json()["data"]["metrics"]["total_return"]["formula_version"] == "metrics-v1"
    with TestClient(create_app(settings)) as reopened:
        assert (
            reopened.get(f"/v1/jobs/{job_id}/result", headers=auth).json()["data"]
            == result.json()["data"]
        )
        cancelled_after_success = reopened.post(f"/v1/jobs/{job_id}/cancel", headers=auth).json()[
            "data"
        ]
        assert cancelled_after_success["state"] == "succeeded"


def test_worker_failure_has_actionable_code_and_trace(client, auth, market_dataset):
    body = {**spec(market_dataset), "end": "2023-01-10"}
    created = client.post("/v1/backtests", json=body, headers={**auth, "Idempotency-Key": "short"})
    assert created.status_code == 202
    job = wait_job(client, auth, created.json()["data"]["task_id"])
    assert job["state"] == "failed"
    assert job["error"]["code"] == "INSUFFICIENT_HISTORY"
    assert job["error"]["trace_id"]


def test_persisted_metrics_use_selected_price_benchmark_and_rate_parameters(
    client, auth, market_dataset
):
    created = client.post(
        "/v1/backtests",
        json={
            **spec(market_dataset),
            "benchmark_id": "XSHG:600001",
            "annualization": 252,
            "risk_free_annual": 0.03,
        },
        headers={**auth, "Idempotency-Key": "benchmark"},
    )
    assert created.status_code == 202
    job_id = created.json()["data"]["task_id"]
    assert wait_job(client, auth, job_id)["state"] == "succeeded"
    artifact = client.get(f"/v1/jobs/{job_id}/result", headers=auth).json()["data"]
    assert artifact["metrics"]["benchmark_return"]["value"] is not None
    assert artifact["metrics"]["benchmark_basis"] == "price"
    assert artifact["metrics"]["annualization"] == 252
    assert artifact["metrics"]["risk_free_annual"] == 0.03


@pytest.fixture
def job_system(tmp_path):
    from portfolio_client.application.jobs import JobService
    from portfolio_client.application.research import ResearchService
    from portfolio_client.infrastructure.db.database import Database
    from portfolio_client.infrastructure.db.jobs import SQLiteJobs
    from portfolio_client.infrastructure.db.market_queries import SQLiteMarketData
    from portfolio_client.infrastructure.db.repositories import SQLiteDatasetRepository
    from portfolio_client.infrastructure.imports.synthetic import generate_dataset

    database = Database(f"sqlite:///{tmp_path}/jobs.db")
    datasets = SQLiteDatasetRepository(database)
    dataset = datasets.commit_preview(generate_dataset(42, 3, 30))
    research = ResearchService(datasets, SQLiteMarketData(database))
    jobs = JobService(SQLiteJobs(database), research)
    yield jobs, research, dataset
    database.close()


def request_for(dataset):
    from portfolio_client.domain.research.backtest import BacktestRequest

    return BacktestRequest.model_validate(
        spec(
            {
                "dataset_id": dataset.dataset_id,
                "calendar_id": dataset.calendar_id,
                "first_session": dataset.first_session.isoformat(),
                "last_session": dataset.last_session.isoformat(),
            }
        )
    )


def test_cancelled_job_cannot_be_overwritten_by_a_late_success(job_system):
    jobs, research, dataset = job_system
    request = request_for(dataset)
    job = jobs.submit("backtest", request, "cancel-race")
    claimed = jobs.claim()
    assert claimed.id == job.id
    result = research.run(request, lambda _: None)
    assert jobs.cancel(job.id).state == "cancelled"
    assert jobs.complete(job.id, result).state == "cancelled"
    assert jobs.get(job.id).result_id is None


def test_restart_marks_running_interrupted_and_keeps_queued_work(job_system):
    from portfolio_client.infrastructure.jobs.runner import JobRunner

    jobs, research, dataset = job_system
    request = request_for(dataset)
    one = jobs.submit("backtest", request, "running")
    two = jobs.submit("backtest", request, "queued")
    assert jobs.claim().id == one.id
    jobs.recover()
    assert jobs.get(one.id).error.code == "JOB_INTERRUPTED"
    assert jobs.get(two.id).state == "queued"
    assert JobRunner(jobs, research).run_once()
    assert jobs.get(two.id).state == "succeeded"


def test_running_worker_stops_at_next_progress_boundary_on_cancel(job_system, monkeypatch):
    from threading import Event

    from portfolio_client.infrastructure.jobs.runner import JobRunner

    jobs, research, dataset = job_system
    job = jobs.submit("backtest", request_for(dataset), "cooperative-cancel")
    entered, release, finished = Event(), Event(), Event()
    observed = []
    original = research.run

    def gated_run(request, progress):
        entered.set()
        assert release.wait(timeout=5)

        def update(value):
            observed.append(value.completed)
            progress(value)

        try:
            return original(request, update)
        finally:
            finished.set()

    monkeypatch.setattr(research, "run", gated_run)
    runner = JobRunner(jobs, research)
    runner.start()
    try:
        assert entered.wait(timeout=5)
        assert jobs.cancel(job.id).state == "cancelled"
        release.set()
        assert finished.wait(timeout=5)
        assert observed == [0]
        assert jobs.get(job.id).result_id is None
    finally:
        release.set()
        runner.stop()
