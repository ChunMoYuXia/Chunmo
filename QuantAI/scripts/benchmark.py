"""Measure the actual local application pipeline on a reproducible synthetic workload."""

import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy
from fastapi.testclient import TestClient
from portfolio_client.config import Settings
from portfolio_client.main import create_app


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / ".cache/benchmark" / uuid4().hex
    directory.mkdir(parents=True)
    settings = Settings(
        database_url=f"sqlite:///{directory}/portfolio.db",
        data_dir=directory,
        session_token="benchmark-local-only",
        allowed_hosts=["testserver"],
    )
    auth = {"Authorization": "Bearer benchmark-local-only"}
    result: dict[str, object] = {
        "at": datetime.now(UTC).isoformat(),
        "method": "FastAPI TestClient, real SQLite, one worker, single cold run",
        "machine": {
            "os": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "numpy": numpy.__version__,
        },
        "workload": {
            "securities": 200,
            "sessions": 756,
            "seed": 42,
            "calendar": "synthetic weekdays only",
            "frequency": "monthly",
        },
    }
    with TestClient(create_app(settings)) as client:
        start = time.perf_counter()
        response = client.post(
            "/v1/datasets/synthetic",
            headers=auth,
            json={"securities": 200, "sessions": 756, "seed": 42},
        )
        response.raise_for_status()
        dataset = response.json()["data"]
        result["generation_seconds"] = time.perf_counter() - start
        query = {
            "dataset_id": dataset["dataset_id"],
            "as_of": dataset["last_session"] + "T15:00:00+08:00",
        }
        start = time.perf_counter()
        snapshot = client.get("/v1/market/snapshot", headers=auth, params=query)
        snapshot.raise_for_status()
        result["snapshot_seconds"] = time.perf_counter() - start
        universe = [row["security_id"] for row in snapshot.json()["data"]]
        start = time.perf_counter()
        screening = client.post(
            "/v1/screening",
            headers=auth,
            json={
                "version": 1,
                **query,
                "universe": universe,
                "filters": [{"field": "momentum_20d", "operator": "gt", "value": "0"}],
                "limit": 20,
            },
        )
        screening.raise_for_status()
        result["screening_seconds"] = time.perf_counter() - start
        payload = {
            "dataset_id": dataset["dataset_id"],
            "calendar_id": dataset["calendar_id"],
            "security_ids": universe,
            "weights": {id: "0.005" for id in universe},
            "start": dataset["first_session"],
            "end": dataset["last_session"],
            "capital": "1000000",
            "frequency": "monthly",
            "seed": 42,
        }
        start = time.perf_counter()
        submitted = client.post(
            "/v1/backtests",
            headers={**auth, "Idempotency-Key": "benchmark"},
            json=payload,
        )
        submitted.raise_for_status()
        job_id = submitted.json()["data"]["task_id"]
        last = -1
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            job = client.get("/v1/jobs/" + job_id, headers=auth).json()["data"]
            progress = job["progress"]
            percent = progress["completed"] * 100 // max(1, progress["total"])
            if percent // 10 != last:
                print(
                    f"Backtest progress {percent}% ({progress['completed']}/{progress['total']})",
                    flush=True,
                )
                last = percent // 10
            if job["state"] != "running" and job["state"] != "queued":
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Benchmark exceeded 900 seconds")
        result["backtest_seconds"] = time.perf_counter() - start
        if job["state"] != "succeeded":
            raise RuntimeError(f"Benchmark failed: {job['error']['code']}")
        artifact = client.get("/v1/jobs/" + job_id + "/result", headers=auth).json()[
            "data"
        ]
        result.update(
            {
                "row_count": dataset["row_count"],
                "data_hash": dataset["content_hash"],
                "nav_points": len(artifact["result"]["nav"]),
                "trades": len(artifact["result"]["trades"]),
                "costs_cny": artifact["result"]["costs"],
                "engine_version": artifact["result"]["evidence"]["engine_version"],
            }
        )
    destination = root / ".cache/benchmark/result.json"
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
