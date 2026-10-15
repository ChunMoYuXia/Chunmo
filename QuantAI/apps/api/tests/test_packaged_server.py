import json
import os
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import httpx
import pytest

local_get = partial(httpx.get, trust_env=False)


@dataclass(frozen=True)
class RunningService:
    url: str
    environment: dict[str, str]
    data_dir: Path
    process: subprocess.Popen[str]
    log: Path


@pytest.fixture
def sidecar(tmp_path):
    static = tmp_path / "web"
    static.mkdir()
    (static / "index.html").write_text("<html><body>research-shell</body></html>", encoding="utf-8")
    (static / "asset.js").write_text('console.log("local")', encoding="utf-8")
    (tmp_path / "private.txt").write_text("do-not-serve", encoding="utf-8")
    data = tmp_path / "data"
    env = {**os.environ, "PORTFOLIO_SESSION_TOKEN": "packaged-session-test-only", "PYTHONUTF8": "1"}
    log = tmp_path / "sidecar.log"
    with log.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "portfolio_client",
                "--sidecar",
                "--parent-stdin",
                "--port",
                "0",
                "--data-dir",
                str(data),
                "--static-dir",
                str(static),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=stderr,
            text=True,
            encoding="utf-8",
        )
        try:
            with ThreadPoolExecutor(1) as reader:
                first = reader.submit(process.stdout.readline).result(timeout=8)
            assert first.strip().startswith("{"), log.read_text(encoding="utf-8")
            handshake = json.loads(first)
            assert handshake["version"] == 1 and "token" not in handshake
            assert env["PORTFOLIO_SESSION_TOKEN"] not in first
            url = handshake["url"]
            assert url.startswith("http://127.0.0.1:") and not url.endswith(":0")
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                try:
                    if local_get(url + "/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.03)
            else:
                raise AssertionError(log.read_text(encoding="utf-8"))
            yield RunningService(url, env, data, process, log)
        finally:
            process.stdin.close()
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True
                    )
                else:
                    process.kill()
                process.wait(timeout=8)
            process.stdout.close()


def test_sidecar_serves_spa_from_owned_directory_and_protects_personal_api(sidecar):
    url, env = sidecar.url, sidecar.environment
    assert local_get(url + "/portfolios").text == "<html><body>research-shell</body></html>"
    assert local_get(url + "/asset.js").text == 'console.log("local")'
    assert local_get(url + "/missing.js").status_code == 404
    assert local_get(url + "/private.txt").status_code == 404
    assert local_get(url + "/%2e%2e/private.txt").status_code == 404
    assert local_get(url + "/v1/datasets").status_code == 401
    headers = {"Authorization": "Bearer " + env["PORTFOLIO_SESSION_TOKEN"]}
    assert local_get(url + "/v1/datasets", headers=headers).status_code == 200
    assert (
        local_get(
            url + "/v1/config", headers={**headers, "Origin": "https://attacker.invalid"}
        ).status_code
        == 403
    )
    assert local_get(url + "/v1/not-a-route", headers=headers).status_code == 404


def test_second_service_cannot_recover_or_modify_a_live_database(sidecar):
    env, data = sidecar.environment, sidecar.data_dir
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "portfolio_client",
            "--sidecar",
            "--port",
            "0",
            "--data-dir",
            str(data),
        ],
        env=env,
        capture_output=True,
        encoding="utf-8",
        text=True,
        timeout=8,
    )
    assert result.returncode != 0 and "DATA_DIRECTORY_IN_USE" in result.stderr


def test_sidecar_requires_parent_token_without_printing_credentials(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "PORTFOLIO_SESSION_TOKEN"}
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "portfolio_client", "--sidecar", "--data-dir", str(tmp_path)],
        env=env,
        capture_output=True,
        encoding="utf-8",
        text=True,
        timeout=8,
    )
    assert result.returncode != 0 and "SIDECAR_TOKEN_REQUIRED" in result.stderr


def test_parent_pipe_eof_stops_the_owned_listener(sidecar):
    url, process = sidecar.url, sidecar.process
    process.stdin.close()
    process.wait(timeout=12)
    with pytest.raises((httpx.ConnectError, httpx.ConnectTimeout)):
        local_get(url + "/health", timeout=1)


def test_unexpected_http_failure_does_not_log_private_sql_parameters(sidecar):
    database = sidecar.data_dir / "portfolio.db"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TABLE portfolios")
    connection.close()
    marker = "PRIVATE_TEST_ONLY_PORTFOLIO_ID"
    response = local_get(
        sidecar.url + "/v1/portfolios/" + marker,
        headers={"Authorization": "Bearer " + sidecar.environment["PORTFOLIO_SESSION_TOKEN"]},
    )
    assert response.status_code == 500 and response.json()["error"]["code"] == "INTERNAL_ERROR"
    sidecar.process.stdin.close()
    sidecar.process.wait(timeout=12)
    logs = sidecar.log.read_text(encoding="utf-8")
    assert marker not in logs
    assert response.json()["error"]["trace_id"] in logs
