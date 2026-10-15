from datetime import UTC, datetime

from portfolio_client.domain.research.metric_models import Metric, PerformanceMetrics
from portfolio_client.domain.research.reports import (
    Evidence,
    build_report,
    report_csv,
    report_markdown,
)


def test_annualized_ratio_is_displayed_as_a_percentage() -> None:
    from portfolio_client.domain.research.reports import format_metric

    metric = Metric(0.12, "annualized_ratio", "metrics-v1", "nav", None)
    assert format_metric(metric) == "12.00%"


def test_rule_report_has_durable_evidence_missing_values_and_no_model() -> None:
    as_of = datetime(2025, 1, 2, 7, tzinfo=UTC)
    known = Metric(0.1, "ratio", "metrics-v1", "nav", as_of)
    missing = Metric(None, "ratio", "metrics-v1", "benchmark", as_of, "benchmark_not_provided")
    metrics = PerformanceMetrics(
        known, known, known, known, known, missing, missing, 244, 0, 30, None, ()
    )
    evidence = [Evidence("nav", "raw_navigation", "d1:v1:hash", as_of, "metrics-v1")]
    report = build_report(metrics, evidence, as_of)
    assert report.kind == "rule_based" and report.model is None
    assert all(section.evidence_ids == ("nav",) for section in report.sections)
    assert "10.00%" in report_markdown(report)
    assert "benchmark_not_provided" in report_markdown(report)
    assert "d1:v1:hash" in report_markdown(report)
    assert "benchmark_not_provided" in report_csv(report)
    assert "0.00" not in report_csv(report).split("benchmark_return")[1].splitlines()[0]


def test_report_and_original_series_export_reuse_persisted_artifact(client, auth, market_dataset):
    from tests.test_jobs import spec, wait_job

    created = client.post(
        "/v1/backtests", headers={**auth, "Idempotency-Key": "report"}, json=spec(market_dataset)
    )
    job = wait_job(client, auth, created.json()["data"]["task_id"])
    report_id = job["result_id"]
    report = client.get(f"/v1/reports/{report_id}", headers=auth).json()["data"]
    assert report["kind"] == "rule_based" and report["model"] is None
    assert all(section["evidence_ids"] == [report_id] for section in report["sections"])
    markdown = client.get(f"/v1/reports/{report_id}/export?format=markdown", headers=auth).json()[
        "data"
    ]["content"]
    assert (
        "显式费用" in markdown
        and "假设与数据缺口" in markdown
        and market_dataset["content_hash"] in markdown
    )
    nav = client.get(f"/v1/reports/{report_id}/export?format=nav", headers=auth).json()["data"][
        "content"
    ]
    assert len(nav.strip().splitlines()) == 31


def test_market_csv_exports_raw_precision_before_javascript_numeric_conversion(client, auth):
    quotes = "代码,日期,开盘,最高,最低,收盘,成交量\n600001,2024-01-02,10.123456789,10.123456789,10.123456789,10.123456789,9007199254740993\n"
    preview = client.post(
        "/v1/datasets/import/preview",
        headers=auth,
        files={
            "file": ("quotes.csv", quotes.encode("utf-8"), "text/csv"),
            "calendar_file": ("calendar.csv", b"date\n2024-01-02\n", "text/csv"),
        },
    ).json()["data"]
    dataset = client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth).json()[
        "data"
    ]
    response = client.get(
        "/v1/market/export",
        headers=auth,
        params={
            "dataset_id": dataset["dataset_id"],
            "security_id": "XSHG:600001",
            "as_of": "2024-01-02T15:00:00+08:00",
        },
    )
    assert response.status_code == 200
    content = response.json()["data"]["content"]
    assert "9007199254740993" in content and "10.123456789" in content
    assert dataset["content_hash"] in content
