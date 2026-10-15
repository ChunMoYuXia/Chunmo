"""Deterministic report content; no model, filesystem, or HTTP dependency."""

import csv
import io
from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal
from typing import Literal

from portfolio_client.domain.research.metric_models import Metric, PerformanceMetrics

METRIC_LABELS = {
    "total_return": "区间收益率",
    "annualized_return": "年化收益率",
    "annualized_volatility": "年化波动率",
    "max_drawdown": "最大回撤",
    "sharpe": "夏普比率",
    "benchmark_return": "基准收益率",
    "excess_return": "超额收益率",
}


@dataclass(frozen=True)
class Evidence:
    id: str
    source: str
    data_version: str
    as_of: datetime
    formula_version: str
    details: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReportSection:
    title: str
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResearchReport:
    kind: Literal["rule_based"]
    model: None
    as_of: datetime
    version: str
    metrics: dict[str, Metric]
    sections: tuple[ReportSection, ...]
    evidence: tuple[Evidence, ...]


def format_metric(metric: Metric) -> str:
    if metric.value is None:
        return f"数据缺失（{metric.reason or 'unknown'}）"
    value = Decimal(str(metric.value))
    return (
        f"{value * 100:.2f}%"
        if metric.unit in {"ratio", "annualized_ratio", "one_day_loss_ratio"}
        else f"{value:.4f} {metric.unit}"
    )


def build_report(
    metrics: PerformanceMetrics, evidence: list[Evidence], as_of: datetime
) -> ResearchReport:
    if not evidence:
        raise ValueError("报告必须引用原始证据")
    citations = tuple(item.id for item in evidence)
    values = {
        field.name: value
        for field in fields(metrics)
        if isinstance(value := getattr(metrics, field.name), Metric)
    }
    sections = tuple(
        ReportSection(METRIC_LABELS[key], format_metric(metric), citations)
        for key, metric in values.items()
    ) + (
        ReportSection(
            "计算口径",
            f"年化交易日 {metrics.annualization}；年无风险利率 {metrics.risk_free_annual}；"
            f"有效收益观测 {metrics.observations}；基准口径 {metrics.benchmark_basis or '未提供'}。",
            citations,
        ),
        ReportSection(
            "指标缺口与警示", "\n".join(metrics.warnings) or "无附加指标警示。", citations
        ),
    )
    return ResearchReport(
        "rule_based", None, as_of, "rule-report-v1", values, sections, tuple(evidence)
    )


def report_markdown(report: ResearchReport) -> str:
    lines = [
        "# 规则分析报告",
        "",
        f"类型：{report.kind}；模型：未使用；版本：{report.version}",
        f"研究截至：{report.as_of.isoformat()}",
        "",
    ]
    for section in report.sections:
        lines.extend(
            [
                f"## {section.title}",
                "",
                section.text,
                "",
                f"证据：{', '.join(section.evidence_ids)}",
                "",
            ]
        )
    lines.extend(["## 证据索引", ""])
    for evidence in report.evidence:
        lines.extend(
            [
                f"- {evidence.id}：{evidence.source}；数据 {evidence.data_version}；"
                f"时间 {evidence.as_of.isoformat()}；公式 {evidence.formula_version}"
            ]
        )
        lines.extend(f"  - {detail}" for detail in evidence.details)
    return "\n".join(lines) + "\n"


def report_csv(report: ResearchReport) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "metric",
            "value",
            "unit",
            "formula_version",
            "source",
            "as_of",
            "missing_reason",
            "data_version",
        ]
    )
    version = "|".join(evidence.data_version for evidence in report.evidence)
    for key, metric in report.metrics.items():
        writer.writerow(
            [
                key,
                metric.value if metric.value is not None else "",
                metric.unit,
                metric.formula_version,
                metric.source,
                metric.as_of.isoformat() if metric.as_of else "",
                metric.reason or "",
                version,
            ]
        )
    return output.getvalue()
