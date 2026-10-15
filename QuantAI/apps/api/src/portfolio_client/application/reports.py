import csv
import io
from dataclasses import replace
from typing import Literal

from portfolio_client.application.exports import ExportDocument
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.research.jobs import JobRepositoryPort
from portfolio_client.domain.research.reports import (
    Evidence,
    ReportSection,
    ResearchReport,
    build_report,
    report_csv,
    report_markdown,
)


class ReportService:
    def __init__(self, jobs: JobRepositoryPort) -> None:
        self.jobs = jobs

    def get(self, artifact_id: str) -> ResearchReport:
        artifact = self.jobs.artifact(artifact_id)
        result, snapshot = artifact.result, artifact.snapshot
        as_of = at_close(result.request.end)
        evidence = Evidence(
            artifact.id,
            "backtest:raw_execution_and_valuation",
            f"{snapshot.dataset_id}:v{snapshot.version}:{snapshot.content_hash}",
            as_of,
            f"{result.evidence.engine_version}/metrics-v1",
            (
                f"来源 {snapshot.source_kind}；日历 {snapshot.calendar_id}",
                f"价格 {result.evidence.price_basis}；费率 {','.join(result.evidence.fee_versions)}",
                f"知识口径 {result.evidence.knowledge_mode}；随机种子 {result.evidence.seed}",
            ),
        )
        report = build_report(artifact.metrics, [evidence], as_of)
        extra = (
            ReportSection(
                "成本与成交",
                f"显式费用 {result.costs} 元；滑点价差 {result.slippage_costs} 元（已体现在成交价，不再扣减）；"
                f"成交 {len(result.trades)} 笔；未成交 {len(result.unfilled)} 笔。",
                (artifact.id,),
            ),
            ReportSection(
                "假设与数据缺口",
                "\n".join((*result.assumptions, *result.warnings, *snapshot.warnings)),
                (artifact.id,),
            ),
            ReportSection(
                "来源说明",
                "合成演示数据仅用于流程验证，不代表真实市场行情。"
                if snapshot.source_kind == "synthetic"
                else "本地导入数据；数据质量和授权由使用者核验。",
                (artifact.id,),
            ),
        )
        return replace(report, sections=(*report.sections, *extra))

    def export(
        self, artifact_id: str, kind: Literal["markdown", "metrics", "nav", "trades"]
    ) -> ExportDocument:
        report = self.get(artifact_id)
        if kind == "markdown":
            return ExportDocument(
                filename=f"research-{artifact_id}.md",
                media_type="text/markdown;charset=utf-8",
                content=report_markdown(report),
            )
        if kind == "metrics":
            content = report_csv(report)
        else:
            result = self.jobs.artifact(artifact_id).result
            output = io.StringIO(newline="")
            writer = csv.writer(output)
            if kind == "nav":
                writer.writerow(
                    ["session", "nav_cny", "cash_cny", "receivables_cny", "stale_securities"]
                )
                writer.writerows(
                    [
                        point.session,
                        point.nav,
                        point.cash,
                        point.receivables,
                        "|".join(point.stale_securities),
                    ]
                    for point in result.nav
                )
            else:
                writer.writerow(
                    [
                        "session",
                        "security_id",
                        "side",
                        "quantity_shares",
                        "price_cny",
                        "commission_cny",
                        "transfer_cny",
                        "stamp_cny",
                        "fee_version",
                    ]
                )
                writer.writerows(
                    [
                        trade.fill.session,
                        trade.fill.security_id,
                        trade.fill.side,
                        trade.fill.quantity,
                        trade.fill.price,
                        trade.fees.commission,
                        trade.fees.transfer,
                        trade.fees.stamp,
                        trade.fees.schedule_version,
                    ]
                    for trade in result.trades
                )
            content = output.getvalue()
        return ExportDocument(
            filename=f"research-{artifact_id}-{kind}.csv",
            media_type="text/csv;charset=utf-8",
            content="\ufeff" + content,
        )
