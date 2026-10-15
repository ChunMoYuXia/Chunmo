import csv
import io

from pydantic import BaseModel

from portfolio_client.application.market import MarketContext
from portfolio_client.domain.market.models import DailyBar


class ExportDocument(BaseModel):
    filename: str
    media_type: str
    content: str


def export_market_csv(
    context: MarketContext, security_id: str, bars: list[DailyBar]
) -> ExportDocument:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "security_id",
            "session",
            "open_raw",
            "high_raw",
            "low_raw",
            "close_raw",
            "volume_shares",
            "amount_cny",
            "available_at",
            "dataset_id",
            "version",
            "content_hash",
            "calendar_id",
            "as_of",
            "knowledge_mode",
        ]
    )
    snapshot = context.snapshot
    writer.writerows(
        [
            bar.security_id,
            bar.trade_date,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
            bar.amount if bar.amount is not None else "",
            bar.available_at.isoformat(),
            snapshot.dataset_id,
            snapshot.version,
            snapshot.content_hash,
            snapshot.calendar_id,
            context.as_of.isoformat(),
            context.knowledge_mode,
        ]
        for bar in bars
    )
    return ExportDocument(
        filename=f"quotes-{security_id.replace(':', '-')}.csv",
        media_type="text/csv;charset=utf-8",
        content="\ufeff" + output.getvalue(),
    )
