import json
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from portfolio_client.domain.execution.audit import AuditEntry
from portfolio_client.infrastructure.db.models import (
    LedgerRecord,
    PaperAccountRecord,
    PaperOrderRecord,
    PaperTradeRecord,
)


def verify_business_rows(
    session: Session, account: PaperAccountRecord, entries: list[AuditEntry]
) -> str | None:
    """Compare materialized financial rows with the already verified audit history."""
    expected_ledger: dict[int, dict[str, Any]] = {}
    expected_orders: dict[str, dict[str, Any]] = {}
    expected_trades: dict[str, dict[str, Any]] = {}
    expected_account: dict[str, Any] = {}
    try:
        for entry in entries:
            transition = json.loads(entry.payload_json)["payload"]
            expected_account = transition["account"]
            for event in expected_account["state"]["events"]:
                if event["seq"] != len(expected_ledger) + 1:
                    return "ledger_sequence_mismatch"
                expected_ledger[event["seq"]] = event
            for order in transition["orders"]:
                expected_orders[order["order"]["order_id"]] = order
            for trade in transition["trades"]:
                expected_trades[trade["order_id"]] = trade
        expected_account = expected_account.copy()
        state = expected_account["state"].copy()
        state["ledger_offset"] += len(state["events"])
        state["events"] = []
        expected_account["state"] = state
    except (ValueError, KeyError, TypeError):
        return "invalid_transition_payload"
    ledger = {
        record.seq: record.payload
        for record in session.scalars(
            select(LedgerRecord).where(LedgerRecord.account_id == account.id)
        )
    }
    if ledger != expected_ledger:
        return "ledger_mismatch"
    if account.payload != expected_account or account.dataset_id != expected_account["dataset_id"]:
        return "account_mismatch"
    orders = list(
        session.scalars(select(PaperOrderRecord).where(PaperOrderRecord.account_id == account.id))
    )
    if {record.id: record.payload for record in orders} != expected_orders:
        return "orders_mismatch"
    for record in orders:
        expected = expected_orders[record.id]
        if record.state != expected["order"]["state"] or record.sequence != expected["sequence"]:
            return "order_projection_mismatch"
    trades = {
        record.order_id: record.payload
        for record in session.scalars(
            select(PaperTradeRecord).where(PaperTradeRecord.account_id == account.id)
        )
    }
    return "trades_mismatch" if trades != expected_trades else None
