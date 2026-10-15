from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel

from portfolio_client.api.schemas import Envelope, Metadata
from portfolio_client.application.paper import PaperTradingService
from portfolio_client.domain.accounting.models import LedgerEvent, Side
from portfolio_client.domain.execution.audit import AuditEntry, AuditVerification
from portfolio_client.domain.execution.models import OrderState
from portfolio_client.domain.execution.paper_models import (
    AccountSnapshot,
    AdvanceRequest,
    CreateAccountRequest,
    EmptyRequest,
    HaltResult,
    OrderRequest,
    PaperOrder,
    ResumeRequest,
)
from portfolio_client.domain.research.backtest_models import Trade

Key = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)]


class OrderDTO(BaseModel):
    id: str
    account_id: str
    security_id: str
    side: Side
    quantity: int
    order_type: Literal["market", "limit"]
    limit_price: Decimal | None
    submitted_at: datetime
    eligible_session: date
    state: OrderState
    sequence: int
    reserved_cash: Decimal
    filled_quantity: int
    reason: str | None


def order_dto(pending: PaperOrder) -> OrderDTO:
    order = pending.order
    return OrderDTO(
        id=order.order_id,
        account_id=order.account_id,
        security_id=order.security_id,
        side=order.side,
        quantity=order.quantity,
        order_type=order.order_type,
        limit_price=order.limit_price,
        submitted_at=order.submitted_at,
        eligible_session=order.eligible_session,
        state=order.state,
        sequence=pending.sequence,
        reserved_cash=pending.reserved_cash,
        filled_quantity=pending.filled_quantity,
        reason=pending.reason,
    )


def create_paper_router(paper: PaperTradingService) -> APIRouter:
    router = APIRouter(prefix="/v1/paper/accounts", tags=["paper"])

    def wrap[T](request: Request, data: T) -> Envelope[T]:
        return Envelope(data=data, meta=Metadata(trace_id=request.state.trace_id))

    @router.post("", response_model=Envelope[AccountSnapshot], status_code=201)
    def create(
        payload: CreateAccountRequest, request: Request, key: Key
    ) -> Envelope[AccountSnapshot]:
        return wrap(request, paper.create_account(payload, key))

    @router.get("", response_model=Envelope[list[AccountSnapshot]])
    def accounts(request: Request) -> Envelope[list[AccountSnapshot]]:
        return wrap(request, paper.list())

    @router.get("/{account_id}", response_model=Envelope[AccountSnapshot])
    def account(account_id: str, request: Request) -> Envelope[AccountSnapshot]:
        return wrap(request, paper.get(account_id))

    @router.post("/{account_id}/orders", response_model=Envelope[OrderDTO], status_code=201)
    def submit(
        account_id: str, payload: OrderRequest, request: Request, key: Key
    ) -> Envelope[OrderDTO]:
        return wrap(request, order_dto(paper.submit_order(account_id, payload, key)))

    @router.get("/{account_id}/orders", response_model=Envelope[list[OrderDTO]])
    def orders(account_id: str, request: Request) -> Envelope[list[OrderDTO]]:
        return wrap(request, [order_dto(item) for item in paper.repository.orders(account_id)])

    @router.post("/{account_id}/orders/{order_id}/cancel", response_model=Envelope[OrderDTO])
    def cancel(
        account_id: str, order_id: str, payload: EmptyRequest, request: Request, key: Key
    ) -> Envelope[OrderDTO]:
        return wrap(request, order_dto(paper.cancel(account_id, order_id, key)))

    @router.post("/{account_id}/advance", response_model=Envelope[AccountSnapshot])
    def advance(
        account_id: str, payload: AdvanceRequest, request: Request, key: Key
    ) -> Envelope[AccountSnapshot]:
        return wrap(request, paper.advance(account_id, payload, key))

    @router.post("/{account_id}/halt", response_model=Envelope[HaltResult])
    def halt(
        account_id: str, payload: EmptyRequest, request: Request, key: Key
    ) -> Envelope[HaltResult]:
        return wrap(request, paper.halt(account_id, key))

    @router.post("/{account_id}/resume", response_model=Envelope[AccountSnapshot])
    def resume(
        account_id: str, payload: ResumeRequest, request: Request, key: Key
    ) -> Envelope[AccountSnapshot]:
        return wrap(request, paper.resume(account_id, payload, key))

    @router.get("/{account_id}/trades", response_model=Envelope[list[Trade]])
    def trades(account_id: str, request: Request) -> Envelope[list[Trade]]:
        return wrap(request, paper.repository.trades(account_id))

    @router.get("/{account_id}/ledger", response_model=Envelope[list[LedgerEvent]])
    def ledger(
        account_id: str,
        request: Request,
        after: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ) -> Envelope[list[LedgerEvent]]:
        return wrap(request, paper.repository.ledger(account_id, after, limit))

    @router.get("/{account_id}/audit", response_model=Envelope[list[AuditEntry]])
    def audit(
        account_id: str,
        request: Request,
        after: int = Query(0, ge=0),
        limit: int = Query(200, ge=1, le=1000),
    ) -> Envelope[list[AuditEntry]]:
        return wrap(request, paper.repository.audit_entries(account_id, after, limit))

    @router.get("/{account_id}/audit/verify", response_model=Envelope[AuditVerification])
    def verify(account_id: str, request: Request) -> Envelope[AuditVerification]:
        return wrap(request, paper.repository.verify(account_id))

    return router
