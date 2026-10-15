import hashlib
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from pydantic import BaseModel, TypeAdapter

from portfolio_client.domain.accounting.costs import default_fee_schedules, schedule_for
from portfolio_client.domain.accounting.ledger import append_event
from portfolio_client.domain.accounting.models import AccountState
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.execution.audit import canonical_json
from portfolio_client.domain.execution.paper_engine import (
    advance_paper,
    cancel_order,
    prepare_order,
    snapshot,
)
from portfolio_client.domain.execution.paper_models import (
    AccountSnapshot,
    AdvanceRequest,
    CancelRequest,
    CreateAccountRequest,
    EmptyRequest,
    HaltResult,
    OrderRequest,
    PaperAccount,
    PaperOrder,
    PaperTransition,
    ResumeRequest,
)
from portfolio_client.domain.execution.paper_ports import (
    IdempotencyEntry,
    PaperRepositoryPort,
    PaperUnitOfWork,
)
from portfolio_client.domain.market.ports import DatasetRepositoryPort, MarketDataPort

ACCOUNT = TypeAdapter(PaperAccount)
TRANSITION = TypeAdapter(PaperTransition)
SNAPSHOT = TypeAdapter(AccountSnapshot)
ORDER = TypeAdapter(PaperOrder)
HALT = TypeAdapter(HaltResult)
type CommandResult = AccountSnapshot | PaperOrder | HaltResult


class PaperTradingService:
    def __init__(
        self,
        repository: PaperRepositoryPort,
        datasets: DatasetRepositoryPort,
        market: MarketDataPort,
    ) -> None:
        self.repository, self.datasets, self.market = repository, datasets, market

    def _command[T: CommandResult](
        self,
        scope: str,
        operation: str,
        key: str,
        payload: BaseModel,
        adapter: TypeAdapter[T],
        action: Callable[[PaperUnitOfWork], T],
    ) -> T:
        if not 1 <= len(key) <= 128:
            raise DomainError("INVALID_IDEMPOTENCY_KEY", "幂等键长度必须为1–128")
        digest = hashlib.sha256(
            canonical_json(payload.model_dump(mode="json")).encode()
        ).hexdigest()
        with self.repository.transaction() as unit:
            prior = unit.prior(scope, operation, key)
            if prior is not None:
                if prior.request_hash != digest:
                    raise DomainError("IDEMPOTENCY_CONFLICT", "相同账户、操作和幂等键对应不同内容")
                return adapter.validate_json(prior.response_json)
            result = action(unit)
            unit.save_response(
                scope, operation, key, IdempotencyEntry(digest, adapter.dump_json(result).decode())
            )
            return result

    def _persist(self, unit: PaperUnitOfWork, transition: PaperTransition, operation: str) -> None:
        unit.save_account(transition.account)
        for pending in transition.orders:
            unit.save_order(pending)
        for trade in transition.trades:
            unit.add_trade(transition.account.id, trade)
        unit.audit(
            transition.account.id,
            operation,
            cast(dict[str, Any], TRANSITION.dump_python(transition, mode="json")),
        )

    def create_account(self, request: CreateAccountRequest, key: str) -> AccountSnapshot:
        def create(unit: PaperUnitOfWork) -> AccountSnapshot:
            dataset = self.datasets.get_dataset(request.dataset_id)
            calendar = self.market.calendar(dataset.dataset_id)
            day = request.start or calendar.sessions[min(20, len(calendar.sessions) - 1)]
            if day not in calendar.sessions:
                raise DomainError("INVALID_SESSION", "起点必须属于数据集交易日历")
            schedules = request.fee_schedules or default_fee_schedules()
            schedule_for(schedules, day)
            state = append_event(
                AccountState(cash=request.capital, withdrawable_cash=request.capital, session=day),
                "deposit",
                delta_cash=request.capital,
            )
            account = PaperAccount(
                uuid4().hex,
                request.name,
                dataset.dataset_id,
                dataset.calendar_id,
                dataset.version,
                dataset.source_kind,
                "active",
                state,
                schedules,
                request.slippage_bps,
                request.participation_rate,
                request.require_price_limits,
                datetime.now(UTC),
            )
            self._persist(unit, PaperTransition(account), "create_account")
            return snapshot(account, [], self.market)

        return self._command("__accounts__", "create_account", key, request, SNAPSHOT, create)

    def get(self, account_id: str) -> AccountSnapshot:
        with self.repository.read() as unit:
            return snapshot(unit.account(account_id), unit.orders(account_id), self.market)

    def list(self) -> list[AccountSnapshot]:
        return [self.get(account.id) for account in self.repository.accounts()]

    def submit_order(self, account_id: str, request: OrderRequest, key: str) -> PaperOrder:
        def submit(unit: PaperUnitOfWork) -> PaperOrder:
            transition = prepare_order(
                unit.account(account_id), request, uuid4().hex, unit.orders(account_id), self.market
            )
            self._persist(unit, transition, "submit_order")
            return transition.orders[0]

        return self._command(account_id, "submit_order", key, request, ORDER, submit)

    def advance(self, account_id: str, request: AdvanceRequest, key: str) -> AccountSnapshot:
        def advance_one(unit: PaperUnitOfWork) -> AccountSnapshot:
            account = unit.account(account_id)
            if account.state.session is None:
                raise DomainError("INVALID_ACCOUNT", "账户当前交易日缺失")
            day = request.session or self.market.calendar(account.dataset_id).next_session(
                account.state.session
            )
            transition = advance_paper(
                account, unit.orders(account_id, open_only=True), self.market, day
            )
            self._persist(unit, transition, "advance")
            return snapshot(transition.account, list(transition.orders), self.market)

        return self._command(account_id, "advance", key, request, SNAPSHOT, advance_one)

    def cancel(self, account_id: str, order_id: str, key: str) -> PaperOrder:
        def cancel_one(unit: PaperUnitOfWork) -> PaperOrder:
            transition = cancel_order(unit.account(account_id), unit.order(account_id, order_id))
            self._persist(unit, transition, "cancel_order")
            return transition.orders[0]

        return self._command(
            account_id, "cancel_order", key, CancelRequest(order_id=order_id), ORDER, cancel_one
        )

    def halt(self, account_id: str, key: str) -> HaltResult:
        def pause(unit: PaperUnitOfWork) -> HaltResult:
            account = replace(unit.account(account_id), status="paused")
            changed: list[PaperOrder] = []
            for pending in unit.orders(account_id, open_only=True):
                transition = cancel_order(account, pending)
                account = transition.account
                changed.extend(transition.orders)
            self._persist(unit, PaperTransition(account, tuple(changed)), "halt")
            return HaltResult(
                snapshot(account, changed, self.market),
                tuple(item.order.order_id for item in changed),
            )

        return self._command(account_id, "halt", key, EmptyRequest(), HALT, pause)

    def resume(self, account_id: str, request: ResumeRequest, key: str) -> AccountSnapshot:
        def resume_one(unit: PaperUnitOfWork) -> AccountSnapshot:
            if request.confirmation != "恢复模拟交易":
                raise DomainError("RESUME_CONFIRMATION_REQUIRED", "请输入恢复模拟交易确认短语")
            account = replace(unit.account(account_id), status="active")
            self._persist(unit, PaperTransition(account), "resume")
            return snapshot(account, unit.orders(account_id), self.market)

        return self._command(account_id, "resume", key, request, SNAPSHOT, resume_one)
