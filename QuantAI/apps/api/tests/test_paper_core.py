from datetime import UTC, date, datetime, timedelta
from decimal import Decimal as D

from portfolio_client.domain.accounting.costs import default_fee_schedules
from portfolio_client.domain.accounting.models import AccountState, Side
from portfolio_client.domain.execution.paper_engine import advance_paper, prepare_order
from portfolio_client.domain.execution.paper_models import OrderRequest, PaperAccount
from portfolio_client.domain.research.backtest import BacktestRequest, run_backtest
from tests.market_helpers import MemoryMarket


def account(market, capital=D("10000")):
    day = market.sessions[20]
    return PaperAccount(
        "paper",
        "测试",
        "test",
        "calendar",
        1,
        "synthetic",
        "active",
        AccountState(cash=capital, withdrawable_cash=capital, session=day),
        default_fee_schedules(),
        D("0"),
        D(".05"),
        True,
        datetime(2024, 1, 1, tzinfo=UTC),
    )


def test_paper_and_backtest_share_exact_fill_fees_and_final_account_money():
    market = MemoryMarket([date(2024, 1, 2) + timedelta(days=i) for i in range(22)])
    spec = BacktestRequest(
        dataset_id="test",
        calendar_id="calendar",
        security_ids=["XSHE:000001"],
        weights={"XSHE:000001": D("1")},
        start=market.sessions[20],
        end=market.sessions[-1],
        frequency="daily",
        capital=D("10000"),
        slippage_bps=D("0"),
    )
    backtest = run_backtest(spec, market, lambda _: None)
    submitted = prepare_order(
        account(market),
        OrderRequest(
            security_id="XSHE:000001",
            side=Side.BUY,
            quantity=900,
            order_type="limit",
            limit_price=D("10"),
        ),
        "o1",
        [],
        market,
    )
    paper = advance_paper(submitted.account, list(submitted.orders), market, market.sessions[-1])
    assert paper.trades[0].fill == backtest.trades[0].fill
    assert paper.trades[0].fees == backtest.trades[0].fees
    assert paper.account.state.cash == backtest.final_account.cash
    assert paper.account.state.positions == backtest.final_account.positions


def test_multiple_orders_share_one_security_capacity_for_the_day():
    market = MemoryMarket([date(2024, 1, 2) + timedelta(days=i) for i in range(22)])
    initial = account(market, D("10000000"))
    request = OrderRequest(
        security_id="XSHE:000001",
        side=Side.BUY,
        quantity=50000,
        order_type="limit",
        limit_price=D("10"),
    )
    one = prepare_order(initial, request, "o1", [], market)
    two = prepare_order(one.account, request, "o2", list(one.orders), market)
    result = advance_paper(two.account, [*one.orders, *two.orders], market, market.sessions[-1])
    assert sum(trade.fill.quantity for trade in result.trades) == 50000
    assert result.orders[1].reason == "liquidity_below_lot"
