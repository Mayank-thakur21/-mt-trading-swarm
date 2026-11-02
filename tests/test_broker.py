"""Paper broker accounting, persistence and audit trail; in-memory broker parity; exits.

Hand-computed reference trade (defaults: 5 bps slippage, commission max($1, 5 bps)):
BUY 10 @ 100 → fill 100.05, notional 1000.50, commission 1.00, cash −1001.50,
avg price (commission-inclusive) 100.15.
SELL 4 @ 110 → fill 109.945, notional 439.78, commission 1.00,
realized = 439.78 − 1.00 − 4 × 100.15 = 38.18.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_swarm.broker import (
    UNSET,
    Broker,
    CostModel,
    InMemoryBroker,
    OrderValidationError,
    PaperBroker,
    evaluate_exit,
    evaluate_exit_bar,
)
from trading_swarm.config import Settings
from trading_swarm.data import InvalidSymbolError
from trading_swarm.models import (
    Action,
    AgentSignal,
    AlertRecord,
    AlertStatus,
    Decision,
    ExitReason,
    OrderStatus,
    RiskAssessment,
    Side,
)

T0 = datetime(2026, 3, 2, 15, 0, tzinfo=UTC)


# ----------------------------------------------------------------------------- cost model


def test_cost_model_formulas() -> None:
    costs = CostModel(commission_rate=0.0005, min_commission=1.0, slippage_bps=5.0)
    assert costs.fill_price(Side.BUY, 100.0) == pytest.approx(100.05)
    assert costs.fill_price(Side.SELL, 100.0) == pytest.approx(99.95)
    assert costs.commission(1_000.0) == 1.0  # minimum applies
    assert costs.commission(10_000.0) == pytest.approx(5.0)
    assert costs.commission(0.0) == 0.0
    fill = costs.quote(Side.BUY, 10, 100.0)
    assert fill.notional == pytest.approx(1000.5)
    assert fill.slippage_cost == pytest.approx(0.5)
    assert fill.cash_delta == pytest.approx(-1001.5)
    sell = costs.quote(Side.SELL, 10, 100.0)
    assert sell.cash_delta == pytest.approx(999.5 - 1.0)


@pytest.mark.parametrize(
    ("cash", "price"),
    [(10_000.0, 100.0), (1_001.5, 100.0), (5.0, 3.0), (250_000.0, 37.13), (0.5, 0.1)],
)
def test_max_affordable_quantity_is_tight(cash: float, price: float) -> None:
    costs = CostModel()
    quantity = costs.max_affordable_quantity(cash, price)
    if quantity:
        assert costs.buy_cash_required(quantity, price) <= cash + 1e-9
    assert costs.buy_cash_required(quantity + 1, price) > cash
    assert costs.max_affordable_quantity(cash, 0.0) == 0


def test_cost_model_from_settings(make_settings: Callable[..., Settings]) -> None:
    costs = CostModel.from_settings(
        make_settings(commission_rate=0.001, min_commission=0, slippage_bps=10)
    )
    assert costs == CostModel(commission_rate=0.001, min_commission=0.0, slippage_bps=10.0)


# ----------------------------------------------------------------------------- fills


def test_buy_fill_and_cost_basis(broker: PaperBroker) -> None:
    order = broker.submit_order(
        "acme",
        "BUY",
        10,
        100.0,
        reason="unit test",
        stop=95.0,
        take_profit=110.0,
        decision_id="d1",
        timestamp=T0,
    )
    assert order.status is OrderStatus.FILLED
    assert order.symbol == "ACME" and order.side is Side.BUY
    assert order.fill_price == pytest.approx(100.05)
    assert order.reference_price == 100.0
    assert order.commission == 1.0 and order.slippage == pytest.approx(0.5)
    assert order.filled_at == T0 and order.decision_id == "d1" and order.realized_pnl is None

    position = broker.position("ACME")
    assert position is not None
    assert position.quantity == 10 and position.avg_price == pytest.approx(100.15)
    assert (position.stop_price, position.take_profit_price) == (95.0, 110.0)
    assert broker.cash == pytest.approx(100_000 - 1001.5)
    [trade] = broker.trades()
    assert trade.order_id == order.id and trade.price == pytest.approx(100.05)
    assert trade.realized_pnl is None


def test_adding_to_a_position_averages_cost(broker: PaperBroker) -> None:
    broker.submit_order("ACME", Side.BUY, 10, 100.0, stop=95.0)
    broker.submit_order("ACME", Side.BUY, 10, 110.0)  # no new stop → keeps 95
    # second leg: fill 110.055, notional 1100.55, commission 1.00 → basis 1101.55
    position = broker.position("ACME")
    assert position is not None and position.quantity == 20
    assert position.avg_price == pytest.approx((1001.5 + 1101.55) / 20)
    assert position.stop_price == 95.0


def test_partial_and_full_sell_realise_pnl_net_of_commissions(broker: PaperBroker) -> None:
    broker.submit_order("ACME", "BUY", 10, 100.0, timestamp=T0)
    partial = broker.submit_order("ACME", "SELL", 4, 110.0, timestamp=T0 + timedelta(days=1))
    assert partial.status is OrderStatus.FILLED
    assert partial.fill_price == pytest.approx(109.945)
    assert partial.realized_pnl == pytest.approx(38.18)
    position = broker.position("ACME")
    assert position is not None and position.quantity == 6
    assert position.avg_price == pytest.approx(100.15)  # sells don't change cost basis

    final = broker.submit_order("ACME", "SELL", 6, 90.0, timestamp=T0 + timedelta(days=2))
    # fill 89.955 × 6 = 539.73; commission 1; basis 6 × 100.15 = 600.90 → −62.17
    assert final.realized_pnl == pytest.approx(539.73 - 1.0 - 600.9)
    assert broker.position("ACME") is None
    snapshot = broker.portfolio()
    assert snapshot.realized_pnl == pytest.approx(38.18 - 62.17)
    assert snapshot.equity == pytest.approx(100_000 + 38.18 - 62.17)
    assert [t.side for t in broker.trades()] == [Side.SELL, Side.SELL, Side.BUY]


def test_total_return_equals_realized_plus_unrealized(broker: PaperBroker) -> None:
    broker.submit_order("ACME", "BUY", 10, 100.0)
    broker.submit_order("ACME", "SELL", 4, 110.0)
    broker.submit_order("BETA", "BUY", 37, 52.31)
    snapshot = broker.portfolio({"ACME": 110.0, "BETA": 49.0})
    assert snapshot.total_return == pytest.approx(
        snapshot.realized_pnl + snapshot.unrealized_pnl, abs=0.011
    )
    assert snapshot.equity == pytest.approx(snapshot.cash + snapshot.positions_value, abs=0.011)
    acme = snapshot.position("acme")
    assert acme is not None and acme.unrealized_pnl == pytest.approx(6 * (110 - 100.15))
    assert acme.unrealized_pnl_pct == pytest.approx((110 / 100.15 - 1) * 100, abs=1e-3)
    assert snapshot.gross_exposure_pct == pytest.approx(
        snapshot.positions_value / snapshot.equity * 100, abs=1e-3
    )


# ----------------------------------------------------------------------------- rejections


def test_insufficient_cash_is_rejected_and_logged(broker: PaperBroker) -> None:
    order = broker.submit_order("ACME", "BUY", 1_000, 100.0)
    assert order.status is OrderStatus.REJECTED
    assert order.reject_reason is not None and "insufficient cash" in order.reject_reason
    assert order.fill_price is None and order.filled_at is None
    assert broker.cash == 100_000.0 and broker.positions() == [] and broker.trades() == []
    assert broker.orders()[0].id == order.id  # rejected orders are kept for the audit trail


def test_cash_must_cover_commission(broker: PaperBroker) -> None:
    affordable = broker.cost_model.max_affordable_quantity(100_000.0, 100.0)
    assert broker.submit_order("ACME", "BUY", affordable + 1, 100.0).status is OrderStatus.REJECTED
    assert broker.submit_order("ACME", "BUY", affordable, 100.0).status is OrderStatus.FILLED
    assert 0 <= broker.cash < 102.0


def test_selling_without_or_beyond_position_is_rejected(broker: PaperBroker) -> None:
    no_position = broker.submit_order("ACME", "SELL", 1, 100.0)
    assert no_position.status is OrderStatus.REJECTED
    assert "no position" in (no_position.reject_reason or "")
    broker.submit_order("ACME", "BUY", 5, 100.0)
    oversell = broker.submit_order("ACME", "SELL", 6, 100.0)
    assert oversell.status is OrderStatus.REJECTED
    assert "hold 5" in (oversell.reject_reason or "")
    assert broker.position("ACME").quantity == 5  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        (("ACME", "BUY", 0, 100.0), {}),
        (("ACME", "BUY", -5, 100.0), {}),
        (("ACME", "BUY", 1.5, 100.0), {}),
        (("ACME", "BUY", True, 100.0), {}),
        (("ACME", "BUY", 1, 0.0), {}),
        (("ACME", "BUY", 1, float("nan")), {}),
        (("ACME", "BUY", 1, float("inf")), {}),
        (("ACME", "HOLD", 1, 100.0), {}),
        (("ACME", "BUY", 1, 100.0), {"stop": 100.0}),
        (("ACME", "BUY", 1, 100.0), {"stop": -1.0}),
        (("ACME", "BUY", 1, 100.0), {"take_profit": 99.0}),
        # A SELL only reduces / closes a position: levels would be silently ignored.
        (("ACME", "SELL", 1, 100.0), {"stop": 90.0}),
        (("ACME", "SELL", 1, 100.0), {"take_profit": 120.0}),
    ],
)
def test_malformed_orders_raise(broker: PaperBroker, args: tuple, kwargs: dict) -> None:
    with pytest.raises(OrderValidationError):
        broker.submit_order(*args, **kwargs)
    assert broker.orders() == []


def test_invalid_symbol_raises(broker: PaperBroker) -> None:
    with pytest.raises(InvalidSymbolError):
        broker.submit_order("not a symbol", "BUY", 1, 10.0)


# ----------------------------------------------------------------------------- marks & peak


def test_unpriced_positions_are_marked_at_cost(broker: PaperBroker) -> None:
    broker.submit_order("ACME", "BUY", 10, 100.0)
    snapshot = broker.portfolio({"OTHER": 5.0, "ACME": float("nan")})
    position = snapshot.positions[0]
    assert position.market_price is None
    assert position.market_value == pytest.approx(1001.5)
    assert position.unrealized_pnl == 0.0
    # cash 98_998.50 + cost basis 1_001.50: marking at cost shows no gain or loss
    assert snapshot.equity == pytest.approx(100_000.0)
    assert snapshot.unrealized_pnl == 0.0


def test_peak_equity_and_drawdown(broker: PaperBroker) -> None:
    broker.submit_order("ACME", "BUY", 100, 100.0)  # cost 10_010.0025
    up = broker.portfolio({"ACME": 120.0})
    assert up.equity == pytest.approx(101_989.9975, abs=0.01)
    assert up.peak_equity == up.equity and up.drawdown_pct == 0.0

    down = broker.portfolio({"ACME": 90.0})
    assert down.peak_equity == pytest.approx(101_989.9975, abs=0.01)
    expected = (101_989.9975 - 98_989.9975) / 101_989.9975
    assert down.drawdown == pytest.approx(expected, abs=1e-6)
    assert down.drawdown_pct == pytest.approx(expected * 100, abs=1e-4)


def test_peak_only_moves_on_fully_priced_marks(broker: PaperBroker) -> None:
    broker.submit_order("ACME", "BUY", 10, 100.0)
    broker.submit_order("BETA", "BUY", 10, 100.0)
    spike = broker.portfolio({"ACME": 10_000.0})  # BETA unpriced → not an observation
    assert spike.peak_equity > 100_000  # reported for this snapshot...
    assert broker.portfolio().peak_equity == pytest.approx(100_000.0)  # ...but not stored


# ----------------------------------------------------------------------------- persistence


def test_state_survives_a_new_broker_on_the_same_file(settings: Settings) -> None:
    first = PaperBroker(settings)
    try:
        first.submit_order("ACME", "BUY", 10, 100.0, stop=95.0, timestamp=T0)
        first.submit_order("ACME", "SELL", 4, 110.0, timestamp=T0 + timedelta(hours=1))
        before = first.portfolio({"ACME": 105.0})
    finally:
        first.close()

    second = PaperBroker(settings)
    try:
        after = second.portfolio({"ACME": 105.0})
        assert after.model_dump(exclude={"updated_at"}) == before.model_dump(exclude={"updated_at"})
        assert [o.side for o in second.orders()] == [Side.SELL, Side.BUY]
        assert second.orders()[0].requested_at == T0 + timedelta(hours=1)
        assert second.orders()[0].requested_at.tzinfo is not None
        assert second.position("ACME").stop_price == 95.0  # type: ignore[union-attr]
    finally:
        second.close()


def test_configured_capital_only_seeds_a_new_account(
    make_settings: Callable[..., Settings],
) -> None:
    with PaperBroker(make_settings(initial_capital=50_000)) as broker:
        assert broker.initial_capital == 50_000
    with PaperBroker(make_settings(initial_capital=75_000)) as broker:
        assert broker.initial_capital == 50_000  # resumed, not overwritten


def test_reset_clears_the_account_but_keeps_history(broker: PaperBroker) -> None:
    broker.submit_order("ACME", "BUY", 10, 100.0)
    broker.record_snapshot({"ACME": 101.0})
    decision = _decision()
    broker.save_decision(decision)
    broker.set_watchlist(["ACME"])

    snapshot = broker.reset(25_000)
    assert snapshot.cash == snapshot.equity == snapshot.initial_capital == 25_000
    assert snapshot.peak_equity == 25_000 and snapshot.positions == []
    assert broker.orders() == [] and broker.trades() == []
    assert len(broker.equity_curve()) == 1
    assert broker.get_decision(decision.id) is not None
    assert broker.get_watchlist() == ["ACME"]
    assert broker.reset().initial_capital == 100_000  # back to the configured capital
    with pytest.raises(ValueError):
        broker.reset(0)


def test_memory_and_nested_file_databases(
    make_settings: Callable[..., Settings], tmp_path: Path
) -> None:
    with PaperBroker(make_settings(database_url="sqlite:///:memory:")) as memory:
        memory.submit_order("ACME", "BUY", 1, 10.0)
        assert memory.position("ACME") is not None
    nested = tmp_path / "deep" / "er" / "swarm.db"
    with PaperBroker(make_settings(database_url=f"sqlite:///{nested}")) as broker:
        broker.submit_order("ACME", "BUY", 1, 10.0)
    assert nested.exists()


# ----------------------------------------------------------------------------- stops


def test_update_stops_set_clear_keep(broker: PaperBroker) -> None:
    broker.submit_order("ACME", "BUY", 10, 100.0, stop=95.0, take_profit=110.0)
    assert broker.update_stops("ACME", stop=97.5).stop_price == 97.5
    kept = broker.update_stops("ACME", take_profit=UNSET)
    assert (kept.stop_price, kept.take_profit_price) == (97.5, 110.0)
    cleared = broker.update_stops("ACME", take_profit=None)
    assert cleared.take_profit_price is None and cleared.stop_price == 97.5
    with pytest.raises(KeyError):
        broker.update_stops("NONE", stop=1.0)


@pytest.mark.parametrize(
    ("stop", "target", "price", "expected"),
    [
        (95.0, 110.0, 100.0, None),
        (95.0, 110.0, 95.0, ExitReason.STOP_LOSS),
        (95.0, 110.0, 80.0, ExitReason.STOP_LOSS),
        (95.0, 110.0, 110.0, ExitReason.TAKE_PROFIT),
        (None, 110.0, 50.0, None),
        (95.0, None, 500.0, None),
        (95.0, 110.0, float("nan"), None),
    ],
)
def test_evaluate_exit(
    stop: float | None, target: float | None, price: float, expected: ExitReason | None
) -> None:
    assert evaluate_exit(stop, target, price) == expected


@pytest.mark.parametrize(
    ("bar", "expected"),
    [
        ((100.0, 105.0, 96.0), None),  # inside the levels
        ((90.0, 99.0, 88.0), (ExitReason.STOP_LOSS, 90.0)),  # gap down → fill at open
        ((115.0, 118.0, 112.0), (ExitReason.TAKE_PROFIT, 115.0)),  # gap up → fill at open
        ((100.0, 104.0, 94.0), (ExitReason.STOP_LOSS, 95.0)),  # intrabar stop
        ((100.0, 111.0, 99.0), (ExitReason.TAKE_PROFIT, 110.0)),  # intrabar target
        ((100.0, 112.0, 93.0), (ExitReason.STOP_LOSS, 95.0)),  # both → stop first
    ],
)
def test_evaluate_exit_bar(
    bar: tuple[float, float, float], expected: tuple[ExitReason, float] | None
) -> None:
    open_, high, low = bar
    assert evaluate_exit_bar(95.0, 110.0, open_, high, low) == expected


# ----------------------------------------------------------------------------- queries


def test_orders_and_trades_are_newest_first_with_filters(broker: PaperBroker) -> None:
    for i, symbol in enumerate(["AAA", "BBB", "AAA", "CCC"]):
        broker.submit_order(symbol, "BUY", 1, 10.0 + i, timestamp=T0 + timedelta(minutes=i))
    assert [o.symbol for o in broker.orders()] == ["CCC", "AAA", "BBB", "AAA"]
    assert [o.reference_price for o in broker.orders(limit=2)] == [13.0, 12.0]
    assert [t.price for t in broker.trades(symbol="aaa")] == pytest.approx([12.006, 10.005])
    assert len(broker.trades(limit=None)) == 4
    order = broker.orders()[0]
    assert broker.get_order(order.id) == order and broker.get_order("missing") is None


def test_equity_curve(broker: PaperBroker) -> None:
    broker.submit_order("ACME", "BUY", 10, 100.0)
    broker.record_snapshot({"ACME": 101.0}, timestamp=T0)
    snapshot = broker.record_snapshot({"ACME": 103.0}, timestamp=T0 + timedelta(days=1))
    curve = broker.equity_curve()
    assert list(curve.columns) == ["equity", "cash", "positions_value"]
    assert isinstance(curve.index, pd.DatetimeIndex) and str(curve.index.tz) == "UTC"
    assert len(curve) == 3  # account creation (wall-clock time) + two snapshots
    assert curve.index.is_monotonic_increasing
    row = curve.loc[pd.Timestamp(T0 + timedelta(days=1))]
    assert row["equity"] == pytest.approx(snapshot.equity, abs=0.01)
    assert row["positions_value"] == pytest.approx(1030.0)
    assert row["cash"] == pytest.approx(100_000 - 1001.5)


def test_empty_equity_curve_has_the_right_shape(broker: PaperBroker) -> None:
    broker.reset()
    curve = broker.equity_curve()
    assert list(curve.columns) == ["equity", "cash", "positions_value"] and len(curve) == 1


# ----------------------------------------------------------------------------- decisions


def _decision(symbol: str = "BULL", when: datetime = T0) -> Decision:
    trend = AgentSignal(
        agent="trend",
        signal=0.62,
        confidence=0.8,
        rationale="price above SMA50/200",
        data={
            "rsi": np.float64(61.2),
            "bars": np.int64(252),
            "adx": float("nan"),
            "flags": [np.bool_(True)],
            "as_of": pd.Timestamp("2026-03-02", tz="UTC"),
        },
    )
    return Decision(
        symbol=symbol,
        timestamp=when,
        price=187.5,
        action=Action.BUY,
        score=0.55,
        confidence=0.7,
        signals=[trend],
        risk=RiskAssessment(
            approved=True,
            action=Action.BUY,
            quantity=37,
            entry_price=187.5,
            stop_price=182.4,
            take_profit_price=197.7,
            risk_score=35,
        ),
        summary="Trend +0.62 (conf 0.80)",
    )


def test_decisions_round_trip_and_upsert(broker: PaperBroker) -> None:
    first = _decision("BULL", T0)
    second = _decision("BEAR", T0 + timedelta(minutes=5))
    broker.save_decision(first)
    broker.save_decision(second)
    loaded = broker.decisions()
    assert [d.symbol for d in loaded] == ["BEAR", "BULL"]
    data = loaded[1].signals[0].data
    assert data == {
        "rsi": 61.2,
        "bars": 252,
        "adx": None,
        "flags": [True],
        "as_of": "2026-03-02T00:00:00+00:00",
    }
    assert loaded[1].timestamp == T0

    first.executed = True
    broker.save_decision(first)  # update in place, not a duplicate
    assert len(broker.decisions(limit=None)) == 2
    assert broker.decisions(symbol="bull")[0].executed is True
    assert broker.get_decision("nope") is None


# ----------------------------------------------------------------------------- alerts & KV


def test_alert_log_and_cooldown_lookup(broker: PaperBroker) -> None:
    sent = broker.log_alert(
        AlertRecord(
            timestamp=T0, rule="signal", symbol="BULL", status=AlertStatus.SENT, subject="BUY BULL"
        )
    )
    assert sent.id is not None and sent.timestamp == T0
    broker.log_alert(
        AlertRecord(
            timestamp=T0 + timedelta(minutes=1),
            rule="signal",
            symbol="BULL",
            status=AlertStatus.SUPPRESSED,
        )
    )
    broker.log_alert(
        AlertRecord(
            timestamp=T0 + timedelta(minutes=2),
            rule="drawdown",
            status=AlertStatus.DISABLED,
            channel="email",
        )
    )
    assert [a.rule for a in broker.alerts()] == ["drawdown", "signal", "signal"]
    assert len(broker.alerts(rule="signal", symbol="bull")) == 2
    assert broker.last_alert_at("signal", "BULL") == T0
    assert broker.last_alert_at("signal", "BULL", statuses=None) == T0 + timedelta(minutes=1)
    assert broker.last_alert_at("signal", "BEAR") is None
    assert broker.last_alert_at("drawdown", None, statuses=["disabled"]) == T0 + timedelta(
        minutes=2
    )


def test_settings_kv_and_watchlist(broker: PaperBroker) -> None:
    assert broker.get_setting("missing", default=7) == 7
    broker.set_setting("flags", {"a": [1, 2]})
    broker.set_setting("flags", {"a": [3]})
    assert broker.get_setting("flags") == {"a": [3]}
    assert broker.get_watchlist() == ["BULL", "BEAR", "FLAT"]  # from settings
    assert broker.get_watchlist(default=["X"]) == ["X"]
    assert broker.set_watchlist(["aapl", " MSFT ", "AAPL"]) == ["AAPL", "MSFT"]
    assert broker.get_watchlist() == ["AAPL", "MSFT"]
    with pytest.raises(InvalidSymbolError):
        broker.set_watchlist(["GOOD", "bad symbol"])


# ----------------------------------------------------------------------------- concurrency


def test_concurrent_orders_are_serialised(broker: PaperBroker) -> None:
    barrier = threading.Barrier(8)

    def buy(_: int) -> OrderStatus:
        if _ < 8:
            barrier.wait(timeout=5)
        return broker.submit_order("ACME", "BUY", 1, 100.0).status

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(buy, range(40)))
    assert statuses == [OrderStatus.FILLED] * 40
    assert broker.position("ACME").quantity == 40  # type: ignore[union-attr]
    assert broker.cash == pytest.approx(100_000 - 40 * 101.05)
    assert len(broker.trades(limit=None)) == 40


# ----------------------------------------------------------------------------- in-memory twin


def _script(target: Broker) -> list:
    steps = [
        ("ACME", "BUY", 10, 100.0, {"stop": 95.0, "take_profit": 120.0}),
        ("BETA", "BUY", 25, 41.37, {}),
        ("ACME", "BUY", 5, 104.2, {}),
        ("ACME", "SELL", 7, 111.9, {}),
        ("BETA", "SELL", 30, 40.0, {}),  # rejected: oversell
        ("BETA", "SELL", 25, 39.1, {}),
        ("GAMMA", "BUY", 10_000, 99.0, {}),  # rejected: cash
    ]
    orders = []
    for i, (symbol, side, qty, price, extra) in enumerate(steps):
        orders.append(
            target.submit_order(
                symbol,
                side,
                qty,
                price,
                reason=f"step {i}",
                timestamp=T0 + timedelta(minutes=i),
                **extra,
            )
        )
    return orders


def test_in_memory_broker_books_identically(broker: PaperBroker) -> None:
    memory = InMemoryBroker(100_000.0, CostModel.from_settings(broker.settings))
    assert isinstance(memory, Broker) and isinstance(broker, Broker)
    paper_orders = _script(broker)
    memory_orders = _script(memory)

    def strip(models: list) -> list[dict]:
        return [m.model_dump(exclude={"id", "order_id"}) for m in models]

    assert strip(paper_orders) == strip(memory_orders)
    assert strip(broker.trades(limit=None)) == strip(memory.trades(limit=None))
    prices = {"ACME": 108.0}
    assert broker.portfolio(prices).model_dump(exclude={"updated_at"}) == memory.portfolio(
        prices
    ).model_dump(exclude={"updated_at"})


def test_in_memory_broker_curve_and_stops() -> None:
    memory = InMemoryBroker(10_000.0)
    with pytest.raises(ValueError):
        InMemoryBroker(0)
    memory.submit_order("ACME", "BUY", 10, 100.0, stop=90.0, timestamp=T0)
    state = memory.position_state("ACME")
    assert state is not None and state.stop_price == 90.0
    memory.update_stops("ACME", stop=None, take_profit=130.0)
    assert memory.position("ACME").take_profit_price == 130.0  # type: ignore[union-attr]
    memory.record_snapshot({"ACME": 105.0}, timestamp=T0)
    memory.record_snapshot({"ACME": 95.0}, timestamp=T0 + timedelta(days=1))
    curve = memory.equity_curve()
    assert curve.index[0] == pd.Timestamp(T0)
    assert curve["equity"].iloc[0] == pytest.approx(10_000 - 1001.5 + 1050)
    assert memory.portfolio({"ACME": 95.0}).drawdown_pct > 0
    with pytest.raises(KeyError):
        memory.update_stops("NONE", stop=1.0)
    assert memory.reset().cash == 10_000.0 and memory.orders() == []
    assert math.isclose(memory.peak_equity, 10_000.0)


# ----------------------------------------------------------------------------- add-on levels


@pytest.fixture(params=["paper", "memory"])
def any_broker(request: pytest.FixtureRequest, broker: PaperBroker) -> Broker:
    """Both brokers: they share the accounting core, so they must agree on every rule."""
    return broker if request.param == "paper" else InMemoryBroker(100_000.0)


def test_add_on_never_loosens_the_stop(any_broker: Broker) -> None:
    any_broker.submit_order("ACME", Side.BUY, 10, 100.0, stop=95.0, take_profit=110.0)
    # The add-on's own stop (90) is lower: the 10 shares already held keep their 95 stop,
    # and the order reports the level that actually protects the position.
    order = any_broker.submit_order("ACME", Side.BUY, 10, 102.0, stop=90.0, take_profit=115.0)
    assert order.status is OrderStatus.FILLED
    assert (order.stop_price, order.take_profit_price) == (95.0, 115.0)
    position = any_broker.position("ACME")
    assert position is not None and position.quantity == 20
    assert position.stop_price == 95.0  # never lowered
    assert position.take_profit_price == 115.0  # the latest target wins


def test_add_on_can_tighten_the_stop(any_broker: Broker) -> None:
    any_broker.submit_order("ACME", Side.BUY, 10, 100.0, stop=95.0, take_profit=110.0)
    order = any_broker.submit_order("ACME", Side.BUY, 5, 104.0, stop=99.0)
    assert (order.stop_price, order.take_profit_price) == (99.0, 110.0)
    position = any_broker.position("ACME")
    assert position is not None
    assert (position.stop_price, position.take_profit_price) == (99.0, 110.0)


def test_add_on_gets_a_stop_when_the_position_had_none(any_broker: Broker) -> None:
    any_broker.submit_order("ACME", Side.BUY, 10, 100.0)
    any_broker.submit_order("ACME", Side.BUY, 5, 101.0, stop=97.0)
    assert any_broker.position("ACME").stop_price == 97.0  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("price", "level"),
    [(94.0, "stop-loss"), (95.0, "stop-loss"), (110.0, "take-profit"), (111.0, "take-profit")],
)
def test_no_add_on_while_a_level_is_breached(any_broker: Broker, price: float, level: str) -> None:
    any_broker.submit_order("ACME", Side.BUY, 10, 100.0, stop=95.0, take_profit=110.0)
    cash = any_broker.portfolio().cash
    order = any_broker.submit_order("ACME", Side.BUY, 5, price, stop=price * 0.9)
    assert order.status is OrderStatus.REJECTED
    assert level in (order.reject_reason or "") and "cannot be added to" in order.reject_reason
    position = any_broker.position("ACME")
    assert position is not None and position.quantity == 10
    # The breached stop was not replaced: an exit check still sees it.
    assert (position.stop_price, position.take_profit_price) == (95.0, 110.0)
    assert any_broker.portfolio().cash == cash


def test_sell_keeps_the_levels_of_the_remaining_shares(any_broker: Broker) -> None:
    any_broker.submit_order("ACME", Side.BUY, 10, 100.0, stop=95.0, take_profit=110.0)
    any_broker.submit_order("ACME", Side.SELL, 4, 101.0)
    position = any_broker.position("ACME")
    assert position is not None and position.quantity == 6
    assert (position.stop_price, position.take_profit_price) == (95.0, 110.0)
