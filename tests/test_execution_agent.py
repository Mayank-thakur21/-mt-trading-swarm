"""ExecutionAgent: entries at the provider's price, protective levels, and stop/target exits."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from trading_swarm.agents import ExecutionAgent
from trading_swarm.broker import CostModel, InMemoryBroker, OrderValidationError, PaperBroker
from trading_swarm.config import Settings
from trading_swarm.data import NoDataError, SyntheticProvider
from trading_swarm.models import (
    Action,
    Decision,
    ExitReason,
    Order,
    OrderStatus,
    RiskAssessment,
    Side,
)

T0 = datetime(2026, 9, 14, 20, 0, tzinfo=UTC)


class QuoteProvider:
    """Minimal provider: fixed latest prices; unknown symbols raise NoDataError."""

    def __init__(self, prices: dict[str, float]) -> None:
        self.prices = prices
        self.calls: list[str] = []

    def latest_price(self, symbol: str) -> float:
        self.calls.append(symbol)
        if symbol not in self.prices:
            raise NoDataError(f"no quote for {symbol}", symbol=symbol)
        return self.prices[symbol]


def make_decision(
    symbol: str = "ACME",
    price: float = 100.0,
    action: Action = Action.BUY,
    quantity: int = 10,
    *,
    stop: float | None = 96.0,
    take_profit: float | None = 108.0,
    approved: bool = True,
) -> Decision:
    risk = RiskAssessment(
        approved=approved,
        action=action,
        quantity=quantity,
        entry_price=price,
        stop_price=stop if action is Action.BUY else None,
        take_profit_price=take_profit if action is Action.BUY else None,
    )
    return Decision(
        symbol=symbol, price=price, action=action, score=0.55, confidence=0.75, risk=risk
    )


# ----------------------------------------------------------------------------- execute


def test_buy_fills_at_the_providers_latest_price(broker: PaperBroker, settings: Settings) -> None:
    provider = QuoteProvider({"ACME": 101.0})
    decision = make_decision()
    order = ExecutionAgent(broker, settings, provider).execute(decision)
    assert order is not None and order.status is OrderStatus.FILLED
    assert order.side is Side.BUY and order.quantity == 10
    assert order.reference_price == pytest.approx(101.0)  # re-quoted, not the decision price
    assert order.fill_price == pytest.approx(101.0 * 1.0005)  # 5 bps slippage
    assert order.commission == pytest.approx(1.0)  # minimum commission
    assert order.decision_id == decision.id
    assert "swarm BUY" in order.reason and "+0.55" in order.reason
    position = broker.position("ACME")
    assert position is not None
    assert (position.stop_price, position.take_profit_price) == (96.0, 108.0)
    assert provider.calls == ["ACME"]


def test_synthetic_provider_quote(
    broker: PaperBroker, settings: Settings, provider: SyntheticProvider
) -> None:
    latest = provider.latest_price("BULL")
    decision = make_decision("BULL", latest, stop=latest * 0.95, take_profit=latest * 1.1)
    order = ExecutionAgent(broker, settings, provider).execute(decision)
    assert order is not None and order.is_filled
    assert order.reference_price == pytest.approx(latest, abs=1e-4)


def test_without_provider_uses_the_decision_price(broker: PaperBroker, settings: Settings) -> None:
    order = ExecutionAgent(broker, settings).execute(make_decision(price=100.0))
    assert order is not None and order.reference_price == pytest.approx(100.0)


def test_failed_requote_falls_back_to_decision_price(
    broker: PaperBroker, settings: Settings
) -> None:
    order = ExecutionAgent(broker, settings, QuoteProvider({})).execute(make_decision(price=100.0))
    assert order is not None and order.is_filled
    assert order.reference_price == pytest.approx(100.0)


@pytest.mark.parametrize(
    "decision",
    [
        make_decision(approved=False),
        make_decision(action=Action.HOLD, quantity=0),
        make_decision(quantity=0),
    ],
    ids=["vetoed", "hold", "zero-quantity"],
)
def test_nothing_to_execute(broker: PaperBroker, settings: Settings, decision: Decision) -> None:
    assert ExecutionAgent(broker, settings).execute(decision) is None
    assert broker.orders() == []


def test_sell_closes_position(broker: PaperBroker, settings: Settings) -> None:
    broker.submit_order("ACME", Side.BUY, 20, 100.0, stop=90.0)
    provider = QuoteProvider({"ACME": 110.0})
    order = ExecutionAgent(broker, settings, provider).execute(
        make_decision(action=Action.SELL, quantity=20, price=110.0)
    )
    assert order is not None and order.is_filled
    assert order.side is Side.SELL and order.quantity == 20
    assert order.realized_pnl is not None and order.realized_pnl > 0
    assert broker.position("ACME") is None


def test_sell_of_shrunken_position_sells_what_is_held(
    broker: PaperBroker, settings: Settings
) -> None:
    broker.submit_order("ACME", Side.BUY, 5, 100.0)
    order = ExecutionAgent(broker, settings).execute(make_decision(action=Action.SELL, quantity=20))
    assert order is not None and order.is_filled and order.quantity == 5


def test_sell_without_position_is_a_rejected_order(broker: PaperBroker, settings: Settings) -> None:
    order = ExecutionAgent(broker, settings).execute(make_decision(action=Action.SELL, quantity=5))
    assert order is not None and order.status is OrderStatus.REJECTED
    assert "no position" in (order.reject_reason or "")


def test_insufficient_cash_is_a_rejected_order(broker: PaperBroker, settings: Settings) -> None:
    order = ExecutionAgent(broker, settings).execute(make_decision(quantity=5_000))
    assert order is not None and order.status is OrderStatus.REJECTED
    assert "insufficient cash" in (order.reject_reason or "")
    assert broker.position("ACME") is None


def test_levels_reanchored_when_quote_moved_past_the_stop(
    broker: PaperBroker, settings: Settings
) -> None:
    provider = QuoteProvider({"ACME": 95.0})  # below the 96 stop computed at 100
    order = ExecutionAgent(broker, settings, provider).execute(make_decision())
    assert order is not None and order.is_filled
    assert order.stop_price == pytest.approx(91.0)  # same 4-point distance below 95
    assert order.take_profit_price == pytest.approx(103.0)


def test_levels_reanchored_when_quote_moved_past_the_target(
    broker: PaperBroker, settings: Settings
) -> None:
    provider = QuoteProvider({"ACME": 110.0})
    order = ExecutionAgent(broker, settings, provider).execute(make_decision())
    assert order is not None and order.is_filled
    assert order.stop_price == pytest.approx(106.0)
    assert order.take_profit_price == pytest.approx(118.0)


def test_valid_levels_are_kept_when_quote_moves_a_little(
    broker: PaperBroker, settings: Settings
) -> None:
    order = ExecutionAgent(broker, settings, QuoteProvider({"ACME": 100.5})).execute(
        make_decision()
    )
    assert order is not None
    assert (order.stop_price, order.take_profit_price) == (96.0, 108.0)


def test_malformed_order_returns_none(settings: Settings) -> None:
    class Strict(InMemoryBroker):
        def submit_order(self, *args: Any, **kwargs: Any) -> Order:
            raise OrderValidationError("bad request")

    agent = ExecutionAgent(Strict(10_000.0), settings)
    assert agent.execute(make_decision()) is None


def test_in_memory_broker_and_timestamp(settings: Settings) -> None:
    memory = InMemoryBroker(100_000.0, CostModel.from_settings(settings))
    order = ExecutionAgent(memory, settings).execute(make_decision(), timestamp=T0)
    assert order is not None and order.is_filled
    assert order.requested_at == T0 and order.filled_at == T0


# ----------------------------------------------------------------------------- exits


def _open(broker: Any, symbol: str, stop: float | None, take_profit: float | None) -> None:
    order = broker.submit_order(symbol, Side.BUY, 10, 100.0, stop=stop, take_profit=take_profit)
    assert order.is_filled


def test_stop_loss_exit(broker: PaperBroker, settings: Settings) -> None:
    _open(broker, "ACME", 95.0, 110.0)
    exits = ExecutionAgent(broker, settings).enforce_exits({"ACME": 94.5})
    assert len(exits) == 1
    order = exits[0]
    assert order.is_filled and order.side is Side.SELL and order.quantity == 10
    assert order.reason == ExitReason.STOP_LOSS.value == "stop_loss"
    assert order.reference_price == pytest.approx(94.5)
    assert order.realized_pnl is not None and order.realized_pnl < 0
    assert broker.position("ACME") is None


def test_take_profit_exit(broker: PaperBroker, settings: Settings) -> None:
    _open(broker, "ACME", 95.0, 110.0)
    (order,) = ExecutionAgent(broker, settings).enforce_exits({"ACME": 110.0})
    assert order.reason == "take_profit"
    assert order.realized_pnl is not None and order.realized_pnl > 0


def test_no_exit_between_levels(broker: PaperBroker, settings: Settings) -> None:
    _open(broker, "ACME", 95.0, 110.0)
    assert ExecutionAgent(broker, settings).enforce_exits({"ACME": 100.0}) == []
    assert broker.position("ACME") is not None


def test_exits_skip_unpriced_and_invalid_quotes(broker: PaperBroker, settings: Settings) -> None:
    _open(broker, "AAA", 95.0, None)
    _open(broker, "BBB", 95.0, None)
    _open(broker, "CCC", 95.0, None)
    exits = ExecutionAgent(broker, settings).enforce_exits({"BBB": float("nan"), "CCC": 0.0})
    assert exits == []
    assert len(broker.positions()) == 3


def test_exits_without_levels_never_fire(broker: PaperBroker, settings: Settings) -> None:
    _open(broker, "ACME", None, None)
    assert ExecutionAgent(broker, settings).enforce_exits({"ACME": 1.0}) == []


def test_exit_price_keys_are_case_insensitive(broker: PaperBroker, settings: Settings) -> None:
    _open(broker, "ACME", 95.0, None)
    (order,) = ExecutionAgent(broker, settings).enforce_exits({" acme ": 90.0})
    assert order.symbol == "ACME"


def test_multiple_exits_in_one_pass(broker: PaperBroker, settings: Settings) -> None:
    _open(broker, "AAA", 95.0, 110.0)
    _open(broker, "BBB", 95.0, 110.0)
    _open(broker, "CCC", 95.0, 110.0)
    exits = ExecutionAgent(broker, settings).enforce_exits(
        {"AAA": 90.0, "BBB": 120.0, "CCC": 100.0}
    )
    assert {o.symbol: o.reason for o in exits} == {"AAA": "stop_loss", "BBB": "take_profit"}
    assert [p.symbol for p in broker.positions()] == ["CCC"]


def test_one_failing_exit_does_not_block_the_others(settings: Settings) -> None:
    class Flaky(InMemoryBroker):
        def submit_order(self, symbol: str, *args: Any, **kwargs: Any) -> Order:
            if symbol == "AAA" and kwargs.get("reason") == "stop_loss":
                raise RuntimeError("exchange hiccup")
            return super().submit_order(symbol, *args, **kwargs)

    memory = Flaky(100_000.0)
    _open(memory, "AAA", 95.0, None)
    _open(memory, "BBB", 95.0, None)
    exits = ExecutionAgent(memory, settings).enforce_exits({"AAA": 90.0, "BBB": 90.0}, timestamp=T0)
    assert [o.symbol for o in exits] == ["BBB"]
    assert exits[0].filled_at == T0
    assert memory.position("AAA") is not None  # retried on the next scan
