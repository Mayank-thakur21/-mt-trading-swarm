"""RiskManagerAgent: sizing math, caps, vetoes, SELL rules and the risk score.

Reference history: 120 constant closes at 100 with highs/lows 1 % away, so every true
range is 2.0 and ATR(14) = 2.0 exactly; realised volatility is 0.

Hand-computed BUY at 100 with 100k equity (defaults: 1 % risk, 2×ATR stop, 2R target):
stop = 100 − 2 × 2 = 96, risk/share = 4, qty = floor(1 000 / 4) = 250, TP = 100 + 2 × 4 =
108, risk amount = 250 × 4 = 1 000. The default 20 % position cap allows 200 shares.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pytest

from trading_swarm.agents import RiskManagerAgent
from trading_swarm.agents.risk import FALLBACK_STOP_PCT, MIN_STOP_PCT
from trading_swarm.broker import CostModel, PaperBroker
from trading_swarm.broker.accounting import PositionState, build_snapshot
from trading_swarm.config import Settings
from trading_swarm.models import Action, Order, OrderStatus, PortfolioSnapshot, RiskAssessment, Side


def make_portfolio(
    cash: float = 100_000.0,
    positions: Sequence[tuple[str, int, float]] = (),
    *,
    peak: float | None = None,
    prices: dict[str, float] | None = None,
    initial: float = 100_000.0,
) -> PortfolioSnapshot:
    """Snapshot via the broker's own accounting (positions given as (symbol, qty, avg))."""
    states = [PositionState(symbol=s, quantity=q, avg_price=p) for s, q, p in positions]
    return build_snapshot(
        initial_capital=initial,
        cash=cash,
        positions=states,
        realized_pnl=0.0,
        peak_equity=peak if peak is not None else initial,
        prices=prices,
    )


@pytest.fixture
def flat_history(make_ohlcv: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    return make_ohlcv([100.0] * 120, spread=0.01)


def assess(
    settings: Settings,
    history: pd.DataFrame,
    portfolio: PortfolioSnapshot | None = None,
    *,
    action: Action = Action.BUY,
    price: float = 100.0,
    confidence: float = 1.0,
    symbol: str = "ACME",
) -> RiskAssessment:
    return RiskManagerAgent(settings).assess(
        symbol=symbol,
        price=price,
        action=action,
        score=0.6 if action is Action.BUY else -0.6,
        confidence=confidence,
        history=history,
        portfolio=portfolio or make_portfolio(),
    )


# ----------------------------------------------------------------------------- sizing


def test_fixed_fractional_sizing_math(
    make_settings: Callable[..., Settings], flat_history: pd.DataFrame
) -> None:
    risk = assess(make_settings(max_position_pct=0.5), flat_history)
    assert risk.approved and risk.action is Action.BUY
    assert risk.quantity == 250
    assert risk.entry_price == 100.0
    assert risk.stop_price == pytest.approx(96.0)
    assert risk.take_profit_price == pytest.approx(108.0)
    assert risk.risk_amount == pytest.approx(1_000.0)
    assert risk.position_value == pytest.approx(25_000.0)
    assert risk.checks == {
        "drawdown_ok": True,
        "volatility_ok": True,
        "stop_from_atr": True,
        "position_limit_ok": True,
        "cash_ok": True,
        "exposure_ok": True,
        "min_quantity_ok": True,
    }
    assert risk.reasons[0].startswith("approved BUY 250 sh")


def test_settings_drive_stop_and_target(
    make_settings: Callable[..., Settings], flat_history: pd.DataFrame
) -> None:
    settings = make_settings(
        max_position_pct=0.5, atr_stop_multiple=3.0, take_profit_r=1.5, risk_per_trade=0.02
    )
    risk = assess(settings, flat_history)
    assert risk.stop_price == pytest.approx(94.0)  # 100 − 3 × 2
    assert risk.take_profit_price == pytest.approx(109.0)  # 100 + 1.5 × 6
    assert risk.quantity == 333  # floor(2 000 / 6)


def test_capped_by_max_position(settings: Settings, flat_history: pd.DataFrame) -> None:
    risk = assess(settings, flat_history)
    assert risk.quantity == 200  # 20 % × 100k / 100
    assert any("capped 250→200" in reason for reason in risk.reasons)


def test_existing_position_counts_against_the_cap(
    settings: Settings, flat_history: pd.DataFrame
) -> None:
    portfolio = make_portfolio(cash=85_000.0, positions=[("ACME", 150, 100.0)])
    risk = assess(settings, flat_history, portfolio)
    assert risk.approved
    assert risk.quantity == 50  # 20k cap − 15k already held


def test_capped_by_cash_including_costs(
    make_settings: Callable[..., Settings], flat_history: pd.DataFrame
) -> None:
    settings = make_settings(max_position_pct=0.5, max_gross_exposure=1.0)
    portfolio = make_portfolio(cash=5_000.0, positions=[("OTHER", 950, 100.0)])
    risk = assess(settings, flat_history, portfolio)
    expected = CostModel.from_settings(settings).max_affordable_quantity(5_000.0, 100.0)
    assert expected == 49  # 5 000 / (100.05 × 1.0005)
    assert risk.quantity == expected
    assert any("by available cash" in reason for reason in risk.reasons)


def test_capped_by_gross_exposure_headroom(
    make_settings: Callable[..., Settings], flat_history: pd.DataFrame
) -> None:
    settings = make_settings(max_position_pct=0.5)
    portfolio = make_portfolio(cash=10_000.0, positions=[("OTHER", 900, 100.0)])  # 90 % exposed
    risk = assess(settings, flat_history, portfolio)
    assert risk.quantity == 50  # (95 % − 90 %) × 100k / 100
    assert any("gross exposure limit" in reason for reason in risk.reasons)


@pytest.mark.parametrize(
    ("confidence", "expected"), [(1.0, 250), (0.8, 200), (0.5, 125), (0.2, 125)]
)
def test_confidence_scaling_has_a_floor(
    make_settings: Callable[..., Settings],
    flat_history: pd.DataFrame,
    confidence: float,
    expected: int,
) -> None:
    risk = assess(make_settings(max_position_pct=0.5), flat_history, confidence=confidence)
    assert risk.quantity == expected


def test_atr_fallback_stop(
    make_settings: Callable[..., Settings], make_ohlcv: Callable[..., pd.DataFrame]
) -> None:
    short = make_ohlcv([100.0] * 10)  # ATR(14) needs 14 bars
    risk = assess(make_settings(max_position_pct=0.5), short)
    assert risk.checks["stop_from_atr"] is False
    assert risk.stop_price == pytest.approx(100.0 * (1 - FALLBACK_STOP_PCT))
    assert risk.take_profit_price == pytest.approx(110.0)
    assert risk.quantity == 200  # floor(1 000 / 5)
    assert any("fallback" in reason for reason in risk.reasons)
    assert any("volatility unavailable" in reason for reason in risk.reasons)


def test_minimum_stop_distance(
    make_settings: Callable[..., Settings], make_ohlcv: Callable[..., pd.DataFrame]
) -> None:
    tight = make_ohlcv([100.0] * 60, spread=0.0001)  # ATR = 0.02
    risk = assess(make_settings(max_position_pct=1.0, max_gross_exposure=1.0), tight)
    assert risk.stop_price == pytest.approx(100.0 * (1 - MIN_STOP_PCT))
    assert risk.quantity == 999  # risk size 2 000 sh capped by cash incl. costs


# ----------------------------------------------------------------------------- vetoes


def _assert_vetoed(risk: RiskAssessment, check: str, text: str) -> None:
    assert not risk.approved
    assert risk.action is Action.HOLD
    assert risk.quantity == 0
    assert risk.stop_price is None and risk.take_profit_price is None
    assert risk.checks[check] is False
    assert any(text in reason for reason in risk.reasons), risk.reasons


def test_drawdown_circuit_breaker(settings: Settings, flat_history: pd.DataFrame) -> None:
    portfolio = make_portfolio(peak=130_000.0)  # 100k equity → 23 % below peak
    assert portfolio.drawdown >= settings.max_drawdown_halt
    _assert_vetoed(assess(settings, flat_history, portfolio), "drawdown_ok", "halt")


def test_drawdown_below_halt_is_allowed(settings: Settings, flat_history: pd.DataFrame) -> None:
    portfolio = make_portfolio(peak=110_000.0)  # 9 % drawdown
    assert assess(settings, flat_history, portfolio).approved


def test_too_volatile(settings: Settings, make_ohlcv: Callable[..., pd.DataFrame]) -> None:
    wild = make_ohlcv([100.0 if i % 2 else 115.0 for i in range(80)])  # ~220 % annualised
    _assert_vetoed(assess(settings, wild), "volatility_ok", "too volatile")


def test_max_volatility_is_configurable(
    make_settings: Callable[..., Settings], make_ohlcv: Callable[..., pd.DataFrame]
) -> None:
    wild = make_ohlcv([100.0 if i % 2 else 115.0 for i in range(80)])
    assert assess(make_settings(max_volatility=5.0), wild).approved


def test_already_at_max_position(settings: Settings, flat_history: pd.DataFrame) -> None:
    portfolio = make_portfolio(cash=75_000.0, positions=[("ACME", 250, 100.0)])  # 25 % > 20 %
    _assert_vetoed(assess(settings, flat_history, portfolio), "position_limit_ok", "max position")


def test_gross_exposure_limit_reached(
    make_settings: Callable[..., Settings], flat_history: pd.DataFrame
) -> None:
    portfolio = make_portfolio(cash=4_000.0, positions=[("OTHER", 960, 100.0)])
    risk = assess(make_settings(max_position_pct=0.5), flat_history, portfolio)
    _assert_vetoed(risk, "exposure_ok", "gross exposure")


def test_insufficient_cash(settings: Settings, flat_history: pd.DataFrame) -> None:
    portfolio = make_portfolio(cash=50.0, positions=[("OTHER", 50, 100.0)])
    risk = assess(settings, flat_history, portfolio)
    _assert_vetoed(risk, "cash_ok", "insufficient cash")
    assert risk.checks["min_quantity_ok"] is False  # a zero cap leaves nothing to buy


def test_size_rounding_to_zero(settings: Settings, make_ohlcv: Callable[..., pd.DataFrame]) -> None:
    pricey = make_ohlcv([5_000.0] * 60, spread=0.1)  # ATR 1 000 → 2 000 risk per share
    risk = assess(settings, pricey, price=5_000.0)
    _assert_vetoed(risk, "min_quantity_ok", "rounds to zero")


def test_every_veto_is_listed(settings: Settings, make_ohlcv: Callable[..., pd.DataFrame]) -> None:
    wild = make_ohlcv([100.0 if i % 2 else 115.0 for i in range(80)])
    portfolio = make_portfolio(cash=50.0, positions=[("OTHER", 50, 100.0)], peak=20_000.0)
    risk = assess(settings, wild, portfolio)
    failed = {name for name, ok in risk.checks.items() if not ok}
    assert {"drawdown_ok", "volatility_ok", "cash_ok"} <= failed
    assert len(risk.reasons) >= 3


@pytest.mark.parametrize("price", [0.0, -5.0, float("nan"), float("inf")])
def test_invalid_price_is_vetoed(
    settings: Settings, flat_history: pd.DataFrame, price: float
) -> None:
    risk = assess(settings, flat_history, price=price)
    assert not risk.approved and risk.action is Action.HOLD
    assert risk.checks["price_valid"] is False


def test_non_positive_equity_is_vetoed(settings: Settings, flat_history: pd.DataFrame) -> None:
    risk = assess(settings, flat_history, make_portfolio(cash=0.0))
    assert not risk.approved
    assert risk.checks["equity_positive"] is False


# ----------------------------------------------------------------------------- SELL / HOLD


def test_sell_without_position_becomes_hold(settings: Settings, flat_history: pd.DataFrame) -> None:
    risk = assess(settings, flat_history, action=Action.SELL)
    assert not risk.approved
    assert risk.action is Action.HOLD
    assert risk.checks == {"position_exists": False}
    assert "no position to sell" in risk.reasons[0]


def test_sell_closes_the_full_position(settings: Settings, flat_history: pd.DataFrame) -> None:
    portfolio = make_portfolio(cash=90_000.0, positions=[("ACME", 120, 80.0)])
    risk = assess(settings, flat_history, portfolio, action=Action.SELL, price=100.0)
    assert risk.approved and risk.action is Action.SELL
    assert risk.quantity == 120
    assert risk.position_value == pytest.approx(12_000.0)
    assert risk.stop_price is None and risk.risk_amount == 0.0


def test_sell_is_allowed_during_drawdown_and_high_volatility(
    settings: Settings, make_ohlcv: Callable[..., pd.DataFrame]
) -> None:
    wild = make_ohlcv([100.0 if i % 2 else 115.0 for i in range(80)])
    portfolio = make_portfolio(cash=50_000.0, positions=[("ACME", 100, 200.0)], peak=200_000.0)
    risk = assess(settings, wild, portfolio, action=Action.SELL)
    assert risk.approved and risk.quantity == 100


def test_hold_proposes_nothing(settings: Settings, flat_history: pd.DataFrame) -> None:
    risk = assess(settings, flat_history, action=Action.HOLD)
    assert not risk.approved
    assert risk.action is Action.HOLD and risk.quantity == 0
    assert "HOLD band" in risk.reasons[0]


# ----------------------------------------------------------------------------- risk score


def test_risk_score_bounded_and_monotone(settings: Settings, flat_history: pd.DataFrame) -> None:
    calm = assess(settings, flat_history)
    stressed = assess(
        settings,
        flat_history,
        make_portfolio(cash=40_000.0, positions=[("OTHER", 600, 100.0)], peak=115_000.0),
    )
    assert 0.0 <= calm.risk_score <= 100.0
    assert 0.0 <= stressed.risk_score <= 100.0
    assert stressed.risk_score > calm.risk_score


def test_risk_score_terms(settings: Settings, make_ohlcv: Callable[..., pd.DataFrame]) -> None:
    """Fully utilised limits on every term give the maximum score."""
    rng = np.random.default_rng(0)
    wild = make_ohlcv(100 * np.exp(np.cumsum(rng.normal(0, 0.2, 80))))
    portfolio = make_portfolio(cash=1.0, positions=[("ACME", 1, 100_000.0)], peak=1_000_000.0)
    risk = assess(settings, wild, portfolio, action=Action.HOLD, price=100_000.0)
    assert risk.risk_score == pytest.approx(100.0)


# ----------------------------------------------------------------------------- integration


def test_uses_the_brokers_cost_model(
    settings: Settings, broker: PaperBroker, flat_history: pd.DataFrame
) -> None:
    agent = RiskManagerAgent(settings, cost_model=broker.cost_model)
    risk = agent.assess(
        symbol="ACME",
        price=100.0,
        action=Action.BUY,
        score=0.5,
        confidence=1.0,
        history=flat_history,
        portfolio=broker.portfolio(),
    )
    assert risk.approved and risk.quantity == 200
    order = broker.submit_order(
        "ACME",
        "BUY",
        risk.quantity,
        100.0,
        stop=risk.stop_price,
        take_profit=risk.take_profit_price,
    )
    assert order.is_filled
    position = broker.position("ACME")
    assert position is not None and position.stop_price == pytest.approx(96.0)


# ----------------------------------------------------------------------------- add-ons


def held_portfolio(
    quantity: int,
    *,
    stop: float | None,
    take_profit: float | None = None,
    price: float = 100.0,
) -> PortfolioSnapshot:
    """100k equity: ``quantity`` ACME shares bought at 100 (with levels), the rest in cash."""
    state = PositionState(
        symbol="ACME",
        quantity=quantity,
        avg_price=100.0,
        stop_price=stop,
        take_profit_price=take_profit,
    )
    return build_snapshot(
        initial_capital=100_000.0,
        cash=100_000.0 - quantity * 100.0,
        positions=[state],
        realized_pnl=0.0,
        peak_equity=100_000.0,
        prices={"ACME": price},
    )


def test_add_on_budget_includes_the_open_risk_of_held_shares(
    make_settings: Callable[..., Settings], flat_history: pd.DataFrame
) -> None:
    # 100 sh held with the same 96 stop: 100 × 4 = $400 of the $1 000 budget is already
    # at risk, leaving room for floor(600 / 4) = 150 more shares (not a fresh 250).
    settings = make_settings(max_position_pct=0.5)
    risk = assess(settings, flat_history, held_portfolio(100, stop=96.0))
    assert risk.approved and risk.quantity == 150
    assert risk.stop_price == pytest.approx(96.0)
    assert risk.checks["risk_budget_ok"] and risk.checks["existing_levels_ok"]
    # The whole position now risks exactly the budget to its stop.
    assert (100 + risk.quantity) * (100.0 - risk.stop_price) == pytest.approx(1_000.0)


def test_add_on_keeps_a_tighter_existing_stop(
    make_settings: Callable[..., Settings], flat_history: pd.DataFrame
) -> None:
    # Existing stop 98 is above the new ATR stop (96): the broker keeps 98 for every share,
    # so sizing uses 2/share: open risk 100 × 2 = 200, room floor(800 / 2) = 400 shares.
    settings = make_settings(max_position_pct=0.5)
    risk = assess(settings, flat_history, held_portfolio(100, stop=98.0))
    assert risk.approved and risk.quantity == 400
    assert risk.stop_price == pytest.approx(98.0)
    assert risk.take_profit_price == pytest.approx(104.0)  # 2R of the 2.0 stop distance
    assert risk.risk_amount == pytest.approx(800.0)
    assert risk.checks["kept_existing_stop"] is True
    assert any("keeps the position's tighter stop 98.00" in r for r in risk.reasons)


def test_add_on_vetoed_when_held_shares_use_the_whole_budget(
    make_settings: Callable[..., Settings], flat_history: pd.DataFrame
) -> None:
    settings = make_settings(max_position_pct=0.5)
    risk = assess(settings, flat_history, held_portfolio(250, stop=96.0))  # 250 × 4 = 1 000
    _assert_vetoed(risk, "risk_budget_ok", "already held use the position's risk budget")


@pytest.mark.parametrize(
    ("levels", "text"),
    [
        ({"stop": 100.0}, "at or below the open position's stop 100.00"),
        ({"stop": 101.0}, "at or below the open position's stop 101.00"),
        ({"stop": 90.0, "take_profit": 99.5}, "at or above the open position's take-profit"),
    ],
)
def test_add_on_vetoed_when_the_position_is_due_to_exit(
    make_settings: Callable[..., Settings],
    flat_history: pd.DataFrame,
    levels: dict[str, float],
    text: str,
) -> None:
    settings = make_settings(max_position_pct=0.5)
    risk = assess(settings, flat_history, held_portfolio(10, **levels))
    _assert_vetoed(risk, "existing_levels_ok", text)


def test_add_on_without_an_existing_stop_counts_held_risk_to_the_new_stop(
    settings: Settings, flat_history: pd.DataFrame
) -> None:
    # No stop yet: the add-on's 96 stop will protect all shares (150 × 4 = $600 open risk).
    risk = assess(settings, flat_history, held_portfolio(150, stop=None))
    assert risk.approved
    assert risk.quantity == 50  # the 20 % position cap binds before the remaining budget (100)
    assert risk.stop_price == pytest.approx(96.0)
    assert "kept_existing_stop" not in risk.checks


# ----------------------------------------------------------------------------- conviction


def _grow(settings: Settings, history: pd.DataFrame, confidence: float, rounds: int = 8) -> int:
    """Re-assess an unchanged BUY signal on the position it produced; return the shares."""
    held = 0
    for _ in range(rounds):
        portfolio = held_portfolio(held, stop=96.0) if held else make_portfolio()
        risk = assess(settings, history, portfolio, confidence=confidence)
        if not risk.approved:
            break
        held += risk.quantity
    return held


@pytest.mark.parametrize(
    ("max_position_pct", "confidence", "unscaled"),
    [
        (0.20, 0.5, 200),  # the position cap binds (20 % of 100k / 100)
        (0.50, 0.5, 250),  # the risk budget binds ($1 000 / $4)
        (0.20, 0.8, 200),
        (0.20, 1.0, 200),
    ],
)
def test_repeated_buy_signals_converge_to_the_confidence_scaled_size(
    make_settings: Callable[..., Settings],
    flat_history: pd.DataFrame,
    max_position_pct: float,
    confidence: float,
    unscaled: int,
) -> None:
    """Regression: scaling only each add-on let a 0.5-confidence signal pyramid 88 → 175 sh
    (2× its intended size, the unscaled cap) in ever smaller min-commission orders."""
    settings = make_settings(max_position_pct=max_position_pct)
    assert _grow(settings, flat_history, confidence) == math.floor(unscaled * confidence)


def test_add_on_tops_the_position_up_to_its_scaled_target(
    settings: Settings, flat_history: pd.DataFrame
) -> None:
    # Full-conviction size 200 sh (20 % cap); at 0.5 confidence the target is 100 sh.
    risk = assess(settings, flat_history, held_portfolio(40, stop=96.0), confidence=0.5)
    assert risk.approved and risk.quantity == 60
    assert any("position target scaled 200→100 sh" in r for r in risk.reasons)

    at_target = assess(settings, flat_history, held_portfolio(100, stop=96.0), confidence=0.5)
    _assert_vetoed(at_target, "min_quantity_ok", "already reach the position's confidence-scaled")
    above = assess(settings, flat_history, held_portfolio(150, stop=96.0), confidence=0.5)
    _assert_vetoed(above, "min_quantity_ok", "target of 100 sh")  # a BUY never trims


def test_tiny_add_ons_are_skipped(settings: Settings, flat_history: pd.DataFrame) -> None:
    # Target 200 sh at full conviction; 192 held → an 8 sh top-up < 10 % of 192 (= 20 sh).
    risk = assess(settings, flat_history, held_portfolio(192, stop=96.0))
    _assert_vetoed(risk, "min_add_on_ok", "below the minimum top-up of 20 sh")
    assert risk.checks["min_quantity_ok"] is True
    # A top-up of at least 10 % of the position goes through.
    ok = assess(settings, flat_history, held_portfolio(180, stop=96.0))
    assert ok.approved and ok.quantity == 20 and ok.checks["min_add_on_ok"] is True


# ----------------------------------------------------------------------------- re-entry


def _stop_exit(symbol: str = "ACME", quantity: int = 10, price: float = 99.5) -> Order:
    return Order(
        symbol=symbol,
        side=Side.SELL,
        quantity=quantity,
        status=OrderStatus.FILLED,
        fill_price=price,
        reason="stop_loss",
    )


def test_buy_right_after_a_protective_exit_is_vetoed(
    settings: Settings, flat_history: pd.DataFrame
) -> None:
    risk = RiskManagerAgent(settings).assess(
        symbol="ACME",
        price=100.0,
        action=Action.BUY,
        score=0.6,
        confidence=1.0,
        history=flat_history,
        portfolio=make_portfolio(),
        exit_order=_stop_exit(),
    )
    _assert_vetoed(
        risk, "reentry_ok", "the stop-loss closed the ACME position just before this assessment"
    )
    assert "sold 10 sh @ 99.50" in risk.reasons[0]
    # Without the exit the very same BUY is approved; an unfilled exit closed nothing.
    assert assess(settings, flat_history).approved
    rejected = _stop_exit().model_copy(update={"status": OrderStatus.REJECTED})
    assert (
        RiskManagerAgent(settings)
        .assess(
            symbol="ACME",
            price=100.0,
            action=Action.BUY,
            score=0.6,
            confidence=1.0,
            history=flat_history,
            portfolio=make_portfolio(),
            exit_order=rejected,
        )
        .approved
    )


def test_sell_after_a_protective_exit_reports_the_position_as_closed(
    settings: Settings, flat_history: pd.DataFrame
) -> None:
    exit_order = _stop_exit().model_copy(update={"reason": "take_profit"})
    risk = RiskManagerAgent(settings).assess(
        symbol="ACME",
        price=100.0,
        action=Action.SELL,
        score=-0.6,
        confidence=1.0,
        history=flat_history,
        portfolio=make_portfolio(),
        exit_order=exit_order,
    )
    assert not risk.approved and risk.action is Action.HOLD
    assert risk.checks == {"position_exists": False}
    assert risk.reasons == [
        "position already closed by its take-profit just before this assessment "
        "(sold 10 sh @ 99.50): nothing left to sell"
    ]
