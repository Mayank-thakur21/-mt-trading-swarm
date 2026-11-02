"""Backtester: no look-ahead, next-open fills, protective exits, the buy-and-hold benchmark,
multi-symbol reservation, rebalance schedules, point-in-time sentiment, serialisation and
the backtest CLI (:mod:`trading_swarm.backtest.cli`, wrapped by ``scripts.run_backtest``).

Mechanics tests use :class:`ScriptedTrend` (votes from a timestamp script) on hand-built
bars, so every fill price can be computed by hand; the real ``TrendAnalyzerAgent`` is
exercised on the synthetic BULL / BEAR scenarios over short windows to keep the suite fast.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.run_backtest as cli_wrapper
import trading_swarm.backtest.cli as cli
from trading_swarm.agents import RiskManagerAgent, TrendAnalyzerAgent
from trading_swarm.analytics import PerformanceMetrics
from trading_swarm.backtest import (
    EQUITY_CURVE_COLUMNS,
    NEWS_CUTOFF,
    SIGNAL_LOG_COLUMNS,
    BacktestConfigError,
    Backtester,
    BacktestResult,
    rebalance_mask,
    reserve_order,
)
from trading_swarm.broker import CostModel, InMemoryBroker
from trading_swarm.config import Settings
from trading_swarm.data import (
    InvalidSymbolError,
    NoDataError,
    ProviderUnavailableError,
    SyntheticProvider,
)
from trading_swarm.models import (
    Action,
    AgentSignal,
    ExitReason,
    OrderStatus,
    RiskAssessment,
    Side,
)

TEST_END = "2026-09-14"  # same pin as conftest.TEST_END_DATE
SLIPPAGE = 5.0 / 10_000  # default SLIPPAGE_BPS
#: Hand-built market: the window starts at bar 60, the scripted BUY is decided at bar 65,
#: fills at bar 66's open, and the scenario event happens on bar 68.
WINDOW_START, ENTRY, FILL, EVENT = 60, 65, 66, 68


# =============================================================================== doubles


class ScriptedTrend:
    """Trend-agent double: votes ``+1``/``-1`` (confidence 1) on scripted bars, abstains
    elsewhere, and records the history it was given at every call."""

    name = "trend"

    def __init__(self, script: dict[Any, float] | None = None, *, default: float | None = None):
        self.script = script or {}
        self.default = default
        self.calls: list[tuple[str, pd.Timestamp, pd.DataFrame]] = []

    def analyze(self, symbol: str, history: pd.DataFrame) -> AgentSignal:
        last = history.index[-1]
        self.calls.append((symbol, last, history))
        value = self.script.get((symbol, last), self.script.get(last, self.default))
        if value is None:
            return AgentSignal.neutral("trend", "scripted: no view")
        return AgentSignal(agent="trend", signal=value, confidence=1.0, rationale="scripted")


class BrokenTrend:
    name = "trend"

    def analyze(self, symbol: str, history: pd.DataFrame) -> AgentSignal:
        raise RuntimeError("indicator blew up")


class SpyRisk(RiskManagerAgent):
    """The real risk agent, recording the history and account it was shown."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.seen: list[tuple[str, pd.Timestamp, Any]] = []

    def assess(self, **kwargs: Any):  # type: ignore[override]
        self.seen.append((kwargs["symbol"], kwargs["history"].index[-1], kwargs["portfolio"]))
        return super().assess(**kwargs)


# =============================================================================== fixtures


@pytest.fixture
def gentle_frame(make_ohlcv: Callable[..., pd.DataFrame]) -> pd.DataFrame:
    """160 bars rising 0.1 % a day (opens = previous close, ±1 % ranges)."""
    return make_ohlcv(100.0 * 1.001 ** np.arange(160), start="2024-01-02")


def serve(frames: dict[str, pd.DataFrame]) -> SyntheticProvider:
    provider = SyntheticProvider(seed=42, end_date=TEST_END)
    for symbol, frame in frames.items():
        provider.set_history(symbol, frame)
    return provider


def set_bar(frame: pd.DataFrame, i: int, **ohlc: float) -> None:
    for column, value in ohlc.items():
        frame.iloc[i, frame.columns.get_loc(column.capitalize())] = value


def run_scripted(
    frame: pd.DataFrame,
    settings: Settings,
    script: dict[Any, float],
    **kwargs: Any,
) -> BacktestResult:
    ts = frame.index
    trend = ScriptedTrend(script)
    backtester = Backtester(serve({"XYZ": frame}), settings, trend=trend)
    return backtester.run("XYZ", ts[WINDOW_START], ts[-1], warmup=WINDOW_START, **kwargs)


def strip_ids(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k not in {"id", "order_id"}}


# =============================================================================== no look-ahead


def test_no_look_ahead_mutating_future_bars_leaves_the_past_identical(
    settings: Settings,
) -> None:
    """Rewrite every bar after ``cut`` (a crash path); everything up to ``cut`` must match."""
    base = SyntheticProvider(seed=42, end_date=TEST_END)
    full = base.history("BULL", start="2021-01-01", end=TEST_END)
    cut = pd.Timestamp("2026-04-15", tz="UTC")
    future = full.index > cut
    shock = np.exp(np.cumsum(np.random.default_rng(7).normal(-0.01, 0.03, future.sum())))
    mutated = full.copy()
    for column in ("Open", "High", "Low", "Close"):
        mutated.loc[future, column] = mutated.loc[future, column] * shock
    altered = SyntheticProvider(seed=42, end_date=TEST_END)
    altered.set_history("BULL", mutated)

    original = Backtester(base, settings).run("BULL", "2026-03-02", "2026-05-29")
    rewritten = Backtester(altered, settings).run("BULL", "2026-03-02", "2026-05-29")

    def log_until(result: BacktestResult) -> pd.DataFrame:
        log = result.signal_log
        return log[log["timestamp"] <= cut].reset_index(drop=True)

    pd.testing.assert_frame_equal(log_until(original), log_until(rewritten))
    pd.testing.assert_frame_equal(original.equity_curve.loc[:cut], rewritten.equity_curve.loc[:cut])

    def decisions_until(result: BacktestResult) -> list[tuple[Any, ...]]:
        return [
            (d.timestamp, d.action, d.score, d.confidence, d.risk.quantity, d.risk.stop_price)
            for d in result.decisions
            if d.timestamp <= cut
        ]

    def trades_until(result: BacktestResult) -> list[dict[str, Any]]:
        return [strip_ids(t.model_dump()) for t in result.trades if t.timestamp <= cut]

    assert decisions_until(original) == decisions_until(rewritten)
    assert trades_until(original) == trades_until(rewritten)
    assert trades_until(original), "the scenario must trade before the cut to be meaningful"
    # ... and the mutation really changed what happened afterwards.
    assert not original.signal_log.equals(rewritten.signal_log)
    assert original.metrics.end_equity != rewritten.metrics.end_equity


def test_agents_only_see_the_live_window_ending_at_the_decision_bar(
    provider: SyntheticProvider, settings: Settings
) -> None:
    trend = ScriptedTrend(default=1.0)  # BUY proposal on every bar → the risk agent runs too
    risk = SpyRisk(settings)
    result = Backtester(provider, settings, trend=trend, risk=risk).run(
        "BULL", "2026-06-01", "2026-07-31"
    )
    window = result.equity_curve.index
    decided = [ts for _, ts, _ in trend.calls]
    assert decided == list(window[:-1]), "one decision per bar, none on the last bar"
    for _, ts, history in trend.calls:
        live = provider.history("BULL", end=ts, period="1y")  # what /analyze would see
        assert history.index.equals(live.index)
        assert history.index.max() == ts
        pd.testing.assert_frame_equal(history, live)
    assert len(trend.calls[0][2]) >= 200  # a full live-sized window from the first bar on
    assert [ts for _, ts, _ in risk.seen] == decided
    for _, ts, portfolio in risk.seen:
        assert portfolio.updated_at == ts  # account marked at that bar's close


# =============================================================================== fills


def test_orders_fill_at_the_next_open_with_shared_costs(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    ts = gentle_frame.index
    result = run_scripted(gentle_frame, settings, {ts[ENTRY]: 1.0})

    [decision] = result.decisions
    assert decision.timestamp == ts[ENTRY]
    assert decision.price == pytest.approx(gentle_frame["Close"].iloc[ENTRY], abs=1e-4)
    order = decision.order
    assert order is not None and order.status is OrderStatus.FILLED and decision.executed
    assert order.filled_at == ts[FILL]
    open_price = gentle_frame["Open"].iloc[FILL]
    assert order.reference_price == pytest.approx(open_price, abs=1e-4)
    assert order.fill_price == pytest.approx(open_price * (1 + SLIPPAGE), abs=1e-4)
    costs = CostModel.from_settings(settings)
    assert order.commission == pytest.approx(
        costs.commission(open_price * (1 + SLIPPAGE) * order.quantity), abs=0.01
    )
    assert order.stop_price == decision.risk.stop_price
    assert order.take_profit_price == decision.risk.take_profit_price
    assert order.decision_id == decision.id
    assert "Executed: BUY" in decision.summary

    entry, *later = result.trades
    assert entry.timestamp == ts[FILL] and entry.side is Side.BUY
    assert entry.order_id == order.id and entry.price == order.fill_price
    curve = result.equity_curve
    assert curve.loc[ts[ENTRY], "positions_value"] == 0.0  # nothing held at the decision close
    assert curve.loc[ts[FILL], "positions_value"] == pytest.approx(
        order.quantity * gentle_frame["Close"].iloc[FILL]
    )
    # The steady 0.1 %/day climb eventually reaches the 2R target; nothing else fires.
    assert [t.reason for t in later] == [ExitReason.TAKE_PROFIT.value]
    assert later[0].timestamp > ts[FILL + 20]


@pytest.mark.parametrize(
    ("bar", "reason", "reference"),
    [
        # Low trades through the stop, open above it → fill at the stop.
        ({"open": 1.0, "high": 1.001, "low": 0.80, "close": 0.90}, ExitReason.STOP_LOSS, "stop"),
        # Gap down through the stop → fill at the (worse) open.
        ({"open": 0.85, "high": 0.86, "low": 0.84, "close": 0.85}, ExitReason.STOP_LOSS, "open"),
        # High trades through the target → fill at the target.
        ({"open": 1.0, "high": 1.25, "low": 0.999, "close": 1.2}, ExitReason.TAKE_PROFIT, "tp"),
        # Gap up through the target → fill at the (better) open.
        ({"open": 1.3, "high": 1.31, "low": 1.29, "close": 1.3}, ExitReason.TAKE_PROFIT, "open"),
        # Both levels inside one bar: the stop is assumed to have traded first.
        ({"open": 1.0, "high": 1.25, "low": 0.80, "close": 1.0}, ExitReason.STOP_LOSS, "stop"),
    ],
    ids=["stop-inside-bar", "gap-through-stop", "target-inside-bar", "gap-over-target", "both"],
)
def test_protective_exits_use_the_bar_range_and_gap_at_the_open(
    gentle_frame: pd.DataFrame,
    settings: Settings,
    bar: dict[str, float],
    reason: ExitReason,
    reference: str,
) -> None:
    prev = gentle_frame["Close"].iloc[EVENT - 1]
    set_bar(gentle_frame, EVENT, **{k: v * prev for k, v in bar.items()})
    ts = gentle_frame.index
    result = run_scripted(gentle_frame, settings, {ts[ENTRY]: 1.0})

    [decision] = result.decisions
    assert decision.order is not None
    stop, target = decision.order.stop_price, decision.order.take_profit_price
    assert stop is not None and target is not None
    exits = [t for t in result.trades if t.side is Side.SELL]
    assert len(exits) == 1, "levels must not trigger before the event bar"
    [exit_trade] = exits
    assert exit_trade.timestamp == ts[EVENT]
    assert exit_trade.reason == reason.value
    assert exit_trade.quantity == decision.order.quantity
    level = {"stop": stop, "tp": target, "open": bar["open"] * prev}[reference]
    assert exit_trade.price == pytest.approx(level * (1 - SLIPPAGE), abs=1e-4)

    # Realised P&L is net of both legs' slippage and commission.
    costs = CostModel.from_settings(settings)
    buy = costs.quote(Side.BUY, exit_trade.quantity, decision.order.reference_price)
    sell = costs.quote(Side.SELL, exit_trade.quantity, level)
    expected = sell.notional - sell.commission - (buy.notional + buy.commission)
    assert exit_trade.realized_pnl == pytest.approx(expected, abs=0.02)
    assert (exit_trade.realized_pnl < 0) is (reason is ExitReason.STOP_LOSS)
    assert result.stats["exits"][reason.value] == 1
    assert result.final_portfolio is not None and result.final_portfolio.positions == []


def test_sell_signal_closes_the_position_at_the_following_open(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    ts = gentle_frame.index
    result = run_scripted(gentle_frame, settings, {ts[ENTRY]: 1.0, ts[ENTRY + 5]: -1.0})

    buy, sell = result.decisions
    assert (buy.action, sell.action) == (Action.BUY, Action.SELL)
    assert sell.risk.quantity == buy.order.quantity  # type: ignore[union-attr]
    assert sell.order is not None and sell.order.filled_at == ts[ENTRY + 6]
    assert sell.order.reason.startswith("swarm SELL")
    assert sell.order.fill_price == pytest.approx(
        gentle_frame["Open"].iloc[ENTRY + 6] * (1 - SLIPPAGE), abs=1e-4
    )
    assert result.stats["exits"] == {"stop_loss": 0, "take_profit": 0, "signal": 1}
    assert result.metrics.num_trades == 1


def test_buy_made_unaffordable_by_a_gap_up_is_trimmed_to_available_cash(
    gentle_frame: pd.DataFrame, make_settings: Callable[..., Settings]
) -> None:
    settings = make_settings(risk_per_trade=0.1, max_position_pct=1.0, max_gross_exposure=1.0)
    set_bar(gentle_frame, FILL, open=gentle_frame["Close"].iloc[ENTRY] * 1.05)
    ts = gentle_frame.index
    result = run_scripted(gentle_frame, settings, {ts[ENTRY]: 1.0})

    [decision] = result.decisions
    assert decision.risk.checks["cash_ok"] is True
    order = decision.order
    assert order is not None and order.status is OrderStatus.FILLED
    assert 0 < order.quantity < decision.risk.quantity
    assert result.stats["orders_trimmed"] == 1
    assert result.final_portfolio is not None and result.final_portfolio.cash >= 0


# =============================================================================== portfolio


def test_same_bar_buys_are_reserved_so_exposure_limits_hold(
    gentle_frame: pd.DataFrame, make_settings: Callable[..., Settings]
) -> None:
    settings = make_settings(max_gross_exposure=0.30)
    ts = gentle_frame.index
    trend = ScriptedTrend({("AAA", ts[ENTRY]): 1.0, ("BBB", ts[ENTRY]): 1.0})
    provider = serve({"AAA": gentle_frame, "BBB": gentle_frame.copy()})
    result = Backtester(provider, settings, trend=trend).run(
        ["AAA", "BBB"], ts[WINDOW_START], ts[-1], warmup=WINDOW_START
    )

    first, second = result.decisions
    assert (first.symbol, second.symbol) == ("AAA", "BBB")
    assert second.risk.quantity < first.risk.quantity  # the headroom was already reserved
    assert any("gross exposure" in reason for reason in second.risk.reasons)
    row = result.equity_curve.loc[ts[FILL]]
    assert row["positions_value"] / row["equity"] <= 0.30 + 0.002
    # Equal-weight benchmark: half of the capital in each symbol.
    assert result.initial_capital * 0.99 < result.equity_curve["benchmark"].iloc[0]


def test_exits_are_assessed_before_entries_on_the_same_bar(
    gentle_frame: pd.DataFrame, make_settings: Callable[..., Settings]
) -> None:
    settings = make_settings(max_gross_exposure=0.25)
    ts = gentle_frame.index
    script = {("AAA", ts[ENTRY]): 1.0, ("AAA", ts[ENTRY + 5]): -1.0, ("BBB", ts[ENTRY + 5]): 1.0}
    provider = serve({"AAA": gentle_frame, "BBB": gentle_frame.copy()})
    result = Backtester(provider, settings, trend=ScriptedTrend(script)).run(
        ["BBB", "AAA"], ts[WINDOW_START], ts[-1], warmup=WINDOW_START
    )

    aaa_buy, aaa_sell, bbb_buy = result.decisions
    assert (aaa_sell.symbol, aaa_sell.action) == ("AAA", Action.SELL)
    assert bbb_buy.symbol == "BBB"
    # The SELL freed AAA's 20 % before BBB was sized: BBB gets a full position, not the
    # 5 % that was left while AAA was still held.
    assert bbb_buy.risk.quantity >= 0.9 * aaa_buy.risk.quantity
    assert bbb_buy.order is not None and bbb_buy.order.is_filled


def test_reserve_order_applies_cash_and_exposure_of_pending_orders() -> None:
    costs = CostModel()
    snapshot = InMemoryBroker(10_000.0, costs).portfolio()
    bought = reserve_order(snapshot, "AAA", Side.BUY, 20, 100.0, costs)
    assert bought.cash == pytest.approx(10_000.0 - costs.buy_cash_required(20, 100.0))
    assert bought.positions_value == pytest.approx(2_000.0)
    assert bought.gross_exposure == pytest.approx(2_000.0 / bought.equity)
    held = bought.position("AAA")
    assert held is not None and held.quantity == 20
    sold = reserve_order(bought, "AAA", Side.SELL, 20, 100.0, costs)
    assert sold.position("AAA") is None and sold.positions_value == 0.0
    assert sold.cash == pytest.approx(bought.cash + costs.quote(Side.SELL, 20, 100.0).cash_delta)
    partly = reserve_order(bought, "AAA", Side.SELL, 5, 100.0, costs)
    assert partly.position("AAA").quantity == 15  # type: ignore[union-attr]
    assert partly.positions_value == pytest.approx(1_500.0)
    assert reserve_order(snapshot, "ZZZ", Side.SELL, 5, 10.0, costs) is snapshot  # nothing held


# =============================================================================== benchmark


def test_benchmark_is_buy_and_hold_from_the_first_open_with_costs(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    result = run_scripted(gentle_frame, settings, {})  # the strategy never trades

    costs = CostModel.from_settings(settings)
    first_open = gentle_frame["Open"].iloc[WINDOW_START]
    quantity = costs.max_affordable_quantity(100_000.0, first_open)
    cash_left = 100_000.0 - costs.buy_cash_required(quantity, first_open)
    curve = result.equity_curve
    assert curve["benchmark"].iloc[0] == pytest.approx(
        cash_left + quantity * gentle_frame["Close"].iloc[WINDOW_START]
    )
    final = cash_left + quantity * gentle_frame["Close"].iloc[-1]
    assert curve["benchmark"].iloc[-1] == pytest.approx(final)
    bench = result.benchmark_metrics
    # Measured from the money invested, not from the first (close-marked) point.
    assert bench.start_equity == 100_000.0
    assert bench.total_return_pct == pytest.approx((final / 100_000.0 - 1) * 100)
    assert bench.exposure_pct == 100.0
    assert (curve["benchmark_drawdown"] <= 0).all()

    assert (curve["equity"] == 100_000.0).all()
    assert result.metrics.total_return_pct == 0.0
    assert result.metrics.num_trades == 0 and result.trades == []
    assert result.excess_return_pct == pytest.approx(-bench.total_return_pct)


# =============================================================================== schedules


def test_rebalance_mask_marks_the_first_bar_of_each_period() -> None:
    calendar = pd.bdate_range("2026-01-01", "2026-03-06", tz="UTC")
    assert rebalance_mask(calendar, "daily").all()
    weekly = calendar[rebalance_mask(calendar, "weekly")]
    assert weekly[0] == calendar[0] and (weekly[1:].day_of_week == 0).all()
    assert len(weekly) == len({tuple(ts.isocalendar())[:2] for ts in calendar})
    monthly = calendar[rebalance_mask(calendar, "monthly")]
    assert [ts.strftime("%Y-%m-%d") for ts in monthly] == ["2026-01-01", "2026-02-02", "2026-03-02"]
    with pytest.raises(BacktestConfigError):
        rebalance_mask(calendar, "hourly")


def test_weekly_rebalance_decides_weekly_but_enforces_stops_every_bar(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    ts = gentle_frame.index
    entry = next(i for i in range(ENTRY, len(ts)) if ts[i].day_of_week == 0)  # a Monday
    event = entry + 3  # the Thursday after the Tuesday fill
    prev = gentle_frame["Close"].iloc[event - 1]
    set_bar(gentle_frame, event, open=prev, high=prev, low=prev * 0.8, close=prev * 0.9)
    trend = ScriptedTrend({ts[entry]: 1.0})
    result = Backtester(serve({"XYZ": gentle_frame}), settings, trend=trend).run(
        "XYZ", ts[WINDOW_START], ts[-1], warmup=WINDOW_START, rebalance="weekly"
    )

    window = ts[WINDOW_START:]
    expected = [
        day
        for k, day in enumerate(window[:-1])
        if k == 0 or day.isocalendar()[:2] != window[k - 1].isocalendar()[:2]
    ]
    assert [day for _, day, _ in trend.calls] == expected
    assert result.parameters["rebalance"] == "weekly"
    exits = [t for t in result.trades if t.reason == ExitReason.STOP_LOSS.value]
    assert [t.timestamp for t in exits] == [ts[event]]


# =============================================================================== real agents


def test_bull_scenario_end_to_end(provider: SyntheticProvider, settings: Settings) -> None:
    result = Backtester(provider, settings).run("BULL", "2026-03-02", "2026-04-30")

    assert result.symbols == ["BULL"]
    assert (result.start.isoformat(), result.end.isoformat()) == ("2026-03-02", "2026-04-30")
    assert result.trades and result.decisions
    assert all(d.action in (Action.BUY, Action.SELL) for d in result.decisions)
    assert all(d.executed and "Executed:" in d.summary for d in result.decisions)
    times = [t.timestamp for t in result.trades]
    assert times == sorted(times)
    assert result.metrics.total_return_pct is not None and result.metrics.total_return_pct > 0
    assert result.benchmark_metrics.total_return_pct > 0  # type: ignore[operator]
    assert result.metrics.exposure_pct is not None and result.metrics.exposure_pct > 0
    assert isinstance(result.metrics, PerformanceMetrics)

    curve = result.equity_curve
    assert list(curve.columns) == list(EQUITY_CURVE_COLUMNS)
    assert len(curve) == result.stats["bars"] and (curve["drawdown"] <= 0).all()
    final = result.final_portfolio
    assert final is not None
    assert final.equity == pytest.approx(curve["equity"].iloc[-1], abs=0.01)
    assert final.realized_pnl + final.unrealized_pnl == pytest.approx(final.total_return, abs=0.05)
    assert result.parameters["warmup_bars"]["BULL"] >= 200
    assert result.warnings == []

    # The vote at any bar equals a live analysis of the same one-year window.
    row = result.signal_log.iloc[10]
    live = TrendAnalyzerAgent(settings).analyze(
        "BULL", provider.history("BULL", end=row["timestamp"], period="1y")
    )
    assert (row["trend_signal"], row["trend_confidence"]) == (live.signal, live.confidence)


def test_bear_scenario_never_buys_and_beats_buy_and_hold(
    provider: SyntheticProvider, settings: Settings
) -> None:
    result = Backtester(provider, settings).run(
        "BEAR", "2026-03-02", "2026-06-30", rebalance="weekly"
    )

    assert result.trades == [] and result.decisions == []
    assert (result.equity_curve["equity"] == 100_000.0).all()
    assert result.benchmark_metrics.total_return_pct < 0  # type: ignore[operator]
    assert result.excess_return_pct is not None and result.excess_return_pct > 0
    log = result.signal_log
    sells = log[log["proposed"] == "SELL"]
    assert not sells.empty and (sells["action"] == "HOLD").all()
    assert sells["note"].str.contains("no position to sell").all()
    assert result.stats["proposals"] == result.stats["vetoed"] == len(sells)


def test_backtests_are_deterministic(provider: SyntheticProvider, settings: Settings) -> None:
    backtester = Backtester(provider, settings)
    first = backtester.run(["BULL", "FLAT"], "2026-03-02", "2026-06-30", rebalance="weekly")
    second = backtester.run(["BULL", "FLAT"], "2026-03-02", "2026-06-30", rebalance="weekly")

    pd.testing.assert_frame_equal(first.signal_log, second.signal_log)
    pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
    assert [strip_ids(t.model_dump()) for t in first.trades] == [
        strip_ids(t.model_dump()) for t in second.trades
    ]
    assert first.metrics == second.metrics and first.stats == second.stats


def test_short_history_is_reported_and_handled(
    provider: SyntheticProvider, settings: Settings
) -> None:
    # The synthetic calendar starts on 2021-01-04: no warm-up history at all.
    result = Backtester(provider, settings).run(
        "BULL", "2021-01-04", "2021-06-30", rebalance="weekly"
    )

    assert result.parameters["warmup_bars"] == {"BULL": 0}
    assert any("only 0 bar(s) of history" in warning for warning in result.warnings)
    first_row = result.signal_log.iloc[0]
    assert first_row["trend_confidence"] == 0.0  # "insufficient data" vote
    assert first_row["note"] == "no conviction: every agent abstained"


def test_a_crashing_agent_abstains_and_is_reported(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    ts = gentle_frame.index
    result = Backtester(serve({"XYZ": gentle_frame}), settings, trend=BrokenTrend()).run(
        "XYZ", ts[WINDOW_START], ts[-1], warmup=WINDOW_START, rebalance="monthly"
    )
    assert result.trades == []
    assert result.stats["agent_failures"] == result.stats["evaluations"] > 0
    assert any("agent failed" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("overrides", "capital", "warning"),
    [
        ({"weight_trend": 0.0}, None, "the strategy cannot trade"),
        ({}, 50.0, "benchmark allocation 50.00 buys no share"),
    ],
    ids=["zero-trend-weight", "capital-below-one-share"],
)
def test_degenerate_setups_run_flat_and_explain_why(
    gentle_frame: pd.DataFrame,
    make_settings: Callable[..., Settings],
    overrides: dict[str, Any],
    capital: float | None,
    warning: str,
) -> None:
    ts = gentle_frame.index
    result = run_scripted(
        gentle_frame, make_settings(**overrides), {ts[ENTRY]: 1.0}, initial_capital=capital
    )
    assert result.trades == []
    assert any(warning in message for message in result.warnings)
    assert result.equity_curve["equity"].nunique() == 1


def test_symbols_with_missing_bars_trade_on_their_own_calendar(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    ts = gentle_frame.index
    holiday = ENTRY + 1  # BBB's exchange is closed the day after its BUY decision
    sparse = gentle_frame.drop(index=ts[holiday])
    trend = ScriptedTrend({("BBB", ts[ENTRY]): 1.0})
    provider = serve({"AAA": gentle_frame, "BBB": sparse})
    result = Backtester(provider, settings, trend=trend).run(
        ["AAA", "BBB"], ts[WINDOW_START], ts[-1], warmup=WINDOW_START
    )

    assert result.equity_curve.index.equals(ts[WINDOW_START:])  # union calendar
    assert ("BBB", ts[holiday]) not in {(s, day) for s, day, _ in trend.calls}
    [decision] = result.decisions
    assert decision.order is not None and decision.order.filled_at == ts[holiday + 1]
    assert decision.order.reference_price == pytest.approx(sparse.loc[ts[holiday + 1], "Open"])
    # On the holiday BBB has no bar: nothing is held yet and equity is unchanged.
    assert result.equity_curve.loc[ts[holiday], "positions_value"] == 0.0


# =============================================================================== sentiment


def test_sentiment_is_point_in_time_and_scored_once_per_headline_set(
    provider: SyntheticProvider, settings: Settings, fake_llm: Any
) -> None:
    # BULL's canned headlines are stamped on the last four bars (09-09 … 09-14, 13:30 UTC).
    backtester = Backtester(provider, settings, llm=fake_llm, trend=ScriptedTrend())
    result = backtester.run("BULL", "2026-08-24", TEST_END, use_sentiment=True)

    bars = [pd.Timestamp(day, tz="UTC") for day in ("2026-09-09", "2026-09-10", "2026-09-11")]
    assert len(fake_llm.calls) == 3, "one LLM call per distinct visible headline set"
    for (symbol, headlines), bar in zip(fake_llm.calls, bars, strict=True):
        assert symbol == "BULL"
        assert all(h.published_at <= bar + NEWS_CUTOFF for h in headlines)
    assert [len(h) for _, h in fake_llm.calls] == [1, 2, 3]
    latest = provider.news("BULL", 10)[0]
    assert all(latest.title not in {h.title for h in hs} for _, hs in fake_llm.calls)

    log = result.signal_log.set_index("timestamp")
    assert (log.loc[:"2026-09-08", "sentiment_confidence"] == 0.0).all()
    assert (log.loc[bars[0] :, "sentiment_signal"] == 0.6).all()
    assert result.parameters["sentiment_method"] == "fake:fake-model"
    assert result.parameters["sentiment_calls"] == 3
    assert result.parameters["weights"]["sentiment"] == settings.weight_sentiment
    # With the trend abstaining, the sentiment vote alone drives the first trade.
    assert result.decisions and result.decisions[0].timestamp == bars[0]
    assert [s.agent for s in result.decisions[0].signals] == ["trend", "sentiment"]


def test_sentiment_disabled_never_calls_the_llm(
    gentle_frame: pd.DataFrame, settings: Settings, fake_llm: Any
) -> None:
    ts = gentle_frame.index
    backtester = Backtester(
        serve({"XYZ": gentle_frame}), settings, llm=fake_llm, trend=ScriptedTrend()
    )
    result = backtester.run("XYZ", ts[WINDOW_START], ts[-1], warmup=WINDOW_START)
    assert fake_llm.calls == []
    assert result.signal_log["sentiment_signal"].isna().all()
    assert result.parameters["weights"]["sentiment"] == 0.0
    assert result.parameters["sentiment_method"] is None


def test_sentiment_without_an_llm_uses_the_lexicon(
    provider: SyntheticProvider, settings: Settings
) -> None:
    backtester = Backtester(provider, settings, trend=ScriptedTrend())
    result = backtester.run("BEAR", "2026-09-01", TEST_END, use_sentiment=True)
    assert result.parameters["sentiment_method"] == "lexicon"
    visible = result.signal_log.set_index("timestamp").loc["2026-09-09":]
    assert (visible["sentiment_signal"] < 0).all()  # BEAR's canned news is bearish


# =============================================================================== serialisation


def test_to_dict_is_json_safe_and_complete(gentle_frame: pd.DataFrame, settings: Settings) -> None:
    ts = gentle_frame.index
    result = run_scripted(gentle_frame, settings, {ts[ENTRY]: 1.0, ts[ENTRY + 5]: -1.0})

    payload = result.to_dict()
    text = json.dumps(payload, allow_nan=False)  # no NaN / inf / numpy types
    assert "signal_log" not in payload
    assert payload["symbols"] == ["XYZ"] and payload["initial_capital"] == 100_000.0
    assert payload["start"] == ts[WINDOW_START].date().isoformat()
    assert len(payload["equity_curve"]) == len(result.equity_curve)
    point = payload["equity_curve"][0]
    assert set(point) == {"t", *EQUITY_CURVE_COLUMNS}
    assert len(payload["trades"]) == 2 and len(payload["decisions"]) == 2
    assert payload["decisions"][0]["order"]["status"] == "FILLED"
    assert payload["metrics"]["num_trades"] == 1
    assert payload["final_portfolio"]["positions"] == []
    assert payload["parameters"]["fill_model"].startswith("market orders at the next bar's open")
    assert json.loads(text) == payload

    with_log = result.to_dict(include_signal_log=True)
    json.dumps(with_log, allow_nan=False)
    assert len(with_log["signal_log"]) == result.stats["evaluations"]
    assert set(with_log["signal_log"][0]) == set(SIGNAL_LOG_COLUMNS)
    assert with_log["signal_log"][0]["sentiment_signal"] is None  # NaN → null


# =============================================================================== validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": "2026-06-01", "end": "2026-05-01"},
        {"start": "not-a-date"},
        {"rebalance": "hourly"},
        {"initial_capital": 0},
        {"initial_capital": -5.0},
        {"initial_capital": math.nan},
        {"initial_capital": "lots"},
        {"warmup": -1},
        {"warmup": True},
        {"warmup": 2.5},
        {"symbols": []},
        {"symbols": " , "},
        {"symbols": [123]},
    ],
)
def test_invalid_parameters_are_rejected(
    provider: SyntheticProvider, settings: Settings, kwargs: dict[str, Any]
) -> None:
    arguments: dict[str, Any] = {"symbols": "BULL", "start": "2026-03-02", "end": "2026-03-31"}
    arguments.update(kwargs)
    with pytest.raises(BacktestConfigError):
        Backtester(provider, settings).run(**arguments)
    assert issubclass(BacktestConfigError, ValueError)


@pytest.mark.parametrize(
    ("symbol", "start", "end", "error"),
    [
        ("BAD SYMBOL!", "2026-03-02", "2026-03-31", InvalidSymbolError),
        ("NOPE", "2026-03-02", "2026-03-31", NoDataError),
        ("DOWN", "2026-03-02", "2026-03-31", ProviderUnavailableError),
        ("BULL", "2026-09-12", "2026-09-13", NoDataError),  # a weekend: no bars in the window
        ("BULL", "2027-01-04", "2027-02-01", NoDataError),  # after the provider's last bar
    ],
)
def test_market_data_problems_raise_clear_errors(
    provider: SyntheticProvider,
    settings: Settings,
    symbol: str,
    start: str,
    end: str,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        Backtester(provider, settings).run(symbol, start, end)


def test_symbols_are_normalised_and_deduplicated(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    ts = gentle_frame.index
    provider = serve({"XYZ": gentle_frame, "ABC": gentle_frame.copy()})
    result = Backtester(provider, settings, trend=ScriptedTrend()).run(
        " xyz,abc, XYZ ", ts[WINDOW_START], ts[-1], warmup=WINDOW_START, rebalance="monthly"
    )
    assert result.symbols == ["XYZ", "ABC"]


# =============================================================================== CLI


@pytest.fixture
def quiet_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the CLI from installing a root log handler bound to pytest's captured stderr."""
    monkeypatch.setattr(cli, "configure_logging", lambda *args, **kwargs: None)


@pytest.mark.usefixtures("quiet_cli")
def test_cli_prints_metrics_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out" / "results.json"
    code = cli.main(
        [
            "--provider",
            "synthetic",
            "--symbol",
            "BULL",
            "-s",
            "flat",
            "--start",
            "2026-03-02",
            "--end",
            "2026-04-30",
            "--capital",
            "50000",
            "--rebalance",
            "weekly",
            "--output",
            str(output),
        ]
    )
    assert code == cli.EXIT_OK
    stdout = capsys.readouterr().out
    assert "Backtest BULL, FLAT: 2026-03-02 to 2026-04-30" in stdout
    for label in ("Metric", "Strategy", "Buy & hold", "Total return", "Sharpe ratio"):
        assert label in stdout
    assert f"Results written to {output}" in stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["symbols"] == ["BULL", "FLAT"]
    assert payload["initial_capital"] == 50_000.0
    assert payload["parameters"]["rebalance"] == "weekly"
    assert payload["signal_log"]


@pytest.mark.usefixtures("quiet_cli")
def test_cli_defaults_to_the_watchlist_and_settings_provider(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("WATCHLIST", "BEAR")
    monkeypatch.setenv("DATA_PROVIDER", "synthetic")
    code = cli.main(["--start", "2026-03-02", "--end", "2026-04-30", "--rebalance", "monthly"])
    assert code == cli.EXIT_OK
    assert "Backtest BEAR:" in capsys.readouterr().out


@pytest.mark.usefixtures("quiet_cli")
@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        (["--symbol", "BAD SYMBOL!"], cli.EXIT_USAGE, "invalid symbol"),
        (["--symbol", "BULL", "--start", "2026-05-01", "--end", "2026-04-01"], 2, "after"),
        (["--symbol", "BULL", "--start", "2026-09-12", "--end", "2026-09-13"], 1, "no price"),
    ],
)
def test_cli_reports_errors_with_exit_codes(
    argv: list[str], code: int, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--provider", "synthetic", *argv]) == code
    stderr = capsys.readouterr().err
    assert stderr.startswith("error:") and message in stderr


@pytest.mark.usefixtures("quiet_cli")
def test_cli_output_write_failure_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["--provider", "synthetic", "-s", "FLAT", "--start", "2026-04-01", "--end", "2026-04-30"]
    assert cli.main([*argv, "--rebalance", "monthly", "--output", str(tmp_path)]) == 1
    assert "could not write" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [["--start", "2026-13-01"], ["--capital", "-1"], ["--warmup", "x"], ["--provider", "csv"]],
)
def test_cli_rejects_malformed_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(argv)
    assert excinfo.value.code == 2


def test_cli_helpers() -> None:
    assert cli.parse_symbols(["aapl, msft", "AAPL", " nvda "]) == ["AAPL", "MSFT", "NVDA"]
    assert cli.parse_symbols(None) == []
    table = cli.format_metrics_table(
        PerformanceMetrics(total_return_pct=12.345, sharpe=1.5, avg_loss=-12.5),
        PerformanceMetrics(),
    )
    lines = table.splitlines()
    assert lines[0].split() == ["Metric", "Strategy", "Buy", "&", "hold"]
    assert "+12.35%" in table and "1.50" in table and "-$12.50" in table and "n/a" in table
    assert len({len(line) for line in lines}) == 1  # aligned columns
    assert isinstance(
        cli.build_provider("synthetic", Settings(_env_file=None), seed=7), SyntheticProvider
    )


def test_scripts_wrapper_is_the_packaged_cli() -> None:
    """``python -m scripts.run_backtest`` and ``mt-swarm-backtest`` run the same code."""
    assert cli_wrapper.main is cli.main
    assert cli_wrapper.build_parser is cli.build_parser
    assert cli_wrapper.EXIT_USAGE == cli.EXIT_USAGE == 2


# =============================================================================== gap vs add-on


class FixedRisk(RiskManagerAgent):
    """Approves every BUY proposal for 10 shares with a 4 % stop / 10 % target, so the
    engine's ordering can be checked without the sizing math."""

    def assess(self, **kwargs: Any):  # type: ignore[override]
        if kwargs["action"] is not Action.BUY:
            return super().assess(**kwargs)
        price = kwargs["price"]
        return RiskAssessment(
            approved=True,
            action=Action.BUY,
            quantity=10,
            entry_price=price,
            stop_price=round(price * 0.96, 4),
            take_profit_price=round(price * 1.10, 4),
            reasons=["fixed test sizing"],
        )


def test_gap_through_the_stop_fires_before_a_pending_add_on(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    """Regression: the add-on used to fill first, re-anchor and *replace* the gapped stop."""
    ts = gentle_frame.index
    gap = EVENT + 1  # the add-on decided at EVENT's close would fill at this open
    gap_open = gentle_frame["Close"].iloc[EVENT] * 0.85  # far below the 4 % stop
    set_bar(gentle_frame, gap, open=gap_open, high=gap_open * 1.005, low=gap_open * 0.995)
    set_bar(gentle_frame, gap, close=gap_open)
    backtester = Backtester(
        serve({"XYZ": gentle_frame}),
        settings,
        trend=ScriptedTrend({ts[ENTRY]: 1.0, ts[EVENT]: 1.0}),
        risk=FixedRisk(settings),
    )
    result = backtester.run("XYZ", ts[WINDOW_START], ts[-1], warmup=WINDOW_START)

    entry, add_on = result.decisions
    stop = entry.order.stop_price  # type: ignore[union-attr]
    assert stop is not None and gap_open < stop
    buy, sell = result.trades
    assert (buy.side, buy.quantity) == (Side.BUY, 10)
    assert (sell.side, sell.quantity, sell.reason) == (Side.SELL, 10, "stop_loss")
    assert sell.timestamp == ts[gap]
    assert sell.price == pytest.approx(gap_open * (1 - SLIPPAGE))  # filled at the gapped open
    # The add-on was sized for a position that no longer exists: cancelled, not filled.
    assert add_on.order is not None and add_on.order.status is OrderStatus.CANCELLED
    assert not add_on.executed
    assert "Order cancelled: cancelled at the open" in add_on.summary
    assert result.stats["orders_cancelled"] == 1
    assert result.stats["exits"]["stop_loss"] == 1
    assert result.final_portfolio is not None and result.final_portfolio.positions == []


def test_add_on_without_a_gap_fills_and_never_loosens_the_stop(
    gentle_frame: pd.DataFrame, settings: Settings
) -> None:
    ts = gentle_frame.index

    class LowerStopRisk(FixedRisk):
        def assess(self, **kwargs: Any):  # type: ignore[override]
            risk = super().assess(**kwargs)
            if kwargs["action"] is Action.BUY and kwargs["portfolio"].position("XYZ"):
                return risk.model_copy(update={"stop_price": round(kwargs["price"] * 0.80, 4)})
            return risk

    backtester = Backtester(
        serve({"XYZ": gentle_frame}),
        settings,
        trend=ScriptedTrend({ts[ENTRY]: 1.0, ts[EVENT]: 1.0}),
        risk=LowerStopRisk(settings),
    )
    # End shortly after the add-on fills (the rising market later reaches the target).
    result = backtester.run("XYZ", ts[WINDOW_START], ts[EVENT + 3], warmup=WINDOW_START)
    entry, add_on = result.decisions
    assert add_on.executed and result.stats["orders_cancelled"] == 0
    first_stop = entry.order.stop_price  # type: ignore[union-attr]
    assert add_on.risk.stop_price is not None and add_on.risk.stop_price < first_stop
    # The add-on's lower stop was not applied to the 10 shares already held.
    assert add_on.order.stop_price == pytest.approx(first_stop)  # type: ignore[union-attr]
    (position,) = result.final_portfolio.positions  # type: ignore[union-attr]
    assert position.quantity == 20 and position.stop_price == pytest.approx(first_stop)


# =============================================================================== reservation


def test_orders_pending_over_a_symbols_holiday_are_reserved(
    make_ohlcv: Callable[..., pd.DataFrame], make_settings: Callable[..., Settings]
) -> None:
    """Regression: an order waiting for its symbol's next bar used to be invisible to the
    sizing of other symbols, so the fills together overshot cash and gross exposure."""
    # Loose per-position limits so the account-wide cash / exposure limits bind.
    settings = make_settings(
        max_position_pct=0.9, risk_per_trade=0.1, atr_stop_multiple=0.5, max_gross_exposure=0.95
    )
    frame = make_ohlcv(100.0 * 1.001 ** np.arange(160), start="2024-01-02")
    ts = frame.index
    holiday = ENTRY + 1  # BBB does not trade the day after its BUY decision
    trend = ScriptedTrend({("BBB", ts[ENTRY]): 1.0, ("AAA", ts[holiday]): 1.0})
    spy = SpyRisk(settings)
    provider = serve({"AAA": frame, "BBB": frame.drop(index=ts[holiday])})
    result = Backtester(provider, settings, trend=trend, risk=spy).run(
        ["AAA", "BBB"], ts[WINDOW_START], ts[-1], warmup=WINDOW_START
    )

    bbb, aaa = result.decisions
    assert (bbb.symbol, aaa.symbol) == ("BBB", "AAA")
    (account,) = [acct for sym, last, acct in spy.seen if (sym, last) == ("AAA", ts[holiday])]
    reserved = account.position("BBB")
    assert reserved is not None and reserved.quantity == bbb.risk.quantity  # still pending
    assert account.cash < 100_000.0 - 0.8 * bbb.risk.position_value
    # Both fill at the same open without any trimming, and the limit holds (up to the
    # market drift between decision and fill).
    assert bbb.executed and aaa.executed and result.stats["orders_trimmed"] == 0
    row = result.equity_curve.loc[ts[holiday + 1]]
    assert row["positions_value"] / row["equity"] <= 0.95 * 1.003


# =============================================================================== benchmark start


@pytest.mark.parametrize("first_day_move", [0.10, -0.08])
def test_benchmark_metrics_start_from_the_initial_capital(
    gentle_frame: pd.DataFrame, settings: Settings, first_day_move: float
) -> None:
    """Regression: the benchmark's metrics used its first *close-marked* point as the start,
    leaving out the entry costs and the first session's open-to-close move."""
    close = gentle_frame["Close"].iloc[WINDOW_START]
    first_open = close / (1 + first_day_move)
    set_bar(
        gentle_frame,
        WINDOW_START,
        open=first_open,
        high=max(first_open, close) * 1.01,
        low=min(first_open, close) * 0.99,
    )
    result = run_scripted(gentle_frame, settings, {})
    bench = result.benchmark_metrics
    final = result.equity_curve["benchmark"].iloc[-1]
    assert bench.start_equity == result.metrics.start_equity == 100_000.0
    assert bench.total_return_pct == pytest.approx((final / 100_000.0 - 1) * 100, abs=1e-5)
    assert result.excess_return_pct == pytest.approx(
        result.metrics.total_return_pct - bench.total_return_pct  # type: ignore[operator]
    )
    first_drawdown = result.equity_curve["benchmark_drawdown"].iloc[0]
    if first_day_move < 0:  # opened at the capital, closed ~8 % lower: already in drawdown
        assert first_drawdown == pytest.approx(result.equity_curve["benchmark"].iloc[0] / 1e5 - 1)
        assert bench.max_drawdown_pct >= -first_drawdown * 100 - 1e-9  # type: ignore[operator]
    else:
        assert first_drawdown == 0.0


# =============================================================================== conviction


def test_a_steady_half_confidence_signal_does_not_pyramid(
    make_ohlcv: Callable[..., pd.DataFrame], settings: Settings
) -> None:
    """Regression: re-deciding the same 0.5-confidence BUY every bar added 88, 44, 22, 11,
    5, 3, 1, 1 … shares (each paying the minimum commission) up to the *unscaled* 20 %
    cap. Conviction now scales the whole position, so it is bought once and held."""

    class HalfConfidence:
        name = "trend"

        def analyze(self, symbol: str, history: pd.DataFrame) -> AgentSignal:
            return AgentSignal(agent="trend", signal=1.0, confidence=0.5, rationale="steady")

    frame = make_ohlcv(100.0 + 0.05 * np.arange(160), spread=0.005)
    ts = frame.index
    backtester = Backtester(serve({"XYZ": frame}), settings, trend=HalfConfidence())
    result = backtester.run("XYZ", ts[WINDOW_START], ts[WINDOW_START + 40], warmup=WINDOW_START)

    buys = [t for t in result.trades if t.side is Side.BUY]
    first = result.decisions[0].risk
    assert any("by confidence (0.50×)" in reason for reason in first.reasons)
    assert len(buys) == 1 and buys[0].quantity == first.quantity
    # Half of the full-conviction size, held to the end (it used to reach the full size).
    assert any(f"→{first.quantity * 2} sh by max position" in r for r in first.reasons)
    (position,) = result.final_portfolio.positions  # type: ignore[union-attr]
    assert position.quantity == first.quantity
    vetoed = result.signal_log[~result.signal_log["approved"]]
    assert len(vetoed) >= 30  # every later bar: the position is already at its target
    assert vetoed["note"].str.contains("confidence-scaled target").all()
