"""SwarmCoordinator end to end (synthetic provider + fake LLM), consensus math, summaries,
error mapping, batches, watchlist scans, alert hooks and concurrency safety."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from trading_swarm.agents import RiskManagerAgent, TrendAnalyzerAgent
from trading_swarm.broker import PaperBroker
from trading_swarm.config import Settings
from trading_swarm.data import ProviderUnavailableError, SyntheticProvider
from trading_swarm.llm import NullLLMClient
from trading_swarm.logging_setup import request_id_ctx
from trading_swarm.models import (
    Action,
    AgentSignal,
    Decision,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    RiskAssessment,
    Side,
)
from trading_swarm.swarm import (
    AlertSink,
    DataUnavailable,
    SwarmCoordinator,
    SymbolNotFound,
    action_for_score,
    compute_consensus,
    summarize_decision,
)


class FakeAlerts:
    """Records every hook call; ``fail=True`` makes each hook raise."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.decisions: list[Decision] = []
        self.exits: list[Order] = []
        self.price_moves: list[tuple[str, float, float | None]] = []
        self.drawdowns: list[PortfolioSnapshot] = []

    def _check(self) -> None:
        if self.fail:
            raise RuntimeError("smtp exploded")

    def on_decision(self, decision: Decision) -> list[Any]:
        self._check()
        self.decisions.append(decision)
        return []

    def on_exit(self, order: Order) -> list[Any]:
        self._check()
        self.exits.append(order)
        return []

    def on_price_move(self, symbol: str, pct: float, *, price: float | None = None) -> list[Any]:
        self._check()
        self.price_moves.append((symbol, pct, price))
        return []

    def on_drawdown(self, snapshot: PortfolioSnapshot) -> list[Any]:
        self._check()
        self.drawdowns.append(snapshot)
        return []


class MinimalAlerts:
    """Only the two required hooks (the optional ones must be skipped silently)."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def on_decision(self, decision: Decision) -> None:
        self.calls.append(f"decision:{decision.symbol}")

    def on_exit(self, order: Order) -> None:
        self.calls.append(f"exit:{order.symbol}")


@pytest.fixture
def alerts() -> FakeAlerts:
    return FakeAlerts()


@pytest.fixture
def coordinator(
    settings: Settings,
    provider: SyntheticProvider,
    broker: PaperBroker,
    fake_llm: Any,
    alerts: FakeAlerts,
) -> SwarmCoordinator:
    return SwarmCoordinator(settings, provider, broker, llm=fake_llm, alerts=alerts)


def sig(agent: str, signal: float, confidence: float) -> AgentSignal:
    return AgentSignal(agent=agent, signal=signal, confidence=confidence)


# ----------------------------------------------------------------------------- consensus


def test_consensus_formula() -> None:
    trend, sentiment = sig("trend", 0.8, 0.9), sig("sentiment", -0.2, 0.5)
    c = compute_consensus([(trend, 0.65), (sentiment, 0.35)], 0.25, -0.25)
    v_t, v_s = 0.65 * 0.9, 0.35 * 0.5
    score = (v_t * 0.8 + v_s * -0.2) / (v_t + v_s)
    mean_conf = (v_t * 0.9 + v_s * 0.5) / (v_t + v_s)
    agreement = abs(v_t * 0.8 - v_s * 0.2) / (v_t * 0.8 + v_s * 0.2)
    assert c.score == pytest.approx(score)
    assert c.mean_confidence == pytest.approx(mean_conf)
    assert c.agreement == pytest.approx(agreement)
    assert c.confidence == pytest.approx(mean_conf * (0.5 + 0.5 * agreement))
    assert c.action is Action.BUY
    assert c.total_weight == pytest.approx(v_t + v_s)


def test_abstaining_agent_neither_moves_nor_dilutes() -> None:
    trend = sig("trend", 0.6, 0.8)
    alone = compute_consensus([(trend, 0.65)], 0.25, -0.25)
    with_abstainer = compute_consensus(
        [(trend, 0.65), (sig("sentiment", 0.0, 0.0), 0.35)], 0.25, -0.25
    )
    assert with_abstainer == alone
    assert alone.score == pytest.approx(0.6)
    assert alone.confidence == pytest.approx(0.8)  # single voter: agreement 1


def test_trend_only_consensus_renormalises() -> None:
    c = compute_consensus([(sig("trend", -0.4, 0.5), 0.65)], 0.25, -0.25)
    assert c.score == pytest.approx(-0.4)
    assert c.action is Action.SELL


def test_disagreement_lowers_confidence() -> None:
    agree = compute_consensus([(sig("a", 0.5, 0.8), 1.0), (sig("b", 0.5, 0.8), 1.0)], 0.25, -0.25)
    clash = compute_consensus([(sig("a", 0.5, 0.8), 1.0), (sig("b", -0.5, 0.8), 1.0)], 0.25, -0.25)
    assert agree.confidence == pytest.approx(0.8)
    assert clash.score == pytest.approx(0.0)
    assert clash.confidence == pytest.approx(0.4)  # agreement 0 halves it
    assert clash.action is Action.HOLD


def test_no_confidence_means_hold() -> None:
    c = compute_consensus([(sig("a", 1.0, 0.0), 1.0), (sig("b", -1.0, 0.0), 1.0)], 0.25, -0.25)
    assert (c.score, c.confidence, c.action) == (0.0, 0.0, Action.HOLD)
    assert compute_consensus([], 0.25, -0.25).action is Action.HOLD


def test_negative_weights_are_ignored() -> None:
    c = compute_consensus([(sig("a", 0.5, 1.0), 1.0), (sig("b", -1.0, 1.0), -3.0)], 0.25, -0.25)
    assert c.score == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("score", "action"),
    [
        (0.25, Action.BUY),
        (0.2499, Action.HOLD),
        (-0.25, Action.SELL),
        (-0.2499, Action.HOLD),
        (0.0, Action.HOLD),
    ],
)
def test_thresholds_are_inclusive(score: float, action: Action) -> None:
    assert action_for_score(score, 0.25, -0.25) is action


# ----------------------------------------------------------------------------- analyze


def test_bullish_end_to_end(
    coordinator: SwarmCoordinator,
    broker: PaperBroker,
    provider: SyntheticProvider,
    fake_llm: Any,
    alerts: FakeAlerts,
) -> None:
    decision = coordinator.analyze("bull")
    assert decision.symbol == "BULL"
    assert decision.action is Action.BUY
    assert decision.price == pytest.approx(provider.latest_price("BULL"), abs=1e-4)
    assert decision.score >= 0.25
    assert 0.6 < decision.confidence <= 1.0
    assert [s.agent for s in decision.signals] == ["trend", "sentiment"]
    trend, sentiment = decision.signal_for("trend"), decision.signal_for("sentiment")
    assert trend is not None and trend.signal > 0.5
    assert sentiment is not None and sentiment.data["method"] == "fake:fake-model"
    assert len(fake_llm.calls) == 1

    risk = decision.risk
    assert risk.approved and risk.action is Action.BUY and risk.quantity > 0
    assert risk.stop_price is not None and risk.stop_price < decision.price
    assert risk.take_profit_price is not None and risk.take_profit_price > decision.price

    # Default AUTO_TRADE=false: nothing executed.
    assert not decision.executed and decision.order is None
    assert broker.orders() == []

    # Persisted and alerted.
    stored = broker.get_decision(decision.id)
    assert stored is not None and stored.summary == decision.summary
    assert alerts.decisions == [decision]
    assert alerts.price_moves and alerts.price_moves[0][0] == "BULL"
    # The last close rides along so the price-move e-mail can quote it.
    assert alerts.price_moves[0][2] == decision.price
    assert coordinator.trade_lock is coordinator._trade_lock


def test_summary_text(coordinator: SwarmCoordinator) -> None:
    decision = coordinator.analyze("BULL")
    summary = decision.summary
    assert summary.startswith(f"BUY BULL @ {decision.price:.2f}: consensus +")
    assert "Trend +" in summary and "price above SMA20/50/200" in summary
    assert "Sentiment +0.60 (fake:fake-model, conf 0.80)" in summary
    assert f"Risk: approved {decision.risk.quantity} shares" in summary
    assert f"stop {decision.risk.stop_price:.2f} (2.0×ATR)" in summary
    assert f"TP {decision.risk.take_profit_price:.2f}" in summary
    assert summary.endswith("Not executed (auto-trade off).")
    # Deterministic: same inputs ⇒ same text.
    assert coordinator.analyze("BULL").summary == summary


def test_auto_trade_executes(coordinator: SwarmCoordinator, broker: PaperBroker) -> None:
    decision = coordinator.analyze("BULL", auto_trade=True)
    assert decision.executed
    order = decision.order
    assert order is not None and order.status is OrderStatus.FILLED
    assert order.side is Side.BUY and order.quantity == decision.risk.quantity
    assert order.decision_id == decision.id
    position = broker.position("BULL")
    assert position is not None and position.quantity == order.quantity
    assert position.stop_price == pytest.approx(decision.risk.stop_price)
    assert position.take_profit_price == pytest.approx(decision.risk.take_profit_price)
    assert f"Executed: BUY {order.quantity} @ {order.fill_price:.2f}." in decision.summary
    stored = broker.get_decision(decision.id)
    assert stored is not None and stored.executed and stored.order is not None


def test_settings_auto_trade_and_override(
    make_settings: Callable[..., Settings], provider: SyntheticProvider, fake_llm: Any
) -> None:
    settings = make_settings(auto_trade=True)
    with PaperBroker(settings) as paper:
        coordinator = SwarmCoordinator(settings, provider, paper, llm=fake_llm)
        assert not coordinator.analyze("BULL", auto_trade=False).executed
        assert coordinator.analyze("BULL").executed


def test_bearish_without_position_is_vetoed_to_hold(
    settings: Settings,
    provider: SyntheticProvider,
    broker: PaperBroker,
    fake_llm_factory: Any,
    make_judgement: Callable[..., Any],
) -> None:
    llm = fake_llm_factory(make_judgement(score=-0.7, confidence=0.8))
    decision = SwarmCoordinator(settings, provider, broker, llm=llm).analyze(
        "BEAR", auto_trade=True
    )
    assert decision.score <= -0.25
    assert decision.action is Action.HOLD
    assert not decision.risk.approved
    assert "→ SELL vetoed by risk" in decision.summary
    assert "no position to sell" in decision.summary
    assert decision.order is None


def test_bearish_with_position_sells(
    settings: Settings,
    provider: SyntheticProvider,
    broker: PaperBroker,
    fake_llm_factory: Any,
    make_judgement: Callable[..., Any],
) -> None:
    broker.submit_order("BEAR", Side.BUY, 50, 80.0)
    llm = fake_llm_factory(make_judgement(score=-0.7, confidence=0.8))
    decision = SwarmCoordinator(settings, provider, broker, llm=llm).analyze(
        "BEAR", auto_trade=True
    )
    assert decision.action is Action.SELL
    assert decision.executed
    assert decision.order is not None and decision.order.quantity == 50
    assert broker.position("BEAR") is None
    assert "Risk: approved SELL of 50 shares" in decision.summary


def test_sideways_is_hold(coordinator: SwarmCoordinator, fake_llm: Any) -> None:
    fake_llm.judgement = None  # LLM failure → lexicon (FLAT news is neutral → abstains)
    decision = coordinator.analyze("FLAT", auto_trade=True)
    assert decision.action is Action.HOLD
    assert "inside the HOLD band" in decision.summary
    assert "Risk: no trade." in decision.summary
    assert decision.signal_for("sentiment").data["method"] == "lexicon"  # type: ignore[union-attr]
    assert decision.order is None


def test_drawdown_breaker_blocks_buys(coordinator: SwarmCoordinator, broker: PaperBroker) -> None:
    # Buy far above the market and mark the loss: equity drops > 20 % below the peak.
    broker.submit_order("BEAR", Side.BUY, 500, 150.0)
    broker.portfolio({"BEAR": 150.0})  # observed peak with real prices
    decision = coordinator.analyze("BULL", auto_trade=True)
    assert decision.action is Action.HOLD
    assert decision.risk.checks["drawdown_ok"] is False
    assert "BUY vetoed" in decision.summary
    assert not decision.executed


@pytest.mark.parametrize(
    ("symbol", "error"),
    [("NOPE", SymbolNotFound), ("bad symbol!", SymbolNotFound), ("DOWN", DataUnavailable)],
)
def test_error_mapping(coordinator: SwarmCoordinator, symbol: str, error: type[Exception]) -> None:
    with pytest.raises(error) as info:
        coordinator.analyze(symbol)
    assert info.value.message  # type: ignore[attr-defined]


def test_symbol_on_errors(coordinator: SwarmCoordinator) -> None:
    with pytest.raises(SymbolNotFound) as info:
        coordinator.analyze("nope")
    assert info.value.symbol == "NOPE"


def test_failing_agent_abstains(
    settings: Settings, provider: SyntheticProvider, broker: PaperBroker, fake_llm: Any
) -> None:
    class Exploding:
        name = "trend"

        def analyze(self, symbol: str, history: pd.DataFrame) -> AgentSignal:
            raise ZeroDivisionError("boom")

    coordinator = SwarmCoordinator(settings, provider, broker, llm=fake_llm, trend=Exploding())
    decision = coordinator.analyze("BULL")
    trend = decision.signal_for("trend")
    assert trend is not None and trend.confidence == 0.0
    assert "failed" in trend.rationale
    assert decision.score == pytest.approx(0.6)  # sentiment alone


def test_agent_returning_garbage_abstains(
    settings: Settings, provider: SyntheticProvider, broker: PaperBroker, fake_llm: Any
) -> None:
    class Garbage:
        name = "trend"

        def analyze(self, symbol: str, history: pd.DataFrame) -> Any:
            return {"signal": 1}

    decision = SwarmCoordinator(settings, provider, broker, llm=fake_llm, trend=Garbage()).analyze(
        "BULL"
    )
    assert decision.signal_for("trend").confidence == 0.0  # type: ignore[union-attr]


def test_zero_weight_agent_is_not_run(
    make_settings: Callable[..., Settings], provider: SyntheticProvider, fake_llm: Any
) -> None:
    settings = make_settings(weight_sentiment=0.0)
    with PaperBroker(settings) as paper:
        decision = SwarmCoordinator(settings, provider, paper, llm=fake_llm).analyze("BULL")
    assert fake_llm.calls == []
    sentiment = decision.signal_for("sentiment")
    assert sentiment is not None and sentiment.data["disabled"] is True
    assert decision.score == pytest.approx(decision.signal_for("trend").signal, abs=1e-3)  # type: ignore[union-attr]


def test_alert_failures_do_not_break_analysis(
    settings: Settings, provider: SyntheticProvider, broker: PaperBroker, fake_llm: Any
) -> None:
    coordinator = SwarmCoordinator(
        settings, provider, broker, llm=fake_llm, alerts=FakeAlerts(fail=True)
    )
    decision = coordinator.analyze("BULL", auto_trade=True)
    assert decision.executed
    assert broker.get_decision(decision.id) is not None


def test_minimal_alert_sink(
    settings: Settings, provider: SyntheticProvider, broker: PaperBroker, fake_llm: Any
) -> None:
    sink = MinimalAlerts()
    assert isinstance(sink, AlertSink)
    SwarmCoordinator(settings, provider, broker, llm=fake_llm, alerts=sink).analyze("BULL")
    assert sink.calls == ["decision:BULL"]


def test_persistence_failure_still_returns_decision(
    coordinator: SwarmCoordinator, broker: PaperBroker, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(decision: Decision) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(broker, "save_decision", broken)
    assert coordinator.analyze("BULL").action is Action.BUY


def test_default_wiring(
    settings: Settings, provider: SyntheticProvider, broker: PaperBroker
) -> None:
    coordinator = SwarmCoordinator(settings, provider, broker)
    assert isinstance(coordinator.llm, NullLLMClient)  # LLM_PROVIDER=none in tests
    assert isinstance(coordinator.trend, TrendAnalyzerAgent)
    assert coordinator.execution.broker is broker
    assert coordinator.risk.cost_model == broker.cost_model
    decision = coordinator.analyze("BULL")
    assert decision.signal_for("sentiment").data["method"] == "lexicon"  # type: ignore[union-attr]


def test_request_id_reaches_agent_threads(
    settings: Settings, provider: SyntheticProvider, broker: PaperBroker, fake_llm: Any
) -> None:
    seen: list[str | None] = []
    real = TrendAnalyzerAgent(settings)

    class Recording:
        name = "trend"

        def analyze(self, symbol: str, history: pd.DataFrame) -> AgentSignal:
            seen.append(request_id_ctx.get())
            return real.analyze(symbol, history)

    coordinator = SwarmCoordinator(settings, provider, broker, llm=fake_llm, trend=Recording())
    token = request_id_ctx.set("rid-123")
    try:
        coordinator.analyze("BULL")
        coordinator.analyze_many(["BULL", "BEAR"])
    finally:
        request_id_ctx.reset(token)
    assert seen == ["rid-123"] * 3


def test_agents_run_concurrently(
    settings: Settings, provider: SyntheticProvider, broker: PaperBroker
) -> None:
    """Each agent waits for the other: this only completes if they run in parallel."""
    barrier = threading.Barrier(2, timeout=5)

    class Waiting:
        def __init__(self, name: str) -> None:
            self.name = name

        def analyze(self, symbol: str, history: Any = None) -> AgentSignal:
            barrier.wait()
            return AgentSignal(agent=self.name, signal=0.5, confidence=0.5)

    coordinator = SwarmCoordinator(
        settings, provider, broker, trend=Waiting("trend"), sentiment=Waiting("sentiment")
    )
    assert coordinator.analyze("BULL").score == pytest.approx(0.5)


# ----------------------------------------------------------------------------- summary unit


def _decision(
    risk: RiskAssessment, order: Order | None = None, action: Action = Action.BUY
) -> Decision:
    return Decision(
        symbol="ACME",
        price=100.0,
        action=action,
        score=0.5,
        confidence=0.7,
        signals=[AgentSignal(agent="trend", signal=0.5, confidence=0.7, rationale="x " * 200)],
        risk=risk,
        order=order,
    )


def test_summary_variants() -> None:
    approved = RiskAssessment(
        approved=True,
        action=Action.BUY,
        quantity=7,
        entry_price=100.0,
        stop_price=95.0,
        take_profit_price=110.0,
        checks={"stop_from_atr": False},
    )
    kwargs: dict[str, Any] = {
        "proposed": Action.BUY,
        "buy_threshold": 0.25,
        "sell_threshold": -0.25,
        "atr_multiple": 2.0,
    }
    text = summarize_decision(_decision(approved), auto_trade=True, **kwargs)
    assert "stop 95.00 (5% fallback)" in text
    assert text.endswith("Execution failed (see logs).")
    assert "…" in text  # long rationale clipped

    rejected = Order(
        symbol="ACME",
        side=Side.BUY,
        quantity=7,
        status=OrderStatus.REJECTED,
        reject_reason="insufficient cash",
    )
    text = summarize_decision(_decision(approved, rejected), auto_trade=True, **kwargs)
    assert text.endswith("Order rejected: insufficient cash.")


# ----------------------------------------------------------------------------- batches


def test_analyze_many(coordinator: SwarmCoordinator) -> None:
    batch = coordinator.analyze_many(["BULL", "nope", "DOWN", "bull", "BEAR", "??"])
    assert [d.symbol for d in batch.results] == ["BULL", "BEAR"]
    assert [(e.symbol, e.code) for e in batch.errors] == [
        ("NOPE", "not_found"),
        ("DOWN", "data_unavailable"),
        ("??", "not_found"),
    ]
    assert all(e.error for e in batch.errors)


def test_analyze_many_empty(coordinator: SwarmCoordinator) -> None:
    batch = coordinator.analyze_many([])
    assert batch.results == [] and batch.errors == []


def test_analyze_many_unexpected_error(
    settings: Settings, broker: PaperBroker, fake_llm: Any
) -> None:
    class Buggy(SyntheticProvider):
        def history(self, symbol: str, *args: Any, **kwargs: Any) -> pd.DataFrame:
            if symbol == "BOOM":
                raise KeyError("Close")
            return super().history(symbol, *args, **kwargs)

    provider = Buggy(seed=42, end_date="2026-09-14")
    batch = SwarmCoordinator(settings, provider, broker, llm=fake_llm).analyze_many(
        ["BOOM", "BULL"]
    )
    assert [d.symbol for d in batch.results] == ["BULL"]
    assert batch.errors[0].code == "error" and "KeyError" in batch.errors[0].error


def test_concurrent_auto_trades_respect_exposure_limit(
    make_settings: Callable[..., Settings],
) -> None:
    """Risk is assessed under a lock, so racing BUYs cannot overshoot gross exposure.

    The slowed risk agent widens the race window: without the coordinator's lock every
    worker would size its trade against the same empty account (exposure ≈ 0.75 here).
    """
    symbols = ["UPA", "UPB", "UPC", "UPD", "UPE"]
    settings = make_settings(max_position_pct=0.5, max_gross_exposure=0.6, auto_trade=True)
    provider = SyntheticProvider(
        seed=42,
        end_date="2026-09-14",
        drift=dict.fromkeys(symbols, 0.8),
        vol=dict.fromkeys(symbols, 0.1),
        start_price=dict.fromkeys(symbols, 100.0),
    )

    class SlowRisk(RiskManagerAgent):
        def assess(self, **kwargs: Any) -> RiskAssessment:
            time.sleep(0.02)
            return super().assess(**kwargs)

    with PaperBroker(settings) as paper:
        coordinator = SwarmCoordinator(
            settings, provider, paper, llm=NullLLMClient(), risk=SlowRisk(settings)
        )
        batch = coordinator.analyze_many(symbols)
        assert len(batch.results) == len(symbols)
        assert sum(d.executed for d in batch.results) >= 2
        quotes = {s: provider.latest_price(s) for s in symbols}
        snapshot = paper.portfolio(quotes)
    assert snapshot.gross_exposure <= settings.max_gross_exposure + 0.01
    assert snapshot.cash >= 0


# ----------------------------------------------------------------------------- scans


def test_scan_watchlist(
    coordinator: SwarmCoordinator,
    broker: PaperBroker,
    alerts: FakeAlerts,
    provider: SyntheticProvider,
) -> None:
    broker.set_watchlist(["BULL", "FLAT", "NOPE"])
    # A BEAR position bought at 70 with a 60 stop: the market (~57) is through the stop.
    broker.submit_order("BEAR", Side.BUY, 100, 70.0, stop=60.0, take_profit=90.0)
    snapshots_before = len(broker.equity_curve())

    report = coordinator.scan_watchlist()

    assert [d.symbol for d in report.decisions] == ["BULL", "FLAT"]
    assert [(e.symbol, e.code) for e in report.errors] == [("NOPE", "not_found")]
    (exit_order,) = report.exits
    assert exit_order.symbol == "BEAR" and exit_order.reason == "stop_loss"
    assert exit_order.reference_price == pytest.approx(provider.latest_price("BEAR"), abs=1e-4)
    assert broker.position("BEAR") is None
    assert alerts.exits == [exit_order]
    assert {d.symbol for d in alerts.decisions} == {"BULL", "FLAT"}

    assert report.snapshot is not None and report.snapshot_recorded
    assert len(broker.equity_curve()) == snapshots_before + 1
    assert alerts.drawdowns == [report.snapshot]
    assert report.finished_at is not None and report.finished_at >= report.started_at
    assert len(broker.decisions()) == 2


def test_scan_uses_settings_watchlist_by_default(coordinator: SwarmCoordinator) -> None:
    report = coordinator.scan_watchlist()
    assert [d.symbol for d in report.decisions] == ["BULL", "BEAR", "FLAT"]
    assert report.exits == [] and report.errors == []


def test_scan_reports_unpriceable_positions(
    coordinator: SwarmCoordinator, broker: PaperBroker
) -> None:
    broker.set_watchlist(["FLAT"])
    broker.submit_order("DOWN", Side.BUY, 10, 50.0, stop=45.0)
    report = coordinator.scan_watchlist()
    assert [(e.symbol, e.code) for e in report.errors] == [("DOWN", "data_unavailable")]
    assert broker.position("DOWN") is not None  # unknown price → no exit
    assert report.snapshot is not None  # still shown (DOWN at cost) …
    assert report.snapshot_recorded is False  # … but not stored


def test_scan_never_records_a_cost_marked_equity_point(
    coordinator: SwarmCoordinator,
    broker: PaperBroker,
    alerts: FakeAlerts,
    provider: SyntheticProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a failed quote put a point valued at cost into the equity curve
    ([100000, 94821, 100000]: a fake recovery) and the drawdown alert saw it."""
    broker.set_watchlist(["FLAT"])
    bear = provider.latest_price("BEAR")
    broker.submit_order("BEAR", Side.BUY, 400, bear * 2)  # bought at twice today's price
    first = coordinator.scan_watchlist()
    assert first.snapshot_recorded and first.snapshot is not None
    priced_equity = first.snapshot.equity
    assert priced_equity < 95_000
    curve_before = broker.equity_curve()["equity"].tolist()
    assert curve_before[-1] == pytest.approx(priced_equity)
    assert alerts.drawdowns == [first.snapshot]

    real = provider.latest_price

    def bear_down(symbol: str) -> float:
        if symbol == "BEAR":
            raise ProviderUnavailableError("quote down", symbol=symbol)
        return real(symbol)

    monkeypatch.setattr(provider, "latest_price", bear_down)
    second = coordinator.scan_watchlist()
    assert [e.symbol for e in second.errors] == ["BEAR"]
    assert second.snapshot_recorded is False
    assert second.snapshot is not None and second.snapshot.equity > priced_equity  # at cost
    assert broker.equity_curve()["equity"].tolist() == curve_before  # no phantom point
    assert alerts.drawdowns == [first.snapshot]  # not evaluated on the cost-marked account


def test_scan_with_auto_trade_opens_positions(
    make_settings: Callable[..., Settings], provider: SyntheticProvider
) -> None:
    settings = make_settings(auto_trade=True)
    with PaperBroker(settings) as paper:
        # Lexicon sentiment: BULL buys, BEAR has nothing to sell, FLAT stays in the HOLD band.
        coordinator = SwarmCoordinator(settings, provider, paper, llm=NullLLMClient())
        report = coordinator.scan_watchlist()
        assert [d.symbol for d in report.decisions if d.executed] == ["BULL"]
        assert report.snapshot is not None and report.snapshot.positions_value > 0
        assert [p.symbol for p in paper.positions()] == ["BULL"]


def test_portfolio_marks_positions(
    coordinator: SwarmCoordinator, broker: PaperBroker, provider: SyntheticProvider
) -> None:
    broker.submit_order("BULL", Side.BUY, 10, 100.0)
    snapshot = coordinator.portfolio()
    assert snapshot.positions[0].market_price == pytest.approx(
        provider.latest_price("BULL"), abs=1e-4
    )


# ----------------------------------------------------------------------------- exits in analyze


def test_analyze_enforces_a_breached_stop_and_does_not_buy_back(
    coordinator: SwarmCoordinator,
    broker: PaperBroker,
    alerts: FakeAlerts,
    provider: SyntheticProvider,
) -> None:
    """Regression: the stop sold 20 BULL and the same call bought 79 back at the same price
    (double costs). The backtester cancels exactly this case, so live must veto it."""
    price = provider.latest_price("BULL")
    broker.submit_order("BULL", Side.BUY, 10, price, stop=price * 0.97)
    broker.update_stops("BULL", stop=price * 1.01)  # the market is already through it
    snapshots_before = len(broker.equity_curve())

    decision = coordinator.analyze("BULL", auto_trade=True)

    # The stop is enforced first: the 10 old shares are sold at the stop check...
    exits = [o for o in broker.orders(limit=None) if o.reason == "stop_loss"]
    assert [(o.symbol, o.quantity, o.status) for o in exits] == [("BULL", 10, OrderStatus.FILLED)]
    assert alerts.exits == exits
    assert [o.id for o in decision.exits] == [exits[0].id]
    # ...and the bullish consensus is not allowed to re-buy at the price the stop sold at.
    assert decision.score >= 0.25 and decision.action is Action.HOLD
    assert not decision.executed and decision.order is None
    assert decision.risk.checks["reentry_ok"] is False
    assert "no re-entry" in decision.risk.reasons[0]
    assert broker.position("BULL") is None
    assert [o.side for o in broker.orders(limit=None)] == [Side.SELL, Side.BUY]  # newest first
    assert "→ BUY vetoed by risk" in decision.summary
    assert "Protective exits before assessment: stop-loss sold 10 BULL @" in decision.summary
    # The exit is reflected in the equity curve without a scan.
    assert len(broker.equity_curve()) == snapshots_before + 1
    # The next analysis (a later scan) may enter again.
    again = coordinator.analyze("BULL", auto_trade=True)
    assert again.executed and again.exits == []
    assert "reentry_ok" not in again.risk.checks


def test_analyze_enforces_stops_even_without_auto_trade(
    coordinator: SwarmCoordinator, broker: PaperBroker, provider: SyntheticProvider
) -> None:
    broker.submit_order("BEAR", Side.BUY, 100, 70.0, stop=60.0)  # BEAR trades ~57
    broker.submit_order("FLAT", Side.BUY, 5, provider.latest_price("FLAT"), stop=1.0)
    snapshots_before = len(broker.equity_curve())

    decision = coordinator.analyze("BULL", auto_trade=False)

    assert not decision.executed
    assert broker.position("BEAR") is None  # another holding, checked with its fresh quote
    assert broker.position("FLAT") is not None  # not breached: untouched
    (exit_order,) = [o for o in broker.orders() if o.reason == "stop_loss"]
    assert exit_order.symbol == "BEAR"
    assert exit_order.reference_price == pytest.approx(provider.latest_price("BEAR"), abs=1e-4)
    assert len(broker.equity_curve()) == snapshots_before + 1
    # Regression: the response used to say nothing about the sale of another holding.
    assert [o.id for o in decision.exits] == [exit_order.id]
    assert decision.exit_for("BEAR") == exit_order and decision.exit_for() is None
    assert "Protective exits before assessment: stop-loss sold 100 BEAR @ " in decision.summary
    assert f"(P&L {exit_order.realized_pnl:+.2f})" in decision.summary
    stored = broker.get_decision(decision.id)
    assert stored is not None and [o.id for o in stored.exits] == [exit_order.id]


def test_sell_signal_on_a_position_its_stop_just_closed_reads_as_done(
    settings: Settings,
    provider: SyntheticProvider,
    broker: PaperBroker,
    fake_llm_factory: Any,
    make_judgement: Callable[..., Any],
) -> None:
    """Regression: the summary said "SELL vetoed — no position to sell" right after the
    analysis itself had sold the position through its stop."""
    price = provider.latest_price("BEAR")
    broker.submit_order("BEAR", Side.BUY, 10, price, stop=price * 0.99)
    broker.update_stops("BEAR", stop=price * 1.05)  # breached at the current price
    llm = fake_llm_factory(make_judgement(score=-0.7, confidence=0.8))

    decision = SwarmCoordinator(settings, provider, broker, llm=llm).analyze(
        "BEAR", auto_trade=False
    )

    assert decision.score <= -0.25 and decision.action is Action.HOLD
    assert not decision.executed and decision.order is None
    (exit_order,) = decision.exits
    assert exit_order.symbol == "BEAR" and exit_order.reason == "stop_loss"
    assert exit_order.is_filled and broker.position("BEAR") is None
    assert "→ SELL already done by the stop-loss" in decision.summary
    assert "Risk: SELL not needed — position already closed by its stop-loss" in decision.summary
    assert "vetoed" not in decision.summary
    assert "no position to sell" not in decision.summary


class _AlwaysBuy:
    """Trend double: a confident +1 vote; ``slow`` symbols take a while to analyse."""

    name = "trend"

    def __init__(self, slow: frozenset[str] = frozenset()) -> None:
        self.slow = slow

    def analyze(self, symbol: str, history: pd.DataFrame) -> AgentSignal:
        if symbol in self.slow:
            time.sleep(0.3)
        return AgentSignal(agent="trend", signal=1.0, confidence=1.0, rationale="stub")


def _breached(paper: PaperBroker, provider: SyntheticProvider, symbol: str) -> None:
    """Hold 10 ``symbol`` with a stop that the current price is already through."""
    price = provider.latest_price(symbol)
    paper.submit_order(symbol, Side.BUY, 10, price, stop=price * 0.99)
    paper.update_stops(symbol, stop=price * 1.05)


def test_batch_analyses_share_exits_so_a_sibling_cannot_rebuy(
    make_settings: Callable[..., Settings], provider: SyntheticProvider, fake_llm: Any
) -> None:
    """An exit made by one analysis of a batch (FLAT quoted while analysing BULL) blocks a
    BUY of FLAT by a later analysis of the same batch."""
    settings = make_settings(auto_trade=True)
    with PaperBroker(settings) as paper:
        _breached(paper, provider, "FLAT")
        trend = _AlwaysBuy(slow=frozenset({"FLAT"}))  # BULL reaches the trade lock first
        coordinator = SwarmCoordinator(settings, provider, paper, llm=fake_llm, trend=trend)
        bull, flat = coordinator.analyze_many(["BULL", "FLAT"]).results
        assert bull.executed
        exit_order = bull.exit_for("FLAT")
        assert exit_order is not None and exit_order.reason == "stop_loss"
        assert flat.exit_for() == exit_order  # listed: it explains the veto
        assert not flat.executed and flat.order is None
        assert flat.risk.checks["reentry_ok"] is False
        assert paper.position("FLAT") is None


def test_scan_does_not_rebuy_what_its_exit_pass_sold(
    make_settings: Callable[..., Settings], provider: SyntheticProvider, fake_llm: Any
) -> None:
    settings = make_settings(auto_trade=True, watchlist="FLAT")
    with PaperBroker(settings) as paper:
        _breached(paper, provider, "FLAT")
        coordinator = SwarmCoordinator(settings, provider, paper, llm=fake_llm, trend=_AlwaysBuy())
        report = coordinator.scan_watchlist()
        (exit_order,) = report.exits
        (decision,) = report.decisions
        assert decision.exit_for() == exit_order
        assert decision.score >= 0.25 and not decision.executed
        assert decision.risk.checks["reentry_ok"] is False
        assert paper.position("FLAT") is None
        assert [o.side for o in paper.orders(limit=None)] == [Side.SELL, Side.BUY]


def test_scan_report_lists_exits_made_inside_the_analyses(
    coordinator: SwarmCoordinator,
    broker: PaperBroker,
    provider: SyntheticProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: ``ScanReport.exits`` left out exits made by the per-symbol analyses."""
    broker.set_watchlist(["FLAT"])
    _breached(broker, provider, "FLAT")
    # The scan's own quote of FLAT fails, so only the analysis (last close) can exit it.
    real = provider.latest_price
    calls = {"n": 0}

    def flaky(symbol: str) -> float:
        calls["n"] += 1
        if symbol == "FLAT" and calls["n"] == 1:
            raise ProviderUnavailableError("quote down", symbol=symbol)
        return real(symbol)

    monkeypatch.setattr(provider, "latest_price", flaky)
    report = coordinator.scan_watchlist()
    (decision,) = report.decisions
    (exit_order,) = decision.exits
    assert exit_order.symbol == "FLAT" and exit_order.is_filled
    assert [o.id for o in report.exits] == [exit_order.id]
    assert broker.position("FLAT") is None


def test_repeated_analyses_of_an_unchanged_signal_do_not_pyramid(
    make_settings: Callable[..., Settings], provider: SyntheticProvider
) -> None:
    """Regression: five analyses on identical data bought 79 → +21 → +6 → +2 shares,
    ending at the unscaled cap; confidence scaling must hold for the whole position."""
    settings = make_settings(auto_trade=True)
    with PaperBroker(settings) as paper:
        coordinator = SwarmCoordinator(settings, provider, paper, llm=NullLLMClient())
        first = coordinator.analyze("BULL")
        assert first.executed and first.confidence < 1.0
        later = [coordinator.analyze("BULL") for _ in range(4)]
        assert not any(d.executed for d in later)
        assert all(d.risk.checks["min_quantity_ok"] is False for d in later)
        assert "already reach the position's confidence-scaled target" in later[0].summary
        position = paper.position("BULL")
        assert position is not None and position.quantity == first.risk.quantity
        assert len([o for o in paper.orders(limit=None) if o.side is Side.BUY]) == 1


def test_analyze_without_fills_records_no_snapshot(
    coordinator: SwarmCoordinator, broker: PaperBroker
) -> None:
    snapshots_before = len(broker.equity_curve())
    decision = coordinator.analyze("BULL", auto_trade=False)  # approved, but not executed
    assert decision.risk.approved and not decision.executed
    assert len(broker.equity_curve()) == snapshots_before


def test_record_snapshot_skips_unpriceable_positions(
    coordinator: SwarmCoordinator, broker: PaperBroker, provider: SyntheticProvider
) -> None:
    broker.submit_order("BULL", Side.BUY, 5, provider.latest_price("BULL"))
    before = len(broker.equity_curve())
    assert coordinator.record_snapshot() is not None  # BULL quoted by the provider
    broker.submit_order("DOWN", Side.BUY, 5, 50.0)  # the provider cannot price DOWN
    assert coordinator.record_snapshot() is None  # a cost-marked point would be misleading
    assert coordinator.snapshot_after_fill({"DOWN": 49.0}) is not None
    assert len(broker.equity_curve()) == before + 2
