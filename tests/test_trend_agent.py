"""TrendAnalyzerAgent: direction on synthetic regimes, insufficient data, explainability,
determinism and absence of look-ahead."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from trading_swarm.agents import TrendAnalyzerAgent
from trading_swarm.agents.trend import COMPONENT_WEIGHTS, adx_label, rsi_component
from trading_swarm.config import Settings
from trading_swarm.data import SyntheticProvider


@pytest.fixture
def agent(settings: Settings) -> TrendAnalyzerAgent:
    return TrendAnalyzerAgent(settings)


def _history(provider: SyntheticProvider, symbol: str, period: str = "1y") -> pd.DataFrame:
    return provider.history(symbol, period=period)


# ----------------------------------------------------------------------------- direction


def test_bullish_scenario_votes_positive(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    signal = agent.analyze("BULL", _history(provider, "BULL"))
    assert signal.agent == "trend"
    assert signal.signal > 0.5
    assert signal.confidence > 0.6
    assert "price above SMA20/50/200" in signal.rationale
    assert "golden cross" in signal.rationale
    assert "overbought" in signal.rationale  # RSI ≈ 85 is flagged …
    assert signal.data["components"]["rsi"] < 0  # … and argues against the trend


def test_bearish_scenario_votes_negative(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    signal = agent.analyze("BEAR", _history(provider, "BEAR"))
    assert signal.signal < -0.5
    assert signal.confidence > 0.6
    assert "price below SMA20/50/200" in signal.rationale
    assert "death cross" in signal.rationale
    assert "oversold" in signal.rationale
    assert signal.data["components"]["rsi"] > 0  # oversold → contrarian bounce component


def test_sideways_scenario_is_weak_and_less_confident(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    flat = agent.analyze("FLAT", _history(provider, "FLAT"))
    bull = agent.analyze("BULL", _history(provider, "BULL"))
    assert abs(flat.signal) < 0.25  # inside the default HOLD band
    assert flat.confidence < bull.confidence
    assert flat.data["adx"] < 25


@pytest.mark.parametrize("symbol", ["BULL", "BEAR", "FLAT"])
def test_signal_and_confidence_bounded(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider, symbol: str
) -> None:
    signal = agent.analyze(symbol, _history(provider, symbol))
    assert -1.0 <= signal.signal <= 1.0
    assert 0.0 <= signal.confidence <= 1.0
    assert all(-1.0 <= v <= 1.0 for v in signal.data["components"].values())


def test_monotone_uptrend_from_builder(
    agent: TrendAnalyzerAgent, make_ohlcv: Callable[..., pd.DataFrame]
) -> None:
    closes = 100 * np.exp(np.linspace(0, 0.5, 250)) * (1 + 0.004 * np.sin(np.arange(250)))
    up = agent.analyze("UP", make_ohlcv(closes))
    down = agent.analyze("DN", make_ohlcv(closes[::-1]))
    assert up.signal > 0.3
    assert down.signal < -0.3


# ----------------------------------------------------------------------------- data sufficiency


def test_insufficient_data_is_neutral(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    history = _history(provider, "BULL").iloc[: TrendAnalyzerAgent.MIN_BARS - 1]
    signal = agent.analyze("BULL", history)
    assert signal.signal == 0.0
    assert signal.confidence == 0.0
    assert signal.rationale == "insufficient data"
    assert signal.data["bars"] == TrendAnalyzerAgent.MIN_BARS - 1


def test_empty_or_missing_history_is_neutral(agent: TrendAnalyzerAgent) -> None:
    empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    assert agent.analyze("X", empty).rationale == "insufficient data"
    assert agent.analyze("X", None).confidence == 0.0  # type: ignore[arg-type]


def test_exactly_min_bars_votes_without_long_term_components(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    history = _history(provider, "BULL").iloc[-TrendAnalyzerAgent.MIN_BARS :]
    signal = agent.analyze("BULL", history)
    assert signal.confidence > 0
    assert signal.signal > 0
    components = signal.data["components"]
    assert "sma200" not in components and "cross" not in components
    assert signal.data["sma200"] is None
    # Weights renormalise over the available components.
    assert sum(signal.data["weights"].values()) == pytest.approx(1.0, abs=1e-3)


def test_short_history_lowers_confidence(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    full = _history(provider, "BULL", period="2y")
    long_vote = agent.analyze("BULL", full)
    short_vote = agent.analyze("BULL", full.iloc[-150:])
    assert long_vote.data["sma200"] is not None
    assert short_vote.data["sma200"] is None
    assert short_vote.confidence < long_vote.confidence


# ----------------------------------------------------------------------------- explainability


def test_data_contains_rounded_indicator_values(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    history = _history(provider, "BULL")
    data = agent.analyze("BULL", history).data
    for key in (
        "close",
        "sma20",
        "sma50",
        "sma200",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr",
        "adx",
        "plus_di",
        "minus_di",
        "momentum_20d_pct",
        "realized_vol",
        "change_1d_pct",
    ):
        assert data[key] is not None, key
        assert isinstance(data[key], float)
    assert data["close"] == pytest.approx(float(history["Close"].iloc[-1]), abs=1e-4)
    assert data["bars"] == len(history)
    assert data["as_of"].startswith("2026-09-14")
    assert data["rsi"] == round(data["rsi"], 2)
    assert set(data["components"]) <= set(COMPONENT_WEIGHTS)
    assert 0.0 <= data["agreement"] <= 1.0
    assert 0.0 <= data["trend_strength"] <= 1.0


def test_signal_is_weighted_mean_of_components(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    signal = agent.analyze("FLAT", _history(provider, "FLAT"))
    components, weights = signal.data["components"], signal.data["weights"]
    expected = sum(components[k] * weights[k] for k in components)
    assert signal.signal == pytest.approx(expected, abs=2e-3)


def test_conflicting_components_reduce_confidence(
    agent: TrendAnalyzerAgent, provider: SyntheticProvider
) -> None:
    signal = agent.analyze("BULL", _history(provider, "BULL"))
    # Every component but RSI is bullish, so agreement < 1. ADX ≈ 63 saturates the strength
    # factor (0.5 + 0.5·1 = 1) and SMA200 exists, so confidence equals the agreement.
    assert signal.data["trend_strength"] == 1.0
    assert signal.data["agreement"] < 1.0
    assert signal.confidence == pytest.approx(signal.data["agreement"], abs=2e-3)


# ----------------------------------------------------------------------------- determinism / causality


def test_deterministic(agent: TrendAnalyzerAgent, provider: SyntheticProvider) -> None:
    history = _history(provider, "BULL")
    assert agent.analyze("BULL", history) == agent.analyze("BULL", history.copy())


def test_no_look_ahead(agent: TrendAnalyzerAgent, provider: SyntheticProvider) -> None:
    """The vote at bar t depends only on bars ≤ t: rewriting the future changes nothing."""
    history = _history(provider, "BULL", period="2y")
    cut = len(history) - 40
    baseline = agent.analyze("BULL", history.iloc[:cut])

    tampered = history.copy()
    tampered.iloc[cut:, tampered.columns.get_loc("Close")] *= 0.5
    tampered.iloc[cut:, tampered.columns.get_loc("Low")] *= 0.5
    assert agent.analyze("BULL", tampered.iloc[:cut]) == baseline
    assert agent.analyze("BULL", tampered).signal < baseline.signal


def test_nan_closes_are_ignored(agent: TrendAnalyzerAgent, provider: SyntheticProvider) -> None:
    history = _history(provider, "BULL").copy()
    history.iloc[5, history.columns.get_loc("Close")] = np.nan
    signal = agent.analyze("BULL", history)
    assert math.isfinite(signal.signal)
    assert signal.data["bars"] == len(history) - 1


# ----------------------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("rsi", "expected"),
    [
        (50.0, 0.0),
        (60.0, 0.15),
        (40.0, -0.15),
        (70.0, 0.3),
        (30.0, -0.3),
        (80.0, -0.65),
        (20.0, 0.65),
        (95.0, -1.0),
        (5.0, 1.0),
    ],
)
def test_rsi_component_regimes(rsi: float, expected: float) -> None:
    assert rsi_component(rsi) == pytest.approx(expected)


def test_rsi_component_flips_at_extremes() -> None:
    assert rsi_component(69.9) > 0 > rsi_component(70.1)
    assert rsi_component(29.9) > 0 > rsi_component(30.1)


@pytest.mark.parametrize(
    ("adx", "label"), [(10, "no trend"), (18, "weak trend"), (30, "trending"), (55, "strong trend")]
)
def test_adx_label(adx: float, label: str) -> None:
    assert adx_label(adx) == label
