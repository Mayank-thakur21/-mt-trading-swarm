"""Tests for the dashboard's chart builders and display helpers (pure functions, no UI).

Payloads come from the real engine (coordinator + paper broker + synthetic data), serialised
exactly as the API returns them, so the dashboard is checked against the true contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from dashboard import charts
from dashboard import formatting as fmt
from dashboard.charts import DARK, LIGHT
from trading_swarm import indicators as ind
from trading_swarm.llm.null_client import NullLLMClient
from trading_swarm.swarm import SwarmCoordinator

THEMES = [LIGHT, DARK]


# =============================================================================== fixtures


@pytest.fixture
def swarm_payloads(settings, provider, broker) -> dict[str, Any]:
    """API-shaped JSON for a BUY (executed), a vetoed SELL and a HOLD, plus account state."""
    coordinator = SwarmCoordinator(settings, provider, broker, llm=NullLLMClient())
    buy = coordinator.analyze("BULL", auto_trade=True)
    vetoed = coordinator.analyze("BEAR", auto_trade=True)
    hold = coordinator.analyze("FLAT", auto_trade=True)
    return {
        "buy": buy.model_dump(mode="json"),
        "vetoed": vetoed.model_dump(mode="json"),
        "hold": hold.model_dump(mode="json"),
        "portfolio": coordinator.portfolio().model_dump(mode="json"),
        "orders": [o.model_dump(mode="json") for o in broker.orders()],
        "trades": [t.model_dump(mode="json") for t in broker.trades()],
    }


@pytest.fixture
def history_payload(provider) -> dict[str, Any]:
    """``GET /history`` shape: compact candles + indicator arrays with leading nulls."""
    df = provider.history("BULL", period="6mo")
    macd = ind.macd(df["Close"])

    def clean(series: pd.Series) -> list[float | None]:
        return [None if not np.isfinite(v) else float(v) for v in series]

    return {
        "symbol": "BULL",
        "candles": [
            {"t": t.isoformat(), "o": r.Open, "h": r.High, "l": r.Low, "c": r.Close, "v": r.Volume}
            for t, r in df.iterrows()
        ],
        "indicators": {
            "sma20": clean(ind.sma(df["Close"], 20)),
            "sma50": clean(ind.sma(df["Close"], 50)),
            "rsi": clean(ind.rsi(df["Close"])),
            "macd": clean(macd["macd"]),
            "signal": clean(macd["signal"]),
        },
    }


@pytest.fixture
def curve() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=120, tz="UTC", name="t")
    rng = np.random.default_rng(7)
    equity = 100_000 * np.cumprod(1 + rng.normal(0.0006, 0.01, len(index)))
    bench = 100_000 * np.cumprod(1 + rng.normal(0.0004, 0.012, len(index)))
    return pd.DataFrame({"equity": equity, "benchmark": bench}, index=index)


def _layout_colors(fig: go.Figure) -> tuple[Any, Any, Any]:
    layout = fig.layout
    return layout.paper_bgcolor, layout.plot_bgcolor, layout.font.color


def _all_colors(fig: go.Figure) -> str:
    return fig.to_json().lower()


# =============================================================================== theme


def test_get_theme_variants() -> None:
    assert charts.get_theme(True) is DARK
    assert charts.get_theme("dark") is DARK
    assert charts.get_theme(" Dark ") is DARK
    assert charts.get_theme("light") is LIGHT
    assert charts.get_theme(None) is LIGHT
    assert charts.get_theme(False) is LIGHT
    assert DARK.is_dark and not LIGHT.is_dark


def test_rgba() -> None:
    assert charts.rgba("#2a78d6", 0.5) == "rgba(42, 120, 214, 0.500)"
    assert charts.rgba("2a78d6", 2) == "rgba(42, 120, 214, 1.000)"
    with pytest.raises(ValueError):
        charts.rgba("#fff", 0.5)


@pytest.mark.parametrize(
    ("value", "expected"), [(0.3, "up"), (-0.2, "down"), (0.0, "neutral"), (None, "neutral")]
)
def test_direction_color(value, expected) -> None:
    assert charts.direction_color(value, LIGHT) == getattr(LIGHT, expected)
    assert charts.direction_color(float("nan"), LIGHT) == LIGHT.neutral


def test_action_color() -> None:
    assert charts.action_color("buy", DARK) == DARK.up
    assert charts.action_color("SELL", DARK) == DARK.down
    assert charts.action_color("HOLD", DARK) == DARK.neutral
    assert charts.action_color(None, DARK) == DARK.neutral


def test_direction_colors_avoid_green_red() -> None:
    """Bullish/bearish use blue/red (CVD-safe), never the green/red trading convention."""
    for theme in THEMES:
        assert theme.up.startswith("#") and theme.up != theme.good
        assert theme.up in theme.series[:1]  # categorical slot 1 (blue)


def test_template_is_self_contained() -> None:
    template = charts.plotly_template(DARK)
    assert template.layout.paper_bgcolor == DARK.surface
    assert template.layout.plot_bgcolor == DARK.surface
    assert template.layout.xaxis.gridcolor == DARK.grid
    assert list(template.layout.colorway) == list(DARK.series)


# =============================================================================== shaping


def test_to_utc_index_formats() -> None:
    idx = charts.to_utc_index(["2026-01-02T00:00:00Z", "2026-01-03T05:00:00+05:00", None, "x"])
    assert str(idx.tz) == "UTC"
    assert idx[0] == pd.Timestamp("2026-01-02", tz="UTC")
    assert idx[1] == pd.Timestamp("2026-01-03", tz="UTC")
    assert idx[2:].isna().all()
    seconds = charts.to_utc_index([1_700_000_000, 1_700_086_400])
    millis = charts.to_utc_index([1_700_000_000_000])
    assert seconds[0] == millis[0] == pd.Timestamp(1_700_000_000, unit="s", tz="UTC")
    assert len(charts.to_utc_index([])) == 0


def test_candles_frame_from_api_payload(history_payload) -> None:
    frame = charts.candles_frame(history_payload)
    assert len(frame) == len(history_payload["candles"])
    assert {"open", "high", "low", "close", "volume", "sma20", "sma50", "rsi"} <= set(frame)
    assert frame.index.is_monotonic_increasing and str(frame.index.tz) == "UTC"
    # hist is derived as macd − signal when the API does not send it
    np.testing.assert_allclose(frame["hist"], frame["macd"] - frame["signal"])
    assert frame["sma20"].isna().sum() == 19  # leading nulls preserved, not shifted


def test_candles_frame_aligns_short_indicators_right_and_cleans_rows() -> None:
    payload = {
        "candles": [
            {"timestamp": "2026-01-05", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
            {"timestamp": "2026-01-02", "open": 1, "high": 2, "low": 0.5, "close": 1.2},
            {"timestamp": "2026-01-06", "open": None, "high": 2, "low": 0.5, "close": 1.5},
            {"timestamp": "garbage", "open": 1, "high": 2, "low": 0.5, "close": 1.5},
            {"timestamp": "2026-01-05", "open": 1, "high": 2, "low": 0.5, "close": 1.7},
        ],
        "indicators": {"RSI_14": [55.0], "macd_hist": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
    }
    frame = charts.candles_frame(payload)
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2026-01-02", "2026-01-05"]
    assert frame.loc["2026-01-05", "close"].item() == 1.7  # duplicate: last one wins
    assert frame["rsi"].iloc[-1] == 55.0  # right-aligned to the latest candle
    assert "sma20" not in frame  # too few bars to compute a fallback
    # Indicators align to the raw rows (last 5 of 6 values) *before* bad rows are dropped
    # and duplicates collapse, so each value stays attached to its own bar.
    assert frame["hist"].tolist() == [0.3, 0.6]


def test_candles_frame_computes_missing_moving_averages() -> None:
    closes = np.linspace(10, 70, 60)
    payload = {
        "candles": [
            {"t": d.isoformat(), "o": c, "h": c + 1, "l": c - 1, "c": c}
            for d, c in zip(pd.bdate_range("2026-01-01", periods=60, tz="UTC"), closes, strict=True)
        ]
    }
    frame = charts.candles_frame(payload)
    assert frame["sma20"].iloc[-1] == pytest.approx(closes[-20:].mean())
    assert frame["sma50"].iloc[-1] == pytest.approx(closes[-50:].mean())


def test_equity_frame() -> None:
    payload = {
        "points": [
            {"t": "2026-01-03T00:00:00Z", "equity": 101.0, "cash": 50.0},
            {"t": "2026-01-02T00:00:00Z", "equity": 100.0, "cash": 100.0},
            {"t": "bad", "equity": 1.0, "cash": 1.0},
            {"t": "2026-01-04T00:00:00Z", "equity": None, "cash": 1.0},
        ]
    }
    frame = charts.equity_frame(payload)
    assert frame["equity"].tolist() == [100.0, 101.0]
    assert charts.equity_frame(payload["points"]).equals(frame)
    assert charts.equity_frame({"points": []}).empty


@pytest.mark.parametrize(
    "encode",
    [
        lambda df: [{"t": t.isoformat(), **row} for t, row in df.to_dict("index").items()],
        lambda df: {
            "index": [t.isoformat() for t in df.index],
            "columns": list(df.columns),
            "data": df.to_numpy().tolist(),
        },
        lambda df: {"date": [t.isoformat() for t in df.index], **df.to_dict("list")},
        lambda df: {c: {t.isoformat(): v for t, v in s.items()} for c, s in df.items()},
        lambda df: {t.isoformat(): row for t, row in df.to_dict("index").items()},
    ],
    ids=["records", "split", "columns-lists", "columns-dict", "index-dict"],
)
def test_backtest_curve_frame_accepts_common_encodings(encode: Callable[..., Any], curve) -> None:
    source = curve.rename(columns={"benchmark": "benchmark_equity"})
    frame = charts.backtest_curve_frame({"equity_curve": encode(source)})
    assert list(frame.columns) == ["equity", "benchmark"]
    assert len(frame) == len(curve)
    np.testing.assert_allclose(frame["equity"].to_numpy(), curve["equity"].to_numpy())
    assert str(frame.index.tz) == "UTC"


def test_backtest_curve_frame_empty_inputs() -> None:
    for payload in ({}, {"equity_curve": None}, {"equity_curve": []}, {"equity_curve": {}}):
        frame = charts.backtest_curve_frame(payload)
        assert frame.empty and "equity" in frame.columns


def test_drawdown_pct_known_series() -> None:
    series = pd.Series([100.0, 110.0, 99.0, 121.0, 0.0])
    assert charts.drawdown_pct(series).round(6).tolist() == [0.0, 0.0, -10.0, 0.0, -100.0]
    assert charts.drawdown_pct(pd.Series([0.0, 0.0])).tolist() == [0.0, 0.0]


# =============================================================================== figures


@pytest.mark.parametrize("theme", THEMES, ids=lambda t: t.name)
def test_every_builder_pins_colors_on_the_layout(theme, swarm_payloads, history_payload, curve):
    """theme=None safety: Streamlit only respects colours set on the layout itself."""
    buy = swarm_payloads["buy"]
    figures = [
        charts.empty_figure("nothing", theme=theme),
        charts.equity_curve_figure(curve[["equity"]], initial_capital=100_000, theme=theme),
        charts.allocation_figure(
            swarm_payloads["portfolio"]["positions"],
            swarm_payloads["portfolio"]["cash"],
            theme=theme,
        ),
        charts.consensus_gauge(buy["score"], theme=theme),
        charts.agent_signals_figure(buy["signals"], consensus=buy["score"], theme=theme),
        charts.price_figure(charts.candles_frame(history_payload), symbol="BULL", theme=theme),
        charts.backtest_equity_figure(curve, initial_capital=100_000, theme=theme),
        charts.drawdown_figure(curve, theme=theme),
    ]
    for fig in figures:
        assert isinstance(fig, go.Figure)
        assert _layout_colors(fig) == (theme.surface, theme.surface, theme.ink)
        assert fig.layout.template.layout.paper_bgcolor == theme.surface
        fig.to_json()  # serialisable (what st.plotly_chart sends)


def test_empty_figure_shows_message() -> None:
    fig = charts.empty_figure("No data here")
    assert fig.layout.annotations[0].text == "No data here"
    assert fig.layout.xaxis.visible is False


def test_equity_curve_figure(curve) -> None:
    fig = charts.equity_curve_figure(curve, initial_capital=100_000)
    line = fig.data[0]
    assert line.line.color == LIGHT.series[0]
    assert line.showlegend is False  # one series: the subheader names it
    texts = [a.text for a in fig.layout.annotations]
    assert any("Initial capital $100,000" in t for t in texts)
    assert any(t.strip().startswith("$") for t in texts)  # end label
    assert fig.data[0].mode == "lines"  # many points: no markers


def test_equity_curve_figure_handles_missing_data() -> None:
    fig = charts.equity_curve_figure(pd.DataFrame())
    assert "No equity snapshots" in fig.layout.annotations[0].text
    two = charts.equity_frame(
        {"points": [{"t": "2026-01-02", "equity": 1.0}, {"t": "2026-01-03", "equity": 2.0}]}
    )
    assert charts.equity_curve_figure(two).data[0].mode == "lines+markers"


def test_allocation_figure_folds_tail_and_keeps_entity_colors() -> None:
    positions = [
        {"symbol": s, "market_value": v}
        for s, v in [("MSFT", 500), ("AAPL", 400), ("NVDA", 300), ("TSLA", 50), ("AMD", 40)]
    ]
    fig = charts.allocation_figure(positions, cash=1_000, max_slices=3)
    pie = fig.data[0]
    assert list(pie.labels) == ["MSFT", "AAPL", "NVDA", "Other", "Cash"]
    assert list(pie.values) == [500, 400, 300, 90, 1_000]
    colors = dict(zip(pie.labels, pie.marker.colors, strict=True))
    # alphabetical slot assignment: colour does not depend on the size ranking
    assert colors["AAPL"] == LIGHT.series[0]
    assert colors["MSFT"] == LIGHT.series[1]
    assert colors["NVDA"] == LIGHT.series[2]
    assert colors["Other"] == LIGHT.muted and colors["Cash"] == LIGHT.cash
    assert "56%" in fig.layout.annotations[0].text  # 1,290 of 2,290 invested
    # re-ranking the same holdings keeps each symbol's colour
    reranked = charts.allocation_figure(
        [{"symbol": "AAPL", "market_value": 900}, {"symbol": "MSFT", "market_value": 10}], cash=0
    )
    recolors = dict(zip(reranked.data[0].labels, reranked.data[0].marker.colors, strict=True))
    assert recolors["AAPL"] == LIGHT.series[0] and recolors["MSFT"] == LIGHT.series[1]


def test_allocation_figure_fallbacks() -> None:
    fig = charts.allocation_figure(
        [{"symbol": "aapl", "quantity": 10, "market_price": None, "avg_price": 5.0}], cash=None
    )
    assert list(fig.data[0].labels) == ["AAPL"]
    assert list(fig.data[0].values) == [50.0]
    empty = charts.allocation_figure([], cash=1_000)
    assert "100% cash" in empty.layout.annotations[0].text


def test_allocation_from_real_portfolio(swarm_payloads) -> None:
    portfolio = swarm_payloads["portfolio"]
    fig = charts.allocation_figure(portfolio["positions"], portfolio["cash"])
    assert list(fig.data[0].labels) == ["BULL", "Cash"]


@pytest.mark.parametrize(
    ("score", "needle"), [(0.6, "up"), (-0.4, "down"), (0.1, "neutral"), (None, "neutral")]
)
def test_consensus_gauge_needle_follows_zone(score, needle) -> None:
    fig = charts.consensus_gauge(score, buy_threshold=0.25, sell_threshold=-0.25)
    gauge = fig.data[0].gauge
    assert gauge.threshold.line.color == getattr(LIGHT, needle)
    assert fig.data[0].value == (0.0 if score is None else score)
    assert gauge.axis.range == (-1, 1)
    assert list(gauge.axis.tickvals) == [-1, -0.25, 0, 0.25, 1]
    assert [s.range for s in gauge.steps] == [(-1, -0.25), (-0.25, 0.25), (0.25, 1)]


def test_consensus_gauge_clips_and_honours_action() -> None:
    fig = charts.consensus_gauge(3.0, action="SELL")
    assert fig.data[0].value == 1.0
    assert fig.data[0].gauge.threshold.line.color == LIGHT.down


def test_agent_signals_figure(swarm_payloads) -> None:
    decision = swarm_payloads["vetoed"]
    fig = charts.agent_signals_figure(
        decision["signals"],
        consensus=decision["score"],
        consensus_confidence=decision["confidence"],
    )
    bar = fig.data[0]
    labels = list(bar.y)
    assert labels[-1].startswith("Trend")  # first agent drawn on top
    assert labels[0].startswith("Consensus")
    assert all("conf" in label for label in labels)  # confidence is in the text too
    assert all(v < 0 for v in bar.x)  # BEAR: every vote bearish
    assert all(c.startswith("rgba(227, 73, 72") for c in bar.marker.color)  # LIGHT.down
    assert list(bar.text) == [f"{v:+.2f}" for v in bar.x]
    dashed = [s for s in fig.layout.shapes if s.line.dash == "dash"]
    assert sorted(s.x0 for s in dashed) == [-0.25, 0.25]


def test_agent_signals_opacity_tracks_confidence() -> None:
    fig = charts.agent_signals_figure(
        [
            {"agent": "trend", "signal": 0.5, "confidence": 1.0},
            {"agent": "sentiment", "signal": 0.5, "confidence": 0.0},
        ]
    )
    alphas = [float(c.rsplit(",", 1)[1].rstrip(")")) for c in fig.data[0].marker.color]
    assert alphas == [0.35, 1.0]  # reversed order: sentiment (bottom), trend (top)


def test_agent_signals_empty() -> None:
    assert "No agent signals" in charts.agent_signals_figure([]).layout.annotations[0].text


def test_price_figure_panels_and_levels(history_payload) -> None:
    frame = charts.candles_frame(history_payload)
    fig = charts.price_figure(
        frame, symbol="BULL", stop_price=170.0, take_profit_price=190.0, entry_price=180.0
    )
    names = [t.name for t in fig.data]
    assert names == ["BULL", "SMA20", "SMA50", "RSI 14", "Histogram", "MACD", "Signal"]
    candle = fig.data[0]
    assert candle.increasing.line.color == LIGHT.up
    assert candle.decreasing.line.color == LIGHT.down
    assert fig.data[2].line.dash == "dash"  # SMA50 distinguished from SMA20 without colour
    level_texts = {a.text for a in fig.layout.annotations}
    assert {"Stop 170.00", "Target 190.00", "Entry 180.00"} <= level_texts
    assert fig.layout.yaxis2.range == (0, 100)
    assert fig.layout.xaxis.rangebreaks[0].bounds == ("sat", "mon")  # business-day data
    assert fig.layout.xaxis.rangeslider.visible is False


def test_price_figure_without_indicators_or_weekend_gaps() -> None:
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")  # trades weekends
    closes = np.linspace(100, 130, 30)
    frame = pd.DataFrame(
        {"open": closes, "high": closes + 1, "low": closes - 1, "close": closes}, index=index
    )
    fig = charts.price_figure(frame, stop_price=float("nan"), take_profit_price=-1)
    assert [t.name for t in fig.data] == ["Price"]
    assert not fig.layout.xaxis.rangebreaks
    assert not fig.layout.annotations  # invalid levels are skipped
    empty = charts.price_figure(pd.DataFrame())
    assert "No price history" in empty.layout.annotations[0].text


def test_backtest_figures(curve) -> None:
    fig = charts.backtest_equity_figure(curve, initial_capital=100_000)
    lines = [t for t in fig.data if t.mode == "lines"]
    assert [t.name for t in lines] == ["Strategy", "Buy & hold"]
    assert [t.line.color for t in lines] == [LIGHT.series[0], LIGHT.series[1]]
    labels = [a.text for a in fig.layout.annotations]
    assert any(t.startswith("Strategy $") for t in labels)
    assert any(t.startswith("Buy & hold $") for t in labels)

    dd = charts.drawdown_figure(curve)
    assert [t.name for t in dd.data] == ["Buy & hold", "Strategy"]  # strategy on top
    strategy = dd.data[1]
    assert max(strategy.y) <= 0
    assert min(strategy.y) == pytest.approx(charts.drawdown_pct(curve["equity"]).min())
    assert dd.layout.annotations[0].text.startswith("Max -")


def test_backtest_figures_without_benchmark(curve) -> None:
    only = curve[["equity"]]
    assert [t.name for t in charts.backtest_equity_figure(only).data if t.mode == "lines"] == [
        "Strategy"
    ]
    assert [t.name for t in charts.drawdown_figure(only).data] == ["Strategy"]
    for builder in (charts.backtest_equity_figure, charts.drawdown_figure):
        assert "No equity curve" in builder(pd.DataFrame()).layout.annotations[0].text


def test_end_labels_do_not_overlap() -> None:
    index = pd.bdate_range("2026-01-01", periods=3, tz="UTC")
    close = pd.DataFrame(
        {"equity": [100.0, 101.0, 102.0], "benchmark": [100.0, 101.0, 102.1]}, index=index
    )
    fig = charts.backtest_equity_figure(close)
    shifts = sorted(a.yshift for a in fig.layout.annotations)
    assert shifts == [-9, 9]


# =============================================================================== formatting


@pytest.mark.parametrize(
    ("value", "kwargs", "expected"),
    [
        (1234.5, {}, "$1,234.50"),
        (-3, {}, "-$3.00"),
        (12.0, {"signed": True}, "+$12.00"),
        (0, {"signed": True}, "$0.00"),
        (None, {}, "—"),
        (float("nan"), {}, "—"),
        (float("inf"), {}, "—"),
        ("abc", {}, "—"),
        (True, {}, "—"),
    ],
)
def test_fmt_money(value, kwargs, expected) -> None:
    assert fmt.fmt_money(value, **kwargs) == expected


@pytest.mark.parametrize(
    ("value", "signed", "expected"),
    [
        (99_980.67, False, "$99,981"),  # fits a quarter-width tile at 1280 px
        (100_000.0, False, "$100,000"),
        (-12_345.5, True, "-$12,346"),
        (9_999.99, False, "$9,999.99"),  # below the cut-off the cents stay
        (-19.33, True, "-$19.33"),
        (None, False, "—"),
    ],
)
def test_fmt_money_kpi(value, signed, expected) -> None:
    assert fmt.fmt_money_kpi(value, signed=signed) == expected


def test_number_formatters() -> None:
    assert fmt.fmt_pct(5.0) == "5.00%"
    assert fmt.fmt_pct(-1.234, signed=True, decimals=1) == "-1.2%"
    assert fmt.fmt_pct(None) == "—"
    assert fmt.fmt_fraction_pct(0.8187) == "82%"
    assert fmt.fmt_signal(0.7427) == "+0.74"
    assert fmt.fmt_signal(-0.5) == "-0.50"
    assert fmt.fmt_number(1234.567, 1) == "1,234.6"
    assert fmt.fmt_price(182.0815) == "182.08"
    assert fmt.fmt_price(0.012345) == "0.0123"
    assert fmt.format_metric(6.0, "int") == "6"
    assert fmt.format_metric(None, "int") == "—"
    assert fmt.format_metric(19.4, "days") == "19"
    assert fmt.format_metric(1.234, "ratio") == "1.23"


def test_timestamps() -> None:
    assert fmt.fmt_timestamp("2026-11-02T17:57:37.955190Z") == "2026-11-02 17:57 UTC"
    assert fmt.fmt_timestamp("2026-11-02T19:57:37+02:00", seconds=True) == "2026-11-02 17:57:37 UTC"
    assert fmt.fmt_timestamp(None) == "—"
    assert fmt.fmt_timestamp("not a date") == "—"
    parsed = fmt.parse_timestamp("2026-01-02T03:04:05")
    assert parsed is not None and parsed.tzinfo is not None and parsed.hour == 3


def test_trade_math_helpers() -> None:
    assert fmt.pct_distance(100, 95) == pytest.approx(-5.0)
    assert fmt.pct_distance(0, 95) is None
    assert fmt.reward_risk(100, 95, 110) == pytest.approx(2.0)
    assert fmt.reward_risk(100, 101, 110) is None
    assert fmt.reward_risk(None, 95, 110) is None


def test_escape_markdown_neutralises_api_text() -> None:
    text = "risk size 268 sh ($1,000 at risk / $3.73 per share) [link](http://x) *bold* <b>"
    escaped = fmt.escape_markdown(text)
    assert "\\$1,000" in escaped and "\\$3.73" in escaped
    assert "\\[link\\]" in escaped and "\\*bold\\*" in escaped and "\\<b\\>" in escaped
    assert fmt.escape_markdown(None) == ""
    assert fmt.escape_markdown("plain text 1.5×ATR") == "plain text 1.5×ATR"


def test_resolve_thresholds_sources() -> None:
    assert fmt.resolve_thresholds({}, {}) == fmt.Thresholds(0.25, -0.25)
    assert fmt.resolve_thresholds({"buy_threshold": 0.3, "sell_threshold": -0.4}, {}) == (
        fmt.Thresholds(0.3, -0.4)
    )
    nested = {"thresholds": {"buy": 0.0, "sell": -0.1}}  # 0.0 is a legitimate threshold
    assert fmt.resolve_thresholds(nested, {}) == fmt.Thresholds(0.0, -0.1)
    env = {"BUY_THRESHOLD": "0.5", "SELL_THRESHOLD": "-0.5"}
    assert fmt.resolve_thresholds(None, env) == fmt.Thresholds(0.5, -0.5)
    inconsistent = {"buy_threshold": -0.5, "sell_threshold": 0.5}
    assert fmt.resolve_thresholds(inconsistent, {}) == fmt.Thresholds()


def test_exit_helpers() -> None:
    stop = {"symbol": "BEAR", "status": "FILLED", "reason": "stop_loss"}
    other = {"symbol": "FLAT", "status": "FILLED", "reason": "take_profit"}
    rejected = {"symbol": "BEAR", "status": "REJECTED", "reason": "stop_loss"}
    assert fmt.exit_label(stop) == "stop-loss"
    assert fmt.exit_label(other) == "take-profit"
    assert fmt.exit_label({"reason": ""}) == "protective exit"
    decision = {"symbol": "bear", "exits": [other, rejected, stop, "junk"]}
    assert fmt.decision_exits(decision) == [other, rejected, stop]
    assert fmt.own_exit(decision) is stop
    assert fmt.own_exit({"symbol": "BEAR", "exits": [rejected]}) is None
    assert fmt.own_exit({"symbol": "BEAR"}) is None  # payloads from older API versions


def test_proposed_action() -> None:
    thresholds = fmt.Thresholds(0.25, -0.25)
    assert fmt.proposed_action(0.25, thresholds) == "BUY"
    assert fmt.proposed_action(-0.72, thresholds) == "SELL"
    assert fmt.proposed_action(0.1, thresholds) == "HOLD"
    assert fmt.proposed_action(None, thresholds) == "HOLD"


def test_labels() -> None:
    assert fmt.humanize("position_limit_ok") == "Position limit ok"
    assert fmt.action_badge_color("buy") == "blue"
    assert fmt.action_badge_color("SELL") == "red"
    assert fmt.action_badge_color(None) == "gray"
    assert fmt.risk_checks({"cash_ok": True, "new_rule": False}) == [
        ("Enough cash (incl. costs)", True),
        ("New rule", False),
    ]


def test_tables_from_real_payloads(swarm_payloads) -> None:
    portfolio = swarm_payloads["portfolio"]
    positions = fmt.positions_table(portfolio)
    assert list(positions.columns) == list(fmt.POSITION_COLUMNS)
    row = positions.iloc[0]
    assert row["Symbol"] == "BULL"
    position = portfolio["positions"][0]
    assert row["To stop %"] == pytest.approx(
        (position["stop_price"] / position["market_price"] - 1) * 100
    )
    assert str(positions["Opened (UTC)"].dtype).startswith("datetime64")

    orders = fmt.orders_table(swarm_payloads["orders"])
    assert orders.iloc[0]["Status"] == "FILLED" and orders.iloc[0]["Side"] == "BUY"
    trades = fmt.trades_table(swarm_payloads["trades"])
    first = trades.iloc[0]
    assert first["Notional"] == pytest.approx(first["Price"] * first["Qty"])

    decisions = [swarm_payloads[k] for k in ("buy", "vetoed", "hold")]
    table = fmt.decisions_table(decisions)
    assert table["Action"].tolist() == ["BUY", "HOLD", "HOLD"]
    assert table["Executed"].tolist() == [True, False, False]
    assert table["Qty"].iloc[1] == 0
    assert table["Trend"].iloc[1] < 0


def test_agent_detail_tables(swarm_payloads) -> None:
    buy = swarm_payloads["buy"]
    trend = fmt.signal_for(buy, "trend")
    components = fmt.trend_components_table(trend)
    assert "MACD histogram" in components["Component"].tolist()
    total = (components["Score"] * components["Weight"]).sum()
    assert components["Contribution"].sum() == pytest.approx(total)
    assert components["Contribution"].sum() == pytest.approx(trend["signal"], abs=1e-3)
    rows = dict(fmt.indicator_rows(trend))
    assert rows["RSI 14"] == f"{trend['data']['rsi']:,.1f}"
    assert rows["Realized vol (annual)"].endswith("%")  # fraction shown as percent
    headlines = fmt.headlines_table(fmt.signal_for(buy, "sentiment"))
    assert len(headlines) == 4
    assert "Relevance" not in headlines  # lexicon has no relevance: column dropped
    assert headlines["Matched terms"].str.len().gt(0).all()
    assert fmt.signal_for(buy, "nope") is None
    assert fmt.headlines_table(None).empty and fmt.trend_components_table(None).empty


def test_alert_and_error_tables() -> None:
    alerts = fmt.alerts_table(
        [
            {
                "timestamp": "2026-11-02T10:00:00Z",
                "rule": "signal",
                "status": "sent",
                "channel": "email",
            }
        ]
    )
    assert alerts.iloc[0]["Status"] == "sent"
    errors = fmt.errors_table(
        [{"symbol": "NOPE", "error": "no data", "code": "not_found"}, {"symbol": "X", "error": "e"}]
    )
    assert errors["Code"].tolist() == ["not_found", "error"]
    assert fmt.alerts_table([]).empty


def test_metrics_table() -> None:
    strategy = {"total_return_pct": 12.345, "sharpe": 1.234, "num_trades": 6, "avg_win": 171.1}
    benchmark = {"total_return_pct": 20.0, "sharpe": None, "num_trades": 0}
    table = fmt.metrics_table(strategy, benchmark)
    rows = {r["Metric"]: (r["Strategy"], r["Buy & hold"]) for r in table.to_dict("records")}
    assert rows["Total return"] == ("12.35%", "20.00%")
    assert rows["Sharpe ratio"] == ("1.23", "—")
    assert rows["Closed trades"] == ("6", "0")
    assert rows["Average win"] == ("$171.10", "—")
    assert "CAGR" not in rows  # absent from both: row skipped
    assert list(fmt.metrics_table(strategy).columns) == ["Metric", "Strategy"]


def test_style_table_formats_and_colours() -> None:
    frame = pd.DataFrame(
        {"P&L": [12.5, -3.0, None], "Score": [0.5, -0.2, 0.0], "Price": [1.5, 2.0, 3.0]}
    )
    html = fmt.style_table(
        frame,
        signed_money=("P&L",),
        signals=("Score",),
        prices=("Price",),
        pnl=("P&L",),
        directional=("Score",),
        theme=DARK,
    ).to_html()
    assert "+$12.50" in html and "-$3.00" in html and "—" in html
    assert DARK.good in html and DARK.bad in html  # money outcome: status colours
    assert DARK.up in html and DARK.down in html  # direction: chart colours
    assert "+0.50" in html
    assert fmt.pnl_color(0) == "" and fmt.direction_css(None) == ""


def test_to_csv_bytes() -> None:
    data = fmt.to_csv_bytes(pd.DataFrame({"a": [1], "b": ["x"]}))
    assert data == b"a,b\n1,x\n"
    assert not math.isnan(len(data))
