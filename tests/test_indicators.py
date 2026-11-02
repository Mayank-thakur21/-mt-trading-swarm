"""Indicator tests against hand-computed values (and the draft's bugs)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from trading_swarm import indicators as ind
from trading_swarm.data import SyntheticProvider


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2024-01-02", periods=len(values), tz="UTC"))


def _ema_by_hand(values: list[float], alpha: float) -> list[float]:
    out = [values[0]]
    for value in values[1:]:
        out.append(alpha * value + (1 - alpha) * out[-1])
    return out


# ----------------------------------------------------------------------------- averages


def test_sma_hand_computed() -> None:
    result = ind.sma(_series([1, 2, 3, 4, 5]), 3)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2:].tolist() == pytest.approx([2.0, 3.0, 4.0])


def test_ema_is_recursive_with_adjust_false() -> None:
    # span 3 → α = 2 / (3 + 1) = 0.5
    result = ind.ema(_series([1, 2, 3, 4, 5]), 3)
    assert result.tolist() == pytest.approx([1.0, 1.5, 2.25, 3.125, 4.0625])


def test_wilder_smoothing_uses_alpha_one_over_n() -> None:
    values = [2.0, 4.0, 6.0, 8.0]
    result = ind.wilder_smooth(_series(values), 2)
    expected = _ema_by_hand(values, 0.5)
    assert math.isnan(result.iloc[0])  # min_periods = n
    assert result.iloc[1:].tolist() == pytest.approx(expected[1:])


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_invalid_windows_raise(bad: object) -> None:
    with pytest.raises(ValueError):
        ind.sma(_series([1.0, 2.0, 3.0]), bad)  # type: ignore[arg-type]


def test_series_type_is_checked() -> None:
    with pytest.raises(TypeError):
        ind.sma([1, 2, 3], 2)  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- RSI


def test_rsi_matches_wilder_recursion() -> None:
    closes = [44.0, 44.5, 44.0, 45.0, 46.0, 45.5]
    n = 3
    result = ind.rsi(_series(closes), n)

    deltas = np.diff(closes)
    gains = np.clip(deltas, 0, None)
    losses = np.clip(-deltas, 0, None)
    avg_gain = _ema_by_hand([float(g) for g in gains], 1 / n)[n - 1 :]
    avg_loss = _ema_by_hand([float(x) for x in losses], 1 / n)[n - 1 :]
    expected = [100 - 100 / (1 + g / x) for g, x in zip(avg_gain, avg_loss, strict=True)]

    assert result.iloc[:n].isna().all()  # first value needs n price changes
    assert result.iloc[n:].tolist() == pytest.approx(expected)
    # Hand check of the first value: avg gain 0.5556 / avg loss 0.1111 → RS 5 → RSI 83.33
    assert result.iloc[3] == pytest.approx(100 - 100 / 6)


def test_rsi_is_not_the_drafts_simple_average_version() -> None:
    rng = np.random.default_rng(7)
    close = _series(list(100 + np.cumsum(rng.normal(0, 1, 120))))
    delta = close.diff()
    sma_gain = delta.clip(lower=0).rolling(14).mean()
    sma_loss = (-delta.clip(upper=0)).rolling(14).mean()
    draft = 100 - 100 / (1 + sma_gain / sma_loss)
    assert not np.allclose(ind.rsi(close).iloc[30:], draft.iloc[30:])


def test_rsi_edge_cases() -> None:
    assert ind.rsi(_series([10.0] * 30)).iloc[-1] == pytest.approx(50.0)
    assert ind.rsi(_series([float(i) for i in range(1, 31)])).iloc[-1] == pytest.approx(100.0)
    assert ind.rsi(_series([float(i) for i in range(30, 0, -1)])).iloc[-1] == pytest.approx(0.0)


def test_rsi_bounds() -> None:
    rng = np.random.default_rng(1)
    close = _series(list(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))))
    values = ind.rsi(close).dropna()
    assert ((values >= 0) & (values <= 100)).all()


# ----------------------------------------------------------------------------- MACD


def test_macd_signal_is_ema_of_macd_line() -> None:
    rng = np.random.default_rng(3)
    values = list(100 + np.cumsum(rng.normal(0, 1, 80)))
    close = _series(values)
    result = ind.macd(close, 12, 26, 9)

    fast = _ema_by_hand(values, 2 / 13)
    slow = _ema_by_hand(values, 2 / 27)
    line = [f - s for f, s in zip(fast, slow, strict=True)]
    signal = _ema_by_hand(line, 2 / 10)

    assert list(result.columns) == ["macd", "signal", "hist"]
    assert result["macd"].tolist() == pytest.approx(line)
    assert result["signal"].tolist() == pytest.approx(signal)
    assert result["hist"].tolist() == pytest.approx(
        [m - s for m, s in zip(line, signal, strict=True)]
    )
    # The draft's bug: signal = EMA(close, 9). Make sure we are nowhere near that.
    assert not np.allclose(result["signal"], ind.ema(close, 9))


def test_macd_rejects_fast_not_below_slow() -> None:
    with pytest.raises(ValueError):
        ind.macd(_series([1.0] * 40), fast=26, slow=12)


# ----------------------------------------------------------------------------- Bollinger


def test_bollinger_uses_population_std() -> None:
    close = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    bands = ind.bollinger(close, n=5, k=2.0)
    std = math.sqrt(2.0)  # population std of 1..5
    last = bands.iloc[-1]
    assert last["middle"] == pytest.approx(3.0)
    assert last["upper"] == pytest.approx(3.0 + 2 * std)
    assert last["lower"] == pytest.approx(3.0 - 2 * std)
    assert last["bandwidth"] == pytest.approx(4 * std / 3.0)
    assert last["percent_b"] == pytest.approx((5.0 - (3.0 - 2 * std)) / (4 * std))


def test_bollinger_flat_series_has_nan_percent_b() -> None:
    bands = ind.bollinger(_series([10.0] * 25))
    assert math.isnan(bands["percent_b"].iloc[-1])
    assert bands["bandwidth"].iloc[-1] == pytest.approx(0.0)


# ----------------------------------------------------------------------------- ATR / ADX


@pytest.fixture
def small_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [10.0, 10.5, 11.5, 10.0],
            "High": [11.0, 12.0, 12.5, 11.0],
            "Low": [9.0, 10.0, 11.0, 9.5],
            "Close": [10.5, 11.5, 12.0, 10.0],
        },
        index=pd.bdate_range("2024-01-02", periods=4, tz="UTC"),
    )


def test_true_range_hand_computed(small_bars: pd.DataFrame) -> None:
    tr = ind.true_range(small_bars)
    # bar0: H-L=2; bar1: max(2, |12-10.5|, |10-10.5|)=2; bar2: max(1.5, 1.0, 0.5)=1.5;
    # bar3: max(1.5, |11-12|, |9.5-12|)=2.5 (gap down through the previous close)
    assert tr.tolist() == pytest.approx([2.0, 2.0, 1.5, 2.5])


def test_atr_is_wilder_smoothed_true_range(small_bars: pd.DataFrame) -> None:
    atr = ind.atr(small_bars, n=2)
    expected = _ema_by_hand([2.0, 2.0, 1.5, 2.5], 0.5)
    assert math.isnan(atr.iloc[0])
    assert atr.iloc[1:].tolist() == pytest.approx(expected[1:])


def test_atr_accepts_lowercase_columns(small_bars: pd.DataFrame) -> None:
    lower = small_bars.rename(columns=str.lower)
    pd.testing.assert_series_equal(ind.atr(lower, 2), ind.atr(small_bars, 2))


def test_missing_column_raises_key_error(small_bars: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="High"):
        ind.atr(small_bars.drop(columns=["High"]))


def test_adx_strong_uptrend(make_ohlcv: Callable[..., pd.DataFrame]) -> None:
    bars = make_ohlcv([100 * 1.01**i for i in range(80)])
    result = ind.adx(bars, 14)
    last = result.iloc[-1]
    assert last["plus_di"] > last["minus_di"]
    assert last["adx"] > 25
    # Warm-up: DI from bar n, ADX from bar 2n - 1.
    assert result["plus_di"].iloc[:14].isna().all()
    assert result["adx"].iloc[:27].isna().all()
    assert not math.isnan(result["adx"].iloc[27])


def test_adx_bounded_and_zero_for_flat_market(make_ohlcv: Callable[..., pd.DataFrame]) -> None:
    rng = np.random.default_rng(11)
    noisy = make_ohlcv(list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))))
    values = ind.adx(noisy).dropna()
    assert ((values >= 0) & (values <= 100)).all().all()

    flat = make_ohlcv([50.0] * 60, spread=0.0)
    assert ind.adx(flat)["adx"].iloc[-1] == pytest.approx(0.0)


# ----------------------------------------------------------------------------- returns


def test_pct_change_is_a_fraction() -> None:
    result = ind.pct_change(_series([100.0, 110.0, 121.0]), 1)
    assert math.isnan(result.iloc[0])
    assert result.iloc[1:].tolist() == pytest.approx([0.1, 0.1])
    assert ind.pct_change(_series([100.0, 110.0, 121.0]), 2).iloc[-1] == pytest.approx(0.21)


def test_realized_volatility_annualises_log_return_std() -> None:
    values = [100.0]
    for i in range(30):
        values.append(values[-1] * (1.01 if i % 2 == 0 else 0.99))
    close = _series(values)
    vol = ind.realized_volatility(close, n=20)
    logs = np.log(np.array(values[1:]) / np.array(values[:-1]))[-20:]
    assert vol.iloc[-1] == pytest.approx(np.std(logs, ddof=1) * math.sqrt(252))
    assert vol.iloc[:20].isna().all()


def test_realized_volatility_zero_for_constant_growth() -> None:
    close = _series([100 * 1.001**i for i in range(40)])
    assert ind.realized_volatility(close).iloc[-1] == pytest.approx(0.0, abs=1e-12)


def test_last_helper() -> None:
    assert ind.last(_series([1.0, 2.0])) == 2.0
    assert ind.last(_series([1.0, float("nan")])) is None
    assert ind.last(pd.Series([], dtype=float)) is None


# ----------------------------------------------------------------------------- causality


def test_indicators_have_no_look_ahead(synthetic_provider: SyntheticProvider) -> None:
    full = synthetic_provider.history("AAPL", period="2y")
    cut = 300
    head = full.iloc[:cut]
    pairs = [
        (ind.sma(full["Close"], 50), ind.sma(head["Close"], 50)),
        (ind.ema(full["Close"], 20), ind.ema(head["Close"], 20)),
        (ind.rsi(full["Close"]), ind.rsi(head["Close"])),
        (ind.atr(full), ind.atr(head)),
        (ind.realized_volatility(full["Close"]), ind.realized_volatility(head["Close"])),
    ]
    for on_full, on_head in pairs:
        pd.testing.assert_series_equal(on_full.iloc[:cut], on_head, check_freq=False)
    for frame_full, frame_head in (
        (ind.macd(full["Close"]), ind.macd(head["Close"])),
        (ind.adx(full), ind.adx(head)),
        (ind.bollinger(full["Close"]), ind.bollinger(head["Close"])),
    ):
        pd.testing.assert_frame_equal(frame_full.iloc[:cut], frame_head, check_freq=False)
