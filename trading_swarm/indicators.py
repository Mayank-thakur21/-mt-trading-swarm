"""Pure, vectorised technical indicators on pandas objects.

Every function is *causal*: the value at bar ``t`` uses bars ``0..t`` only, so computing an
indicator on a truncated history gives exactly the prefix of the full-history result. This
is what lets the backtester slice history bar-by-bar without look-ahead bias.

Smoothing conventions
---------------------
* :func:`ema` is the recursive exponential average ``EMAₜ = α·xₜ + (1 − α)·EMAₜ₋₁`` with
  ``α = 2 / (n + 1)``, seeded with the first observation (pandas ``adjust=False``).
* Wilder smoothing (RSI, ATR, ADX) is the same recursion with ``α = 1 / n`` — J. Welles
  Wilder's "modified moving average" from *New Concepts in Technical Trading Systems*.
  Values are reported once ``n`` observations are available (``min_periods=n``).

The draft implementation computed the MACD signal line as ``EMA(close, 9)`` and RSI with a
simple moving average; both are fixed here (signal = ``EMA(MACD, 9)``; Wilder RSI).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# =============================================================================== helpers


def _check_window(n: int, name: str = "n") -> None:
    if not isinstance(n, int | np.integer) or isinstance(n, bool) or n < 1:
        raise ValueError(f"{name} must be a positive integer, got {n!r}")


def _as_float_series(series: pd.Series, name: str = "series") -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(series).__name__}")
    return series.astype(float)


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """Fetch an OHLCV column by name, tolerating lower-case column labels."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a DataFrame with a {name!r} column, got {type(df).__name__}")
    if name in df.columns:
        return df[name].astype(float)
    lowered = {str(c).lower(): c for c in df.columns}
    if name.lower() in lowered:
        return df[lowered[name.lower()]].astype(float)
    raise KeyError(f"DataFrame is missing the {name!r} column (has {list(df.columns)})")


def wilder_smooth(series: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average: EMA with ``α = 1/n``, reported after ``n`` observations."""
    _check_window(n)
    return _as_float_series(series).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def last(series: pd.Series) -> float | None:
    """The final value of ``series`` as a float, or ``None`` if empty / NaN / infinite."""
    if series is None or len(series) == 0:
        return None
    value = float(series.iloc[-1])
    return value if math.isfinite(value) else None


# =============================================================================== averages


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average over ``n`` bars (NaN until ``n`` bars exist)."""
    _check_window(n)
    return _as_float_series(series).rolling(window=n, min_periods=n).mean()


def ema(series: pd.Series, n: int, *, min_periods: int = 0) -> pd.Series:
    """Exponential moving average with span ``n`` (``α = 2/(n+1)``, ``adjust=False``)."""
    _check_window(n)
    return _as_float_series(series).ewm(span=n, adjust=False, min_periods=min_periods).mean()


# =============================================================================== oscillators


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index in ``[0, 100]``.

    ``RSI = 100 − 100 / (1 + RS)`` with ``RS = WilderAvg(gains) / WilderAvg(losses)``.
    Edge cases: only gains → 100, only losses → 0, a perfectly flat series → 50.
    The first value appears at bar ``n`` (``n`` price changes are needed).
    """
    _check_window(n)
    close = _as_float_series(close, "close")
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)
    avg_gain = gains.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_loss = losses.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100.0 - 100.0 / (1.0 + rs)
    no_losses = avg_loss == 0
    out = out.mask(no_losses & (avg_gain > 0), 100.0)
    out = out.mask(no_losses & (avg_gain == 0), 50.0)
    return out.rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence/Divergence.

    * ``macd   = EMA(close, fast) − EMA(close, slow)``
    * ``signal = EMA(macd, signal)``  ← an EMA *of the MACD line*, not of price
    * ``hist   = macd − signal``      (positive and rising = strengthening up-momentum)
    """
    for value, label in ((fast, "fast"), (slow, "slow"), (signal, "signal")):
        _check_window(value, label)
    if fast >= slow:
        raise ValueError(f"fast period ({fast}) must be shorter than slow period ({slow})")
    close = _as_float_series(close, "close")
    line = ema(close, fast) - ema(close, slow)
    signal_line = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: ``middle = SMA(n)``, ``upper/lower = middle ± k·σ``.

    σ is the *population* standard deviation of the window (Bollinger's definition).
    Also returns ``bandwidth = (upper − lower) / middle`` and
    ``percent_b = (close − lower) / (upper − lower)`` (NaN when the bands collapse).
    """
    _check_window(n)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    close = _as_float_series(close, "close")
    middle = close.rolling(window=n, min_periods=n).mean()
    std = close.rolling(window=n, min_periods=n).std(ddof=0)
    upper = middle + k * std
    lower = middle - k * std
    width = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        bandwidth = (width / middle).where(middle != 0)
        percent_b = ((close - lower) / width).where(width > 0)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "percent_b": percent_b,
        }
    )


# =============================================================================== volatility


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's true range: ``max(H − L, |H − Cₜ₋₁|, |L − Cₜ₋₁|)`` (first bar: ``H − L``)."""
    high, low, close = _column(df, "High"), _column(df, "Low"), _column(df, "Close")
    prev_close = close.shift(1)
    ranges = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1)
    return ranges.max(axis=1, skipna=True).rename("true_range")


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range: Wilder-smoothed true range (in price units)."""
    return wilder_smooth(true_range(df), n).rename("atr")


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Average Directional Index with the directional indicators.

    * ``+DM = Hₜ − Hₜ₋₁`` if it exceeds ``Lₜ₋₁ − Lₜ`` and is positive, else 0 (``−DM``
      mirrors it); both and the true range are Wilder-smoothed.
    * ``±DI = 100 · smoothed(±DM) / smoothed(TR)``
    * ``DX = 100 · |+DI − −DI| / (+DI + −DI)``; ``ADX = WilderAvg(DX)``.

    ADX measures trend *strength* regardless of direction: below ~20 is a weak/absent
    trend, above ~25 a trend worth following. Returns ``plus_di``, ``minus_di``, ``adx``.
    """
    _check_window(n)
    high, low, close = _column(df, "High"), _column(df, "Low"), _column(df, "Close")
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1, skipna=False
    )
    # Directional movement needs a previous bar: start every series at bar 1.
    plus_dm.iloc[:1] = np.nan
    minus_dm.iloc[:1] = np.nan

    smoothed_tr = wilder_smooth(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = (100.0 * wilder_smooth(plus_dm, n) / smoothed_tr).where(smoothed_tr > 0, 0.0)
        minus_di = (100.0 * wilder_smooth(minus_dm, n) / smoothed_tr).where(smoothed_tr > 0, 0.0)
        di_sum = plus_di + minus_di
        dx = (100.0 * (plus_di - minus_di).abs() / di_sum).where(di_sum > 0, 0.0)
    # ``where(..., 0.0)`` above also overwrote warm-up NaNs; restore them.
    warm_up = smoothed_tr.isna()
    plus_di = plus_di.mask(warm_up)
    minus_di = minus_di.mask(warm_up)
    dx = dx.mask(warm_up)
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": wilder_smooth(dx, n)})


# =============================================================================== returns


def pct_change(close: pd.Series, n: int = 1) -> pd.Series:
    """Simple return over ``n`` bars as a fraction: ``Cₜ / Cₜ₋ₙ − 1``."""
    _check_window(n)
    close = _as_float_series(close, "close")
    return (close / close.shift(n) - 1.0).rename(f"pct_change_{n}")


def log_returns(close: pd.Series) -> pd.Series:
    """One-bar log returns ``ln(Cₜ / Cₜ₋₁)``."""
    close = _as_float_series(close, "close")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(close / close.shift(1)).rename("log_return")


def realized_volatility(
    close: pd.Series, n: int = 20, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Annualised realised volatility: rolling sample std of log returns × √periods.

    Returned as a fraction (``0.25`` = 25 % a year).
    """
    _check_window(n)
    _check_window(periods_per_year, "periods_per_year")
    returns = log_returns(close)
    vol = returns.rolling(window=n, min_periods=n).std(ddof=1) * math.sqrt(periods_per_year)
    return vol.rename("realized_volatility")


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "adx",
    "atr",
    "bollinger",
    "ema",
    "last",
    "log_returns",
    "macd",
    "pct_change",
    "realized_volatility",
    "rsi",
    "sma",
    "true_range",
    "wilder_smooth",
]
