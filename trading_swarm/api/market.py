"""Quote and chart payloads for ``GET /quote`` and ``GET /history``.

Indicator warm-up
-----------------
A 6-month daily chart has ~126 bars, but SMA50 needs 50 bars and the MACD's EMAs need a
few dozen more to converge. Computing indicators on the displayed window alone would leave
SMA50 blank for the first 49 candles and show an unconverged MACD. Instead the history is
fetched with :data:`INDICATOR_WARMUP_BARS` extra bars of look-back (:func:`lookback_period`),
indicators are computed on the long frame, and both candles and indicators are sliced to the
requested period. Every indicator is causal (see :mod:`trading_swarm.indicators`), so the
extra history only *fills in* early values; it never changes a value at bar ``t`` that
could not have been known at ``t``.

Values that are still undefined (the first bars of ``period=max``) are ``null``. The MACD
line is reported from bar ``slow`` onwards and the signal/histogram from bar
``slow + signal − 1``, the points from which their recursions carry real information.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_swarm import indicators
from trading_swarm.data import MarketDataError, MarketDataProvider, daily_change_pct
from trading_swarm.data.base import apply_period

logger = logging.getLogger(__name__)

#: Extra bars fetched before the displayed window (SMA50 + MACD(12, 26, 9) convergence).
INDICATOR_WARMUP_BARS: Final = 100
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
RSI_PERIOD: Final = 14

#: Calendar days spanned by each period (upper bounds; ``ytd`` ≤ one year).
_PERIOD_DAYS: Final[Mapping[str, float]] = {
    "1d": 1,
    "5d": 7,
    "1mo": 31,
    "3mo": 92,
    "6mo": 183,
    "ytd": 366,
    "1y": 366,
    "2y": 731,
    "5y": 1827,
    "10y": 3653,
    "max": math.inf,
}
#: Periods we may *fetch*, shortest first.
_FETCH_PERIODS: Final = ("1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max")
#: Calendar days per bar, rounded up (weekends and holidays included for daily bars).
_DAYS_PER_BAR: Final[Mapping[str, float]] = {"1d": 1.5, "1wk": 7.0, "1mo": 31.0}


def lookback_period(period: str, interval: str) -> str:
    """Smallest fetchable period covering ``period`` plus the indicator warm-up."""
    needed = _PERIOD_DAYS[period] + INDICATOR_WARMUP_BARS * _DAYS_PER_BAR[interval]
    return next((p for p in _FETCH_PERIODS if _PERIOD_DAYS[p] >= needed), "max")


def _series_payload(series: pd.Series, digits: int = 4) -> list[float | None]:
    values = series.to_numpy(dtype=float, na_value=np.nan)
    return [round(float(v), digits) if math.isfinite(v) else None for v in values]


def indicator_frame(full: pd.DataFrame) -> pd.DataFrame:
    """SMA20/50, Wilder RSI(14) and MACD(12, 26, 9) for every bar of ``full``."""
    close = full["Close"].astype(float)
    frame = pd.DataFrame(index=full.index)
    frame["sma20"] = indicators.sma(close, 20)
    frame["sma50"] = indicators.sma(close, 50)
    frame["rsi"] = indicators.rsi(close, RSI_PERIOD)
    macd = indicators.macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    # The EMAs are seeded with the first close; mask the bars before they mean anything.
    macd.iloc[: MACD_SLOW - 1, macd.columns.get_loc("macd")] = np.nan
    signal_start = MACD_SLOW + MACD_SIGNAL - 2
    macd.iloc[:signal_start, [macd.columns.get_loc("signal"), macd.columns.get_loc("hist")]] = (
        np.nan
    )
    frame["macd"] = macd["macd"]
    frame["signal"] = macd["signal"]
    frame["hist"] = macd["hist"]
    return frame


def history_payload(
    provider: MarketDataProvider, symbol: str, period: str, interval: str
) -> dict[str, Any]:
    """Candles of the trailing ``period`` with warm-started indicator series.

    Raises:
        MarketDataError: from the provider (mapped to 404 / 422 / 502 by the API).
    """
    full = provider.history(symbol, period=lookback_period(period, interval), interval=interval)
    window = apply_period(full, period)
    if window.empty:  # pragma: no cover - providers raise NoDataError instead
        window = full
    studies = indicator_frame(full).loc[window.index]
    candles = [
        {
            "t": ts.to_pydatetime(),
            "o": round(float(row.Open), 4),
            "h": round(float(row.High), 4),
            "l": round(float(row.Low), 4),
            "c": round(float(row.Close), 4),
            "v": float(round(float(row.Volume))),
        }
        for ts, row in zip(window.index, window.itertuples(index=False), strict=True)
    ]
    return {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "candles": candles,
        "indicators": {
            name: _series_payload(studies[name], 2 if name == "rsi" else 4)
            for name in studies.columns
        },
    }


def quote_payload(provider: MarketDataProvider, symbol: str) -> dict[str, Any]:
    """Latest price and the 1-day change for ``symbol``.

    ``price`` is :meth:`MarketDataProvider.latest_price` (the price a market order would
    use); if only the quote fails, the last daily close is used. ``change_pct_1d`` is the
    last close versus the previous close, in percent.

    Raises:
        MarketDataError: when the daily history itself is unavailable.
    """
    history = provider.history(symbol, period="5d", interval="1d")
    last_close = float(history["Close"].iloc[-1])
    previous_close = float(history["Close"].iloc[-2]) if len(history) >= 2 else None
    try:
        price = float(provider.latest_price(symbol))
    except MarketDataError as exc:
        logger.warning("quote for %s unavailable (%s); using the last close", symbol, exc)
        price = last_close
    if not (math.isfinite(price) and price > 0):
        logger.warning("discarding invalid quote %r for %s; using the last close", price, symbol)
        price = last_close
    change = daily_change_pct(history)
    return {
        "symbol": symbol,
        "price": round(price, 4),
        "previous_close": round(previous_close, 4) if previous_close is not None else None,
        "change_pct_1d": round(change, 4) if change is not None and math.isfinite(change) else None,
        "as_of": history.index[-1].to_pydatetime(),
    }


def latest_trade_price(provider: MarketDataProvider, symbol: str) -> float:
    """The reference price for a manual order: the provider's latest price.

    Raises:
        MarketDataError: no usable quote (the order must not be placed blind).
    """
    price = float(provider.latest_price(symbol))
    if not (math.isfinite(price) and price > 0):
        raise MarketDataError(f"invalid latest price {price!r} for {symbol}", symbol=symbol)
    return price


__all__ = [
    "INDICATOR_WARMUP_BARS",
    "history_payload",
    "indicator_frame",
    "latest_trade_price",
    "lookback_period",
    "quote_payload",
]
