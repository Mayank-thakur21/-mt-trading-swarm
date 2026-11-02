"""Market-data contract shared by every provider.

A provider returns *adjusted* daily (or coarser) OHLCV bars in a canonical shape:

* index: tz-aware UTC :class:`pandas.DatetimeIndex`, ascending, unique. Daily-or-coarser
  bars are stamped at **midnight UTC of the exchange trading date**, so a bar's date is the
  same regardless of the exchange's time zone;
* columns: exactly ``Open, High, Low, Close, Volume`` as floats, with
  ``Low ≤ min(Open, Close)`` and ``High ≥ max(Open, Close)``.

Date ranges are **inclusive on both ends** (``end="2024-03-01"`` includes the bar for
1 March), unlike yfinance whose ``end`` is exclusive; providers translate.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Final, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from trading_swarm.models import NewsItem

logger = logging.getLogger(__name__)

SYMBOL_PATTERN: Final = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
OHLCV_COLUMNS: Final = ("Open", "High", "Low", "Close", "Volume")
VALID_PERIODS: Final = frozenset(
    {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
)
DAILY_OR_COARSER: Final = frozenset({"1d", "5d", "1wk", "1mo", "3mo"})

DateLike = date | datetime | str | pd.Timestamp


# =============================================================================== errors


class MarketDataError(Exception):
    """Base class for market-data failures (always carries a human-readable message)."""

    def __init__(self, message: str, *, symbol: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.symbol = symbol


class InvalidSymbolError(MarketDataError, ValueError):
    """The ticker is syntactically invalid (rejected before any network call)."""


class NoDataError(MarketDataError):
    """The provider answered but returned no bars: unknown/delisted symbol or empty range."""


class ProviderUnavailableError(MarketDataError):
    """The provider could not be reached or kept failing after retries (network, 5xx, rate
    limiting). Usually transient — map to HTTP 502/503, not 404."""


# =============================================================================== protocol


def validate_symbol(symbol: str) -> str:
    """Normalise a ticker (strip + upper-case) and validate it.

    Accepts the characters used by Yahoo tickers: letters, digits and ``. - ^ =``
    (``BRK-B``, ``^GSPC``, ``EURUSD=X``), 1-15 characters.

    Raises:
        InvalidSymbolError: if the symbol is empty or contains other characters.
    """
    if not isinstance(symbol, str):
        raise InvalidSymbolError(f"symbol must be a string, got {type(symbol).__name__}")
    normalised = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(normalised):
        raise InvalidSymbolError(
            f"invalid symbol {symbol!r}: use 1-15 characters from A-Z, 0-9 and . - ^ =",
            symbol=normalised or None,
        )
    return normalised


@runtime_checkable
class MarketDataProvider(Protocol):
    """What the swarm needs from a market-data source.

    Implementations must be safe to call from several threads at once (the coordinator
    analyses symbols concurrently).
    """

    #: Short provider id reported by ``/health`` (``"yahoo"``, ``"synthetic"``).
    name: str

    def history(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Adjusted OHLCV bars (canonical shape, see module docstring).

        ``start``/``end`` are inclusive; when ``start`` is given ``period`` is ignored.

        Raises:
            InvalidSymbolError, NoDataError, ProviderUnavailableError (all MarketDataError).
        """
        ...

    def latest_price(self, symbol: str) -> float:
        """Most recent trade/close price (positive, finite). Raises MarketDataError."""
        ...

    def news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        """Up to ``limit`` recent headlines, newest first.

        Missing news is not an error: returns ``[]`` (and logs) when nothing is available
        or the news endpoint fails. Raises :class:`InvalidSymbolError` for bad tickers.
        """
        ...

    def validate_symbol(self, symbol: str) -> str:
        """Normalised symbol or :class:`InvalidSymbolError`."""
        ...


# =============================================================================== helpers


def to_utc_timestamp(value: DateLike) -> pd.Timestamp:
    """Parse a date-like value into an aware UTC timestamp (naive values are UTC)."""
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot interpret {value!r} as a date") from exc
    if pd.isna(ts):
        raise ValueError(f"cannot interpret {value!r} as a date")
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def to_utc_date(value: DateLike) -> pd.Timestamp:
    """Like :func:`to_utc_timestamp` but floored to midnight UTC (a trading *date*)."""
    return to_utc_timestamp(value).normalize()


def normalize_ohlcv(df: pd.DataFrame | None, *, symbol: str, interval: str = "1d") -> pd.DataFrame:
    """Coerce a provider frame into the canonical OHLCV shape.

    Handles yfinance quirks: MultiIndex columns, lower-case names, exchange-local
    timestamps, duplicated rows, NaN rows and occasional ``High < Close`` glitches.

    Raises:
        NoDataError: if nothing usable is left.
        MarketDataError: if price columns are missing altogether.
    """
    if df is None or df.empty:
        raise NoDataError(f"no price data returned for {symbol}", symbol=symbol)
    frame = df.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    canonical = {c.lower(): c for c in OHLCV_COLUMNS}
    frame = frame.rename(columns=lambda c: canonical.get(str(c).strip().lower(), c))
    frame = frame.loc[:, ~frame.columns.duplicated()]
    missing = [c for c in ("Open", "High", "Low", "Close") if c not in frame.columns]
    if missing:
        raise MarketDataError(
            f"price data for {symbol} is missing columns {missing}", symbol=symbol
        )
    if "Volume" not in frame.columns:
        frame["Volume"] = 0.0

    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if interval in DAILY_OR_COARSER:
        # Keep the *local* trading date, then stamp it at midnight UTC.
        local_dates = index.tz_localize(None) if index.tz is not None else index
        index = local_dates.normalize().tz_localize("UTC")
    else:
        index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    frame.index = index
    frame.index.name = "Date"

    frame = frame[list(OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce").astype(float)
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame[np.isfinite(frame["Close"]) & (frame["Close"] > 0)]
    if frame.empty:
        raise NoDataError(f"no valid closing prices for {symbol}", symbol=symbol)

    close = frame["Close"]
    frame["Open"] = frame["Open"].where(frame["Open"] > 0, close)
    frame["High"] = frame[["High", "Open", "Close"]].max(axis=1)
    frame["Low"] = frame[["Low", "Open", "Close"]].where(lambda x: x > 0).min(axis=1)
    frame["Volume"] = frame["Volume"].fillna(0.0).clip(lower=0.0)
    return frame


def apply_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Keep the trailing ``period`` of an (ascending) frame, Yahoo-style.

    ``"1d"``/``"5d"`` count trading bars; ``"ytd"`` starts on 1 January of the last bar's
    year; ``"max"`` keeps everything; month/year periods are calendar look-backs.
    """
    if period not in VALID_PERIODS:
        raise ValueError(f"unsupported period {period!r}; choose one of {sorted(VALID_PERIODS)}")
    if df.empty or period == "max":
        return df
    if period in {"1d", "5d"}:
        return df.tail(int(period[0]))
    last_ts = df.index[-1]
    if period == "ytd":
        start = pd.Timestamp(year=last_ts.year, month=1, day=1, tz="UTC")
        return df[df.index >= start]
    if period.endswith("mo"):
        offset = pd.DateOffset(months=int(period[:-2]))
    else:  # "Ny"
        offset = pd.DateOffset(years=int(period[:-1]))
    return df[df.index > last_ts - offset]


def slice_range(
    df: pd.DataFrame,
    start: DateLike | None = None,
    end: DateLike | None = None,
    period: str = "1y",
) -> pd.DataFrame:
    """Select ``[start, end]`` (inclusive dates), or the trailing ``period`` ending at ``end``."""
    out = df
    if end is not None:
        end_ts = to_utc_date(end) + pd.Timedelta(days=1)
        out = out[out.index < end_ts]
    if start is not None:
        out = out[out.index >= to_utc_date(start)]
        return out
    return apply_period(out, period)


def daily_change_pct(df: pd.DataFrame) -> float | None:
    """Last close versus the previous close, in percent (``None`` with < 2 bars)."""
    if df is None or len(df) < 2:
        return None
    prev, last = float(df["Close"].iloc[-2]), float(df["Close"].iloc[-1])
    if prev <= 0:
        return None
    return (last / prev - 1.0) * 100.0


__all__ = [
    "DAILY_OR_COARSER",
    "OHLCV_COLUMNS",
    "SYMBOL_PATTERN",
    "VALID_PERIODS",
    "DateLike",
    "InvalidSymbolError",
    "MarketDataError",
    "MarketDataProvider",
    "NoDataError",
    "ProviderUnavailableError",
    "apply_period",
    "daily_change_pct",
    "normalize_ohlcv",
    "slice_range",
    "to_utc_date",
    "to_utc_timestamp",
    "validate_symbol",
]
