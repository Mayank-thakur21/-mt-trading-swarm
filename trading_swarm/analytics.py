"""Performance analytics shared by the paper account (``/equity``) and the backtester.

All functions tolerate empty, single-point and constant series: undefined ratios are
reported as ``None`` (never ``NaN``/``inf``, which are not valid JSON) and counts as 0.

Definitions (``r`` = per-period simple returns of the equity curve, ``P`` = periods per
year, ``rf`` = annual risk-free rate, ``rf_p = rf / P``):

* total return = ``E_end / E_start − 1``
* CAGR = ``(E_end / E_start)^(1 / years) − 1`` with ``years`` from the calendar span of a
  ``DatetimeIndex`` (365.25-day years), otherwise ``len(r) / P``
* volatility = ``std(r, ddof=1) · √P``
* Sharpe = ``mean(r − rf_p) / std(r) · √P``
* Sortino = ``mean(r − rf_p) / DD · √P`` with downside deviation
  ``DD = √mean(min(r − rf_p, 0)²)``
* max drawdown = ``max(1 − E / running_peak)``; its duration is the longest stretch from a
  peak until equity regains that peak (or until the last point if it never does)
* Calmar = ``CAGR / max drawdown``
* trade statistics use realised P&L of closed (SELL) trades: win rate, profit factor
  ``Σ wins / |Σ losses|``, average win/loss and expectancy (mean P&L per closed trade)
* exposure = share of periods with capital in the market

Starting equity
---------------
A curve whose first point is already marked *after* trading began has a problem. A
backtest's buy-and-hold benchmark, for example, buys at the first open and is first
marked at that close. Its first point then already contains the entry costs and the first
session's move. Pass ``start_equity`` (the initial capital) to measure from the money
actually invested instead. ``E_start`` becomes ``start_equity``, the return series gains a
first period ``E_0 / start_equity − 1``, and ``start_equity`` is the initial running peak
of the drawdown. The time span (and so the CAGR exponent) still starts at the first
timestamp, because the inception is the start of the first period.

Percent-valued fields (``*_pct``) are percentages: ``12.5`` means 12.5 %.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from trading_swarm.models import Side, Trade

TRADING_DAYS_PER_YEAR = 252
_SECONDS_PER_YEAR = 365.25 * 24 * 3600


class PerformanceMetrics(BaseModel):
    """Summary statistics of an equity curve and its closed trades (JSON-safe)."""

    start: datetime | None = None
    end: datetime | None = None
    periods: int = 0
    start_equity: float | None = None
    end_equity: float | None = None
    total_return_pct: float | None = None
    cagr_pct: float | None = None
    volatility_pct: float | None = None
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown_pct: float | None = None
    max_drawdown_duration_days: float | None = None
    calmar: float | None = None
    num_trades: int = 0
    win_rate_pct: float | None = None
    profit_factor: float | None = None
    avg_win: float | None = None
    avg_loss: float | None = None
    expectancy: float | None = None
    exposure_pct: float | None = None


# =============================================================================== helpers


def _finite(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _clean_equity(equity: pd.Series) -> pd.Series:
    if not isinstance(equity, pd.Series):
        raise TypeError(f"equity must be a pandas Series, got {type(equity).__name__}")
    series = pd.to_numeric(equity, errors="coerce").astype(float)
    series = series[np.isfinite(series)]
    if isinstance(series.index, pd.DatetimeIndex):
        series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _start_value(start_equity: float | None) -> float | None:
    """A usable starting equity (positive and finite), else ``None``."""
    if start_equity is None:
        return None
    value = float(start_equity)
    return value if math.isfinite(value) and value > 0 else None


def returns_from_equity(equity: pd.Series, start_equity: float | None = None) -> pd.Series:
    """Per-period simple returns (non-positive equity skipped).

    Without ``start_equity`` the first point only serves as the base of the second
    period. With it, the first period's return ``E_0 / start_equity − 1`` is included,
    labelled with the first timestamp.
    """
    series = _clean_equity(equity)
    prev = series.shift(1)
    start = _start_value(start_equity)
    if start is not None and len(series):
        prev.iloc[0] = start
    returns = (series / prev - 1.0).where(prev > 0)
    return returns.dropna()


def drawdown_series(equity: pd.Series, start_equity: float | None = None) -> pd.Series:
    """Drawdown from the running peak as a non-positive fraction (``-0.1`` = 10 % down).

    ``start_equity`` (when given) is the initial peak, so a curve that starts below the
    money invested starts in drawdown.
    """
    series = _clean_equity(equity)
    if series.empty:
        return series
    peak = series.cummax()
    start = _start_value(start_equity)
    if start is not None:
        peak = peak.clip(lower=start)
    return (series / peak - 1.0).where(peak > 0, 0.0).rename("drawdown")


def max_drawdown(equity: pd.Series, start_equity: float | None = None) -> tuple[float, float]:
    """``(max_drawdown_fraction, longest_underwater_duration)``.

    Duration is in calendar days for a ``DatetimeIndex``, otherwise in periods.
    ``start_equity`` (when given) is the initial peak, dated at the first point.
    """
    series = _clean_equity(equity)
    start = _start_value(start_equity)
    if len(series) < (1 if start is not None else 2):
        return 0.0, 0.0
    drawdowns = drawdown_series(series, start_equity=start)
    worst = float(-drawdowns.min()) if len(drawdowns) else 0.0
    is_datetime = isinstance(series.index, pd.DatetimeIndex)
    positions = range(len(series))
    values = series.to_numpy()
    peak = values[0] if start is None else max(start, values[0])
    peak_at = 0
    underwater = False
    longest = 0.0

    def span(start: int, end: int) -> float:
        if is_datetime:
            delta = series.index[end] - series.index[start]
            return delta.total_seconds() / 86_400.0
        return float(end - start)

    for i in positions:
        value = values[i]
        if value >= peak - 1e-12 * abs(peak):
            if underwater:
                longest = max(longest, span(peak_at, i))
                underwater = False
            if value >= peak:
                peak = value
            peak_at = i
        else:
            underwater = True
    if underwater:
        longest = max(longest, span(peak_at, len(series) - 1))
    return max(worst, 0.0), longest


def cagr(
    equity: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    start_equity: float | None = None,
) -> float | None:
    """Compound annual growth rate as a fraction (``None`` if undefined).

    ``start_equity`` (when given) replaces the first point as the base. Without a
    ``DatetimeIndex`` it also adds the first period to the span.
    """
    series = _clean_equity(equity)
    start = _start_value(start_equity)
    base = start if start is not None else (float(series.iloc[0]) if len(series) else 0.0)
    if len(series) < 2 or base <= 0:
        return None
    if isinstance(series.index, pd.DatetimeIndex):
        years = (series.index[-1] - series.index[0]).total_seconds() / _SECONDS_PER_YEAR
    else:
        periods = len(series) if start is not None else len(series) - 1
        years = periods / periods_per_year
    if years <= 0:
        return None
    growth = series.iloc[-1] / base
    if growth <= 0:
        return -1.0
    try:
        return float(growth ** (1.0 / years) - 1.0)
    except OverflowError:
        return None


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Annualised Sharpe ratio (``None`` with < 2 returns or zero volatility)."""
    returns = returns.dropna()
    if len(returns) < 2:
        return None
    excess = returns - risk_free_rate / periods_per_year
    std = float(returns.std(ddof=1))
    if not math.isfinite(std) or std <= 1e-15:
        return None
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Annualised Sortino ratio (``None`` with < 2 returns or no downside)."""
    returns = returns.dropna()
    if len(returns) < 2:
        return None
    excess = returns - risk_free_rate / periods_per_year
    downside = float(np.sqrt(np.mean(np.minimum(excess.to_numpy(), 0.0) ** 2)))
    if downside <= 1e-15:
        return None
    return float(excess.mean() / downside * math.sqrt(periods_per_year))


def _realized_pnls(trades: Iterable[Trade | float | Mapping[str, Any]] | None) -> list[float]:
    """Realised P&L of each closed trade (SELL fills); floats are taken as-is."""
    pnls: list[float] = []
    for trade in trades or []:
        if isinstance(trade, Trade):
            if trade.side is Side.SELL and trade.realized_pnl is not None:
                pnls.append(float(trade.realized_pnl))
        elif isinstance(trade, Mapping):
            value = trade.get("realized_pnl")
            side = str(trade.get("side", "SELL")).upper()
            if value is not None and side == "SELL":
                pnls.append(float(value))
        elif isinstance(trade, int | float) and not isinstance(trade, bool):
            pnls.append(float(trade))
    return [p for p in pnls if math.isfinite(p)]


def trade_statistics(trades: Iterable[Trade | float | Mapping[str, Any]] | None) -> dict[str, Any]:
    """Win rate, profit factor, average win/loss and expectancy of closed trades."""
    pnls = _realized_pnls(trades)
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = -sum(losses)
    return {
        "num_trades": n,
        "win_rate_pct": len(wins) / n * 100.0 if n else None,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "avg_win": gross_profit / len(wins) if wins else None,
        "avg_loss": -gross_loss / len(losses) if losses else None,
        "expectancy": sum(pnls) / n if n else None,
    }


# =============================================================================== main entry


def compute_metrics(
    equity: pd.Series | pd.DataFrame,
    trades: Iterable[Trade | float | Mapping[str, Any]] | None = None,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    exposure: pd.Series | None = None,
    start_equity: float | None = None,
) -> PerformanceMetrics:
    """Compute :class:`PerformanceMetrics` for an equity curve and its trades.

    Args:
        equity: Equity values indexed by time — or a DataFrame with an ``equity`` column
            (and optionally ``positions_value``, used for exposure), such as
            ``PaperBroker.equity_curve()``.
        trades: :class:`~trading_swarm.models.Trade` fills (SELLs carry realised P&L),
            plain realised-P&L floats, or mappings with ``realized_pnl``.
        risk_free_rate: Annual risk-free rate as a fraction.
        periods_per_year: 252 for daily bars.
        exposure: Per-period amount (or boolean) of capital in the market.
        start_equity: Account value *before* the first point (e.g. a backtest's initial
            capital); returns, CAGR and drawdown are then measured from it (see the module
            docstring). ``periods`` and ``exposure_pct`` still count the recorded points.
    """
    start = _start_value(start_equity)
    if isinstance(equity, pd.DataFrame):
        if "equity" not in equity.columns:
            raise KeyError("equity DataFrame needs an 'equity' column")
        if exposure is None and "positions_value" in equity.columns:
            exposure = equity["positions_value"]
        equity = equity["equity"]
    series = _clean_equity(equity)
    stats = trade_statistics(trades)
    metrics: dict[str, Any] = {
        key: _finite(val) if isinstance(val, float) else val for key, val in stats.items()
    }

    if series.empty:
        return PerformanceMetrics(**metrics)

    metrics["periods"] = len(series)
    first = start if start is not None else float(series.iloc[0])
    metrics["start_equity"] = _finite(first, 2)
    metrics["end_equity"] = _finite(series.iloc[-1], 2)
    if isinstance(series.index, pd.DatetimeIndex):
        metrics["start"] = series.index[0].to_pydatetime()
        metrics["end"] = series.index[-1].to_pydatetime()

    if first > 0:
        metrics["total_return_pct"] = _finite((float(series.iloc[-1]) / first - 1.0) * 100.0)
    growth = cagr(series, periods_per_year, start_equity=start)
    metrics["cagr_pct"] = _finite(growth * 100.0) if growth is not None else None

    returns = returns_from_equity(series, start_equity=start)
    if len(returns) >= 2:
        metrics["volatility_pct"] = _finite(
            float(returns.std(ddof=1)) * math.sqrt(periods_per_year) * 100.0
        )
    metrics["sharpe"] = _finite(sharpe_ratio(returns, risk_free_rate, periods_per_year))
    metrics["sortino"] = _finite(sortino_ratio(returns, risk_free_rate, periods_per_year))

    mdd, duration = max_drawdown(series, start_equity=start)
    metrics["max_drawdown_pct"] = _finite(mdd * 100.0)
    metrics["max_drawdown_duration_days"] = _finite(duration, 2)
    if growth is not None and mdd > 0:
        metrics["calmar"] = _finite(growth / mdd)

    if exposure is not None:
        aligned = (
            pd.Series(exposure).reindex(series.index)
            if isinstance(series.index, pd.DatetimeIndex)
            else pd.Series(exposure)
        )
        in_market = pd.to_numeric(aligned, errors="coerce").fillna(0.0).astype(float) > 0
        if len(in_market):
            metrics["exposure_pct"] = _finite(float(in_market.mean()) * 100.0)
    return PerformanceMetrics(**metrics)


__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "PerformanceMetrics",
    "cagr",
    "compute_metrics",
    "drawdown_series",
    "max_drawdown",
    "returns_from_equity",
    "sharpe_ratio",
    "sortino_ratio",
    "trade_statistics",
]
