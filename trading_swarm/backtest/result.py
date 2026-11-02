"""The outcome of a backtest and its JSON representation.

:class:`BacktestResult` keeps rich Python objects (a pandas equity curve, pydantic trades
and decisions) for programmatic use; :meth:`BacktestResult.to_dict` flattens everything
into plain JSON types for the API (``POST /backtest``), the dashboard and the CLI's
``--output`` file.

Units follow the project convention: ``*_pct`` values are percentages (``5.0`` = 5 %);
the equity curve's ``drawdown`` columns are **non-positive fractions** (``-0.12`` = 12 %
below the running peak), the convention of :func:`trading_swarm.analytics.drawdown_series`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

import pandas as pd

from trading_swarm.analytics import PerformanceMetrics
from trading_swarm.broker.accounting import money
from trading_swarm.models import Decision, PortfolioSnapshot, Trade, to_jsonable

#: Columns of :attr:`BacktestResult.equity_curve`, in order.
EQUITY_CURVE_COLUMNS: Final = (
    "equity",
    "cash",
    "positions_value",
    "drawdown",
    "benchmark",
    "benchmark_drawdown",
)

#: Columns of :attr:`BacktestResult.signal_log` (one row per symbol and decision bar).
SIGNAL_LOG_COLUMNS: Final = (
    "timestamp",
    "symbol",
    "price",
    "trend_signal",
    "trend_confidence",
    "sentiment_signal",
    "sentiment_confidence",
    "score",
    "confidence",
    "proposed",
    "action",
    "approved",
    "quantity",
    "note",
)


def empty_signal_log() -> pd.DataFrame:
    """An empty signal log with the canonical columns."""
    return pd.DataFrame({column: [] for column in SIGNAL_LOG_COLUMNS})


def _round_or_none(value: Any, digits: int) -> float | None:
    """``round(value, digits)`` for finite numbers, ``None`` for NaN / missing values."""
    if value is None:
        return None
    number = float(value)
    return round(number, digits) if math.isfinite(number) else None


@dataclass
class BacktestResult:
    """Outcome of a backtest run.

    Attributes:
        symbols: Symbols traded, in the order given.
        start: First bar date inside the requested window (effective start).
        end: Last bar date inside the requested window (effective end).
        initial_capital: Starting cash of both the strategy and the benchmark.
        parameters: Every knob that influenced the run (thresholds, risk limits, costs,
            fill model …) so a result is reproducible from its own JSON.
        equity_curve: Indexed by bar date (UTC midnight) with the columns
            :data:`EQUITY_CURVE_COLUMNS`: strategy ``equity``/``cash``/``positions_value``
            marked at each close, its ``drawdown`` (non-positive fraction), the
            equal-weight buy-and-hold ``benchmark`` equity and ``benchmark_drawdown``.
        trades: Strategy fills in chronological order (SELLs carry realised P&L).
        metrics: :class:`~trading_swarm.analytics.PerformanceMetrics` of the strategy.
        benchmark_metrics: The same metrics for the buy-and-hold benchmark.
        decisions: Every approved BUY/SELL decision, with the resulting order attached
            (``order`` / ``executed``) and a summary written after the fill.
        signal_log: One row per symbol and decision bar (including HOLDs and vetoed
            proposals) with the votes, consensus, proposed and final action; columns
            :data:`SIGNAL_LOG_COLUMNS`.
        final_portfolio: Account at the last close (open positions are *not*
            liquidated; they are marked to market).
        stats: Counters (bars, proposals, vetoes, fills, exits by reason …).
        warnings: Human-readable caveats (short warm-up history, agent failures …).
    """

    symbols: list[str]
    start: date
    end: date
    initial_capital: float
    parameters: dict[str, Any]
    equity_curve: pd.DataFrame
    trades: list[Trade]
    metrics: PerformanceMetrics
    benchmark_metrics: PerformanceMetrics
    decisions: list[Decision] = field(default_factory=list)
    signal_log: pd.DataFrame = field(default_factory=empty_signal_log)
    final_portfolio: PortfolioSnapshot | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ derived values
    @property
    def excess_return_pct(self) -> float | None:
        """Strategy total return minus buy-and-hold total return, in percentage points."""
        strategy, benchmark = self.metrics.total_return_pct, self.benchmark_metrics.total_return_pct
        if strategy is None or benchmark is None:
            return None
        return round(strategy - benchmark, 6)

    # ------------------------------------------------------------------ serialisation
    def equity_points(self) -> list[dict[str, Any]]:
        """The equity curve as ``[{t, equity, cash, positions_value, drawdown, …}]``."""
        points: list[dict[str, Any]] = []
        for ts, row in self.equity_curve.iterrows():
            points.append(
                {
                    "t": pd.Timestamp(ts).isoformat(),
                    "equity": money(row["equity"]),
                    "cash": money(row["cash"]),
                    "positions_value": money(row["positions_value"]),
                    "drawdown": _round_or_none(row["drawdown"], 6),
                    "benchmark": _round_or_none(row["benchmark"], 2),
                    "benchmark_drawdown": _round_or_none(row["benchmark_drawdown"], 6),
                }
            )
        return points

    def signal_records(self) -> list[dict[str, Any]]:
        """The signal log as JSON-ready records."""
        records: list[dict[str, Any]] = []
        for row in self.signal_log.to_dict(orient="records"):
            row["timestamp"] = pd.Timestamp(row["timestamp"]).isoformat()
            records.append(row)
        return to_jsonable(records)

    def to_dict(self, *, include_signal_log: bool = False) -> dict[str, Any]:
        """JSON-serialisable representation (for the API and ``--output``).

        Contains only JSON types (no NaN/inf: undefined values are ``null``). The per-bar
        signal log can be large (one row per symbol per bar), so it is included only
        when ``include_signal_log`` is true.
        """
        payload: dict[str, Any] = {
            "symbols": list(self.symbols),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "initial_capital": money(self.initial_capital),
            "parameters": self.parameters,
            "metrics": self.metrics.model_dump(mode="json"),
            "benchmark_metrics": self.benchmark_metrics.model_dump(mode="json"),
            "excess_return_pct": self.excess_return_pct,
            "stats": self.stats,
            "warnings": list(self.warnings),
            "equity_curve": self.equity_points(),
            "trades": [trade.model_dump(mode="json") for trade in self.trades],
            "decisions": [decision.model_dump(mode="json") for decision in self.decisions],
            "final_portfolio": (
                self.final_portfolio.model_dump(mode="json")
                if self.final_portfolio is not None
                else None
            ),
        }
        if include_signal_log:
            payload["signal_log"] = self.signal_records()
        return to_jsonable(payload)


__all__ = [
    "EQUITY_CURVE_COLUMNS",
    "SIGNAL_LOG_COLUMNS",
    "BacktestResult",
    "empty_signal_log",
]
