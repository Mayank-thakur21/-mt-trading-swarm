"""Run a swarm backtest from the command line and print strategy vs buy-and-hold metrics.

The implementation lives in the package so the installed ``mt-swarm-backtest`` console
script does not need a generic top-level ``scripts`` package in site-packages;
``scripts/run_backtest.py`` is a thin wrapper kept for ``python -m scripts.run_backtest``
from a checkout. All three invocations are equivalent::

    python -m scripts.run_backtest --symbol AAPL --symbol MSFT --start 2022-01-01 --end 2024-12-31
    python -m trading_swarm.backtest.cli --provider synthetic --symbol BULL,FLAT --output r.json
    mt-swarm-backtest --symbol NVDA --capital 50000 --rebalance weekly

Without ``--symbol`` the ``WATCHLIST`` setting is used, without ``--provider`` the
``DATA_PROVIDER`` setting, and without ``--start`` the year before ``--end``
(default: today). Every other knob (thresholds, weights, risk limits, costs) comes
from the usual environment / ``.env`` settings, so a backtest uses exactly the
configuration the live swarm would.

``--output`` writes :meth:`BacktestResult.to_dict` as JSON, including the per-bar
signal log.

Exit codes: ``0`` success, ``1`` market-data or output failure, ``2`` invalid arguments
or configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from trading_swarm.analytics import PerformanceMetrics
from trading_swarm.backtest import (
    REBALANCE_FREQUENCIES,
    BacktestConfigError,
    Backtester,
    BacktestResult,
)
from trading_swarm.config import Settings, get_settings
from trading_swarm.data import (
    InvalidSymbolError,
    MarketDataError,
    MarketDataProvider,
    SyntheticProvider,
    YahooFinanceProvider,
)
from trading_swarm.logging_setup import configure_logging

EXIT_OK: Final = 0
EXIT_FAILURE: Final = 1
EXIT_USAGE: Final = 2
PROVIDERS: Final = ("synthetic", "yahoo")
LOG_LEVELS: Final = ("DEBUG", "INFO", "WARNING", "ERROR")

# =============================================================================== argument types


def _iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}: use YYYY-MM-DD") from exc


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from exc
    if not number > 0 or number == float("inf"):
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value!r}")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from exc
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value!r}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """The CLI's argument parser."""
    parser = argparse.ArgumentParser(
        prog="mt-swarm-backtest",
        description=(
            "Backtest the swarm (trend + risk gate, optional point-in-time sentiment) on "
            "daily bars: decisions at each close, fills at the next open, stops checked "
            "against each bar's range. Prints strategy vs buy-and-hold metrics."
        ),
    )
    parser.add_argument(
        "-s",
        "--symbol",
        dest="symbols",
        action="append",
        metavar="SYMBOL",
        help="ticker to trade; repeatable, comma lists accepted (default: WATCHLIST)",
    )
    parser.add_argument(
        "--start", type=_iso_date, help="first date, inclusive (default: one year before --end)"
    )
    parser.add_argument("--end", type=_iso_date, help="last date, inclusive (default: today)")
    parser.add_argument(
        "--capital", type=_positive_float, help="initial capital (default: INITIAL_CAPITAL)"
    )
    parser.add_argument(
        "--provider", choices=PROVIDERS, help="market-data provider (default: DATA_PROVIDER)"
    )
    parser.add_argument("--output", type=Path, help="write the full result as JSON to this path")
    parser.add_argument(
        "--use-sentiment",
        action="store_true",
        help="add the point-in-time news-sentiment vote (headlines published by each bar)",
    )
    parser.add_argument(
        "--warmup",
        type=_non_negative_int,
        default=200,
        help="bars of history required before --start (default: %(default)s)",
    )
    parser.add_argument(
        "--rebalance",
        choices=REBALANCE_FREQUENCIES,
        default="daily",
        help="decision schedule (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, help="seed of the synthetic provider (default: SYNTHETIC_SEED)"
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default="WARNING",
        help="log verbosity on stderr (default: %(default)s)",
    )
    return parser


# =============================================================================== helpers


def parse_symbols(values: Sequence[str] | None) -> list[str]:
    """Flatten repeated / comma-separated ``--symbol`` values (order kept, duplicates dropped)."""
    symbols: list[str] = []
    for value in values or []:
        for part in value.split(","):
            symbol = part.strip().upper()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
    return symbols


def build_provider(name: str, settings: Settings, seed: int | None = None) -> MarketDataProvider:
    """The market-data provider selected on the command line."""
    if name == "synthetic":
        return SyntheticProvider(seed=settings.synthetic_seed if seed is None else seed)
    return YahooFinanceProvider.from_settings(settings)


def _fmt_pct(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}%" if signed else f"{value:.2f}%"


def _fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _fmt_days(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} d"


def _fmt_int(value: Any) -> str:
    return "n/a" if value is None else str(int(value))


#: ``(label, PerformanceMetrics field, formatter)`` rows of the metrics table.
METRIC_ROWS: Final[tuple[tuple[str, str, Callable[[Any], str]], ...]] = (
    ("Total return", "total_return_pct", lambda v: _fmt_pct(v, signed=True)),
    ("CAGR", "cagr_pct", lambda v: _fmt_pct(v, signed=True)),
    ("Volatility (annualised)", "volatility_pct", _fmt_pct),
    ("Sharpe ratio", "sharpe", _fmt_ratio),
    ("Sortino ratio", "sortino", _fmt_ratio),
    ("Max drawdown", "max_drawdown_pct", _fmt_pct),
    ("Max drawdown duration", "max_drawdown_duration_days", _fmt_days),
    ("Calmar ratio", "calmar", _fmt_ratio),
    ("Closed trades", "num_trades", _fmt_int),
    ("Win rate", "win_rate_pct", _fmt_pct),
    ("Profit factor", "profit_factor", _fmt_ratio),
    ("Average win", "avg_win", _fmt_money),
    ("Average loss", "avg_loss", _fmt_money),
    ("Expectancy per trade", "expectancy", _fmt_money),
    ("Exposure (time in market)", "exposure_pct", _fmt_pct),
    ("Final equity", "end_equity", _fmt_money),
)


def format_metrics_table(strategy: PerformanceMetrics, benchmark: PerformanceMetrics) -> str:
    """A plain-text table of the strategy's metrics next to buy-and-hold."""
    header = ("Metric", "Strategy", "Buy & hold")
    rows = [
        (label, fmt(getattr(strategy, key)), fmt(getattr(benchmark, key)))
        for label, key, fmt in METRIC_ROWS
    ]
    widths = [max(len(row[col]) for row in [header, *rows]) for col in range(3)]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            [cells[0].ljust(widths[0]), cells[1].rjust(widths[1]), cells[2].rjust(widths[2])]
        )

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([line(header), separator, *(line(row) for row in rows)])


def format_report(result: BacktestResult) -> str:
    """Human-readable summary of a backtest: header, metrics table, trade counts, warnings."""
    params = result.parameters
    stats = result.stats
    exits = stats.get("exits", {})
    lines = [
        f"Backtest {', '.join(result.symbols)}: {result.start} to {result.end} "
        f"({stats.get('bars', 0)} bars, rebalance {params.get('rebalance')}, "
        f"provider {params.get('data_provider')}, sentiment "
        f"{params.get('sentiment_method') or 'off'})",
        f"Initial capital {_fmt_money(result.initial_capital)}",
        "",
        format_metrics_table(result.metrics, result.benchmark_metrics),
        "",
        f"Fills: {stats.get('fills', 0)} ({stats.get('round_trips', 0)} closing sells: "
        f"{exits.get('stop_loss', 0)} stop-loss, {exits.get('take_profit', 0)} take-profit, "
        f"{exits.get('signal', 0)} signal); proposals {stats.get('proposals', 0)}, "
        f"vetoed {stats.get('vetoed', 0)}",
    ]
    open_positions = stats.get("open_positions") or []
    lines.append(f"Open positions at the end: {', '.join(open_positions) or 'none'}")
    excess = result.excess_return_pct
    if excess is not None:
        lines.append(f"Excess return vs buy & hold: {excess:+.2f} percentage points")
    lines.extend(f"warning: {message}" for message in result.warnings)
    return "\n".join(lines)


def write_json(result: BacktestResult, path: Path) -> None:
    """Write the full result (with the per-bar signal log) as pretty-printed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(include_signal_log=True), indent=2, allow_nan=False)
    path.write_text(payload + "\n", encoding="utf-8")


# =============================================================================== entry point


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code.

    The report goes to stdout and errors to stderr (``print`` is deliberate here: this is
    the command-line front end, not library code). Logging goes to stderr at
    ``--log-level``.
    """
    args = build_parser().parse_args(argv)
    try:
        settings = get_settings()
    except ValidationError as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return EXIT_USAGE
    configure_logging(args.log_level, json=settings.log_json)

    end = args.end or datetime.now(UTC).date()
    start = args.start or end - timedelta(days=365)
    symbols = parse_symbols(args.symbols) or settings.watchlist_symbols
    provider = build_provider(args.provider or settings.data_provider, settings, args.seed)

    try:
        result = Backtester(provider, settings).run(
            symbols,
            start,
            end,
            initial_capital=args.capital,
            use_sentiment=args.use_sentiment,
            warmup=args.warmup,
            rebalance=args.rebalance,
        )
    except (BacktestConfigError, InvalidSymbolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except MarketDataError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return EXIT_FAILURE

    print(format_report(result))
    if args.output is not None:
        try:
            write_json(result, args.output)
        except OSError as exc:
            print(f"error: could not write {args.output}: {exc}", file=sys.stderr)
            return EXIT_FAILURE
        print(f"Results written to {args.output}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
