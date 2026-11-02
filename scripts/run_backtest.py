"""Backtest CLI wrapper: ``python -m scripts.run_backtest ...`` from a checkout.

The implementation is :mod:`trading_swarm.backtest.cli` (also installed as the
``mt-swarm-backtest`` console script and runnable as ``python -m
trading_swarm.backtest.cli``). Run ``python -m scripts.run_backtest --help`` for the flags.

Examples::

    python -m scripts.run_backtest --provider synthetic --symbol BULL --symbol FLAT \\
        --start 2025-01-02 --end 2026-06-30 --output results.json
    python -m scripts.run_backtest --symbol AAPL,MSFT --start 2022-01-01 --end 2024-12-31

Exit codes: ``0`` success, ``1`` market-data or output failure, ``2`` invalid arguments
or configuration.
"""

from __future__ import annotations

from trading_swarm.backtest.cli import (
    EXIT_FAILURE,
    EXIT_OK,
    EXIT_USAGE,
    build_parser,
    format_metrics_table,
    format_report,
    main,
    parse_symbols,
    write_json,
)

__all__ = [
    "EXIT_FAILURE",
    "EXIT_OK",
    "EXIT_USAGE",
    "build_parser",
    "format_metrics_table",
    "format_report",
    "main",
    "parse_symbols",
    "write_json",
]

if __name__ == "__main__":
    raise SystemExit(main())
