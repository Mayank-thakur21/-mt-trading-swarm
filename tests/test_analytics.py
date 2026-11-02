"""Performance analytics on known series, edge cases and trade statistics."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from trading_swarm.analytics import (
    PerformanceMetrics,
    cagr,
    compute_metrics,
    drawdown_series,
    max_drawdown,
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    trade_statistics,
)
from trading_swarm.models import Side, Trade


def _daily(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D", tz="UTC"))


KNOWN = [100.0, 110.0, 99.0, 120.0]


def test_known_series_metrics() -> None:
    equity = _daily(KNOWN)
    metrics = compute_metrics(equity)
    returns = np.array([0.1, -0.1, 120 / 99 - 1])
    sharpe = returns.mean() / returns.std(ddof=1) * math.sqrt(252)
    downside = math.sqrt(np.mean(np.minimum(returns, 0) ** 2))
    sortino = returns.mean() / downside * math.sqrt(252)
    growth = 1.2 ** (365.25 / 3) - 1

    assert metrics.periods == 4
    assert metrics.start_equity == 100.0 and metrics.end_equity == 120.0
    assert metrics.start == equity.index[0].to_pydatetime()
    assert metrics.total_return_pct == pytest.approx(20.0)
    assert metrics.volatility_pct == pytest.approx(
        returns.std(ddof=1) * math.sqrt(252) * 100, rel=1e-6
    )
    assert metrics.sharpe == pytest.approx(sharpe, rel=1e-6)
    assert metrics.sortino == pytest.approx(sortino, rel=1e-6)
    assert metrics.max_drawdown_pct == pytest.approx(10.0)
    assert metrics.max_drawdown_duration_days == pytest.approx(2.0)  # Jan 2 peak → Jan 4
    assert metrics.cagr_pct == pytest.approx(growth * 100, rel=1e-6)
    assert metrics.calmar == pytest.approx(growth / 0.1, rel=1e-6)
    assert metrics.num_trades == 0 and metrics.win_rate_pct is None


def test_helpers_on_known_series() -> None:
    equity = _daily(KNOWN)
    assert returns_from_equity(equity).tolist() == pytest.approx([0.1, -0.1, 120 / 99 - 1])
    assert drawdown_series(equity).tolist() == pytest.approx([0.0, 0.0, -0.1, 0.0])
    assert max_drawdown(equity) == pytest.approx((0.1, 2.0))


def test_cagr_uses_periods_without_a_datetime_index() -> None:
    assert cagr(pd.Series([100.0, 110.0, 121.0]), periods_per_year=2) == pytest.approx(0.21)
    two_years = pd.Series(
        [100.0, 121.0], index=pd.to_datetime(["2020-01-01", "2022-01-01"], utc=True)
    )
    years = (two_years.index[1] - two_years.index[0]).days / 365.25
    assert cagr(two_years) == pytest.approx(1.21 ** (1 / years) - 1)


def test_unrecovered_drawdown_runs_to_the_end() -> None:
    mdd, duration = max_drawdown(_daily([100.0, 120.0, 90.0, 100.0]))
    assert mdd == pytest.approx(0.25)
    assert duration == pytest.approx(2.0)


def test_longest_underwater_period_wins() -> None:
    equity = _daily([100, 95, 101, 90, 92, 94, 96, 102])
    mdd, duration = max_drawdown(equity)
    assert mdd == pytest.approx(1 - 90 / 101)
    assert duration == pytest.approx(5.0)  # 101 on day 2 → recovered on day 7


def test_constant_series_has_no_ratios() -> None:
    metrics = compute_metrics(_daily([100.0] * 10))
    assert metrics.total_return_pct == 0.0
    assert metrics.volatility_pct == 0.0
    assert metrics.sharpe is None and metrics.sortino is None
    assert metrics.max_drawdown_pct == 0.0 and metrics.max_drawdown_duration_days == 0.0
    assert metrics.calmar is None
    assert metrics.cagr_pct == pytest.approx(0.0)


@pytest.mark.parametrize("values", [[], [100.0]])
def test_empty_and_single_point_series(values: list[float]) -> None:
    metrics = compute_metrics(_daily(values))
    assert metrics.periods == len(values)
    assert metrics.sharpe is None and metrics.sortino is None and metrics.cagr_pct is None
    assert metrics.volatility_pct is None and metrics.calmar is None
    if values:
        assert metrics.total_return_pct == 0.0 and metrics.max_drawdown_pct == 0.0
    else:
        assert metrics.total_return_pct is None and metrics.start is None


def test_wiped_out_account() -> None:
    metrics = compute_metrics(_daily([100.0, 50.0, 0.0]))
    assert metrics.total_return_pct == pytest.approx(-100.0)
    assert metrics.cagr_pct == pytest.approx(-100.0)
    assert metrics.max_drawdown_pct == pytest.approx(100.0)


def test_nan_and_duplicate_points_are_cleaned() -> None:
    equity = _daily([100.0, float("nan"), 110.0, float("inf"), 121.0])
    assert compute_metrics(equity).periods == 3
    dup = pd.Series(
        [100.0, 105.0, 110.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"], utc=True),
    )
    assert compute_metrics(dup).start_equity == 105.0


def test_risk_free_rate_lowers_sharpe() -> None:
    rng = np.random.default_rng(4)
    equity = _daily(list(100 * np.cumprod(1 + rng.normal(0.001, 0.01, 300))))
    returns = returns_from_equity(equity)
    assert sharpe_ratio(returns, 0.05) < sharpe_ratio(returns, 0.0)  # type: ignore[operator]
    assert sortino_ratio(returns, 0.05) < sortino_ratio(returns, 0.0)  # type: ignore[operator]
    with_rf = compute_metrics(equity, risk_free_rate=0.05)
    assert with_rf.sharpe == pytest.approx(sharpe_ratio(returns, 0.05), rel=1e-6)


def test_sortino_without_downside_is_none() -> None:
    returns = pd.Series([0.01, 0.02, 0.015])
    assert sortino_ratio(returns) is None
    assert sharpe_ratio(pd.Series([0.01])) is None


# ----------------------------------------------------------------------------- trades


def _trade(side: Side, pnl: float | None) -> Trade:
    return Trade(order_id="o", symbol="ACME", side=side, quantity=1, price=10.0, realized_pnl=pnl)


def test_trade_statistics_from_floats() -> None:
    stats = trade_statistics([100.0, -50.0, 200.0, -25.0, 0.0])
    assert stats == {
        "num_trades": 5,
        "win_rate_pct": pytest.approx(40.0),
        "profit_factor": pytest.approx(4.0),
        "avg_win": pytest.approx(150.0),
        "avg_loss": pytest.approx(-37.5),
        "expectancy": pytest.approx(45.0),
    }


def test_trade_statistics_from_models_and_mappings() -> None:
    trades = [
        _trade(Side.BUY, None),  # entries carry no P&L
        _trade(Side.SELL, 120.0),
        _trade(Side.SELL, -40.0),
        {"side": "SELL", "realized_pnl": 20.0},
        {"side": "BUY", "realized_pnl": 999.0},  # ignored: not a closing fill
        float("nan"),
    ]
    stats = trade_statistics(trades)  # type: ignore[arg-type]
    assert stats["num_trades"] == 3
    assert stats["profit_factor"] == pytest.approx(140 / 40)
    assert stats["expectancy"] == pytest.approx(100 / 3)


def test_no_losses_means_undefined_profit_factor() -> None:
    stats = trade_statistics([10.0, 5.0])
    assert stats["profit_factor"] is None and stats["avg_loss"] is None
    assert stats["win_rate_pct"] == 100.0
    empty = trade_statistics(None)
    assert empty["num_trades"] == 0 and empty["expectancy"] is None


def test_metrics_include_trades_and_exposure_from_a_curve_frame() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC")
    curve = pd.DataFrame(
        {
            "equity": [100.0, 101.0, 103.0, 102.0],
            "cash": [100.0, 1.0, 1.0, 102.0],
            "positions_value": [0.0, 100.0, 102.0, 0.0],
        },
        index=index,
    )
    metrics = compute_metrics(curve, [_trade(Side.SELL, 2.0)])
    assert metrics.exposure_pct == pytest.approx(50.0)
    assert metrics.num_trades == 1 and metrics.win_rate_pct == 100.0
    explicit = compute_metrics(
        curve["equity"], exposure=pd.Series([True, True, True, False], index=index)
    )
    assert explicit.exposure_pct == pytest.approx(75.0)
    with pytest.raises(KeyError):
        compute_metrics(curve.drop(columns=["equity"]))
    with pytest.raises(TypeError):
        compute_metrics([100.0, 101.0])  # type: ignore[arg-type]


def test_metrics_are_json_safe() -> None:
    metrics = compute_metrics(_daily([100.0] * 3), [1.0])
    payload = json.loads(metrics.model_dump_json())
    assert payload["sharpe"] is None and payload["profit_factor"] is None
    assert isinstance(PerformanceMetrics.model_validate(payload), PerformanceMetrics)
    assert "NaN" not in metrics.model_dump_json() and "Infinity" not in metrics.model_dump_json()


# ----------------------------------------------------------------------------- start equity


def test_start_equity_measures_from_the_money_invested() -> None:
    # Bought at 100 and first marked at 110 after the first session.
    curve = _daily([110.0, 121.0])
    plain = compute_metrics(curve)
    funded = compute_metrics(curve, start_equity=100.0)
    assert plain.total_return_pct == pytest.approx(10.0)  # misses the first session
    assert funded.start_equity == 100.0 and funded.end_equity == 121.0
    assert funded.total_return_pct == pytest.approx(21.0)
    assert funded.periods == 2  # the recorded points
    returns = returns_from_equity(curve, start_equity=100.0)
    assert returns.tolist() == pytest.approx([0.10, 0.10])
    assert returns.index[0] == curve.index[0]
    # Same calendar span (one day), measured from 100 instead of 110.
    assert cagr(curve, start_equity=100.0) == pytest.approx(1.21**365.25 - 1, rel=1e-9)


def test_start_equity_is_the_initial_drawdown_peak() -> None:
    curve = _daily([90.0, 99.0, 101.0])  # opened at 100, first close 10 % lower
    drawdown = drawdown_series(curve, start_equity=100.0)
    assert drawdown.tolist() == pytest.approx([-0.10, -0.01, 0.0])
    worst, duration = max_drawdown(curve, start_equity=100.0)
    assert worst == pytest.approx(0.10) and duration == pytest.approx(2.0)
    assert compute_metrics(curve, start_equity=100.0).max_drawdown_pct == pytest.approx(10.0)
    assert compute_metrics(curve).max_drawdown_pct == 0.0  # without it the loss is invisible


def test_start_equity_none_or_invalid_changes_nothing() -> None:
    curve = _daily(KNOWN)
    baseline = compute_metrics(curve).model_dump()
    for start in (None, 0.0, -5.0, float("nan")):
        assert compute_metrics(curve, start_equity=start).model_dump() == baseline
    # A start equal to the first point only adds a flat first period.
    flat_first = compute_metrics(curve, start_equity=KNOWN[0])
    assert flat_first.total_return_pct == baseline["total_return_pct"]
    assert flat_first.max_drawdown_pct == baseline["max_drawdown_pct"]


def test_start_equity_without_datetime_index_adds_a_period() -> None:
    curve = pd.Series([110.0, 121.0])
    assert cagr(curve, periods_per_year=2, start_equity=100.0) == pytest.approx(0.21)
    assert cagr(curve, periods_per_year=2) == pytest.approx(1.1**2 - 1)
