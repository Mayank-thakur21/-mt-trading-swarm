"""Event-driven daily backtester for the swarm strategy, with no look-ahead.

The backtester replays the live decision logic over historical daily bars. It uses the
same :class:`~trading_swarm.agents.TrendAnalyzerAgent`,
:func:`~trading_swarm.swarm.compute_consensus`,
:class:`~trading_swarm.agents.RiskManagerAgent` and
:class:`~trading_swarm.agents.ExecutionAgent` as the live coordinator. Orders go to an
:class:`~trading_swarm.broker.InMemoryBroker`, which books fills through the accounting
core shared with the SQLite :class:`~trading_swarm.broker.PaperBroker`
(:mod:`trading_swarm.broker.accounting`). Commission and slippage are therefore computed
by the same code in a backtest as in paper trading.

Timeline of one bar ``t``
-------------------------
All symbols are processed together on the union of their trading calendars.

1. **Open.** Every fill below happens at this bar's open, in this order:

   a. Pending SELLs decided at the previous decision bar fill.
   b. *Gap exits.* Every open position whose stop-loss or take-profit the open has
      gapped through is closed at the open. This uses the levels that were in force
      overnight, before any pending add-on can merge its own levels into the position.
   c. Pending BUYs fill. The live ExecutionAgent places them with this bar's open as its
      quote, so the protective levels are re-anchored if the open gapped past them. An
      add-on to a position that step (b) just closed is **cancelled** (an order with
      status ``CANCELLED`` is attached to the decision). It was sized against a position
      that no longer exists, and buying back at the very open the stop sold at would only
      churn costs. The live coordinator applies the same rule: a symbol closed by its
      stop or target is not bought back in the same analysis or scan (the risk agent's
      ``reentry_ok`` veto). A BUY that the open price makes unaffordable is trimmed to the
      shares the cash can still pay for.

   SELLs and gap exits fill before BUYs, so the cash they free is available.
2. **Intrabar.** Each open position's stop-loss and take-profit are checked against
   the bar with :func:`~trading_swarm.broker.evaluate_exit_bar`:

   * a gap through a level at the open fills at the open;
   * otherwise a level inside ``[low, high]`` fills at the level;
   * if both levels are inside the bar, the stop is assumed to have traded first.

   The order ``reason`` is the :class:`~trading_swarm.models.ExitReason` value
   (``stop_loss`` / ``take_profit``). Exits are checked on every bar, including
   non-rebalance bars.
3. **Close.** Positions are marked at the close and the equity curve gets a point.
4. **Decision** (rebalance bars only). For each symbol the agents receive the trailing
   one-year window of bars ending at ``t`` (``apply_period(history[: t + 1], "1y")``).
   That is exactly what the live coordinator analyses, and contains nothing after bar
   ``t``. The steps are:

   * trend vote, plus the point-in-time sentiment vote when enabled;
   * consensus;
   * risk gate, against the account marked at the close.

   Approved trades become orders for the next bar's open. No decision is taken on a
   symbol's last bar, because there is no next open to fill at.

Within one decision bar, symbols proposing a SELL are assessed before those proposing a
BUY. Every approved order is *reserved* in a pro-forma copy of the account (its cash and
exposure are applied) before the next symbol is assessed. Orders still pending from an
earlier bar are reserved first: a symbol with no bar today, such as one on an exchange
holiday, keeps its order pending until its next bar. This mirrors the live coordinator's
trade lock. BUYs that have not filled yet still cannot together overshoot the cash or
gross-exposure limits.

Sentiment
---------
Disabled by default. Historical news is not available point-in-time, so the consensus
uses the trend vote alone with renormalised weights, which is identical to the live math
with an abstaining sentiment agent. With ``use_sentiment=True``:

* the provider's headlines are fetched once per symbol;
* at each bar the sentiment agent (LLM, with the lexicon as fallback) only sees headlines
  published before 20:00 UTC on the bar's date (about the New York close);
* headlines without a timestamp are never shown;
* each distinct headline set is scored once and cached, which bounds the number of LLM
  calls.

Providers typically return only *recent* headlines, so sentiment mostly abstains in
historical windows.

Warm-up and look-back
---------------------
At least one year before ``start`` is fetched, or ``warmup`` bars if that is longer, so
the first decision already sees a full live-sized window. If the provider has fewer than
``warmup`` bars before ``start``, the run proceeds, because the agents handle short
history themselves (the trend agent is neutral below 60 bars), and the shortfall is
reported in :attr:`BacktestResult.warnings`.

Rebalance frequency
-------------------
``"daily"`` decides at every close. ``"weekly"`` and ``"monthly"`` decide at the close of
the first bar of each ISO week or calendar month. This is detected causally, by comparing
with the previous bar, so it needs no future data. Protective exits are checked every bar
regardless.

Benchmark
---------
Equal-weight buy-and-hold. The initial capital is split equally across the symbols, and
each slice buys whole shares at that symbol's first open in the window. Costs come from
the same cost model. The benchmark is then held to the end.

Both curves are marked at each close, so the benchmark's first point already includes
its entry costs and the first session's open-to-close move. The metrics of *both* curves
(total return, CAGR, Sharpe, drawdown …) and the drawdown columns of the equity curve
are therefore measured from the initial capital
(``compute_metrics(..., start_equity=capital)``). Strategy and benchmark then cover the
same periods, starting from the same money.

Open positions are not liquidated at the end: they are marked at the last close, and the
trade statistics count closed (SELL) trades only.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from typing import Any, Final

import numpy as np
import pandas as pd

from trading_swarm.agents import (
    ExecutionAgent,
    RiskManagerAgent,
    SentimentAgent,
    TrendAnalyzerAgent,
)
from trading_swarm.agents.base import SignalAgent
from trading_swarm.analytics import compute_metrics, drawdown_series
from trading_swarm.backtest.result import (
    EQUITY_CURVE_COLUMNS,
    SIGNAL_LOG_COLUMNS,
    BacktestResult,
    empty_signal_log,
)
from trading_swarm.broker.accounting import CostModel, evaluate_exit, evaluate_exit_bar
from trading_swarm.broker.memory import InMemoryBroker
from trading_swarm.config import Settings
from trading_swarm.data.base import (
    DateLike,
    MarketDataError,
    MarketDataProvider,
    NoDataError,
    apply_period,
    normalize_ohlcv,
    to_utc_date,
    validate_symbol,
)
from trading_swarm.llm.base import LLMClient, is_llm_enabled, llm_method_label
from trading_swarm.models import (
    Action,
    AgentSignal,
    Decision,
    ExitReason,
    NewsItem,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    RiskAssessment,
    Side,
)
from trading_swarm.swarm import HISTORY_PERIOD, Consensus, compute_consensus, summarize_decision

logger = logging.getLogger(__name__)

#: Accepted values of ``rebalance``.
REBALANCE_FREQUENCIES: Final = ("daily", "weekly", "monthly")
#: Look-back the agents analyse at every bar: the live coordinator's history window.
AGENT_HISTORY_PERIOD: Final = HISTORY_PERIOD
#: Headlines published before ``bar date + NEWS_CUTOFF`` are visible at that bar's close
#: (20:00 UTC = 16:00 New York in daylight-saving time; an hour early in winter).
NEWS_CUTOFF: Final = pd.Timedelta(hours=20)
#: Pre-start history always requested (calendar days): one year plus slack, so the first
#: decision sees a full live-sized window.
MIN_LOOKBACK_DAYS: Final = 372
#: Calendar days requested per warm-up bar (5 trading days a week, plus holidays).
_CALENDAR_DAYS_PER_BAR: Final = 1.6
_WARMUP_SLACK_DAYS: Final = 14


class BacktestConfigError(ValueError):
    """Invalid backtest parameters (symbols, dates, capital, warm-up or rebalance)."""


# =============================================================================== helpers


def rebalance_mask(calendar: pd.DatetimeIndex, frequency: str) -> np.ndarray:
    """Boolean array: ``True`` where a decision is taken at that bar's close.

    ``"daily"`` marks every bar; ``"weekly"`` and ``"monthly"`` mark the first bar of each
    ISO week or calendar month. Each flag depends only on the bar and the one before it,
    so the schedule is causal (no future bar needed).
    """
    if frequency not in REBALANCE_FREQUENCIES:
        raise BacktestConfigError(
            f"rebalance must be one of {', '.join(REBALANCE_FREQUENCIES)}, got {frequency!r}"
        )
    n = len(calendar)
    if frequency == "daily" or n == 0:
        return np.ones(n, dtype=bool)
    if frequency == "weekly":
        iso = calendar.isocalendar()
        key = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    else:
        key = np.asarray(calendar.year * 100 + calendar.month)
    mask = np.ones(n, dtype=bool)
    mask[1:] = key[1:] != key[:-1]
    return mask


def reserve_order(
    snapshot: PortfolioSnapshot,
    symbol: str,
    side: Side,
    quantity: int,
    price: float,
    costs: CostModel,
) -> PortfolioSnapshot:
    """Pro-forma account after an approved, still-pending order fills at ``price``.

    The cash moves by the order's full cost or proceeds (slippage and commission
    included). ``positions_value`` and the position in ``symbol`` move by the shares at
    ``price``, and equity and gross exposure are recomputed. The drawdown (and peak) are
    left unchanged: the circuit breaker always looks at the real account.
    """
    held = snapshot.position(symbol)
    held_qty = held.quantity if held is not None else 0
    positions = [p for p in snapshot.positions if p.symbol != symbol]
    if side is Side.BUY:
        fill = costs.quote(side, quantity, price)
        new_qty = held_qty + quantity
        basis = (held.cost_basis if held is not None else 0.0) + fill.cash_required
        positions.append(
            Position(
                symbol=symbol,
                quantity=new_qty,
                avg_price=basis / new_qty,
                stop_price=held.stop_price if held is not None else None,
                take_profit_price=held.take_profit_price if held is not None else None,
                opened_at=held.opened_at if held is not None else snapshot.updated_at,
                market_price=price,
                market_value=new_qty * price,
            )
        )
        cash_delta = fill.cash_delta
        value_delta = quantity * price
    else:
        sold = min(quantity, held_qty)
        if sold <= 0:
            return snapshot
        cash_delta = costs.quote(side, sold, price).cash_delta
        value_delta = -sold * price
        if held is not None and held_qty > sold:
            remaining = held_qty - sold
            positions.append(
                held.model_copy(update={"quantity": remaining, "market_value": remaining * price})
            )
    cash = snapshot.cash + cash_delta
    positions_value = max(0.0, snapshot.positions_value + value_delta)
    equity = cash + positions_value
    exposure = positions_value / equity * 100.0 if equity > 0 else 0.0
    return snapshot.model_copy(
        update={
            "cash": cash,
            "equity": equity,
            "positions_value": positions_value,
            "gross_exposure_pct": exposure,
            "positions": sorted(positions, key=lambda p: p.symbol),
        }
    )


class _BarOpenQuotes:
    """Quote source for the ExecutionAgent in a backtest: the fill bar's open price.

    Implements the :class:`~trading_swarm.data.base.MarketDataProvider` surface the
    execution agent touches (``latest_price``); the other methods are inert.
    """

    name = "backtest"

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}

    def set(self, symbol: str, price: float) -> None:
        self._prices[symbol.upper()] = float(price)

    def latest_price(self, symbol: str) -> float:
        try:
            return self._prices[symbol.upper()]
        except KeyError as exc:
            raise MarketDataError(f"no simulated quote for {symbol}", symbol=symbol) from exc

    def history(
        self,
        symbol: str,
        start: DateLike | None = None,
        end: DateLike | None = None,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        raise MarketDataError("history is not served by the backtest quote source", symbol=symbol)

    def news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        return []

    def validate_symbol(self, symbol: str) -> str:
        return validate_symbol(symbol)


class _PointInTimeNews(_BarOpenQuotes):
    """News source for the SentimentAgent: only the headlines visible at the bar's close."""

    name = "backtest-news"

    def __init__(self) -> None:
        super().__init__()
        self.items: list[NewsItem] = []

    def news(self, symbol: str, limit: int = 10) -> list[NewsItem]:
        return list(self.items[: max(0, limit)])


class _SentimentReplay:
    """Point-in-time sentiment votes with one agent call per distinct headline set."""

    def __init__(
        self,
        provider: MarketDataProvider,
        llm: LLMClient,
        settings: Settings,
        symbols: Sequence[str],
        warnings: list[str],
    ) -> None:
        self.view = _PointInTimeNews()
        self.agent = SentimentAgent(self.view, llm, settings)
        self.method = llm_method_label(llm) if is_llm_enabled(llm) else "lexicon"
        self.headlines: dict[str, list[NewsItem]] = {}
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], AgentSignal] = {}
        for sym in symbols:
            try:
                items = list(provider.news(sym, SentimentAgent.MAX_HEADLINES))
            except MarketDataError as exc:
                logger.warning("backtest: news unavailable for %s (%s)", sym, exc)
                items = []
            dated = [item for item in items if item.published_at is not None]
            if len(dated) < len(items):
                warnings.append(
                    f"{sym}: {len(items) - len(dated)} headline(s) without a publication time "
                    "were ignored (they cannot be placed in time)"
                )
            self.headlines[sym] = dated

    @property
    def calls(self) -> int:
        """Distinct non-empty headline sets scored so far (= LLM requests at most)."""
        return sum(1 for _, headlines in self._cache if headlines)

    def vote(self, symbol: str, bar: pd.Timestamp) -> AgentSignal:
        cutoff = bar + NEWS_CUTOFF
        visible = [
            item
            for item in self.headlines.get(symbol, [])
            if item.published_at is not None and item.published_at <= cutoff
        ]
        key = (
            symbol,
            tuple((item.title, str(item.published_at)) for item in visible),
        )
        cached = self._cache.get(key)
        if cached is None:
            self.view.items = visible
            cached = self.agent.analyze(symbol)
            self._cache[key] = cached
        return cached


@dataclass
class _SymbolData:
    """One symbol's bars (look-back + window) as fast numpy columns."""

    symbol: str
    frame: pd.DataFrame
    #: Position of the first bar inside ``[start, end]`` (= bars of pre-start history).
    first: int
    #: Window bar timestamp → position in ``frame``.
    positions: dict[pd.Timestamp, int]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    @classmethod
    def build(cls, symbol: str, frame: pd.DataFrame, first: int) -> _SymbolData:
        window = frame.index[first:]
        return cls(
            symbol=symbol,
            frame=frame,
            first=first,
            positions={ts: first + k for k, ts in enumerate(window)},
            open=frame["Open"].to_numpy(dtype=float),
            high=frame["High"].to_numpy(dtype=float),
            low=frame["Low"].to_numpy(dtype=float),
            close=frame["Close"].to_numpy(dtype=float),
        )

    @property
    def last(self) -> int:
        return len(self.frame) - 1

    @property
    def window_index(self) -> pd.DatetimeIndex:
        return self.frame.index[self.first :]

    def history(self, position: int) -> pd.DataFrame:
        """What the agents see at bar ``position``: the trailing live-sized window."""
        return apply_period(self.frame.iloc[: position + 1], AGENT_HISTORY_PERIOD)


@dataclass
class _Candidate:
    """A symbol's votes and consensus at one decision bar, before the risk gate."""

    symbol: str
    price: float
    history: pd.DataFrame
    signals: list[AgentSignal]
    consensus: Consensus


@dataclass
class _Pending:
    """An approved decision waiting for the next bar's open."""

    decision: Decision
    proposed: Action


# =============================================================================== backtester


class Backtester:
    """Replays the swarm's trend (+ optional sentiment), risk and execution logic over
    historical daily bars (see the module docstring for the event timeline).

    Args:
        provider: Market-data provider for the bars (and headlines, with sentiment).
        settings: Thresholds, weights, risk limits and costs, exactly as used live.
        llm: LLM client for ``use_sentiment=True`` (default: built from ``settings``
            with :func:`trading_swarm.llm.get_llm_client` on first use).
        trend: Trend agent override (any :class:`~trading_swarm.agents.base.SignalAgent`).
        risk: Risk agent override.

    A ``Backtester`` holds no per-run state; concurrent :meth:`run` calls are safe as
    long as the agents are stateless (the built-in ones are).
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        settings: Settings,
        *,
        llm: LLMClient | None = None,
        trend: TrendAnalyzerAgent | SignalAgent | None = None,
        risk: RiskManagerAgent | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.llm = llm
        self.cost_model = CostModel.from_settings(settings)
        self.trend: TrendAnalyzerAgent | SignalAgent = (
            trend if trend is not None else TrendAnalyzerAgent(settings)
        )
        self.risk = risk if risk is not None else RiskManagerAgent(settings, self.cost_model)

    # ------------------------------------------------------------------ public API
    def run(
        self,
        symbols: str | Sequence[str],
        start: DateLike,
        end: DateLike,
        initial_capital: float | None = None,
        use_sentiment: bool = False,
        warmup: int = 200,
        rebalance: str = "daily",
    ) -> BacktestResult:
        """Backtest ``symbols`` between ``start`` and ``end`` (inclusive dates).

        Args:
            symbols: One ticker, a comma-separated string, or a sequence of tickers
                traded from one shared account.
            start: First date of the simulation window (inclusive).
            end: Last date of the simulation window (inclusive).
            initial_capital: Starting cash (default ``settings.initial_capital``).
            use_sentiment: Add the point-in-time sentiment vote (see module docstring).
            warmup: Bars of history required before ``start`` (shortfall → warning).
            rebalance: ``"daily"``, ``"weekly"`` or ``"monthly"`` decision schedule.

        Raises:
            BacktestConfigError: invalid parameters (a ``ValueError``).
            InvalidSymbolError, NoDataError, ProviderUnavailableError: market-data
                failures (all :class:`~trading_swarm.data.MarketDataError`); a symbol
                with no bars inside the window raises ``NoDataError``.
        """
        syms = self._parse_symbols(symbols)
        start_ts, end_ts = self._parse_window(start, end)
        capital = self._parse_capital(initial_capital)
        warmup_bars = self._parse_warmup(warmup)
        frequency = self._parse_rebalance(rebalance)

        warnings: list[str] = []
        s = self.settings
        if s.weight_trend <= 0 and not (use_sentiment and s.weight_sentiment > 0):
            message = "WEIGHT_TREND is 0 and sentiment is disabled: the strategy cannot trade"
            logger.warning("backtest: %s", message)
            warnings.append(message)

        data = {sym: self._load(sym, start_ts, end_ts, warmup_bars, warnings) for sym in syms}
        calendar = reduce(
            lambda left, right: left.union(right), (d.window_index for d in data.values())
        )

        sentiment: _SentimentReplay | None = None
        if use_sentiment:
            if self.llm is None:
                from trading_swarm.llm import get_llm_client

                self.llm = get_llm_client(s)
            sentiment = _SentimentReplay(self.provider, self.llm, s, syms, warnings)

        logger.info(
            "backtest %s %s → %s: %d bars, capital %.2f, rebalance %s, sentiment %s",
            ",".join(syms),
            calendar[0].date(),
            calendar[-1].date(),
            len(calendar),
            capital,
            frequency,
            sentiment.method if sentiment is not None else "off",
        )
        simulation = _Simulation(self, data, calendar, capital, frequency, sentiment, warnings)
        simulation.run()

        parameters = self._parameters(
            syms, start_ts, end_ts, capital, use_sentiment, warmup_bars, frequency, data, sentiment
        )
        result = simulation.result(syms, parameters)
        logger.info(
            "backtest %s done: return %s%% vs buy & hold %s%%, %d fills, max drawdown %s%%",
            ",".join(syms),
            result.metrics.total_return_pct,
            result.benchmark_metrics.total_return_pct,
            len(result.trades),
            result.metrics.max_drawdown_pct,
        )
        return result

    # ------------------------------------------------------------------ validation
    def _parse_symbols(self, symbols: str | Sequence[str]) -> list[str]:
        raw: list[Any] = symbols.split(",") if isinstance(symbols, str) else list(symbols)
        parsed: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise BacktestConfigError(f"symbols must be strings, got {item!r}")
            if not item.strip():
                continue
            sym = self.provider.validate_symbol(item)
            if sym not in parsed:
                parsed.append(sym)
        if not parsed:
            raise BacktestConfigError("at least one symbol is required")
        return parsed

    @staticmethod
    def _parse_window(start: DateLike, end: DateLike) -> tuple[pd.Timestamp, pd.Timestamp]:
        try:
            start_ts, end_ts = to_utc_date(start), to_utc_date(end)
        except ValueError as exc:
            raise BacktestConfigError(f"invalid backtest dates: {exc}") from exc
        if start_ts > end_ts:
            raise BacktestConfigError(
                f"start {start_ts.date()} must not be after end {end_ts.date()}"
            )
        return start_ts, end_ts

    def _parse_capital(self, initial_capital: float | None) -> float:
        raw = self.settings.initial_capital if initial_capital is None else initial_capital
        try:
            capital = float(raw)
        except (TypeError, ValueError) as exc:
            raise BacktestConfigError(f"initial capital must be a number, got {raw!r}") from exc
        if not (math.isfinite(capital) and capital > 0):
            raise BacktestConfigError(f"initial capital must be positive, got {raw!r}")
        return capital

    @staticmethod
    def _parse_warmup(warmup: int) -> int:
        if isinstance(warmup, bool) or not isinstance(warmup, int | np.integer) or warmup < 0:
            raise BacktestConfigError(f"warmup must be a non-negative integer, got {warmup!r}")
        return int(warmup)

    @staticmethod
    def _parse_rebalance(rebalance: str) -> str:
        frequency = rebalance.strip().lower() if isinstance(rebalance, str) else rebalance
        if frequency not in REBALANCE_FREQUENCIES:
            raise BacktestConfigError(
                f"rebalance must be one of {', '.join(REBALANCE_FREQUENCIES)}, got {rebalance!r}"
            )
        return frequency

    # ------------------------------------------------------------------ data
    def _load(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        warmup: int,
        warnings: list[str],
    ) -> _SymbolData:
        """Fetch look-back + window bars for ``symbol`` (canonical, de-duplicated)."""
        lookback_days = max(
            MIN_LOOKBACK_DAYS, math.ceil(warmup * _CALENDAR_DAYS_PER_BAR) + _WARMUP_SLACK_DAYS
        )
        raw = self.provider.history(
            symbol, start=start - pd.Timedelta(days=lookback_days), end=end, interval="1d"
        )
        # Re-normalising is idempotent for canonical frames, guards custom providers, and
        # gives the backtest its own copy (providers may hand out shared/cached frames).
        frame = normalize_ohlcv(raw, symbol=symbol, interval="1d")
        frame = frame[frame.index < end + pd.Timedelta(days=1)]
        first = int(frame.index.searchsorted(start))
        if first >= len(frame):
            raise NoDataError(
                f"no price data for {symbol} between {start.date()} and {end.date()}",
                symbol=symbol,
            )
        if first < warmup:
            message = (
                f"{symbol}: only {first} bar(s) of history before {start.date()} "
                f"(warm-up asks for {warmup}); early decisions use a shorter look-back"
            )
            logger.warning("backtest: %s", message)
            warnings.append(message)
        return _SymbolData.build(symbol, frame, first)

    def _parameters(
        self,
        symbols: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        capital: float,
        use_sentiment: bool,
        warmup: int,
        frequency: str,
        data: dict[str, _SymbolData],
        sentiment: _SentimentReplay | None,
    ) -> dict[str, Any]:
        s = self.settings
        return {
            "symbols": symbols,
            "requested_start": start.date().isoformat(),
            "requested_end": end.date().isoformat(),
            "initial_capital": capital,
            "data_provider": getattr(self.provider, "name", type(self.provider).__name__),
            "use_sentiment": use_sentiment,
            "sentiment_method": sentiment.method if sentiment is not None else None,
            "sentiment_calls": sentiment.calls if sentiment is not None else 0,
            "warmup": warmup,
            "warmup_bars": {sym: d.first for sym, d in data.items()},
            "rebalance": frequency,
            "agent_history_period": AGENT_HISTORY_PERIOD,
            "weights": {
                "trend": s.weight_trend,
                "sentiment": s.weight_sentiment if use_sentiment else 0.0,
            },
            "buy_threshold": s.buy_threshold,
            "sell_threshold": s.sell_threshold,
            "risk_per_trade": s.risk_per_trade,
            "max_position_pct": s.max_position_pct,
            "max_gross_exposure": s.max_gross_exposure,
            "atr_stop_multiple": s.atr_stop_multiple,
            "take_profit_r": s.take_profit_r,
            "max_drawdown_halt": s.max_drawdown_halt,
            "max_volatility": s.max_volatility,
            "commission_rate": s.commission_rate,
            "min_commission": s.min_commission,
            "slippage_bps": s.slippage_bps,
            "risk_free_rate": s.risk_free_rate,
            "fill_model": "market orders at the next bar's open, with slippage and commission",
            "exit_model": (
                "stop-loss / take-profit checked against each bar's low / high; gaps fill "
                "at the open (checked before pending add-ons fill); the stop is assumed "
                "first when both levels are inside a bar"
            ),
            "benchmark": "equal-weight buy & hold, bought at each symbol's first open",
        }


# =============================================================================== simulation


class _Simulation:
    """Mutable state of one backtest run (created fresh by every :meth:`Backtester.run`)."""

    def __init__(
        self,
        backtester: Backtester,
        data: dict[str, _SymbolData],
        calendar: pd.DatetimeIndex,
        capital: float,
        frequency: str,
        sentiment: _SentimentReplay | None,
        warnings: list[str],
    ) -> None:
        self.bt = backtester
        self.settings = backtester.settings
        self.data = data
        self.calendar = calendar
        self.capital = capital
        self.mask = rebalance_mask(calendar, frequency)
        self.sentiment = sentiment
        self.warnings = warnings
        self.costs = backtester.cost_model
        self.broker = InMemoryBroker(capital, self.costs)
        self.benchmark = InMemoryBroker(capital, self.costs)
        self.quotes = _BarOpenQuotes()
        self.execution = ExecutionAgent(self.broker, self.settings, self.quotes)
        self.allocation = capital / len(data)
        self.pending: dict[str, _Pending] = {}
        self.last_close: dict[str, float] = {}
        self.benchmark_bought: set[str] = set()
        self.decisions: list[Decision] = []
        self.rows: list[dict[str, Any]] = []
        self.proposals = 0
        self.clipped = 0
        self.cancelled = 0
        self.agent_failures = 0

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        for k, ts in enumerate(self.calendar):
            bars = {sym: d.positions[ts] for sym, d in self.data.items() if ts in d.positions}
            when = ts.to_pydatetime()
            # The open: signal exits, then gaps through the overnight stops / targets,
            # then entries. A gap exit must see the levels in force *before* any add-on
            # fills, or the add-on's merged (re-anchored) levels would hide the breach.
            self._fill_pending(when, bars, Action.SELL)
            closed_at_open = self._protective_exits(when, bars, at_open=True)
            self._fill_pending(when, bars, Action.BUY, closed_at_open=closed_at_open)
            self._protective_exits(when, bars)
            for sym, i in bars.items():
                self.last_close[sym] = float(self.data[sym].close[i])
            snapshot = self.broker.record_snapshot(self.last_close, timestamp=when)
            self._benchmark_bar(when, bars)
            if self.mask[k]:
                self._decide(ts, bars, snapshot)

    # ------------------------------------------------------------------ 1. open
    def _fill_pending(
        self,
        when: datetime,
        bars: dict[str, int],
        side: Action,
        *,
        closed_at_open: dict[str, ExitReason] | None = None,
    ) -> None:
        """Fill the pending ``side`` orders of symbols trading on this bar, at its open.

        A BUY for a symbol in ``closed_at_open`` (its position was just closed by a gap
        exit) is cancelled instead of filled; see the module docstring.
        """
        due = [sym for sym, p in self.pending.items() if sym in bars and p.decision.action is side]
        for sym in due:
            pending = self.pending.pop(sym)
            decision = pending.decision
            open_price = float(self.data[sym].open[bars[sym]])
            exit_reason = (closed_at_open or {}).get(sym)
            if side is Action.BUY and exit_reason is not None:
                order: Order | None = self._cancel(decision, open_price, exit_reason, when)
            else:
                self.quotes.set(sym, open_price)
                order = self.execution.execute(
                    self._affordable(decision, open_price), timestamp=when
                )
            decision.order = order
            decision.executed = order is not None and order.is_filled
            decision.summary = summarize_decision(
                decision,
                proposed=pending.proposed,
                buy_threshold=self.settings.buy_threshold,
                sell_threshold=self.settings.sell_threshold,
                atr_multiple=self.settings.atr_stop_multiple,
                auto_trade=True,
            )
            logger.debug("backtest fill %s: %s", sym, decision.summary)

    def _cancel(
        self, decision: Decision, open_price: float, reason: ExitReason, when: datetime
    ) -> Order:
        """A CANCELLED order for an add-on whose position a gap exit closed at the open."""
        self.cancelled += 1
        label = "stop-loss" if reason is ExitReason.STOP_LOSS else "take-profit"
        message = (
            f"cancelled at the open: the position's {label} was gapped through and closed "
            f"it at {open_price:.4f}, so this add-on (sized against that position) is stale"
        )
        logger.info("backtest %s on %s: %s", decision.symbol, when.date(), message)
        risk = decision.risk
        return Order(
            symbol=decision.symbol,
            side=Side.BUY,
            quantity=risk.quantity,
            status=OrderStatus.CANCELLED,
            requested_at=when,
            reference_price=round(open_price, 4),
            stop_price=risk.stop_price,
            take_profit_price=risk.take_profit_price,
            reason=f"swarm BUY (score {decision.score:+.2f}, conf {decision.confidence:.2f})",
            decision_id=decision.id,
            reject_reason=message,
        )

    def _affordable(self, decision: Decision, open_price: float) -> Decision:
        """Trim a BUY that the open price made unaffordable to what the cash can pay."""
        risk = decision.risk
        if risk.action is not Action.BUY:
            return decision
        affordable = self.costs.max_affordable_quantity(self.broker.cash, open_price)
        if affordable >= risk.quantity or affordable < 1:
            return decision  # fits, or nothing fits (the broker will reject it)
        self.clipped += 1
        logger.info(
            "backtest %s: open %.4f leaves cash for %d of %d approved shares; trimming",
            decision.symbol,
            open_price,
            affordable,
            risk.quantity,
        )
        trimmed = risk.model_copy(update={"quantity": affordable})
        return decision.model_copy(update={"risk": trimmed})

    # ------------------------------------------------------------------ 2. intrabar
    def _protective_exits(
        self, when: datetime, bars: dict[str, int], *, at_open: bool = False
    ) -> dict[str, ExitReason]:
        """Close positions whose stop-loss / take-profit this bar hits; returns the exits.

        ``at_open=True`` checks only the open (gap-throughs, filled at the open), before
        pending BUYs fill. Otherwise the whole bar is checked
        (:func:`~trading_swarm.broker.evaluate_exit_bar`).
        """
        closed: dict[str, ExitReason] = {}
        for sym, i in bars.items():
            state = self.broker.position_state(sym)
            if state is None or (state.stop_price is None and state.take_profit_price is None):
                continue
            d = self.data[sym]
            open_price = float(d.open[i])
            hit: tuple[ExitReason, float] | None
            if at_open:
                gap = evaluate_exit(state.stop_price, state.take_profit_price, open_price)
                hit = None if gap is None else (gap, open_price)
            else:
                hit = evaluate_exit_bar(
                    state.stop_price,
                    state.take_profit_price,
                    open_price,
                    float(d.high[i]),
                    float(d.low[i]),
                )
            if hit is None:
                continue
            reason, level = hit
            order = self.broker.submit_order(
                sym, Side.SELL, state.quantity, level, reason=reason.value, timestamp=when
            )
            if order.is_filled:
                closed[sym] = reason
            logger.debug(
                "backtest exit %s on %s: %s %d @ %s (%s)",
                sym,
                when.date(),
                reason.value,
                order.quantity,
                order.fill_price,
                order.status.value,
            )
        return closed

    # ------------------------------------------------------------------ 3. benchmark
    def _benchmark_bar(self, when: datetime, bars: dict[str, int]) -> None:
        for sym, i in bars.items():
            if sym in self.benchmark_bought:
                continue
            self.benchmark_bought.add(sym)
            price = float(self.data[sym].open[i])
            budget = min(self.allocation, self.benchmark.cash)
            quantity = self.costs.max_affordable_quantity(budget, price)
            if quantity < 1:
                message = f"{sym}: benchmark allocation {budget:.2f} buys no share at {price:.4f}"
                logger.warning("backtest: %s", message)
                self.warnings.append(message)
                continue
            self.benchmark.submit_order(
                sym, Side.BUY, quantity, price, reason="benchmark buy & hold", timestamp=when
            )
        self.benchmark.record_snapshot(self.last_close, timestamp=when)

    # ------------------------------------------------------------------ 4. decisions
    def _decide(self, ts: pd.Timestamp, bars: dict[str, int], snapshot: PortfolioSnapshot) -> None:
        s = self.settings
        candidates: list[_Candidate] = []
        for sym, i in bars.items():
            d = self.data[sym]
            if i >= d.last:
                continue  # no next open inside the window to fill at
            history = d.history(i)
            signals = [self._vote(self.bt.trend, sym, history)]
            votes = [(signals[0], s.weight_trend)]
            if self.sentiment is not None:
                sentiment = self._sentiment_vote(sym, ts)
                signals.append(sentiment)
                votes.append((sentiment, s.weight_sentiment))
            consensus = compute_consensus(votes, s.buy_threshold, s.sell_threshold)
            candidates.append(_Candidate(sym, float(d.close[i]), history, signals, consensus))

        # Exits before entries; the stable sort keeps the caller's order otherwise.
        candidates.sort(key=lambda c: 0 if c.consensus.action is Action.SELL else 1)
        account = self._reserve_pending(snapshot)
        for c in candidates:
            risk: RiskAssessment | None = None
            if c.consensus.action is not Action.HOLD:
                self.proposals += 1
                risk = self.bt.risk.assess(
                    symbol=c.symbol,
                    price=c.price,
                    action=c.consensus.action,
                    score=c.consensus.score,
                    confidence=c.consensus.confidence,
                    history=c.history,
                    portfolio=account,
                )
                if risk.approved and risk.action is not Action.HOLD and risk.quantity >= 1:
                    decision = Decision(
                        symbol=c.symbol,
                        timestamp=ts.to_pydatetime(),
                        price=round(c.price, 4),
                        action=risk.action,
                        score=round(c.consensus.score, 4),
                        confidence=round(c.consensus.confidence, 4),
                        signals=c.signals,
                        risk=risk,
                    )
                    self.pending[c.symbol] = _Pending(decision, c.consensus.action)
                    self.decisions.append(decision)
                    side = Side.BUY if risk.action is Action.BUY else Side.SELL
                    account = reserve_order(
                        account, c.symbol, side, risk.quantity, c.price, self.costs
                    )
            self.rows.append(self._log_row(ts, c, risk))

    def _reserve_pending(self, snapshot: PortfolioSnapshot) -> PortfolioSnapshot:
        """The account with every still-pending order applied pro forma.

        At decision time ``self.pending`` only holds orders whose symbol had no bar today
        (an exchange holiday, a halt …), so they are still waiting for their next open.
        Their cash and exposure are already committed. Sizing other symbols as if that
        cash were free would overshoot the cash and gross-exposure limits once everything
        fills.
        """
        account = snapshot
        for sym, pending in self.pending.items():
            decision = pending.decision
            side = Side.BUY if decision.action is Action.BUY else Side.SELL
            account = reserve_order(
                account, sym, side, decision.risk.quantity, decision.price, self.costs
            )
        return account

    def _vote(self, agent: SignalAgent, symbol: str, history: pd.DataFrame) -> AgentSignal:
        """Run a signal agent; a crashing agent abstains (as in the live coordinator)."""
        name = getattr(agent, "name", type(agent).__name__)
        try:
            signal = agent.analyze(symbol, history)
            if not isinstance(signal, AgentSignal):
                raise TypeError(f"{name} agent returned {type(signal).__name__}, not AgentSignal")
            return signal
        except Exception as exc:
            self._agent_failed(name, symbol, exc)
            return AgentSignal.neutral(
                name, f"{name} agent failed ({type(exc).__name__})", error=str(exc)
            )

    def _sentiment_vote(self, symbol: str, ts: pd.Timestamp) -> AgentSignal:
        assert self.sentiment is not None
        try:
            return self.sentiment.vote(symbol, ts)
        except Exception as exc:
            self._agent_failed("sentiment", symbol, exc)
            return AgentSignal.neutral(
                "sentiment", f"sentiment agent failed ({type(exc).__name__})", error=str(exc)
            )

    def _agent_failed(self, name: str, symbol: str, exc: Exception) -> None:
        self.agent_failures += 1
        if self.agent_failures == 1:
            logger.exception("backtest: %s agent failed on %s; it abstains", name, symbol)
        else:
            logger.debug("backtest: %s agent failed on %s: %s", name, symbol, exc)

    def _log_row(
        self, ts: pd.Timestamp, c: _Candidate, risk: RiskAssessment | None
    ) -> dict[str, Any]:
        trend = c.signals[0]
        sentiment = c.signals[1] if len(c.signals) > 1 else None
        if risk is None:
            if c.consensus.total_weight <= 0:
                note = "no conviction: every agent abstained"
            else:
                note = (
                    f"inside the HOLD band ({self.settings.sell_threshold:+.2f}, "
                    f"{self.settings.buy_threshold:+.2f})"
                )
            action, approved, quantity = Action.HOLD, False, 0
        else:
            note = risk.reasons[0] if risk.reasons else ""
            approved = risk.approved and risk.action is not Action.HOLD
            action, quantity = risk.action, risk.quantity if approved else 0
        return {
            "timestamp": ts,
            "symbol": c.symbol,
            "price": round(c.price, 4),
            "trend_signal": trend.signal,
            "trend_confidence": trend.confidence,
            "sentiment_signal": sentiment.signal if sentiment is not None else math.nan,
            "sentiment_confidence": sentiment.confidence if sentiment is not None else math.nan,
            "score": round(c.consensus.score, 4),
            "confidence": round(c.consensus.confidence, 4),
            "proposed": c.consensus.action.value,
            "action": action.value,
            "approved": approved,
            "quantity": quantity,
            "note": note,
        }

    # ------------------------------------------------------------------ result
    def result(self, symbols: list[str], parameters: dict[str, Any]) -> BacktestResult:
        rf = self.settings.risk_free_rate
        capital = self.capital
        curve = self.broker.equity_curve()
        bench = self.benchmark.equity_curve()
        # Both curves start from the initial capital: the benchmark's first close-marked
        # point already includes its entry costs and the first session's move.
        curve["drawdown"] = drawdown_series(curve["equity"], start_equity=capital).reindex(
            curve.index
        )
        curve["benchmark"] = bench["equity"].reindex(curve.index)
        curve["benchmark_drawdown"] = drawdown_series(
            bench["equity"], start_equity=capital
        ).reindex(curve.index)
        curve = curve[list(EQUITY_CURVE_COLUMNS)]

        trades = list(reversed(self.broker.trades(limit=None)))  # chronological
        bench_trades = list(reversed(self.benchmark.trades(limit=None)))
        metrics = compute_metrics(
            curve[["equity", "positions_value"]],
            trades,
            risk_free_rate=rf,
            start_equity=capital,
        )
        benchmark_metrics = compute_metrics(
            bench[["equity", "positions_value"]],
            bench_trades,
            risk_free_rate=rf,
            start_equity=capital,
        )
        final = self.broker.portfolio(self.last_close, timestamp=self.calendar[-1].to_pydatetime())

        signal_log = (
            pd.DataFrame(self.rows, columns=list(SIGNAL_LOG_COLUMNS))
            if self.rows
            else empty_signal_log()
        )
        if self.agent_failures:
            self.warnings.append(
                f"an agent failed {self.agent_failures} time(s) and abstained (see logs)"
            )
        orders = self.broker.orders(limit=None)
        sells = [t for t in trades if t.side is Side.SELL]
        protective = {reason.value for reason in ExitReason}
        exit_reasons = Counter(t.reason if t.reason in protective else "signal" for t in sells)
        stats: dict[str, Any] = {
            "bars": len(self.calendar),
            "decision_bars": int(self.mask.sum()),
            "evaluations": len(self.rows),
            "proposals": self.proposals,
            "approved": len(self.decisions),
            "vetoed": self.proposals - len(self.decisions),
            "orders_filled": sum(1 for o in orders if o.status is OrderStatus.FILLED),
            "orders_rejected": sum(1 for o in orders if o.status is OrderStatus.REJECTED),
            "orders_trimmed": self.clipped,
            "orders_cancelled": self.cancelled,
            "fills": len(trades),
            "round_trips": len(sells),
            "exits": {
                ExitReason.STOP_LOSS.value: exit_reasons.get(ExitReason.STOP_LOSS.value, 0),
                ExitReason.TAKE_PROFIT.value: exit_reasons.get(ExitReason.TAKE_PROFIT.value, 0),
                "signal": exit_reasons.get("signal", 0),
            },
            "open_positions": [p.symbol for p in final.positions],
            "final_equity": final.equity,
            "agent_failures": self.agent_failures,
        }
        return BacktestResult(
            symbols=symbols,
            start=self.calendar[0].date(),
            end=self.calendar[-1].date(),
            initial_capital=self.capital,
            parameters=parameters,
            equity_curve=curve,
            trades=trades,
            metrics=metrics,
            benchmark_metrics=benchmark_metrics,
            decisions=self.decisions,
            signal_log=signal_log,
            final_portfolio=final,
            stats=stats,
            warnings=self.warnings,
        )


__all__ = [
    "AGENT_HISTORY_PERIOD",
    "MIN_LOOKBACK_DAYS",
    "NEWS_CUTOFF",
    "REBALANCE_FREQUENCIES",
    "BacktestConfigError",
    "BacktestResult",
    "Backtester",
    "rebalance_mask",
    "reserve_order",
]
