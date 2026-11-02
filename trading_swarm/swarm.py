"""Swarm coordinator: runs the agents for a symbol and turns their votes into a decision.

Pipeline for :meth:`SwarmCoordinator.analyze`
---------------------------------------------
1. Validate the ticker and fetch one year of daily bars. ``InvalidSymbolError`` /
   ``NoDataError`` → :class:`SymbolNotFound` (HTTP 404); ``ProviderUnavailableError`` or
   any other ``MarketDataError`` → :class:`DataUnavailable` (HTTP 502).
2. Run the trend and sentiment agents concurrently (``ThreadPoolExecutor``). An agent
   that raises abstains (signal 0, confidence 0) instead of failing the decision; an
   agent whose weight is 0 is not run at all (no wasted LLM call).
3. Consensus (:func:`compute_consensus`) with vote weights ``vᵢ = wᵢ·cᵢ`` (agent weight
   × agent confidence)::

       score      = Σ vᵢ·sᵢ / Σ vᵢ                       (0 when Σ vᵢ = 0)
       mean_conf  = Σ vᵢ·cᵢ / Σ vᵢ                       (confidence-weighted mean)
       agreement  = |Σ vᵢ·sᵢ| / Σ vᵢ·|sᵢ|                (1 = all agents same direction)
       confidence = mean_conf × (0.5 + 0.5·agreement)

   so an abstaining agent (``cᵢ = 0``) neither moves the score nor dilutes the
   confidence, and agents that contradict each other lower the confidence (by up to
   half). ``score ≥ buy_threshold`` → BUY, ``score ≤ sell_threshold`` → SELL, else HOLD.
4. Protective exits: every open position whose stop-loss or take-profit is breached by
   the prices just fetched (the analysed symbol's last close, a fresh quote for every
   other holding) is closed (:meth:`~trading_swarm.agents.ExecutionAgent.enforce_exits`).
   This runs whether or not auto-trading is on, the same as in a watchlist scan. A breached
   stop is therefore enforced before any new trade is sized, instead of being overwritten
   by an add-on BUY. It also means stops are enforced on every analysis, not only when the
   scheduler or ``POST /scan`` runs. The exit orders are returned in
   :attr:`Decision.exits <trading_swarm.models.Decision.exits>` and described in the
   summary, so a caller always sees what the analysis sold.
5. Risk gate (:class:`~trading_swarm.agents.RiskManagerAgent`) against the portfolio
   marked to current prices. The decision's ``action`` is the risk agent's final action
   (a vetoed BUY is a HOLD; the summary says what was proposed and why it was vetoed).
   If a protective exit closed the analysed symbol earlier in the same run (this
   analysis, a sibling analysis of the same batch, or the scan's exit pass), the risk
   agent gets that exit order: a BUY is vetoed, because buying back at the price the
   exit just sold at would only pay costs twice. The backtester cancels a pending BUY
   whose position the open gapped out of for the same reason. A SELL proposal is
   reported as already done by the exit.
6. If auto-trading (argument, else ``settings.auto_trade``) and approved →
   :meth:`~trading_swarm.agents.ExecutionAgent.execute`.
7. After any fill (exit or entry) an equity snapshot is recorded, so the equity curve and
   ``GET /equity`` metrics reflect trading even without scheduled scans.
8. ``alerts.on_exit`` for each filled exit, ``alerts.on_decision(decision)`` (and
   ``on_price_move`` when available), then ``broker.save_decision(decision)``. Alert
   failures are logged, never raised.

Steps 4-7 run under a coordinator-wide lock, so concurrent analyses (``analyze_many``,
the API's thread pool, the scheduler) size every trade against an account that already
includes the previous fills — the gross-exposure and cash limits cannot be overshot by
two analyses racing each other. Data fetching and the agents themselves run unlocked.
"""

from __future__ import annotations

import contextvars
import logging
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import pandas as pd

from trading_swarm.agents import (
    ExecutionAgent,
    RiskManagerAgent,
    SentimentAgent,
    TrendAnalyzerAgent,
)
from trading_swarm.agents.base import SignalAgent
from trading_swarm.agents.risk import FALLBACK_STOP_PCT, describe_exit_fill, exit_label
from trading_swarm.config import Settings
from trading_swarm.data.base import (
    InvalidSymbolError,
    MarketDataError,
    MarketDataProvider,
    NoDataError,
    daily_change_pct,
)
from trading_swarm.models import (
    Action,
    AgentSignal,
    BatchAnalysis,
    Decision,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    RiskAssessment,
    ScanReport,
    SymbolError,
    utc_now,
)

if TYPE_CHECKING:
    from trading_swarm.broker.paper import PaperBroker
    from trading_swarm.llm.base import LLMClient

logger = logging.getLogger(__name__)

#: Maximum number of symbols analysed in parallel by :meth:`SwarmCoordinator.analyze_many`.
MAX_BATCH_WORKERS: Final = 4
#: History requested for every analysis (enough bars for SMA200).
HISTORY_PERIOD: Final = "1y"
#: Agent rationales are clipped to this many characters inside the decision summary.
SUMMARY_RATIONALE_CHARS: Final = 160

# =============================================================================== errors


class SwarmError(Exception):
    """Base class for coordinator errors that the API maps to HTTP status codes."""

    def __init__(self, message: str, *, symbol: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.symbol = symbol


class SymbolNotFound(SwarmError):
    """Invalid ticker or no market data for it (HTTP 404)."""


class DataUnavailable(SwarmError):
    """The market-data provider failed or is unreachable (HTTP 502)."""


# =============================================================================== alerts


@runtime_checkable
class AlertSink(Protocol):
    """What the coordinator needs from an alert manager (duck-typed).

    :class:`~trading_swarm.alerts.manager.AlertManager` satisfies it. Optional extra hooks
    are called when present: ``on_price_move(symbol, pct, price=last_close)`` after each
    analysis (``pct`` is the 1-day close-to-close change in percent) and
    ``on_drawdown(snapshot)`` after each watchlist scan.
    """

    def on_decision(self, decision: Decision) -> Any:
        """Called once per analysed decision (after execution)."""
        ...

    def on_exit(self, order: Order) -> Any:
        """Called for every filled stop-loss / take-profit exit."""
        ...


# =============================================================================== consensus


@dataclass(frozen=True)
class Consensus:
    """The combined vote of the signal agents (see the module docstring)."""

    score: float
    confidence: float
    action: Action
    #: ``|Σ vᵢ·sᵢ| / Σ vᵢ·|sᵢ|`` (1 when the voting agents agree on direction).
    agreement: float
    #: Confidence-weighted mean confidence ``Σ vᵢ·cᵢ / Σ vᵢ``.
    mean_confidence: float
    #: ``Σ wᵢ·cᵢ``: total voting weight (0 = every agent abstained).
    total_weight: float


def action_for_score(score: float, buy_threshold: float, sell_threshold: float) -> Action:
    """BUY at or above ``buy_threshold``, SELL at or below ``sell_threshold``, else HOLD."""
    if score >= buy_threshold:
        return Action.BUY
    if score <= sell_threshold:
        return Action.SELL
    return Action.HOLD


def compute_consensus(
    votes: Sequence[tuple[AgentSignal, float]],
    buy_threshold: float,
    sell_threshold: float,
) -> Consensus:
    """Combine ``(signal, agent_weight)`` pairs into a :class:`Consensus`.

    Pure function shared by the live coordinator and the backtester. Passing only the
    trend vote gives the trend-only consensus with renormalised weights.
    """
    weighted = [(sig, max(0.0, float(w)) * sig.confidence) for sig, w in votes]
    total = sum(v for _, v in weighted)
    if total <= 0:
        return Consensus(0.0, 0.0, Action.HOLD, 0.0, 0.0, 0.0)
    directional = sum(v * sig.signal for sig, v in weighted)
    magnitude = sum(v * abs(sig.signal) for sig, v in weighted)
    score = max(-1.0, min(1.0, directional / total))
    mean_confidence = sum(v * sig.confidence for sig, v in weighted) / total
    agreement = abs(directional) / magnitude if magnitude > 0 else 1.0
    confidence = max(0.0, min(1.0, mean_confidence * (0.5 + 0.5 * agreement)))
    return Consensus(
        score=score,
        confidence=confidence,
        action=action_for_score(score, buy_threshold, sell_threshold),
        agreement=agreement,
        mean_confidence=mean_confidence,
        total_weight=total,
    )


# =============================================================================== summary


def _clip_text(text: str, limit: int = SUMMARY_RATIONALE_CHARS) -> str:
    """Collapse whitespace and clip ``text`` at a word boundary (with an ellipsis)."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ;,:") + "…"


def _sentence(part: str) -> str:
    part = part.rstrip()
    return part if part.endswith((".", "…")) else part + "."


def _describe_signal(signal: AgentSignal) -> str:
    method = signal.data.get("method")
    label = f"{method}, " if isinstance(method, str) and method not in {"", "none"} else ""
    text = f"{signal.agent.capitalize()} {signal.signal:+.2f} ({label}conf {signal.confidence:.2f})"
    return f"{text}: {_clip_text(signal.rationale)}" if signal.rationale else text


def _describe_exits(exits: Sequence[Order]) -> str | None:
    """``"Protective exits before assessment: stop-loss sold 10 BEAR @ 57.00 (P&L …)"``."""
    parts: list[str] = []
    for order in exits:
        label = exit_label(order)
        if order.is_filled:
            pnl = f" (P&L {order.realized_pnl:+.2f})" if order.realized_pnl is not None else ""
            fill = describe_exit_fill(order).replace(" sh", f" {order.symbol}", 1)
            parts.append(f"{label} {fill}{pnl}")
        else:
            why = order.reject_reason or "see logs"
            parts.append(f"{label} exit of {order.symbol} {order.status.value.lower()}: {why}")
    return "Protective exits before assessment: " + "; ".join(parts) if parts else None


def _describe_risk(
    risk: RiskAssessment, proposed: Action, atr_multiple: float, own_exit: Order | None
) -> str:
    if proposed is Action.SELL and own_exit is not None and not risk.approved:
        reason = risk.reasons[0] if risk.reasons else "position already closed"
        return f"Risk: SELL not needed — {reason}"
    if risk.approved and risk.action is Action.BUY:
        if risk.checks.get("kept_existing_stop"):
            basis = "position's existing stop"
        elif risk.checks.get("stop_from_atr", True):
            basis = f"{atr_multiple:.1f}×ATR"
        else:
            basis = f"{FALLBACK_STOP_PCT:.0%} fallback"
        stop = f", stop {risk.stop_price:.2f} ({basis})" if risk.stop_price is not None else ""
        target = f", TP {risk.take_profit_price:.2f}" if risk.take_profit_price is not None else ""
        return f"Risk: approved {risk.quantity} shares{stop}{target}"
    if risk.approved and risk.action is Action.SELL:
        return f"Risk: approved SELL of {risk.quantity} shares (full position)"
    if proposed is Action.HOLD:
        return "Risk: no trade"
    reason = risk.reasons[0] if risk.reasons else "rule failed"
    return f"Risk: {proposed.value} vetoed — {reason}"


def summarize_decision(
    decision: Decision,
    *,
    proposed: Action,
    buy_threshold: float,
    sell_threshold: float,
    atr_multiple: float,
    auto_trade: bool,
) -> str:
    """Concise, deterministic explanation of ``decision`` (no timestamps or randomness).

    Example: ``"BUY BULL @ 182.08: consensus +0.69 (conf 0.81). Trend +0.74 (conf
    0.82): price above SMA20/50/200; … Sentiment +0.60 (claude-opus-5-5, conf 0.80): …
    Risk: approved 109 shares, stop 178.35 (2.0×ATR), TP 189.53. Executed: BUY 109 @
    182.17."``

    Protective exits in :attr:`Decision.exits` get their own sentence right after the
    headline. A SELL proposal whose position such an exit closed reads "SELL already
    done by the stop-loss", not "vetoed".
    """
    own_exit = decision.exit_for()
    head = (
        f"{decision.action.value} {decision.symbol} @ {decision.price:.2f}: "
        f"consensus {decision.score:+.2f} (conf {decision.confidence:.2f})"
    )
    if proposed is Action.HOLD:
        head += f", inside the HOLD band ({sell_threshold:+.2f}, {buy_threshold:+.2f})"
    elif decision.action is Action.HOLD:
        if proposed is Action.SELL and own_exit is not None:
            head += f" → SELL already done by the {exit_label(own_exit)}"
        else:
            head += f" → {proposed.value} vetoed by risk"
    parts = [head]
    exits_text = _describe_exits(decision.exits)
    if exits_text:
        parts.append(exits_text)
    parts.extend(_describe_signal(signal) for signal in decision.signals)
    parts.append(_describe_risk(decision.risk, proposed, atr_multiple, own_exit))

    order = decision.order
    if order is not None and order.status is OrderStatus.FILLED:
        parts.append(f"Executed: {order.side.value} {order.quantity} @ {order.fill_price:.2f}")
    elif order is not None:
        parts.append(f"Order {order.status.value.lower()}: {order.reject_reason or 'see logs'}")
    elif decision.risk.approved and decision.risk.action is not Action.HOLD:
        parts.append(
            "Execution failed (see logs)" if auto_trade else "Not executed (auto-trade off)"
        )
    return " ".join(_sentence(part) for part in parts)


# =============================================================================== coordinator


def _submit[T](pool: ThreadPoolExecutor, fn: Callable[..., T], *args: Any) -> Future[T]:
    """Submit ``fn`` with a copy of the caller's context (keeps the request id in logs)."""
    return pool.submit(contextvars.copy_context().run, fn, *args)


class _ExitLedger:
    """Protective exits filled during one run (an analysis, a batch or a scan).

    Every analysis of the run reads and extends it while holding the coordinator's trade
    lock, so an exit made by one analysis (or by the scan's exit pass) is seen by every
    later analysis of the same run, even when they run concurrently.
    """

    __slots__ = ("orders",)

    def __init__(self, orders: Iterable[Order] = ()) -> None:
        self.orders: list[Order] = []
        self.record(orders)

    def record(self, orders: Iterable[Order]) -> None:
        self.orders.extend(o for o in orders if o.is_filled)

    def closed(self, symbol: str) -> Order | None:
        """The run's filled exit of ``symbol``, if any."""
        return next((o for o in self.orders if o.symbol == symbol), None)


def _unique_orders(orders: Iterable[Order]) -> list[Order]:
    """``orders`` without repeats (by order id), in first-seen order."""
    seen: set[str] = set()
    unique: list[Order] = []
    for order in orders:
        if order.id not in seen:
            seen.add(order.id)
            unique.append(order)
    return unique


class SwarmCoordinator:
    """Runs the agents for a symbol and combines their votes into a :class:`Decision`.

    Agents not supplied are built from ``settings`` (``llm`` defaults to
    :func:`trading_swarm.llm.get_llm_client`); ``alerts`` is optional and duck-typed
    (:class:`AlertSink`). The coordinator is thread-safe.
    """

    def __init__(
        self,
        settings: Settings,
        provider: MarketDataProvider,
        broker: PaperBroker,
        *,
        llm: LLMClient | None = None,
        alerts: AlertSink | None = None,
        trend: TrendAnalyzerAgent | SignalAgent | None = None,
        sentiment: SentimentAgent | SignalAgent | None = None,
        risk: RiskManagerAgent | None = None,
        execution: ExecutionAgent | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.broker = broker
        self.alerts = alerts
        if sentiment is None:
            if llm is None:
                from trading_swarm.llm import get_llm_client

                llm = get_llm_client(settings)
            sentiment = SentimentAgent(provider, llm, settings)
        self.llm = llm if llm is not None else getattr(sentiment, "llm", None)
        self.trend = trend if trend is not None else TrendAnalyzerAgent(settings)
        self.sentiment = sentiment
        self.risk = (
            risk
            if risk is not None
            else RiskManagerAgent(settings, cost_model=getattr(broker, "cost_model", None))
        )
        self.execution = (
            execution if execution is not None else ExecutionAgent(broker, settings, provider)
        )
        self._trade_lock = threading.RLock()

    @property
    def trade_lock(self) -> threading.RLock:
        """Re-entrant lock held around every portfolio read → risk check → fill sequence.

        Anything else that mutates the account (manual ``POST /orders``, ``POST
        /portfolio/reset``) must take it too, so it cannot land between an automatic
        trade's risk sizing and its fill and push exposure or cash past the limits.
        """
        return self._trade_lock

    # ------------------------------------------------------------------ single symbol
    def analyze(self, symbol: str, auto_trade: bool | None = None) -> Decision:
        """Full swarm analysis of one symbol (optionally paper-trading the result).

        Whatever ``auto_trade`` says, the analysis first enforces the stop-loss and
        take-profit of every holding it can price, so it may sell positions, in this
        symbol or others. Those exits are returned in ``Decision.exits``.

        Raises:
            SymbolNotFound: invalid symbol / no data.
            DataUnavailable: provider failure.
        """
        return self._analyze(symbol, auto_trade, _ExitLedger())

    def _analyze(self, symbol: str, auto_trade: bool | None, ledger: _ExitLedger) -> Decision:
        """:meth:`analyze` within a run whose exits so far are in ``ledger``."""
        sym = self._validate(symbol)
        history = self._history(sym)
        price = float(history["Close"].iloc[-1])
        if not (math.isfinite(price) and price > 0):
            raise DataUnavailable(f"invalid last price {price!r} for {sym}", symbol=sym)
        trade = self.settings.auto_trade if auto_trade is None else bool(auto_trade)

        signals = self._run_signal_agents(sym, history)
        consensus = compute_consensus(
            [
                (signals[0], self.settings.weight_trend),
                (signals[1], self.settings.weight_sentiment),
            ],
            self.settings.buy_threshold,
            self.settings.sell_threshold,
        )
        marks = self._mark_prices(sym, price)

        with self._trade_lock:
            # A breached stop / target is enforced before anything is sized: otherwise an
            # add-on BUY would be assessed against (and replace the levels of) a position
            # that its own stop has already closed.
            exits = self.execution.enforce_exits(marks)
            # An exit of this symbol earlier in the run (the scan's exit pass, or a
            # sibling analysis of the batch that quoted it) counts like one made here.
            earlier = ledger.closed(sym)
            ledger.record(exits)
            exit_order = next((o for o in exits if o.symbol == sym and o.is_filled), earlier)
            portfolio = self.broker.portfolio(marks)
            risk = self.risk.assess(
                symbol=sym,
                price=price,
                action=consensus.action,
                score=consensus.score,
                confidence=consensus.confidence,
                history=history,
                portfolio=portfolio,
                exit_order=exit_order,
            )
            decision = Decision(
                symbol=sym,
                timestamp=utc_now(),
                price=round(price, 4),
                action=risk.action,
                score=round(consensus.score, 4),
                confidence=round(consensus.confidence, 4),
                signals=signals,
                risk=risk,
                exits=_unique_orders([*([earlier] if earlier is not None else []), *exits]),
            )
            if trade and risk.approved and risk.action is not Action.HOLD:
                order = self.execution.execute(decision)
                decision.order = order
                decision.executed = order is not None and order.is_filled
            if decision.executed or any(o.is_filled for o in exits):
                self.snapshot_after_fill(marks, fetch_missing=False)
        for exit_order in exits:
            if exit_order.is_filled:
                self._call_alert("on_exit", exit_order)

        decision.summary = summarize_decision(
            decision,
            proposed=consensus.action,
            buy_threshold=self.settings.buy_threshold,
            sell_threshold=self.settings.sell_threshold,
            atr_multiple=self.settings.atr_stop_multiple,
            auto_trade=trade,
        )
        logger.info("decision %s", decision.summary)
        self._notify_decision(decision, history)
        self._persist(decision)
        return decision

    # ------------------------------------------------------------------ batches
    def analyze_many(self, symbols: Sequence[str], auto_trade: bool | None = None) -> BatchAnalysis:
        """Analyse several symbols concurrently; failures are reported per symbol.

        Symbols are de-duplicated (case-insensitively) and results keep the input order.
        The analyses form one run: a symbol whose position another analysis of the batch
        closed by its stop-loss / take-profit is not bought back in the same batch.
        """
        return self._analyze_many(symbols, auto_trade, _ExitLedger())

    def _analyze_many(
        self, symbols: Sequence[str], auto_trade: bool | None, ledger: _ExitLedger
    ) -> BatchAnalysis:
        unique = list(dict.fromkeys(str(s).strip().upper() for s in symbols))
        if not unique:
            return BatchAnalysis()
        workers = min(MAX_BATCH_WORKERS, len(unique))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swarm-batch") as pool:
            futures = [
                (sym, _submit(pool, self._analyze, sym, auto_trade, ledger)) for sym in unique
            ]
            batch = BatchAnalysis()
            for sym, future in futures:
                try:
                    batch.results.append(future.result())
                except SymbolNotFound as exc:
                    batch.errors.append(
                        SymbolError(symbol=sym, error=exc.message, code="not_found")
                    )
                except DataUnavailable as exc:
                    batch.errors.append(
                        SymbolError(symbol=sym, error=exc.message, code="data_unavailable")
                    )
                except Exception as exc:
                    logger.exception("analysis of %s failed", sym)
                    batch.errors.append(
                        SymbolError(symbol=sym, error=f"{type(exc).__name__}: {exc}", code="error")
                    )
        logger.info(
            "batch analysis: %d ok, %d failed (%s)",
            len(batch.results),
            len(batch.errors),
            ", ".join(unique),
        )
        return batch

    def scan_watchlist(self) -> ScanReport:
        """Enforce stops/targets, analyse the watchlist and record an equity snapshot.

        Protective exits run *first*, with fresh quotes for every open position: a
        breached stop must not wait for (or be overwritten by) a new entry signal on the
        same symbol, and a symbol stopped out here is not bought back by the analyses of
        the same scan. The watchlist is the persisted one (``PUT /watchlist``), falling
        back to ``settings.watchlist_symbols``.

        The equity point goes through :meth:`record_snapshot`: if some open position has
        no price (its quote failed and it was not analysed), no point is stored, because
        a point marked at cost would show a phantom gain or loss in the curve, and the
        drawdown alert is not evaluated on it. ``ScanReport.snapshot`` then still shows
        the account (that position at cost) with ``snapshot_recorded = False``.
        """
        started = utc_now()
        symbols = self.broker.get_watchlist(self.settings.watchlist_symbols)
        quotes, quote_errors = self._position_quotes()

        with self._trade_lock:
            exits = self.execution.enforce_exits(quotes)
        for order in exits:
            if order.is_filled:
                self._call_alert("on_exit", order)

        batch = self._analyze_many(symbols, None, _ExitLedger(exits))
        failed = {error.symbol for error in batch.errors}
        errors = list(batch.errors) + [e for e in quote_errors if e.symbol not in failed]
        all_exits = _unique_orders([*exits, *(o for d in batch.results for o in d.exits)])

        marks = {d.symbol: d.price for d in batch.results}
        marks.update(quotes)
        recorded = self.record_snapshot(marks, fetch_missing=False)
        if recorded is not None:
            snapshot = recorded
            self._call_alert("on_drawdown", snapshot, optional=True)
        else:
            snapshot = self.broker.portfolio(marks)  # for display only: not stored

        report = ScanReport(
            decisions=batch.results,
            exits=all_exits,
            errors=errors,
            snapshot=snapshot,
            snapshot_recorded=recorded is not None,
            started_at=started,
            finished_at=utc_now(),
        )
        logger.info(
            "scan: %d symbols, %d decisions (%d executed), %d exits, %d errors; equity %.2f%s",
            len(symbols),
            len(report.decisions),
            sum(1 for d in report.decisions if d.executed),
            len(all_exits),
            len(errors),
            snapshot.equity,
            "" if recorded is not None else " (not recorded: unpriced position)",
        )
        return report

    # ------------------------------------------------------------------ data
    def _validate(self, symbol: str) -> str:
        try:
            return self.provider.validate_symbol(symbol)
        except InvalidSymbolError as exc:
            raise SymbolNotFound(exc.message, symbol=exc.symbol) from exc

    def _history(self, sym: str) -> pd.DataFrame:
        try:
            history = self.provider.history(sym, period=HISTORY_PERIOD, interval="1d")
        except (InvalidSymbolError, NoDataError) as exc:
            raise SymbolNotFound(exc.message, symbol=sym) from exc
        except MarketDataError as exc:
            raise DataUnavailable(exc.message, symbol=sym) from exc
        if history is None or history.empty:
            raise SymbolNotFound(f"no price data for {sym}", symbol=sym)
        return history

    def _mark_prices(self, sym: str, price: float) -> dict[str, float]:
        """Prices to mark the portfolio with: ``sym`` plus a fresh quote per open position.

        Accurate marks matter for the drawdown circuit breaker and the exposure caps; a
        failed quote just leaves that position at its cost basis.
        """
        marks = {sym: price}
        for position in self.broker.positions():
            if position.symbol in marks:
                continue
            try:
                marks[position.symbol] = float(self.provider.latest_price(position.symbol))
            except MarketDataError as exc:
                logger.debug("could not mark %s: %s", position.symbol, exc)
        return marks

    def _position_quotes(self) -> tuple[dict[str, float], list[SymbolError]]:
        quotes: dict[str, float] = {}
        errors: list[SymbolError] = []
        for position in self.broker.positions():
            try:
                quotes[position.symbol] = float(self.provider.latest_price(position.symbol))
            except MarketDataError as exc:
                logger.warning("no quote for open position %s: %s", position.symbol, exc)
                errors.append(
                    SymbolError(
                        symbol=position.symbol,
                        error=f"price unavailable for the stop/target check: {exc.message}",
                        code="data_unavailable",
                    )
                )
        return quotes, errors

    # ------------------------------------------------------------------ agents
    def _run_signal_agents(self, sym: str, history: pd.DataFrame) -> list[AgentSignal]:
        """``[trend, sentiment]`` votes, computed concurrently; weight-0 agents are skipped."""
        slots = (
            (self.trend, self.settings.weight_trend),
            (self.sentiment, self.settings.weight_sentiment),
        )
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="swarm-agent") as pool:
            futures = [
                _submit(pool, self._safe_vote, agent, sym, history) if weight > 0 else None
                for agent, weight in slots
            ]
            return [
                future.result()
                if future is not None
                else AgentSignal.neutral(
                    getattr(agent, "name", "agent"), "disabled (weight 0)", disabled=True
                )
                for (agent, _), future in zip(slots, futures, strict=True)
            ]

    @staticmethod
    def _safe_vote(agent: SignalAgent, sym: str, history: pd.DataFrame) -> AgentSignal:
        """Run one agent; a crashing agent abstains rather than failing the decision."""
        name = getattr(agent, "name", type(agent).__name__)
        try:
            signal = agent.analyze(sym, history)
            if not isinstance(signal, AgentSignal):
                raise TypeError(f"{name} agent returned {type(signal).__name__}, not AgentSignal")
            return signal
        except Exception as exc:
            logger.exception("%s agent failed on %s; it abstains", name, sym)
            return AgentSignal.neutral(
                name, f"{name} agent failed ({type(exc).__name__})", error=str(exc)
            )

    # ------------------------------------------------------------------ side effects
    def _call_alert(self, hook: str, *args: Any, optional: bool = False, **kwargs: Any) -> None:
        """Invoke ``alerts.<hook>(*args, **kwargs)``; alert failures never propagate."""
        if self.alerts is None:
            return
        method = getattr(self.alerts, hook, None)
        if not callable(method):
            if not optional:
                logger.warning("alert manager has no %s() hook", hook)
            return
        try:
            method(*args, **kwargs)
        except Exception:
            logger.exception("alert hook %s failed", hook)

    def _notify_decision(self, decision: Decision, history: pd.DataFrame) -> None:
        self._call_alert("on_decision", decision)
        change = daily_change_pct(history)
        if change is not None and math.isfinite(change):
            self._call_alert(
                "on_price_move", decision.symbol, change, optional=True, price=decision.price
            )

    def _persist(self, decision: Decision) -> None:
        try:
            self.broker.save_decision(decision)
        except Exception:
            # The decision (and any fill, which the broker already booked) is still
            # returned to the caller; only the audit-trail row is missing.
            logger.exception("could not persist decision %s for %s", decision.id, decision.symbol)

    # ------------------------------------------------------------------ equity curve
    def record_snapshot(
        self, marks: Mapping[str, float] | None = None, *, fetch_missing: bool = True
    ) -> PortfolioSnapshot | None:
        """Append a point to the equity curve after a fill (analysis, manual order, exit).

        Positions are marked with ``marks``. With ``fetch_missing`` a fresh quote is
        fetched for every other open position. If some position still has no price, no
        point is recorded and ``None`` is returned. A point marked at cost basis would
        show a phantom gain or loss in the curve and its metrics, and the next scan
        records a properly marked one anyway.
        """
        prices = {str(sym).upper(): float(p) for sym, p in (marks or {}).items()}
        if fetch_missing:  # network I/O: outside the trade lock
            for position in self.broker.positions():
                if position.symbol in prices:
                    continue
                try:
                    prices[position.symbol] = float(self.provider.latest_price(position.symbol))
                except MarketDataError as exc:
                    logger.debug("snapshot: no quote for %s: %s", position.symbol, exc)
        with self._trade_lock:
            missing = [p.symbol for p in self.broker.positions() if p.symbol not in prices]
            if missing:
                logger.info(
                    "equity snapshot skipped: no price for open position(s) %s",
                    ", ".join(missing),
                )
                return None
            return self.broker.record_snapshot(prices)

    def snapshot_after_fill(
        self, marks: Mapping[str, float] | None = None, *, fetch_missing: bool = True
    ) -> PortfolioSnapshot | None:
        """:meth:`record_snapshot` that logs instead of raising.

        The fill it follows is already booked, so a failed audit point (a database
        hiccup) must not turn a successful trade into an error.
        """
        try:
            return self.record_snapshot(marks, fetch_missing=fetch_missing)
        except Exception:
            logger.exception("could not record the equity snapshot after a fill")
            return None

    # ------------------------------------------------------------------ introspection
    def portfolio(self) -> PortfolioSnapshot:
        """Portfolio marked with fresh quotes for every open position (failures → cost)."""
        quotes, _ = self._position_quotes()
        return self.broker.portfolio(quotes)


__all__ = [
    "MAX_BATCH_WORKERS",
    "AlertSink",
    "Consensus",
    "DataUnavailable",
    "SwarmCoordinator",
    "SwarmError",
    "SymbolNotFound",
    "action_for_score",
    "compute_consensus",
    "summarize_decision",
]
