"""Risk gate and position sizing.

The draft confused *per-trade risk* with *gross exposure* and refused every trade once
exposure reached 10 %. Here the two are separate limits, each with its own setting.

BUY sizing (fixed-fractional, volatility-based stop)
----------------------------------------------------
1. ``stop = price − ATR(14) × atr_stop_multiple`` — the stop sits a multiple of the
   average daily range below the entry, so noisy stocks get wider stops (fallback: 5 %
   below price when ATR is unavailable; the distance is floored at 0.5 % of price so a
   near-zero ATR cannot produce a stop inside the bid/ask noise).
2. ``risk_per_share = price − stop``.
3. ``qty = floor(equity × risk_per_trade / risk_per_share)`` — if the stop is hit the
   account loses ``risk_per_trade`` of its equity (1 % by default), whatever the stock.
4. Caps (each a whole-share floor):

   * position limit: ``(max_position_pct × equity − current position value) / price``;
   * cash: :meth:`~trading_swarm.broker.CostModel.max_affordable_quantity` (includes
     slippage and the minimum commission);
   * gross exposure: ``(max_gross_exposure × equity − positions_value) / price``.

5. Scale by conviction: ``qty = floor(qty × clip(confidence, 0.5, 1))``.
6. ``take_profit = price + take_profit_r × risk_per_share`` (2R by default).

Adding to an open position
--------------------------
The broker gives a position one stop for all of its shares, and an add-on never loosens
it (:func:`~trading_swarm.broker.accounting.merge_protective_levels`). Sizing therefore
treats ``risk_per_trade`` as the budget of the *whole position*, measured forward from the
current price:

* ``stop = max(existing stop, price − ATR × atr_stop_multiple)``: the stop the position
  will actually have after the add-on. ``risk_per_share = price − stop``, floored at
  ``MIN_STOP_PCT`` of the price.
* ``open_risk = held_qty × risk_per_share``: what the shares already held lose if that
  stop is hit.
* ``room = floor((equity × risk_per_trade − open_risk) / risk_per_share)``, then the caps
  of step 4. A position that already uses its whole budget is vetoed
  (``risk_budget_ok``).
* If the price is already at or through the position's stop or target, the add-on is
  vetoed (``existing_levels_ok``): that position is due to be closed, not added to.

Conviction scales the **whole position**, not the increment. ``held_qty + room`` is the
size a full-conviction entry would have now; the position's target is
``floor((held_qty + room) × clip(confidence, 0.5, 1))`` and the add-on is
``target − held_qty``. (For a fresh entry, ``held_qty = 0``, this is exactly step 5.)
Scaling only the increment would leave the held-back part free again on the next
analysis, so an unchanged signal re-scanned every hour (or every backtest bar) would
creep up to the unscaled size in a string of ever smaller orders: the ``min 0.5×`` rule
would have no lasting effect. With the target rule, repeating an assessment on the
position it produced proposes nothing. The cash, exposure and position caps are
consistent with this: ``held + cap`` does not change when shares are bought at the
assessed price.

An add-on smaller than ``MIN_ADD_ON_FRACTION`` (10 %) of the shares held is skipped
(``min_add_on_ok``). Small drifts of the target (equity, ATR or the price moving a
little) would otherwise send one- or two-share top-ups, each paying the minimum
commission.

Re-entry after a protective exit
--------------------------------
``assess(..., exit_order=...)`` says that this symbol's position was closed by its
stop-loss or take-profit earlier in the same analysis (or watchlist scan). A BUY is then
vetoed (``reentry_ok``). Buying back at the price the exit just sold at would only pay
slippage and commission twice and reopen the trade the exit closed. This is the rule the
backtester applies when the open gaps through a position's stop and a BUY is pending for
that open: the BUY is cancelled. The symbol can be bought again from the next analysis
on. A SELL is reported as already done ("position already closed by its stop-loss")
rather than as "no position to sell".

Vetoes (``approved=False``, action → HOLD, every reason listed)
---------------------------------------------------------------
drawdown from peak ≥ ``max_drawdown_halt`` (circuit breaker), realised volatility >
``max_volatility`` (120 % annualised), already at the position limit, gross exposure
limit reached, insufficient cash, a final size below one share, a re-entry right after
a protective exit, and for add-ons a breached stop/target, an exhausted position risk
budget, a position already at its (confidence-scaled) target or a top-up below the
minimum add-on.

SELL
----
Approved when a position exists, for the full position (exits are never blocked by the
drawdown or volatility rules — reducing risk is always allowed). Without a position the
proposal becomes HOLD: "no position to sell" (long-only, no shorting).

Risk score (0-100, higher = riskier)
------------------------------------
``100 × (0.35·v + 0.25·e + 0.25·d + 0.15·c)`` where each term is a utilisation ratio in
``[0, 1]``: ``v = vol / max_volatility``, ``e = gross_exposure / max_gross_exposure``,
``d = drawdown / max_drawdown_halt`` and ``c = position value after the trade /
(max_position_pct × equity)``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Final

import pandas as pd

from trading_swarm import indicators as ind
from trading_swarm.broker.accounting import CostModel
from trading_swarm.config import Settings
from trading_swarm.models import Action, ExitReason, Order, PortfolioSnapshot, RiskAssessment

logger = logging.getLogger(__name__)

#: Stop distance used when ATR cannot be computed (fraction of price).
FALLBACK_STOP_PCT: Final = 0.05
#: Minimum stop distance (fraction of price).
MIN_STOP_PCT: Final = 0.005
#: Confidence scaling never shrinks a position below this fraction of its risk size.
MIN_CONFIDENCE_SCALE: Final = 0.5
#: An add-on must buy at least this fraction of the shares already held (≥ 1 share);
#: smaller top-ups are skipped as commission churn.
MIN_ADD_ON_FRACTION: Final = 0.10
ATR_PERIOD: Final = 14
VOL_WINDOW: Final = 20
#: Weights of the risk-score terms (volatility, exposure, drawdown, concentration).
RISK_SCORE_WEIGHTS: Final = {
    "volatility": 0.35,
    "exposure": 0.25,
    "drawdown": 0.25,
    "concentration": 0.15,
}


def _last(series: pd.Series) -> float | None:
    value = ind.last(series)
    return value if value is not None and value > 0 else None


def _price4(value: float) -> float:
    return round(value, 4)


def _money(value: float) -> float:
    return round(value, 2)


def exit_label(order: Order) -> str:
    """``"stop-loss"`` / ``"take-profit"`` for a protective exit order (else its reason)."""
    try:
        reason = ExitReason(order.reason)
    except ValueError:
        return order.reason or "protective exit"
    return "stop-loss" if reason is ExitReason.STOP_LOSS else "take-profit"


def describe_exit_fill(order: Order) -> str:
    """``"sold 10 sh @ 57.00"`` (the fill of a protective exit)."""
    price = f" @ {order.fill_price:.2f}" if order.fill_price is not None else ""
    return f"sold {order.quantity} sh{price}"


@dataclass
class _Context:
    """Market and account facts shared by the BUY/SELL/HOLD branches."""

    price: float
    equity: float
    cash: float
    positions_value: float
    drawdown: float
    gross_exposure: float
    atr: float | None
    vol: float | None
    held_quantity: int
    held_value: float
    #: Protective levels of the position already held in the symbol (if any).
    held_stop: float | None = None
    held_take_profit: float | None = None
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)


class RiskManagerAgent:
    """Turns a proposed action into an approved/vetoed, sized :class:`RiskAssessment`."""

    name = "risk"

    def __init__(self, settings: Settings, cost_model: CostModel | None = None) -> None:
        self.settings = settings
        self.cost_model = cost_model or CostModel.from_settings(settings)

    # ------------------------------------------------------------------ public API
    def assess(
        self,
        *,
        symbol: str,
        price: float,
        action: Action,
        score: float,
        confidence: float,
        history: pd.DataFrame,
        portfolio: PortfolioSnapshot,
        exit_order: Order | None = None,
    ) -> RiskAssessment:
        """Size and gate ``action`` for ``symbol`` at ``price`` given the account state.

        ``history`` must end at the decision bar (it supplies ATR and realised
        volatility); ``portfolio`` should be marked to current prices so that the
        drawdown and exposure rules see the real account. ``exit_order`` is the filled
        stop-loss / take-profit exit that closed ``symbol``'s position earlier in the
        same analysis, if any (see "Re-entry after a protective exit" in the module
        docstring); ``portfolio`` must already reflect it.
        """
        ctx = self._context(symbol, price, history, portfolio)
        if exit_order is not None and not exit_order.is_filled:
            exit_order = None  # only a fill closed anything
        if not (math.isfinite(price) and price > 0):
            ctx.checks["price_valid"] = False
            return self._veto(ctx, Action.HOLD, [f"invalid price {price!r}"])
        if action is Action.BUY:
            assessment = self._assess_buy(symbol, ctx, confidence, exit_order)
        elif action is Action.SELL:
            assessment = self._assess_sell(symbol, ctx, exit_order)
        else:
            assessment = self._finish(
                ctx,
                approved=False,
                action=Action.HOLD,
                reasons=[f"consensus {score:+.2f} is inside the HOLD band: no trade proposed"],
            )
        logger.debug(
            "risk %s %s → %s approved=%s qty=%d score=%.1f: %s",
            symbol,
            action.value,
            assessment.action.value,
            assessment.approved,
            assessment.quantity,
            assessment.risk_score,
            "; ".join(assessment.reasons),
        )
        return assessment

    # ------------------------------------------------------------------ context
    def _context(
        self, symbol: str, price: float, history: pd.DataFrame, portfolio: PortfolioSnapshot
    ) -> _Context:
        atr = vol = None
        if history is not None and len(history) > 1:
            atr = _last(ind.atr(history, ATR_PERIOD))
            vol = ind.last(ind.realized_volatility(history["Close"], VOL_WINDOW))
        held = portfolio.position(symbol)
        held_quantity = held.quantity if held is not None else 0
        mark = price if math.isfinite(price) and price > 0 else (held.avg_price if held else 0.0)
        return _Context(
            price=float(price),
            equity=float(portfolio.equity),
            cash=float(portfolio.cash),
            positions_value=float(portfolio.positions_value),
            drawdown=portfolio.drawdown,
            gross_exposure=portfolio.gross_exposure,
            atr=atr,
            vol=vol,
            held_quantity=held_quantity,
            held_value=held_quantity * mark,
            held_stop=held.stop_price if held is not None and held_quantity > 0 else None,
            held_take_profit=(
                held.take_profit_price if held is not None and held_quantity > 0 else None
            ),
        )

    # ------------------------------------------------------------------ BUY
    def _assess_buy(
        self, symbol: str, ctx: _Context, confidence: float, exit_order: Order | None
    ) -> RiskAssessment:
        s = self.settings
        vetoes: list[str] = []
        notes: list[str] = []

        if ctx.equity <= 0:
            ctx.checks["equity_positive"] = False
            return self._veto(ctx, Action.HOLD, ["account equity is not positive"])

        if exit_order is not None:
            # Same-run re-entry: the protective exit sold at (about) this very price.
            ctx.checks["reentry_ok"] = False
            vetoes.append(
                f"the {exit_label(exit_order)} closed the {symbol} position just before this "
                f"assessment ({describe_exit_fill(exit_order)}): no re-entry at the price it "
                "sold at (it would only pay costs twice); the next analysis can buy again"
            )

        # Circuit breaker: no new longs while the account is in a deep drawdown.
        ctx.checks["drawdown_ok"] = ctx.drawdown < s.max_drawdown_halt
        if not ctx.checks["drawdown_ok"]:
            vetoes.append(
                f"drawdown {ctx.drawdown:.1%} ≥ {s.max_drawdown_halt:.0%} halt: no new buys"
            )

        ctx.checks["volatility_ok"] = ctx.vol is None or ctx.vol <= s.max_volatility
        if ctx.vol is None:
            notes.append("realised volatility unavailable (short history)")
        elif not ctx.checks["volatility_ok"]:
            vetoes.append(
                f"realised volatility {ctx.vol:.0%} > {s.max_volatility:.0%} limit (too volatile)"
            )

        # Protective stop and per-share risk.
        ctx.checks["stop_from_atr"] = ctx.atr is not None
        if ctx.atr is not None:
            distance = ctx.atr * s.atr_stop_multiple
            stop_note = f"stop {s.atr_stop_multiple:.1f}×ATR (ATR {ctx.atr:.2f})"
        else:
            distance = ctx.price * FALLBACK_STOP_PCT
            stop_note = f"ATR unavailable: {FALLBACK_STOP_PCT:.0%} fallback stop"
        if distance < ctx.price * MIN_STOP_PCT:
            distance = ctx.price * MIN_STOP_PCT
            stop_note += f", widened to the {MIN_STOP_PCT:.1%} minimum"
        if distance >= ctx.price:  # an ATR larger than the price: cap the stop at −50 %
            distance = ctx.price * 0.5
            stop_note += ", capped at 50% below price"
        stop = ctx.price - distance
        notes.append(stop_note)
        adding = ctx.held_quantity > 0
        if adding:
            stop, distance = self._add_on_stop(ctx, stop, distance, vetoes, notes)
        take_profit = ctx.price + s.take_profit_r * distance

        # Fixed-fractional sizing: lose risk_per_trade of equity if the stop is hit.
        risk_budget = ctx.equity * s.risk_per_trade
        if adding:
            # The budget covers the whole position: the shares already held share the
            # (never-loosened) stop, so their open risk is spent before any new share.
            open_risk = ctx.held_quantity * distance
            qty_risk = max(0, math.floor((risk_budget - open_risk) / distance))
            ctx.checks["risk_budget_ok"] = qty_risk >= 1
            notes.append(
                f"risk size {qty_risk} sh (${risk_budget:,.0f} position budget − "
                f"${open_risk:,.0f} open risk on the {ctx.held_quantity} sh held, / "
                f"${distance:.2f} per share)"
            )
            if not ctx.checks["risk_budget_ok"]:
                vetoes.append(
                    f"the {ctx.held_quantity} sh already held use the position's risk budget "
                    f"(${open_risk:,.0f} of ${risk_budget:,.0f} at risk to the stop {stop:.2f})"
                )
        else:
            qty_risk = math.floor(risk_budget / distance)
            notes.append(
                f"risk size {qty_risk} sh (${risk_budget:,.0f} at risk / ${distance:.2f} per share)"
            )

        position_room = s.max_position_pct * ctx.equity - ctx.held_value
        qty_position = max(0, math.floor(position_room / ctx.price))
        ctx.checks["position_limit_ok"] = qty_position >= 1
        if not ctx.checks["position_limit_ok"]:
            vetoes.append(
                f"already at the max position ({s.max_position_pct:.0%} of equity) in {symbol}"
            )

        qty_cash = self.cost_model.max_affordable_quantity(ctx.cash, ctx.price)
        ctx.checks["cash_ok"] = qty_cash >= 1
        if not ctx.checks["cash_ok"]:
            vetoes.append(f"insufficient cash (${ctx.cash:,.2f}) for one share incl. costs")

        exposure_room = s.max_gross_exposure * ctx.equity - ctx.positions_value
        qty_exposure = max(0, math.floor(exposure_room / ctx.price))
        ctx.checks["exposure_ok"] = qty_exposure >= 1
        if not ctx.checks["exposure_ok"]:
            vetoes.append(
                f"gross exposure {ctx.gross_exposure:.0%} at the {s.max_gross_exposure:.0%} limit"
            )

        quantity = qty_risk
        for cap, label in (
            (qty_position, f"max position {s.max_position_pct:.0%} of equity"),
            (qty_cash, "available cash"),
            (qty_exposure, f"gross exposure limit {s.max_gross_exposure:.0%}"),
        ):
            if cap < quantity:
                if cap >= 1:  # a zero cap is already reported as a veto above
                    notes.append(f"capped {quantity}→{cap} sh by {label}")
                quantity = cap

        # Conviction scales the whole position (held + room), never just the increment:
        # otherwise the held-back part is free again at the next analysis and repeated
        # identical signals creep up to the unscaled size (see the module docstring).
        held = ctx.held_quantity
        full_size = held + quantity
        target = full_size
        scale = min(1.0, max(MIN_CONFIDENCE_SCALE, confidence))
        if scale < 1.0:
            target = math.floor(full_size * scale)
            if adding:
                notes.append(
                    f"position target scaled {full_size}→{target} sh by confidence "
                    f"({scale:.2f}×), {held} sh already held"
                )
            else:
                notes.append(f"scaled {quantity}→{target} sh by confidence ({scale:.2f}×)")
            quantity = max(0, target - held)
        ctx.checks["min_quantity_ok"] = quantity >= 1
        if not ctx.checks["min_quantity_ok"] and not vetoes:
            if adding:
                vetoes.append(
                    f"the {held} sh held already reach the position's "
                    f"{'confidence-scaled ' if scale < 1.0 else ''}target of {target} sh"
                )
            else:
                vetoes.append("position size rounds to zero shares")
        elif adding and quantity >= 1:
            minimum = max(1, math.ceil(held * MIN_ADD_ON_FRACTION))
            ctx.checks["min_add_on_ok"] = quantity >= minimum
            if not ctx.checks["min_add_on_ok"]:
                vetoes.append(
                    f"add-on of {quantity} sh is below the minimum top-up of {minimum} sh "
                    f"({MIN_ADD_ON_FRACTION:.0%} of the {held} sh held): not worth the commission"
                )

        if vetoes:
            return self._veto(ctx, Action.HOLD, vetoes + notes)
        return self._finish(
            ctx,
            approved=True,
            action=Action.BUY,
            quantity=quantity,
            stop=stop,
            take_profit=take_profit,
            risk_amount=quantity * distance,
            reasons=[
                f"approved BUY {quantity} sh: stop {stop:.2f}, target {take_profit:.2f} "
                f"({s.take_profit_r:g}R)",
                *notes,
            ],
        )

    @staticmethod
    def _add_on_stop(
        ctx: _Context,
        stop: float,
        distance: float,
        vetoes: list[str],
        notes: list[str],
    ) -> tuple[float, float]:
        """The stop an add-on leaves on the whole position, and its sizing distance.

        The broker keeps ``max(existing stop, new stop)``, so a tighter existing stop is
        the one every share (old and new) will have. A price already at or through the
        existing stop or target is a veto (``existing_levels_ok``): that position is due
        to be closed by its protective level, not enlarged.
        """
        existing_stop, existing_target = ctx.held_stop, ctx.held_take_profit
        stop_breached = existing_stop is not None and ctx.price <= existing_stop
        target_breached = existing_target is not None and ctx.price >= existing_target
        ctx.checks["existing_levels_ok"] = not (stop_breached or target_breached)
        if stop_breached:
            vetoes.append(
                f"price {ctx.price:.2f} is at or below the open position's stop "
                f"{existing_stop:.2f}: the position is due to be closed, not added to"
            )
        elif target_breached:
            vetoes.append(
                f"price {ctx.price:.2f} is at or above the open position's take-profit "
                f"{existing_target:.2f}: the position is due to be closed, not added to"
            )
        elif existing_stop is not None and existing_stop > stop:
            notes.append(
                f"keeps the position's tighter stop {existing_stop:.2f} (an add-on never "
                f"loosens it; the new stop would have been {stop:.2f})"
            )
            ctx.checks["kept_existing_stop"] = True
            stop = existing_stop
            distance = max(ctx.price - existing_stop, ctx.price * MIN_STOP_PCT)
        return stop, distance

    # ------------------------------------------------------------------ SELL
    def _assess_sell(
        self, symbol: str, ctx: _Context, exit_order: Order | None = None
    ) -> RiskAssessment:
        ctx.checks["position_exists"] = ctx.held_quantity > 0
        if not ctx.checks["position_exists"]:
            if exit_order is not None:
                # The SELL the consensus asks for has already happened, by the exit.
                return self._veto(
                    ctx,
                    Action.HOLD,
                    [
                        f"position already closed by its {exit_label(exit_order)} just before "
                        f"this assessment ({describe_exit_fill(exit_order)}): nothing left to sell"
                    ],
                )
            return self._veto(
                ctx, Action.HOLD, [f"no position to sell in {symbol} (short selling not supported)"]
            )
        return self._finish(
            ctx,
            approved=True,
            action=Action.SELL,
            quantity=ctx.held_quantity,
            reasons=[f"approved SELL of the full position ({ctx.held_quantity} sh)"],
        )

    # ------------------------------------------------------------------ helpers
    def _veto(self, ctx: _Context, action: Action, reasons: list[str]) -> RiskAssessment:
        return self._finish(ctx, approved=False, action=action, reasons=reasons)

    def _finish(
        self,
        ctx: _Context,
        *,
        approved: bool,
        action: Action,
        reasons: list[str],
        quantity: int = 0,
        stop: float | None = None,
        take_profit: float | None = None,
        risk_amount: float = 0.0,
    ) -> RiskAssessment:
        position_value = quantity * ctx.price if ctx.price > 0 else 0.0
        return RiskAssessment(
            approved=approved,
            action=action,
            quantity=quantity,
            entry_price=_price4(ctx.price) if ctx.price > 0 else 0.0,
            stop_price=None if stop is None else _price4(stop),
            take_profit_price=None if take_profit is None else _price4(take_profit),
            position_value=_money(position_value),
            risk_amount=_money(risk_amount),
            risk_score=round(self._risk_score(ctx, action, position_value), 2),
            reasons=reasons,
            checks=dict(ctx.checks),
        )

    def _risk_score(self, ctx: _Context, action: Action, position_value: float) -> float:
        """0-100 utilisation-weighted risk score (see module docstring)."""
        s = self.settings

        def ratio(value: float | None, limit: float) -> float:
            if value is None or not math.isfinite(value) or limit <= 0:
                return 0.0
            return max(0.0, min(1.0, value / limit))

        after = ctx.held_value + (position_value if action is Action.BUY else 0.0)
        if action is Action.SELL:
            after = 0.0
        concentration = after / ctx.equity if ctx.equity > 0 else 0.0
        terms = {
            "volatility": ratio(ctx.vol, s.max_volatility),
            "exposure": ratio(ctx.gross_exposure, s.max_gross_exposure),
            "drawdown": ratio(ctx.drawdown, s.max_drawdown_halt),
            "concentration": ratio(concentration, s.max_position_pct),
        }
        return 100.0 * sum(RISK_SCORE_WEIGHTS[k] * v for k, v in terms.items())


__all__ = [
    "FALLBACK_STOP_PCT",
    "MIN_ADD_ON_FRACTION",
    "MIN_CONFIDENCE_SCALE",
    "MIN_STOP_PCT",
    "RISK_SCORE_WEIGHTS",
    "RiskManagerAgent",
    "describe_exit_fill",
    "exit_label",
]
