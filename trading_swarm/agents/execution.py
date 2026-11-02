"""Execution agent: turns approved decisions into broker orders and enforces exits.

Entry/exit orders are market orders at the market-data provider's latest price — never a
client-supplied price (one of the draft's bugs). The broker applies slippage and
commission (:mod:`trading_swarm.broker.accounting`).

Protective levels
-----------------
The risk agent computes the stop-loss and take-profit from the *decision* price. If the
latest quote has moved so far that a level is on the wrong side of the execution price
(stop ≥ price or target ≤ price), both levels are re-anchored to the execution price
keeping the same distances, so the position keeps the risk the risk agent sized it for.

For a BUY into an open position (an add-on) the broker merges the levels: the position
keeps the higher of its current stop and the new one (an add-on never loosens the stop),
and the new target replaces the old one. A BUY while the position's own stop or target
is already breached at the execution price is rejected by the broker. That position must
be closed by :meth:`ExecutionAgent.enforce_exits` first, which is why the coordinator
enforces exits before it sizes a trade
(:func:`~trading_swarm.broker.accounting.merge_protective_levels`).

Exits
-----
:meth:`ExecutionAgent.enforce_exits` sells the whole position when the last price is at
or through the stop (``price ≤ stop``) or the target (``price ≥ take_profit``) — the
draft computed stops but never enforced them. The order ``reason`` is exactly the
:class:`~trading_swarm.models.ExitReason` value (``"stop_loss"`` / ``"take_profit"``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import datetime

from trading_swarm.broker.accounting import OrderValidationError, evaluate_exit
from trading_swarm.broker.base import Broker
from trading_swarm.config import Settings
from trading_swarm.data.base import MarketDataError, MarketDataProvider
from trading_swarm.models import Action, Decision, ExitReason, Order, OrderStatus, Side

logger = logging.getLogger(__name__)


def _valid_price(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


class ExecutionAgent:
    """Turns approved decisions into broker orders and enforces protective exits.

    Args:
        broker: Any :class:`~trading_swarm.broker.base.Broker` (paper or in-memory).
        settings: Application settings.
        provider: Used to re-quote the latest price before executing; without it the
            decision price is used.
    """

    name = "execution"

    def __init__(
        self,
        broker: Broker,
        settings: Settings,
        provider: MarketDataProvider | None = None,
    ) -> None:
        self.broker = broker
        self.settings = settings
        self.provider = provider

    # ------------------------------------------------------------------ entries
    def execute(self, decision: Decision, *, timestamp: datetime | None = None) -> Order | None:
        """Submit the order for an approved BUY/SELL ``decision`` (``None`` otherwise).

        Returns the broker's order — FILLED, or REJECTED with ``reject_reason`` (e.g. the
        cash was spent by another order in the meantime) — or ``None`` when there is
        nothing to execute or the request was malformed (logged).
        """
        risk = decision.risk
        if not risk.approved or risk.action is Action.HOLD or risk.quantity < 1:
            logger.debug("execution %s: nothing to execute (%s)", decision.symbol, risk.action)
            return None
        side = Side.BUY if risk.action is Action.BUY else Side.SELL
        price = self._execution_price(decision)
        if price is None:
            logger.error("execution %s: no valid price; order not sent", decision.symbol)
            return None

        quantity = risk.quantity
        stop = take_profit = None
        if side is Side.BUY:
            stop, take_profit = self._levels(decision, price)
        else:
            held = self.broker.position(decision.symbol)
            if held is not None and held.quantity < quantity:
                logger.info(
                    "execution %s: position shrank to %d sh since assessment; selling all",
                    decision.symbol,
                    held.quantity,
                )
                quantity = held.quantity

        reason = f"swarm {side.value} (score {decision.score:+.2f}, conf {decision.confidence:.2f})"
        try:
            order = self.broker.submit_order(
                decision.symbol,
                side,
                quantity,
                price,
                reason=reason,
                stop=stop,
                take_profit=take_profit,
                decision_id=decision.id,
                timestamp=timestamp,
            )
        except OrderValidationError as exc:
            logger.error("execution %s: order rejected as malformed: %s", decision.symbol, exc)
            return None
        if order.status is OrderStatus.REJECTED:
            logger.warning(
                "execution %s: %s %d rejected: %s",
                decision.symbol,
                side.value,
                quantity,
                order.reject_reason,
            )
        return order

    def _execution_price(self, decision: Decision) -> float | None:
        """Latest provider quote (falls back to the decision price if re-quoting fails)."""
        if self.provider is not None:
            try:
                quote = float(self.provider.latest_price(decision.symbol))
            except MarketDataError as exc:
                logger.warning(
                    "execution %s: re-quote failed (%s); using decision price %.4f",
                    decision.symbol,
                    exc,
                    decision.price,
                )
            else:
                if _valid_price(quote):
                    return quote
                logger.warning("execution %s: invalid quote %r", decision.symbol, quote)
        return decision.price if _valid_price(decision.price) else None

    @staticmethod
    def _levels(decision: Decision, price: float) -> tuple[float | None, float | None]:
        """Stop/target for a BUY at ``price``, re-anchored if the quote moved past them."""
        stop, target = decision.risk.stop_price, decision.risk.take_profit_price
        stop_ok = stop is None or stop < price
        target_ok = target is None or target > price
        if stop_ok and target_ok:
            return stop, target
        shift = price - decision.price
        new_stop = None if stop is None else stop + shift
        new_target = None if target is None else target + shift
        if new_stop is not None and not 0 < new_stop < price:
            new_stop = None
        if new_target is not None and not new_target > price:
            new_target = None
        logger.warning(
            "execution %s: quote %.4f moved past the protective levels (stop %s, target %s); "
            "re-anchored to stop %s, target %s",
            decision.symbol,
            price,
            stop,
            target,
            None if new_stop is None else round(new_stop, 4),
            None if new_target is None else round(new_target, 4),
        )
        return new_stop, new_target

    # ------------------------------------------------------------------ exits
    def enforce_exits(
        self, prices: Mapping[str, float], *, timestamp: datetime | None = None
    ) -> list[Order]:
        """Sell positions whose stop-loss / take-profit is breached by ``prices``.

        Positions without a (valid) price are skipped. A failure on one position is
        logged and does not prevent the others from being checked.
        """
        quotes = {str(symbol).strip().upper(): price for symbol, price in prices.items()}
        orders: list[Order] = []
        for position in self.broker.positions():
            price = quotes.get(position.symbol)
            if not _valid_price(price):
                continue
            assert price is not None  # narrowed by _valid_price
            reason = evaluate_exit(position.stop_price, position.take_profit_price, price)
            if reason is None:
                continue
            level = (
                position.stop_price
                if reason is ExitReason.STOP_LOSS
                else position.take_profit_price
            )
            logger.info(
                "exit %s: %s hit (price %.4f vs level %s); selling %d sh",
                position.symbol,
                reason.value,
                price,
                level,
                position.quantity,
            )
            try:
                order = self.broker.submit_order(
                    position.symbol,
                    Side.SELL,
                    position.quantity,
                    price,
                    reason=reason.value,
                    timestamp=timestamp,
                )
            except Exception:
                # Keep enforcing the remaining positions; this one is retried next scan.
                logger.exception("exit %s: could not submit the %s order", position.symbol, reason)
                continue
            if order.status is OrderStatus.REJECTED:
                logger.error("exit %s rejected: %s", position.symbol, order.reject_reason)
            orders.append(order)
        return orders


__all__ = ["ExecutionAgent"]
