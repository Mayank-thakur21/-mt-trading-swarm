"""Pure accounting core shared by :class:`PaperBroker` (SQLite) and :class:`InMemoryBroker`
(backtests), so live paper trading and backtests book fills identically.

Cost model
----------
* Slippage moves the fill against us: ``BUY fill = price × (1 + bps/10⁴)``,
  ``SELL fill = price × (1 − bps/10⁴)``.
* Commission per order: ``max(min_commission, fill_notional × commission_rate)``.

Position accounting (long-only)
-------------------------------
* ``avg_price`` is the **commission-inclusive** cost basis per share:
  ``avg' = (qty·avg + fill_qty·fill_price + buy_commission) / (qty + fill_qty)``.
* A SELL realises ``fill_qty·fill_price − sell_commission − fill_qty·avg`` — i.e. P&L net
  of the commissions of *both* legs.
* Consequently ``equity − initial_capital == realized_pnl + unrealized_pnl`` exactly,
  where ``unrealized = Σ qty·(market_price − avg)``.

Protective levels when adding to a position
-------------------------------------------
A position has one stop-loss and one take-profit for all of its shares. A BUY into an
existing position (an *add-on*) merges its levels with the position's
(:func:`merge_protective_levels`):

* **The stop never loosens.** The position keeps ``max(current stop, new stop)``. The
  shares already held were sized against their stop; moving it down would raise their risk
  beyond what was sized. A stop is lowered only explicitly, via ``update_stops``.
* **The take-profit follows the latest plan.** The add-on's target replaces the current one
  when it is given (else the current one is kept). Moving a target adds no risk.
* **No adding to a breached position.** If the price is at or through the position's stop
  (``price ≤ stop``) or target (``price ≥ take_profit``), the BUY is rejected: the position
  is due to be closed by that level, and filling the add-on would silently replace a
  breached level that was never enforced.

A SELL only reduces or closes a position, so it cannot carry protective levels
(:class:`OrderValidationError`); the remaining shares keep the position's levels.

Protective exits
----------------
:func:`evaluate_exit` checks a last price against stop/take-profit levels;
:func:`evaluate_exit_bar` checks a whole OHLC bar (backtests): a gap through a level at
the open fills at the open; if both levels are inside the bar's range the stop is assumed
to have been hit first (the conservative choice, since intrabar order is unknown).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Final

from trading_swarm.models import (
    ExitReason,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
    Trade,
    ensure_utc,
    new_id,
    utc_now,
)

if TYPE_CHECKING:
    from trading_swarm.config import Settings

_EPS: Final = 1e-9


class OrderValidationError(ValueError):
    """The order request itself is malformed (bad quantity, price or stop levels).

    Business rejections (insufficient cash, nothing to sell) are *not* exceptions: they
    produce an order with ``status=REJECTED`` and a ``reject_reason``.
    """


def money(value: float) -> float:
    """Round a currency amount to cents (output only — storage keeps full precision)."""
    return round(float(value), 2)


def price4(value: float) -> float:
    """Round a per-share price to 4 decimals for output."""
    return round(float(value), 4)


def _opt(value: float | None, rounder: Callable[[float], float] = money) -> float | None:
    return None if value is None else rounder(value)


def _valid_price(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def is_valid_price(value: float | None) -> bool:
    """``True`` for a positive, finite price."""
    return _valid_price(value)


# =============================================================================== costs


@dataclass(frozen=True)
class CostModel:
    """Commission and slippage parameters (see module docstring for the formulas)."""

    commission_rate: float = 0.0005
    min_commission: float = 1.0
    slippage_bps: float = 5.0

    @classmethod
    def from_settings(cls, settings: Settings) -> CostModel:
        return cls(
            commission_rate=float(settings.commission_rate),
            min_commission=float(settings.min_commission),
            slippage_bps=float(settings.slippage_bps),
        )

    def fill_price(self, side: Side, price: float) -> float:
        """Price after adverse slippage for ``side``."""
        adjustment = self.slippage_bps / 10_000.0
        return price * (1.0 + adjustment) if side is Side.BUY else price * (1.0 - adjustment)

    def commission(self, notional: float) -> float:
        """Commission for an order of ``notional`` value (0 for an empty order)."""
        if notional <= 0:
            return 0.0
        return max(self.min_commission, notional * self.commission_rate)

    def quote(self, side: Side, quantity: int, price: float) -> Fill:
        """The fill an order of ``quantity`` at reference ``price`` would receive."""
        fill = self.fill_price(side, price)
        return Fill(
            side=side,
            quantity=quantity,
            reference_price=price,
            fill_price=fill,
            commission=self.commission(fill * quantity),
        )

    def buy_cash_required(self, quantity: int, price: float) -> float:
        """Cash consumed by buying ``quantity`` at reference ``price`` (incl. costs)."""
        return self.quote(Side.BUY, quantity, price).cash_required

    def max_affordable_quantity(self, cash: float, price: float) -> int:
        """Largest whole-share BUY that ``cash`` can pay for, including all costs.

        Solves ``q·f + max(min_c, q·f·r) ≤ cash`` with ``f`` the slipped fill price:
        ``q ≤ cash / (f·(1 + r))`` and ``q ≤ (cash − min_c) / f``.
        """
        if cash <= 0 or not _valid_price(price):
            return 0
        fill = self.fill_price(Side.BUY, price)
        q_rate = math.floor(cash / (fill * (1.0 + self.commission_rate)))
        q_min = math.floor((cash - self.min_commission) / fill)
        quantity = max(0, min(q_rate, q_min))
        while quantity > 0 and self.buy_cash_required(quantity, price) > cash + _EPS:
            quantity -= 1  # floating-point guard; runs at most once or twice
        return quantity


@dataclass(frozen=True)
class Fill:
    """An executed quantity with its costs (full precision)."""

    side: Side
    quantity: int
    reference_price: float
    fill_price: float
    commission: float

    @property
    def notional(self) -> float:
        return self.fill_price * self.quantity

    @property
    def slippage_cost(self) -> float:
        return abs(self.fill_price - self.reference_price) * self.quantity

    @property
    def cash_required(self) -> float:
        """Cash leaving the account for a BUY (notional + commission)."""
        return self.notional + self.commission

    @property
    def cash_delta(self) -> float:
        """Signed change in cash: negative for buys, positive for sells."""
        if self.side is Side.BUY:
            return -(self.notional + self.commission)
        return self.notional - self.commission


# =============================================================================== records


@dataclass
class PositionState:
    """Full-precision open position (storage representation)."""

    symbol: str
    quantity: int
    avg_price: float
    stop_price: float | None = None
    take_profit_price: float | None = None
    opened_at: datetime = field(default_factory=utc_now)

    def mark(self, market_price: float | None) -> Position:
        """Output model marked at ``market_price`` (falls back to ``avg_price``)."""
        price = market_price if _valid_price(market_price) else None
        mark = price if price is not None else self.avg_price
        unrealized = self.quantity * (mark - self.avg_price)
        pct = (mark / self.avg_price - 1.0) * 100.0 if self.avg_price > 0 else 0.0
        return Position(
            symbol=self.symbol,
            quantity=self.quantity,
            avg_price=price4(self.avg_price),
            stop_price=_opt(self.stop_price, price4),
            take_profit_price=_opt(self.take_profit_price, price4),
            opened_at=self.opened_at,
            market_price=_opt(price, price4),
            market_value=money(self.quantity * mark),
            unrealized_pnl=money(unrealized),
            unrealized_pnl_pct=round(pct, 4),
        )


@dataclass
class OrderRecord:
    """Full-precision order (storage representation of :class:`~trading_swarm.models.Order`)."""

    symbol: str
    side: Side
    quantity: int
    status: OrderStatus
    requested_at: datetime
    reference_price: float | None = None
    fill_price: float | None = None
    filled_at: datetime | None = None
    commission: float = 0.0
    slippage: float = 0.0
    stop_price: float | None = None
    take_profit_price: float | None = None
    realized_pnl: float | None = None
    reason: str = ""
    decision_id: str | None = None
    reject_reason: str | None = None
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    id: str = field(default_factory=new_id)

    def to_model(self) -> Order:
        return Order(
            id=self.id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            type=self.type,
            limit_price=_opt(self.limit_price, price4),
            reference_price=_opt(self.reference_price, price4),
            status=self.status,
            requested_at=self.requested_at,
            filled_at=self.filled_at,
            fill_price=_opt(self.fill_price, price4),
            commission=money(self.commission),
            slippage=money(self.slippage),
            stop_price=_opt(self.stop_price, price4),
            take_profit_price=_opt(self.take_profit_price, price4),
            realized_pnl=_opt(self.realized_pnl),
            reason=self.reason,
            decision_id=self.decision_id,
            reject_reason=self.reject_reason,
        )


@dataclass
class TradeRecord:
    """Full-precision fill (storage representation of :class:`~trading_swarm.models.Trade`)."""

    order_id: str
    symbol: str
    side: Side
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    realized_pnl: float | None = None
    reason: str = ""
    id: str = field(default_factory=new_id)

    def to_model(self) -> Trade:
        return Trade(
            id=self.id,
            order_id=self.order_id,
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            price=price4(self.price),
            commission=money(self.commission),
            realized_pnl=_opt(self.realized_pnl),
            timestamp=self.timestamp,
            reason=self.reason,
        )


@dataclass(frozen=True)
class OrderOutcome:
    """Everything a broker must persist after processing one order."""

    order: OrderRecord
    trade: TradeRecord | None
    position: PositionState | None  # resulting position; None = flat
    cash: float  # resulting cash
    realized_pnl: float  # P&L realised by this order (0 for buys / rejections)

    @property
    def filled(self) -> bool:
        return self.order.status is OrderStatus.FILLED


# =============================================================================== processing


def validate_order_request(
    side: Side | str,
    quantity: int,
    price: float,
    stop: float | None = None,
    take_profit: float | None = None,
) -> Side:
    """Check an order request's shape; returns the parsed :class:`Side`.

    Raises:
        OrderValidationError: non-positive/non-integer quantity, invalid price,
            protective levels on the wrong side of the price for a long position, or
            protective levels on a SELL (which only reduces or closes the position).
    """
    try:
        parsed = Side(side.upper() if isinstance(side, str) else side)
    except ValueError as exc:
        raise OrderValidationError(f"side must be BUY or SELL, got {side!r}") from exc
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise OrderValidationError(f"quantity must be a positive integer, got {quantity!r}")
    if not _valid_price(price):
        raise OrderValidationError(f"price must be a positive finite number, got {price!r}")
    if parsed is Side.SELL and (stop is not None or take_profit is not None):
        # Accepting (and echoing) them would tell the caller a stop protects the remaining
        # shares when none was applied: a SELL keeps the position's existing levels.
        raise OrderValidationError(
            "stop-loss / take-profit can only be set on BUY orders: a SELL reduces or "
            "closes the position, and the remaining shares keep their current levels"
        )
    if stop is not None:
        if not _valid_price(stop):
            raise OrderValidationError(f"stop price must be positive and finite, got {stop!r}")
        if parsed is Side.BUY and stop >= price:
            raise OrderValidationError(
                f"stop-loss {stop:.4f} must be below the entry price {price:.4f} (long only)"
            )
    if take_profit is not None:
        if not _valid_price(take_profit):
            raise OrderValidationError(
                f"take-profit price must be positive and finite, got {take_profit!r}"
            )
        if parsed is Side.BUY and take_profit <= price:
            raise OrderValidationError(
                f"take-profit {take_profit:.4f} must be above the entry price {price:.4f}"
            )
    return parsed


def merge_protective_levels(
    position: PositionState | None,
    stop: float | None,
    take_profit: float | None,
) -> tuple[float | None, float | None]:
    """Stop and take-profit of a long position after a BUY with ``stop`` / ``take_profit``.

    For a new position the order's levels are used as given. For an add-on (see the
    module docstring):

    * stop = ``max(current, new)``: protection only tightens. Either one alone is kept.
    * take-profit = the new target when given, else the current one.
    """
    if position is None:
        return stop, take_profit
    current_stop, current_target = position.stop_price, position.take_profit_price
    if stop is None:
        merged_stop = current_stop
    elif current_stop is None:
        merged_stop = stop
    else:
        merged_stop = max(current_stop, stop)
    merged_target = take_profit if take_profit is not None else current_target
    return merged_stop, merged_target


def _breached_level_reason(position: PositionState, price: float) -> str | None:
    """Why a BUY at ``price`` may not add to ``position`` (a breached level), else ``None``."""
    breach = evaluate_exit(position.stop_price, position.take_profit_price, price)
    if breach is ExitReason.STOP_LOSS:
        return (
            f"price {price:.4f} is at or below the {position.symbol} position's stop-loss "
            f"{position.stop_price:.4f}: the position is due to be closed by its stop, so it "
            "cannot be added to"
        )
    if breach is ExitReason.TAKE_PROFIT:
        return (
            f"price {price:.4f} is at or above the {position.symbol} position's take-profit "
            f"{position.take_profit_price:.4f}: the position is due to be closed at its "
            "target, so it cannot be added to"
        )
    return None


def execute_market_order(
    *,
    symbol: str,
    side: Side,
    quantity: int,
    price: float,
    cash: float,
    position: PositionState | None,
    costs: CostModel,
    reason: str = "",
    stop: float | None = None,
    take_profit: float | None = None,
    decision_id: str | None = None,
    timestamp: datetime | None = None,
) -> OrderOutcome:
    """Fill (or reject) a market order against the given cash/position state.

    Pure function: it does not mutate ``position``; brokers persist the outcome.
    Assumes :func:`validate_order_request` already passed.

    A filled BUY's order record carries the protective levels *in force on the position
    after the fill* (:func:`merge_protective_levels`), which can differ from the requested
    ones for an add-on. A rejected order keeps the requested levels (nothing was applied).
    """
    now = ensure_utc(timestamp) if timestamp is not None else utc_now()
    fill = costs.quote(side, quantity, price)
    held = position.quantity if position is not None else 0

    reject_reason: str | None = None
    if side is Side.BUY:
        if position is not None and held > 0:
            reject_reason = _breached_level_reason(position, price)
        if reject_reason is None and fill.cash_required > cash + _EPS:
            reject_reason = (
                f"insufficient cash: need ${fill.cash_required:,.2f} "
                f"(incl. ${fill.commission:,.2f} commission), have ${cash:,.2f}"
            )
    elif held == 0:
        reject_reason = f"no position in {symbol} to sell (short selling is not supported)"
    elif quantity > held:
        reject_reason = f"insufficient position in {symbol}: hold {held}, asked to sell {quantity}"

    order = OrderRecord(
        symbol=symbol,
        side=side,
        quantity=quantity,
        status=OrderStatus.REJECTED if reject_reason else OrderStatus.FILLED,
        requested_at=now,
        reference_price=price,
        stop_price=stop,
        take_profit_price=take_profit,
        reason=reason,
        decision_id=decision_id,
        reject_reason=reject_reason,
    )
    if reject_reason:
        return OrderOutcome(order=order, trade=None, position=position, cash=cash, realized_pnl=0.0)

    realized = 0.0
    new_position: PositionState | None
    if side is Side.BUY:
        new_qty = held + quantity
        basis = (held * position.avg_price if position else 0.0) + fill.notional + fill.commission
        merged_stop, merged_target = merge_protective_levels(position, stop, take_profit)
        new_position = PositionState(
            symbol=symbol,
            quantity=new_qty,
            avg_price=basis / new_qty,
            stop_price=merged_stop,
            take_profit_price=merged_target,
            opened_at=position.opened_at if position else now,
        )
        # Report what actually protects the position, not just what was requested.
        order.stop_price, order.take_profit_price = merged_stop, merged_target
    else:
        assert position is not None  # guaranteed by the rejection checks above
        realized = fill.notional - fill.commission - quantity * position.avg_price
        remaining = held - quantity
        new_position = replace(position, quantity=remaining) if remaining > 0 else None

    order.fill_price = fill.fill_price
    order.filled_at = now
    order.commission = fill.commission
    order.slippage = fill.slippage_cost
    order.realized_pnl = realized if side is Side.SELL else None
    trade = TradeRecord(
        order_id=order.id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=fill.fill_price,
        commission=fill.commission,
        timestamp=now,
        realized_pnl=order.realized_pnl,
        reason=reason,
    )
    return OrderOutcome(
        order=order,
        trade=trade,
        position=new_position,
        cash=cash + fill.cash_delta,
        realized_pnl=realized,
    )


# =============================================================================== exits


def evaluate_exit(stop: float | None, take_profit: float | None, price: float) -> ExitReason | None:
    """Which protective level ``price`` has breached (the stop wins ties)."""
    if not _valid_price(price):
        return None
    if stop is not None and price <= stop:
        return ExitReason.STOP_LOSS
    if take_profit is not None and price >= take_profit:
        return ExitReason.TAKE_PROFIT
    return None


def evaluate_exit_bar(
    stop: float | None,
    take_profit: float | None,
    open_: float,
    high: float,
    low: float,
) -> tuple[ExitReason, float] | None:
    """Exit triggered within an OHLC bar and its fill price (before slippage).

    * Gap through a level at the open → fill at the open (worse than the stop on a gap
      down, better than the target on a gap up).
    * Otherwise a level inside ``[low, high]`` fills at the level; if both are inside,
      the stop is assumed to have traded first.
    """
    if stop is not None and open_ <= stop:
        return ExitReason.STOP_LOSS, open_
    if take_profit is not None and open_ >= take_profit:
        return ExitReason.TAKE_PROFIT, open_
    if stop is not None and low <= stop:
        return ExitReason.STOP_LOSS, stop
    if take_profit is not None and high >= take_profit:
        return ExitReason.TAKE_PROFIT, take_profit
    return None


# =============================================================================== snapshots


def mark_to_market(
    positions: Iterable[PositionState], prices: Mapping[str, float] | None = None
) -> tuple[float, float]:
    """Full-precision ``(positions_value, unrealized_pnl)``; unpriced positions use avg."""
    prices = prices or {}
    positions_value = 0.0
    unrealized = 0.0
    for state in positions:
        quote = prices.get(state.symbol)
        mark = quote if quote is not None and _valid_price(quote) else state.avg_price
        positions_value += state.quantity * mark
        unrealized += state.quantity * (mark - state.avg_price)
    return positions_value, unrealized


def build_snapshot(
    *,
    initial_capital: float,
    cash: float,
    positions: Iterable[PositionState],
    realized_pnl: float,
    peak_equity: float,
    prices: Mapping[str, float] | None = None,
    timestamp: datetime | None = None,
) -> PortfolioSnapshot:
    """Mark positions to ``prices`` (missing/invalid → ``avg_price``) and summarise.

    The reported ``peak_equity`` is ``max(peak_equity, equity)``.
    """
    prices = prices or {}
    states = sorted(positions, key=lambda p: p.symbol)
    marked = [p.mark(prices.get(p.symbol)) for p in states]
    # Totals come from full-precision state, not from the rounded output models.
    positions_value, unrealized = mark_to_market(states, prices)
    equity = cash + positions_value
    peak = max(peak_equity, equity)
    drawdown = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
    exposure = positions_value / equity * 100.0 if equity > 0 else 0.0
    total_return = equity - initial_capital
    return PortfolioSnapshot(
        initial_capital=money(initial_capital),
        cash=money(cash),
        equity=money(equity),
        positions_value=money(positions_value),
        gross_exposure_pct=round(exposure, 4),
        realized_pnl=money(realized_pnl),
        unrealized_pnl=money(unrealized),
        total_return=money(total_return),
        total_return_pct=round(total_return / initial_capital * 100.0, 4)
        if initial_capital > 0
        else 0.0,
        peak_equity=money(peak),
        drawdown_pct=round(max(drawdown, 0.0), 4),
        positions=marked,
        updated_at=timestamp or utc_now(),
    )


def all_priced(positions: Iterable[PositionState], prices: Mapping[str, float] | None) -> bool:
    """``True`` if every position has a valid market price in ``prices``."""
    prices = prices or {}
    return all(_valid_price(prices.get(p.symbol)) for p in positions)


__all__ = [
    "CostModel",
    "Fill",
    "OrderOutcome",
    "OrderRecord",
    "OrderValidationError",
    "PositionState",
    "TradeRecord",
    "all_priced",
    "build_snapshot",
    "evaluate_exit",
    "evaluate_exit_bar",
    "execute_market_order",
    "is_valid_price",
    "mark_to_market",
    "merge_protective_levels",
    "money",
    "price4",
    "validate_order_request",
]
