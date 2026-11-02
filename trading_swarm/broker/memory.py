"""In-memory broker with exactly the same accounting as :class:`PaperBroker`.

Built for the backtester: no database, explicit timestamps (simulated bar times), and an
equity curve recorded per bar. Both brokers delegate fills, costs, P&L and snapshots to
:mod:`trading_swarm.broker.accounting`, so a strategy books identical numbers whether it
is backtested or paper-traded.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from datetime import datetime

import pandas as pd

from trading_swarm.broker.accounting import (
    CostModel,
    OrderRecord,
    PositionState,
    TradeRecord,
    all_priced,
    build_snapshot,
    execute_market_order,
    mark_to_market,
    validate_order_request,
)
from trading_swarm.broker.base import UNSET, Unset
from trading_swarm.data.base import validate_symbol
from trading_swarm.models import (
    Order,
    PortfolioSnapshot,
    Position,
    Side,
    Trade,
    ensure_utc,
    utc_now,
)


class InMemoryBroker:
    """Volatile long-only account (implements :class:`~trading_swarm.broker.base.Broker`).

    Args:
        initial_capital: Starting cash.
        cost_model: Commission/slippage model (use ``CostModel.from_settings``).
    """

    def __init__(self, initial_capital: float, cost_model: CostModel | None = None) -> None:
        if not initial_capital > 0:
            raise ValueError(f"initial capital must be positive, got {initial_capital!r}")
        self.cost_model = cost_model or CostModel()
        self._lock = threading.RLock()
        self.reset(initial_capital)

    # ------------------------------------------------------------------ state
    def reset(self, initial_capital: float | None = None) -> PortfolioSnapshot:
        with self._lock:
            capital = float(
                initial_capital if initial_capital is not None else self.initial_capital
            )
            if not capital > 0:
                raise ValueError(f"initial capital must be positive, got {capital!r}")
            self.initial_capital = capital
            self.cash = capital
            self.realized_pnl = 0.0
            self.peak_equity = capital
            self._positions: dict[str, PositionState] = {}
            self._orders: list[OrderRecord] = []
            self._trades: list[TradeRecord] = []
            self._curve: list[tuple[datetime, float, float, float]] = []
            return self.portfolio()

    # ------------------------------------------------------------------ orders
    def submit_order(
        self,
        symbol: str,
        side: Side | str,
        quantity: int,
        price: float,
        reason: str = "",
        stop: float | None = None,
        take_profit: float | None = None,
        decision_id: str | None = None,
        *,
        timestamp: datetime | None = None,
    ) -> Order:
        sym = validate_symbol(symbol)
        parsed_side = validate_order_request(side, quantity, price, stop, take_profit)
        with self._lock:
            outcome = execute_market_order(
                symbol=sym,
                side=parsed_side,
                quantity=quantity,
                price=float(price),
                cash=self.cash,
                position=self._positions.get(sym),
                costs=self.cost_model,
                reason=reason,
                stop=stop,
                take_profit=take_profit,
                decision_id=decision_id,
                timestamp=timestamp,
            )
            self._orders.append(outcome.order)
            if outcome.filled and outcome.trade is not None:
                self._trades.append(outcome.trade)
                if outcome.position is None:
                    self._positions.pop(sym, None)
                else:
                    self._positions[sym] = outcome.position
                self.cash = outcome.cash
                self.realized_pnl += outcome.realized_pnl
            return outcome.order.to_model()

    def update_stops(
        self,
        symbol: str,
        stop: float | Unset | None = UNSET,
        take_profit: float | Unset | None = UNSET,
    ) -> Position:
        sym = validate_symbol(symbol)
        with self._lock:
            state = self._positions.get(sym)
            if state is None:
                raise KeyError(f"no open position in {sym}")
            if stop is not UNSET:
                state.stop_price = None if stop is None else float(stop)
            if take_profit is not UNSET:
                state.take_profit_price = None if take_profit is None else float(take_profit)
            return state.mark(None)

    # ------------------------------------------------------------------ queries
    def portfolio(
        self, prices: Mapping[str, float] | None = None, *, timestamp: datetime | None = None
    ) -> PortfolioSnapshot:
        with self._lock:
            states = list(self._positions.values())
            snapshot = build_snapshot(
                initial_capital=self.initial_capital,
                cash=self.cash,
                positions=states,
                realized_pnl=self.realized_pnl,
                peak_equity=self.peak_equity,
                prices=prices,
                timestamp=ensure_utc(timestamp) if timestamp is not None else None,
            )
            positions_value, _ = mark_to_market(states, prices)
            equity = self.cash + positions_value
            if equity > self.peak_equity and all_priced(states, prices):
                self.peak_equity = equity
            return snapshot

    def positions(self) -> list[Position]:
        with self._lock:
            return [self._positions[s].mark(None) for s in sorted(self._positions)]

    def position(self, symbol: str) -> Position | None:
        with self._lock:
            state = self._positions.get(validate_symbol(symbol))
            return state.mark(None) if state is not None else None

    def position_state(self, symbol: str) -> PositionState | None:
        """Full-precision state (for exit checks against raw stop/target levels)."""
        with self._lock:
            return self._positions.get(validate_symbol(symbol))

    def orders(self, limit: int | None = 50) -> list[Order]:
        with self._lock:
            records = self._orders[::-1] if limit is None else self._orders[::-1][: max(0, limit)]
            return [r.to_model() for r in records]

    def trades(self, limit: int | None = 100) -> list[Trade]:
        """Most recent fills first (``limit=None`` returns all)."""
        with self._lock:
            records = self._trades[::-1] if limit is None else self._trades[::-1][: max(0, limit)]
            return [r.to_model() for r in records]

    # ------------------------------------------------------------------ equity curve
    def record_snapshot(
        self, prices: Mapping[str, float] | None = None, *, timestamp: datetime | None = None
    ) -> PortfolioSnapshot:
        """Mark to ``prices`` at ``timestamp`` (a bar time) and append to the equity curve."""
        when = ensure_utc(timestamp) if timestamp is not None else utc_now()
        with self._lock:
            snapshot = self.portfolio(prices, timestamp=when)
            positions_value, _ = mark_to_market(self._positions.values(), prices)
            self._curve.append((when, self.cash + positions_value, self.cash, positions_value))
            return snapshot

    def equity_curve(self) -> pd.DataFrame:
        """UTC-indexed ``equity, cash, positions_value`` for every recorded snapshot."""
        with self._lock:
            rows = list(self._curve)
        frame = pd.DataFrame(rows, columns=["timestamp", "equity", "cash", "positions_value"])
        frame.index = pd.DatetimeIndex(
            pd.to_datetime(frame.pop("timestamp"), utc=True), name="timestamp"
        )
        return frame.astype(float)


__all__ = ["InMemoryBroker"]
