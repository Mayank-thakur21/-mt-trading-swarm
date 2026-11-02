"""Broker contract implemented by :class:`PaperBroker` (SQLite) and :class:`InMemoryBroker`.

Agents (notably :class:`~trading_swarm.agents.execution.ExecutionAgent`) depend on this
protocol rather than on a concrete broker, so the same execution logic drives both the
persistent paper account and the backtester's in-memory account.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Final, Literal, Protocol, runtime_checkable

from trading_swarm.broker.accounting import CostModel
from trading_swarm.models import Order, PortfolioSnapshot, Position, Side, Trade


class _Unset(Enum):
    UNSET = "UNSET"


#: Sentinel for "leave unchanged" in :meth:`Broker.update_stops` (``None`` clears a level).
UNSET: Final = _Unset.UNSET
Unset = Literal[_Unset.UNSET]


@runtime_checkable
class Broker(Protocol):
    """A long-only paper account that fills market orders immediately."""

    cost_model: CostModel

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
        """Fill a market order at reference ``price`` ± slippage, charging commission.

        Returns a FILLED order, or a REJECTED one (``reject_reason`` set) for business
        rule failures: insufficient cash, no/insufficient position to sell, or a BUY
        into a position whose stop / target ``price`` has already breached.

        A BUY into an existing position never loosens its stop (the position keeps the
        higher of the two stops); its take-profit becomes the new one when given. The
        filled order reports the levels in force afterwards. See
        :func:`~trading_swarm.broker.accounting.merge_protective_levels`.

        Raises:
            OrderValidationError: malformed request (quantity ≤ 0, bad price/levels,
                levels on a SELL).
        """
        ...

    def portfolio(self, prices: Mapping[str, float] | None = None) -> PortfolioSnapshot:
        """Account snapshot marked to ``prices`` (positions without a price → avg price)."""
        ...

    def positions(self) -> list[Position]:
        """Open positions (marked at their average price)."""
        ...

    def position(self, symbol: str) -> Position | None:
        """The open position in ``symbol``, if any."""
        ...

    def update_stops(
        self,
        symbol: str,
        stop: float | Unset | None = UNSET,
        take_profit: float | Unset | None = UNSET,
    ) -> Position:
        """Set (float), clear (``None``) or keep (``UNSET``) a position's protective levels.

        Raises:
            KeyError: no open position in ``symbol``.
        """
        ...

    def orders(self, limit: int | None = 50) -> list[Order]:
        """Most recent orders first."""
        ...

    def trades(self, limit: int | None = 100) -> list[Trade]:
        """Most recent fills first."""
        ...


__all__ = ["UNSET", "Broker", "Unset"]
