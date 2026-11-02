"""Agent protocols. Every signal agent returns an :class:`~trading_swarm.models.AgentSignal`
and is deterministic given its inputs (same history + headlines ⇒ same vote)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from trading_swarm.models import AgentSignal


@runtime_checkable
class Agent(Protocol):
    """Anything in the swarm with a stable short name (``"trend"``, ``"risk"`` …)."""

    name: str


@runtime_checkable
class SignalAgent(Agent, Protocol):
    """An agent that votes on a symbol.

    ``history`` is the daily OHLCV frame (canonical provider shape) *up to and including*
    the decision bar; agents must not look beyond its last row. Agents that do not need
    prices (sentiment) accept and ignore it.
    """

    def analyze(self, symbol: str, history: pd.DataFrame) -> AgentSignal:
        """Return this agent's vote for ``symbol``."""
        ...


__all__ = ["Agent", "AgentSignal", "SignalAgent"]
