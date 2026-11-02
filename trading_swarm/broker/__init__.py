"""Paper brokerage: persistent SQLite account plus an in-memory twin for backtests."""

from trading_swarm.broker.accounting import (
    CostModel,
    Fill,
    OrderValidationError,
    PositionState,
    evaluate_exit,
    evaluate_exit_bar,
    merge_protective_levels,
)
from trading_swarm.broker.base import UNSET, Broker
from trading_swarm.broker.db import Database, create_db_engine
from trading_swarm.broker.memory import InMemoryBroker
from trading_swarm.broker.paper import PaperBroker

__all__ = [
    "UNSET",
    "Broker",
    "CostModel",
    "Database",
    "Fill",
    "InMemoryBroker",
    "OrderValidationError",
    "PaperBroker",
    "PositionState",
    "create_db_engine",
    "evaluate_exit",
    "evaluate_exit_bar",
    "merge_protective_levels",
]
