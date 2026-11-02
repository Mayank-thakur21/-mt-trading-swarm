"""Alerting: SMTP e-mail notifier, e-mail templates and the rule-based alert manager.

Typical wiring (API lifespan / scheduler)::

    notifier = EmailNotifier(settings)
    alerts = AlertManager(notifier, broker, settings)
    alerts.on_decision(decision)  # confident signals + executed orders
    alerts.on_exit(order)  # stop-loss / take-profit fills
    alerts.on_drawdown(snapshot)  # circuit-breaker breach
    alerts.on_price_move("AAPL", 6.2)  # |1-day move| ≥ ALERT_PRICE_MOVE_PCT
"""

from trading_swarm.alerts.base import AlertLog, DeliveryResult, Notifier
from trading_swarm.alerts.email import EmailNotifier
from trading_swarm.alerts.manager import (
    COOLDOWN_RULES,
    EVENT_RULES,
    RULE_DRAWDOWN,
    RULE_EXIT,
    RULE_ORDER,
    RULE_PRICE_MOVE,
    RULE_SIGNAL,
    RULE_TEST,
    AlertManager,
    exit_reason_of,
    parse_detail,
)
from trading_swarm.alerts.templates import RenderedEmail

__all__ = [
    "COOLDOWN_RULES",
    "EVENT_RULES",
    "RULE_DRAWDOWN",
    "RULE_EXIT",
    "RULE_ORDER",
    "RULE_PRICE_MOVE",
    "RULE_SIGNAL",
    "RULE_TEST",
    "AlertLog",
    "AlertManager",
    "DeliveryResult",
    "EmailNotifier",
    "Notifier",
    "RenderedEmail",
    "exit_reason_of",
    "parse_detail",
]
