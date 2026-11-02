"""Contracts shared by the alert channels and the :class:`~trading_swarm.alerts.AlertManager`.

Two small protocols decouple the rule engine from its collaborators:

* :class:`Notifier` — a delivery channel (today only SMTP e-mail). ``deliver`` reports
  *how* a message fared (sent / failed / channel disabled) so the audit log can record
  the precise outcome; ``send`` is the boolean convenience form required by the spec.
* :class:`AlertLog` — where alert records are persisted and queried for cooldown
  dedupe. :class:`~trading_swarm.broker.paper.PaperBroker` satisfies it structurally
  (``log_alert`` / ``last_alert_at`` / ``alerts`` on the ``alerts_log`` table), so the
  manager never touches SQL itself.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from trading_swarm.models import AlertRecord, AlertStatus


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    """Outcome of one delivery attempt on a channel.

    Attributes:
        status: :attr:`AlertStatus.SENT`, :attr:`AlertStatus.FAILED` or
            :attr:`AlertStatus.DISABLED` (channel not configured). Channels never return
            ``SUPPRESSED`` — dedupe is the manager's job.
        detail: Human-readable explanation (recipient count, error message, or which
            settings are missing). Never contains credentials.
    """

    status: AlertStatus
    detail: str = ""

    @property
    def sent(self) -> bool:
        """``True`` when the channel accepted the message for delivery."""
        return self.status is AlertStatus.SENT


@runtime_checkable
class Notifier(Protocol):
    """A notification channel. Implementations must never raise from ``deliver``/``send``."""

    channel: str

    @property
    def enabled(self) -> bool:
        """Whether the channel is configured well enough to attempt delivery."""
        ...

    def deliver(self, subject: str, text: str, html: str | None = None) -> DeliveryResult:
        """Deliver a message and report the detailed outcome."""
        ...

    def send(self, subject: str, text: str, html: str | None = None) -> bool:
        """Deliver a message; ``True`` on success."""
        ...


@runtime_checkable
class AlertLog(Protocol):
    """Persistence for alert records (implemented by :class:`PaperBroker`)."""

    def log_alert(self, record: AlertRecord) -> AlertRecord:
        """Append ``record`` and return it with its storage id."""
        ...

    def last_alert_at(
        self,
        rule: str,
        symbol: str | None = None,
        statuses: Iterable[AlertStatus | str] | None = (AlertStatus.SENT,),
    ) -> datetime | None:
        """Timestamp of the latest ``rule`` alert for ``symbol`` with one of ``statuses``."""
        ...

    def alerts(
        self, limit: int | None = 50, rule: str | None = None, symbol: str | None = None
    ) -> list[AlertRecord]:
        """Most recent records first."""
        ...


__all__ = ["AlertLog", "DeliveryResult", "Notifier"]
