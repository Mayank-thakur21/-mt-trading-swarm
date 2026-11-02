"""Alert rules, dedupe and audit log.

Rules
-----
=================  ==========================================================  ===========
rule               fires when                                                  dedupe
=================  ==========================================================  ===========
``signal``         :meth:`AlertManager.on_decision`: action is BUY/SELL and     cooldown
                   ``confidence ≥ ALERT_MIN_CONFIDENCE``                         (per action)
``order``          :meth:`AlertManager.on_decision`: the decision carries a     once per
                   paper order that was FILLED (or REJECTED, which needs        order id
                   attention: an approved auto-trade did not happen)
``exit``           :meth:`AlertManager.on_exit`: a stop-loss / take-profit      once per
                   exit order was filled                                        order id
``drawdown``       :meth:`AlertManager.on_drawdown`:                            cooldown
                   ``snapshot.drawdown ≥ MAX_DRAWDOWN_HALT`` (fractions)
``price_move``     :meth:`AlertManager.on_price_move`:                          cooldown
                   ``|1-day move %| ≥ ALERT_PRICE_MOVE_PCT`` (percent)          (per direction)
``test``           :meth:`AlertManager.send_test` (``POST /alerts/test``)       none
=================  ==========================================================  ===========

Dedupe
------
*Condition* rules (signal, drawdown, price move) are re-evaluated on every scan, so a
persistent condition would otherwise e-mail on every scan. They use a cooldown of
``ALERT_COOLDOWN_MINUTES`` per ``(rule, symbol)``: an alert is suppressed when the
same rule last **sent** for the same symbol less than the cooldown ago
(``broker.last_alert_at``; the account-level drawdown rule uses ``symbol=None``).
Failed and disabled deliveries do not start a cooldown, so a transient SMTP outage is
retried on the next scan. A *reversal* is never suppressed: a SELL signal right after a
BUY alert (or a move down right after a move-up alert) is new information and is sent.

*Event* rules (orders, exits) describe one-off fills; every fill is reported, and the
same order id is never reported twice by this process.

Audit trail
-----------
Every alert that fires — sent, failed, disabled channel or suppressed — is written to
``alerts_log`` via ``broker.log_alert`` with the channel and :class:`AlertStatus`.
``detail`` holds ``key=value`` facts (``action=BUY; score=+0.620; …; outcome=…``) so the
log is both human-readable and machine-parseable (:func:`parse_detail`).

Robustness
----------
Alerting is a side channel of trading: no rule method ever raises. Rendering,
delivery and persistence errors are logged; a failed database write still returns the
(unsaved, ``id=None``) record. Concurrent calls (the coordinator analyses symbols on a
thread pool) are serialised per ``(rule, symbol)`` so the cooldown check and the log
write are atomic and two threads cannot both send the same alert.
"""

from __future__ import annotations

import functools
import logging
import math
import re
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any, Concatenate, Final

from trading_swarm.alerts.base import AlertLog, DeliveryResult, Notifier
from trading_swarm.alerts.email import EmailNotifier, clean_header
from trading_swarm.alerts.templates import (
    SUBJECT_PREFIX,
    RenderedEmail,
    render_decision,
    render_drawdown,
    render_exit,
    render_order,
    render_price_move,
    render_test,
)
from trading_swarm.config import Settings
from trading_swarm.models import (
    Action,
    AlertRecord,
    AlertStatus,
    Decision,
    ExitReason,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Side,
    utc_now,
)

logger = logging.getLogger(__name__)

RULE_SIGNAL: Final = "signal"
RULE_ORDER: Final = "order"
RULE_EXIT: Final = "exit"
RULE_DRAWDOWN: Final = "drawdown"
RULE_PRICE_MOVE: Final = "price_move"
RULE_TEST: Final = "test"

#: Condition rules: repeats within ``ALERT_COOLDOWN_MINUTES`` are suppressed.
COOLDOWN_RULES: Final = frozenset({RULE_SIGNAL, RULE_DRAWDOWN, RULE_PRICE_MOVE})
#: Event rules: each order id is alerted once.
EVENT_RULES: Final = frozenset({RULE_ORDER, RULE_EXIT})

#: How many recent log rows to inspect when looking for the last *sent* alert's variant.
_VARIANT_LOOKBACK: Final = 50
#: Bound on the in-process memory of already-alerted order ids.
_EVENT_MEMORY: Final = 4096
#: Order statuses worth an e-mail when attached to a decision.
_ALERTABLE_ORDER_STATUSES: Final = frozenset({OrderStatus.FILLED, OrderStatus.REJECTED})

_EXIT_PATTERN: Final = re.compile(
    r"\b(" + "|".join(re.escape(reason.value) for reason in ExitReason) + r")\b"
)


# =============================================================================== helpers


def exit_reason_of(order: Order) -> ExitReason | None:
    """The :class:`ExitReason` recorded in ``order.reason`` (``"stop_loss"`` …), if any."""
    match = _EXIT_PATTERN.search(order.reason or "")
    return ExitReason(match.group(1)) if match else None


def format_detail(facts: Mapping[str, object]) -> str:
    """``{"action": "BUY", "score": "+0.62"}`` → ``"action=BUY; score=+0.62"``."""
    parts = []
    for key, value in facts.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={clean_header(str(value), max_chars=500)}")
    return "; ".join(parts)


def parse_detail(detail: str) -> dict[str, str]:
    """Inverse of :func:`format_detail` (fragments without ``=`` extend the previous value)."""
    facts: dict[str, str] = {}
    last_key: str | None = None
    for token in detail.split("; "):
        key, sep, value = token.partition("=")
        if sep and key and " " not in key:
            facts[key] = value
            last_key = key
        elif last_key is not None:
            facts[last_key] = f"{facts[last_key]}; {token}"
    return facts


def _normalise_symbol(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    cleaned = symbol.strip().upper()
    return cleaned or None


def _minutes(delta: timedelta) -> str:
    minutes = max(0.0, delta.total_seconds() / 60.0)
    return f"{minutes:.0f} min" if minutes >= 1 else f"{delta.total_seconds():.0f} s"


def _guarded[**P](
    method: Callable[Concatenate[AlertManager, P], list[AlertRecord]],
) -> Callable[Concatenate[AlertManager, P], list[AlertRecord]]:
    """Make a rule method total: log any unexpected exception and return ``[]``."""

    @functools.wraps(method)
    def wrapper(self: AlertManager, *args: P.args, **kwargs: P.kwargs) -> list[AlertRecord]:
        try:
            return method(self, *args, **kwargs)
        except Exception:
            logger.exception("alert rule %s failed; continuing without alert", method.__name__)
            return []

    return wrapper


# =============================================================================== manager


class AlertManager:
    """Evaluates alert rules, dedupes them and dispatches notifications.

    Args:
        notifier: Delivery channel (normally :class:`EmailNotifier`).
        broker: Alert-log store — the :class:`~trading_swarm.broker.paper.PaperBroker`
            (anything implementing :class:`~trading_swarm.alerts.base.AlertLog`).
        settings: Thresholds, cooldown and the dashboard URL used in e-mail links.
        clock: Source of "now" (UTC); injectable so tests can move time forward.
    """

    def __init__(
        self,
        notifier: Notifier | EmailNotifier,
        broker: AlertLog,
        settings: Settings,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.notifier = notifier
        self.broker = broker
        self.settings = settings
        self._clock = clock
        self._guard = threading.Lock()
        self._key_locks: dict[tuple[str, str | None], threading.Lock] = {}
        self._alerted_events: OrderedDict[str, None] = OrderedDict()

    # ------------------------------------------------------------------ configuration
    @property
    def channel(self) -> str:
        return str(getattr(self.notifier, "channel", "email"))

    @property
    def cooldown(self) -> timedelta:
        """Per-(rule, symbol) cooldown for condition rules."""
        return timedelta(minutes=self.settings.alert_cooldown_minutes)

    def config_summary(self) -> dict[str, Any]:
        """Non-secret alerting configuration (for the API / dashboard Alerts page)."""
        s = self.settings
        return {
            "channel": self.channel,
            "enabled": bool(getattr(self.notifier, "enabled", False)),
            "min_confidence": s.alert_min_confidence,
            "price_move_pct": s.alert_price_move_pct,
            "drawdown_halt_pct": s.max_drawdown_halt * 100.0,
            "cooldown_minutes": s.alert_cooldown_minutes,
            "recipients": len(s.alert_recipients),
            "notifier": self._channel_info(),
        }

    # ------------------------------------------------------------------ rules
    @_guarded
    def on_decision(self, decision: Decision) -> list[AlertRecord]:
        """Alert on confident BUY/SELL signals and on executed (or rejected) orders."""
        records: list[AlertRecord] = []
        if (
            decision.action in (Action.BUY, Action.SELL)
            and decision.confidence >= self.settings.alert_min_confidence
        ):
            facts = {
                "action": decision.action.value,
                "score": f"{decision.score:+.3f}",
                "confidence": f"{decision.confidence:.2f}",
                "price": f"{decision.price:.2f}",
                "approved": str(decision.risk.approved).lower(),
                "executed": str(decision.executed).lower(),
                "decision_id": decision.id,
            }
            records.append(
                self._dispatch(
                    RULE_SIGNAL,
                    decision.symbol,
                    lambda: render_decision(decision, self.settings),
                    facts,
                    variant_field="action",
                )
            )
        else:
            logger.debug(
                "no signal alert for %s: %s at confidence %.2f (min %.2f)",
                decision.symbol,
                decision.action.value,
                decision.confidence,
                self.settings.alert_min_confidence,
            )

        order = decision.order
        if order is not None and order.status in _ALERTABLE_ORDER_STATUSES:
            facts = {
                "order_id": order.id,
                "status": order.status.value,
                "side": order.side.value,
                "quantity": order.quantity,
                "fill_price": f"{order.fill_price:.4f}" if order.fill_price is not None else None,
                "reject_reason": order.reject_reason,
                "decision_id": decision.id,
            }
            records.append(
                self._dispatch(
                    RULE_ORDER,
                    order.symbol,
                    lambda: render_order(order, self.settings, decision),
                    facts,
                    event_id=order.id,
                )
            )
        return records

    @_guarded
    def on_exit(self, order: Order) -> list[AlertRecord]:
        """Alert when a stop-loss or take-profit exit was filled."""
        if order.status is not OrderStatus.FILLED or order.side is not Side.SELL:
            logger.debug(
                "no exit alert for %s order %s (%s %s)",
                order.symbol,
                order.id,
                order.side.value,
                order.status.value,
            )
            return []
        reason = exit_reason_of(order)
        facts = {
            "order_id": order.id,
            "exit": reason.value if reason else "exit",
            "quantity": order.quantity,
            "fill_price": f"{order.fill_price:.4f}" if order.fill_price is not None else None,
            "realized_pnl": f"{order.realized_pnl:+.2f}"
            if order.realized_pnl is not None
            else None,
        }
        return [
            self._dispatch(
                RULE_EXIT,
                order.symbol,
                lambda: render_exit(order, self.settings, reason),
                facts,
                event_id=order.id,
            )
        ]

    @_guarded
    def on_drawdown(self, snapshot: PortfolioSnapshot) -> list[AlertRecord]:
        """Alert when the account drawdown breaches ``max_drawdown_halt``.

        Compares fractions (``snapshot.drawdown`` vs ``MAX_DRAWDOWN_HALT``), never the
        percentage field against the fractional limit.
        """
        limit = self.settings.max_drawdown_halt
        if not math.isfinite(snapshot.drawdown) or snapshot.drawdown < limit:
            return []
        facts = {
            "drawdown_pct": f"{snapshot.drawdown_pct:.2f}",
            "limit_pct": f"{limit * 100:.2f}",
            "equity": f"{snapshot.equity:.2f}",
            "peak_equity": f"{snapshot.peak_equity:.2f}",
        }
        return [
            self._dispatch(
                RULE_DRAWDOWN, None, lambda: render_drawdown(snapshot, self.settings), facts
            )
        ]

    @_guarded
    def on_price_move(
        self, symbol: str, pct: float | None, *, price: float | None = None
    ) -> list[AlertRecord]:
        """Alert on a 1-day move of at least ``alert_price_move_pct`` percent.

        Args:
            symbol: Ticker.
            pct: 1-day change **in percent** (``5.0`` = +5 %), e.g. from
                :func:`trading_swarm.data.daily_change_pct`; ``None``/NaN is ignored.
            price: Optional latest price shown in the e-mail.
        """
        if pct is None or not math.isfinite(pct):
            return []
        if abs(pct) < self.settings.alert_price_move_pct:
            return []
        sym = _normalise_symbol(symbol)
        facts = {
            "direction": "up" if pct > 0 else "down",
            "change_pct": f"{pct:+.2f}",
            "threshold_pct": f"{self.settings.alert_price_move_pct:.2f}",
            "price": f"{price:.2f}" if price is not None and math.isfinite(price) else None,
        }
        return [
            self._dispatch(
                RULE_PRICE_MOVE,
                sym,
                lambda: render_price_move(sym or symbol, pct, self.settings, price),
                facts,
                variant_field="direction",
            )
        ]

    def send_test(self) -> AlertRecord:
        """Send a test e-mail (``POST /alerts/test``) and log the outcome.

        Never raises; the returned record's ``status`` is SENT, FAILED or DISABLED and
        ``detail`` explains the outcome (e.g. which settings are missing).
        """
        now = self._clock()
        try:
            email = render_test(self.settings, self._channel_info())
            subject = email.subject
            result = self._deliver(email)
        except Exception as exc:  # pragma: no cover - defensive, render_test is static
            logger.exception("test alert failed")
            subject = f"{SUBJECT_PREFIX} Test alert"
            result = DeliveryResult(AlertStatus.FAILED, f"unexpected error: {exc}")
        record = AlertRecord(
            timestamp=now,
            rule=RULE_TEST,
            symbol=None,
            channel=self.channel,
            status=result.status,
            subject=subject,
            detail=result.detail,
        )
        logger.info("test alert: %s (%s)", result.status.value, result.detail)
        return self._log(record)

    # ------------------------------------------------------------------ engine
    def _dispatch(
        self,
        rule: str,
        symbol: str | None,
        render: Callable[[], RenderedEmail],
        facts: Mapping[str, object],
        *,
        variant_field: str | None = None,
        event_id: str | None = None,
    ) -> AlertRecord:
        """Dedupe → render → deliver → log, atomically per ``(rule, symbol)``."""
        sym = _normalise_symbol(symbol)
        with self._lock_for(rule, sym):
            now = self._clock()
            if event_id is not None and self._already_alerted(event_id):
                suppressed: str | None = f"duplicate: order {event_id} was already alerted"
            elif rule in COOLDOWN_RULES:
                variant = str(facts[variant_field]) if variant_field else None
                suppressed = self._cooldown_reason(rule, sym, now, variant_field, variant)
            else:
                suppressed = None

            email = self._render(rule, sym, render, facts)
            if suppressed is not None:
                result = DeliveryResult(AlertStatus.SUPPRESSED, suppressed)
            else:
                result = self._deliver(email)
            if event_id is not None and result.sent:
                self._remember_event(event_id)

            record = AlertRecord(
                timestamp=now,
                rule=rule,
                symbol=sym,
                channel=self.channel,
                status=result.status,
                subject=email.subject,
                detail=format_detail({**facts, "outcome": result.detail}),
            )
            stored = self._log(record)
        log = logger.debug if result.status is AlertStatus.SUPPRESSED else logger.info
        log("alert %s/%s %s: %s", rule, sym or "-", result.status.value, result.detail)
        return stored

    def _cooldown_reason(
        self,
        rule: str,
        symbol: str | None,
        now: datetime,
        variant_field: str | None,
        variant: str | None,
    ) -> str | None:
        """Why this alert must be suppressed, or ``None`` when it may be sent."""
        window = self.cooldown
        if window <= timedelta(0):
            return None
        try:
            last_sent = self.broker.last_alert_at(rule, symbol)
        except Exception:
            # Fail open: a duplicate e-mail is better than a silently dropped alert.
            logger.warning("cooldown lookup failed for %s/%s; sending", rule, symbol, exc_info=True)
            return None
        if last_sent is None:
            return None
        age = now - last_sent
        if age >= window:
            return None
        if variant_field is not None and variant is not None:
            previous = self._last_sent_variant(rule, symbol, variant_field)
            if previous is not None and previous != variant:
                logger.info(
                    "%s/%s reversed (%s → %s) within the cooldown; alerting",
                    rule,
                    symbol,
                    previous,
                    variant,
                )
                return None
        return (
            f"cooldown: last sent {_minutes(age)} ago "
            f"(ALERT_COOLDOWN_MINUTES={self.settings.alert_cooldown_minutes})"
        )

    def _last_sent_variant(self, rule: str, symbol: str | None, field: str) -> str | None:
        try:
            recent = self.broker.alerts(limit=_VARIANT_LOOKBACK, rule=rule, symbol=symbol)
        except Exception:
            logger.warning("alert history lookup failed for %s/%s", rule, symbol, exc_info=True)
            return None
        for record in recent:
            if record.symbol == symbol and record.status is AlertStatus.SENT:
                return parse_detail(record.detail).get(field)
        return None

    def _render(
        self,
        rule: str,
        symbol: str | None,
        render: Callable[[], RenderedEmail],
        facts: Mapping[str, object],
    ) -> RenderedEmail:
        try:
            return render()
        except Exception:
            # A template bug must not swallow the alert: fall back to a bare-bones message.
            logger.exception(
                "could not render %s alert for %s; sending plain fallback", rule, symbol
            )
            subject = f"{SUBJECT_PREFIX} {rule.replace('_', ' ')} alert" + (
                f": {symbol}" if symbol else ""
            )
            text = "\n".join(f"{key}: {value}" for key, value in facts.items() if value)
            return RenderedEmail(subject=subject, text=text or subject, html="")

    def _deliver(self, email: RenderedEmail) -> DeliveryResult:
        """Call the notifier, tolerating notifiers that only implement ``send``/raise."""
        try:
            deliver = getattr(self.notifier, "deliver", None)
            if callable(deliver):
                result = deliver(email.subject, email.text, email.html or None)
                if isinstance(result, DeliveryResult):
                    return result
                return DeliveryResult(AlertStatus.SENT if result else AlertStatus.FAILED)
            if not getattr(self.notifier, "enabled", True):
                return DeliveryResult(AlertStatus.DISABLED, f"{self.channel} alerts disabled")
            ok = self.notifier.send(email.subject, email.text, email.html or None)
            return DeliveryResult(AlertStatus.SENT if ok else AlertStatus.FAILED)
        except Exception as exc:
            logger.exception("notifier %s raised while sending %r", self.channel, email.subject)
            return DeliveryResult(AlertStatus.FAILED, f"notifier error: {type(exc).__name__}")

    def _log(self, record: AlertRecord) -> AlertRecord:
        try:
            return self.broker.log_alert(record)
        except Exception:
            logger.exception("could not persist %s alert to alerts_log", record.rule)
            return record

    def _channel_info(self) -> dict[str, object]:
        describe = getattr(self.notifier, "describe", None)
        if callable(describe):
            try:
                info = describe()
                if isinstance(info, Mapping):
                    return dict(info)
            except Exception:  # pragma: no cover - defensive
                logger.debug("notifier.describe() failed", exc_info=True)
        return {"channel": self.channel, "enabled": bool(getattr(self.notifier, "enabled", False))}

    # ------------------------------------------------------------------ concurrency
    def _lock_for(self, rule: str, symbol: str | None) -> threading.Lock:
        key = (rule, symbol)
        with self._guard:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = self._key_locks[key] = threading.Lock()
            return lock

    def _already_alerted(self, event_id: str) -> bool:
        with self._guard:
            return event_id in self._alerted_events

    def _remember_event(self, event_id: str) -> None:
        with self._guard:
            self._alerted_events[event_id] = None
            self._alerted_events.move_to_end(event_id)
            while len(self._alerted_events) > _EVENT_MEMORY:
                self._alerted_events.popitem(last=False)


__all__ = [
    "COOLDOWN_RULES",
    "EVENT_RULES",
    "RULE_DRAWDOWN",
    "RULE_EXIT",
    "RULE_ORDER",
    "RULE_PRICE_MOVE",
    "RULE_SIGNAL",
    "RULE_TEST",
    "AlertManager",
    "exit_reason_of",
    "format_detail",
    "parse_detail",
]
