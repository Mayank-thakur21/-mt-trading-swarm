"""SMTP e-mail notifier.

``EmailNotifier.send(subject, text, html=None) -> bool`` delivers a
``multipart/alternative`` message (plain text first, HTML second, so clients that cannot
render HTML still show a readable alert) to every address in ``ALERT_EMAIL_TO``.

Transport selection follows the usual provider conventions:

* ``SMTP_USE_SSL=true`` → implicit TLS (``smtplib.SMTP_SSL``, normally port 465);
* else ``SMTP_USE_TLS=true`` → plain connection upgraded with ``STARTTLS`` (port 587);
* else an unencrypted connection (only sensible for a local relay / test server).

``login`` is attempted whenever ``SMTP_USERNAME`` is set (Gmail needs an *app password*).
Every network operation uses a 15 s timeout.

Contract: the notifier **never raises**. When the channel is not configured (no host,
sender or recipient) it logs ``"email alerts disabled"`` at INFO and returns ``False``;
SMTP / socket / TLS errors are logged at WARNING and reported as a failed
:class:`~trading_swarm.alerts.base.DeliveryResult`. Credentials are never logged.

``smtplib.SMTP`` / ``smtplib.SMTP_SSL`` are looked up on the module at call time, which
is what lets the test-suite monkeypatch them with recording fakes.
"""

from __future__ import annotations

import logging
import re
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import format_datetime, getaddresses, make_msgid, parseaddr

from trading_swarm import __version__
from trading_swarm.alerts.base import DeliveryResult
from trading_swarm.config import Settings
from trading_swarm.models import AlertStatus, utc_now

logger = logging.getLogger(__name__)

#: Longest subject we emit; RFC 5322 has no hard limit but clients truncate around here.
MAX_SUBJECT_CHARS = 200

_WHITESPACE = re.compile(r"\s+")


def clean_header(value: str, max_chars: int = MAX_SUBJECT_CHARS) -> str:
    """Collapse whitespace (including CR/LF, which would allow header injection) and cap length."""
    cleaned = _WHITESPACE.sub(" ", value).strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


def _bare_address(value: str) -> str:
    """``"Swarm <bot@example.com>"`` → ``"bot@example.com"`` (envelope addresses)."""
    return parseaddr(value)[1] or value.strip()


class EmailNotifier:
    """Sends alert e-mails through the configured SMTP server (implements ``Notifier``)."""

    channel = "email"
    TIMEOUT_SECONDS = 15.0

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.smtp_use_ssl and settings.smtp_port == 587:
            logger.warning(
                "SMTP_USE_SSL=true with port 587: implicit TLS normally uses port 465 "
                "(port 587 expects STARTTLS, i.e. SMTP_USE_TLS=true)"
            )
        if settings.smtp_username and not settings.smtp_password:
            logger.warning("SMTP_USERNAME is set but SMTP_PASSWORD is empty; login will fail")

    # ------------------------------------------------------------------ configuration
    @property
    def enabled(self) -> bool:
        """SMTP host, sender and at least one recipient are configured."""
        return self.settings.email_enabled

    @property
    def recipients(self) -> list[str]:
        """Addresses from ``ALERT_EMAIL_TO`` (as configured, display names allowed)."""
        return self.settings.alert_recipients

    def missing_settings(self) -> list[str]:
        """Names of the environment variables that must be set to enable e-mail."""
        missing = []
        if not self.settings.smtp_host:
            missing.append("SMTP_HOST")
        if not self.settings.alert_email_from:
            missing.append("ALERT_EMAIL_FROM")
        if not self.settings.alert_recipients:
            missing.append("ALERT_EMAIL_TO")
        return missing

    def describe(self) -> dict[str, object]:
        """Non-secret summary of the channel configuration (for status pages / logs)."""
        s = self.settings
        if s.smtp_use_ssl:
            security = "ssl"
        elif s.smtp_use_tls:
            security = "starttls"
        else:
            security = "none"
        return {
            "channel": self.channel,
            "enabled": self.enabled,
            "host": s.smtp_host,
            "port": s.smtp_port,
            "security": security,
            "auth": bool(s.smtp_username),
            "sender": s.alert_email_from,
            "recipients": len(self.recipients),
            "missing": self.missing_settings(),
        }

    # ------------------------------------------------------------------ message
    def build_message(self, subject: str, text: str, html: str | None = None) -> EmailMessage:
        """Assemble the MIME message (``text/plain`` or ``multipart/alternative``).

        Raises:
            ValueError: an address header is malformed (e.g. contains a line break).
        """
        sender = self.settings.alert_email_from
        message = EmailMessage()
        message["Subject"] = clean_header(subject)
        message["From"] = sender
        message["To"] = ", ".join(self.recipients)
        message["Date"] = format_datetime(utc_now())
        # An explicit domain keeps make_msgid from calling socket.getfqdn() (a DNS lookup).
        domain = _bare_address(sender).rpartition("@")[2] or "localhost"
        message["Message-ID"] = make_msgid(idstring="mtswarm", domain=domain)
        message["X-Mailer"] = f"MT Trading Swarm {__version__}"
        # RFC 3834: machine-generated mail — stops out-of-office auto-replies.
        message["Auto-Submitted"] = "auto-generated"
        message.set_content(text if text.endswith("\n") else text + "\n")
        if html:
            message.add_alternative(html, subtype="html")
        return message

    # ------------------------------------------------------------------ delivery
    def send(self, subject: str, text: str, html: str | None = None) -> bool:
        """Send an e-mail to ``ALERT_EMAIL_TO``; ``True`` on success, never raises."""
        return self.deliver(subject, text, html).sent

    def deliver(self, subject: str, text: str, html: str | None = None) -> DeliveryResult:
        """Like :meth:`send` but reports the outcome (sent / failed / disabled) in detail."""
        try:
            return self._deliver(subject, text, html)
        except Exception as exc:  # last line of defence: alerting must never break callers
            logger.exception("unexpected error while sending alert email %r", subject)
            return DeliveryResult(AlertStatus.FAILED, f"unexpected error: {_describe(exc)}")

    def _deliver(self, subject: str, text: str, html: str | None) -> DeliveryResult:
        missing = self.missing_settings()
        if missing:
            detail = f"email alerts disabled: set {', '.join(missing)}"
            logger.info("email alerts disabled (missing %s); not sending %r", missing, subject)
            return DeliveryResult(AlertStatus.DISABLED, detail)

        s = self.settings
        endpoint = f"{s.smtp_host}:{s.smtp_port}"
        try:
            message = self.build_message(subject, text, html)
        except (ValueError, TypeError) as exc:
            logger.warning("could not build alert email %r: %s", subject, exc)
            return DeliveryResult(AlertStatus.FAILED, f"invalid message: {_describe(exc)}")

        envelope_to = [addr for _, addr in getaddresses(self.recipients) if addr]
        try:
            refused = self._transmit(message, envelope_to)
        except smtplib.SMTPAuthenticationError as exc:
            detail = (
                f"SMTP authentication failed at {endpoint} ({_smtp_code(exc)}); check "
                "SMTP_USERNAME / SMTP_PASSWORD (Gmail requires an app password)"
            )
        except smtplib.SMTPNotSupportedError as exc:
            detail = (
                f"{endpoint} does not support a required extension ({exc}); for STARTTLS "
                "problems try SMTP_USE_SSL=true with port 465, or SMTP_USE_TLS=false"
            )
        except smtplib.SMTPRecipientsRefused as exc:
            detail = f"all recipients refused by {endpoint}: {', '.join(exc.recipients)}"
        except smtplib.SMTPSenderRefused as exc:
            detail = f"sender {exc.sender!r} refused by {endpoint} ({_smtp_code(exc)})"
        except smtplib.SMTPException as exc:
            detail = f"SMTP error from {endpoint}: {_describe(exc)}"
        except ssl.SSLError as exc:
            detail = f"TLS handshake with {endpoint} failed: {_describe(exc)}"
        except OSError as exc:  # DNS failure, connection refused, timeouts (TimeoutError)
            detail = f"could not connect to {endpoint}: {_describe(exc)}"
        else:
            delivered = len(envelope_to) - len(refused)
            detail = f"delivered to {delivered}/{len(envelope_to)} recipient(s) via {endpoint}"
            if refused:
                detail += f"; refused: {', '.join(sorted(refused))}"
                logger.warning("alert email %r: some recipients refused: %s", subject, refused)
            logger.info("alert email sent: %r (%s)", message["Subject"], detail)
            return DeliveryResult(AlertStatus.SENT, detail)

        logger.warning("alert email %r not sent: %s", subject, detail)
        return DeliveryResult(AlertStatus.FAILED, detail)

    def _transmit(self, message: EmailMessage, envelope_to: list[str]) -> dict[str, object]:
        """Open the configured connection, authenticate and hand the message over.

        Returns the recipients the server refused (empty when all were accepted).
        """
        s = self.settings
        context = ssl.create_default_context()
        if s.smtp_use_ssl:
            connection: smtplib.SMTP = smtplib.SMTP_SSL(
                s.smtp_host, s.smtp_port, timeout=self.TIMEOUT_SECONDS, context=context
            )
        else:
            connection = smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=self.TIMEOUT_SECONDS)
        with connection as smtp:
            if s.smtp_use_tls and not s.smtp_use_ssl:
                smtp.starttls(context=context)
            if s.smtp_username:
                smtp.login(s.smtp_username, s.smtp_password)
            refused = smtp.send_message(
                message,
                from_addr=_bare_address(s.alert_email_from),
                to_addrs=envelope_to,
            )
        return dict(refused or {})


def _smtp_code(exc: smtplib.SMTPResponseException) -> str:
    """``"535 5.7.8 Username and Password not accepted"`` from an SMTP response error."""
    message = exc.smtp_error
    text = message.decode("utf-8", "replace") if isinstance(message, bytes) else str(message)
    return clean_header(f"{exc.smtp_code} {text}", max_chars=160)


def _describe(exc: BaseException) -> str:
    text = clean_header(str(exc), max_chars=200)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


__all__ = ["MAX_SUBJECT_CHARS", "EmailNotifier", "clean_header"]
