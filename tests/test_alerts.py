"""E-mail notifier, alert templates and alert-manager rules (fake SMTP, temp SQLite)."""

from __future__ import annotations

import logging
import math
import smtplib
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from email.message import EmailMessage
from html.parser import HTMLParser
from typing import Any

import pytest

from trading_swarm.alerts import (
    RULE_DRAWDOWN,
    RULE_EXIT,
    RULE_ORDER,
    RULE_PRICE_MOVE,
    RULE_SIGNAL,
    RULE_TEST,
    AlertManager,
    DeliveryResult,
    EmailNotifier,
    Notifier,
    exit_reason_of,
    parse_detail,
)
from trading_swarm.alerts import manager as manager_module
from trading_swarm.alerts.base import AlertLog
from trading_swarm.alerts.email import clean_header
from trading_swarm.alerts.manager import format_detail
from trading_swarm.alerts.templates import (
    dashboard_link,
    fmt_confidence,
    fmt_money,
    fmt_pct,
    fmt_price,
    fmt_signed,
    render_decision,
    render_drawdown,
    render_exit,
    render_order,
    render_price_move,
    render_test,
)
from trading_swarm.broker import PaperBroker
from trading_swarm.config import Settings
from trading_swarm.models import (
    Action,
    AgentSignal,
    AlertStatus,
    Decision,
    ExitReason,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    Position,
    RiskAssessment,
    Side,
)

T0 = datetime(2026, 9, 14, 15, 0, tzinfo=UTC)
PASSWORD = "s3cret-app-password"
EMAIL_CONFIG: dict[str, Any] = {
    "smtp_host": "smtp.test.invalid",
    "smtp_port": 587,
    "smtp_username": "bot@example.com",
    "smtp_password": PASSWORD,
    "alert_email_from": "MT Swarm <alerts@example.com>",
    "alert_email_to": "ops@example.com, PM <pm@example.com>",
    "dashboard_url": "https://swarm.example.com/dash",
}


# ============================================================================== helpers


class Clock:
    """A controllable UTC clock for cooldown tests."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


class _TagBalance(HTMLParser):
    """Asserts every non-void tag is closed in order."""

    VOID = frozenset({"meta", "br", "hr", "img", "link", "input"})

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        assert self.stack and self.stack[-1] == tag, f"unbalanced </{tag}> after {self.stack[-3:]}"
        self.stack.pop()


def assert_well_formed(html: str) -> None:
    parser = _TagBalance()
    parser.feed(html)
    parser.close()
    assert parser.stack == []


def bodies(message: EmailMessage) -> tuple[str, str | None]:
    """``(plain_text, html_or_None)`` of a sent message."""
    plain = message.get_body(preferencelist=("plain",))
    html = message.get_body(preferencelist=("html",))
    assert plain is not None
    return plain.get_content(), html.get_content() if html is not None else None


def make_decision(
    action: Action = Action.BUY,
    confidence: float = 0.8,
    *,
    symbol: str = "BULL",
    score: float | None = None,
    order: Order | None = None,
    rationale: str = "price above SMA20/50/200, MACD histogram rising",
) -> Decision:
    sign = {Action.BUY: 1.0, Action.SELL: -1.0, Action.HOLD: 0.0}[action]
    return Decision(
        symbol=symbol,
        timestamp=T0,
        price=100.0,
        action=action,
        score=score if score is not None else 0.62 * sign,
        confidence=confidence,
        signals=[
            AgentSignal(agent="trend", signal=0.71 * sign, confidence=0.85, rationale=rationale),
            AgentSignal(
                agent="sentiment",
                signal=0.2 * sign,
                confidence=0.4,
                rationale="2 bullish / 1 neutral headline",
                data={"method": "lexicon"},
            ),
        ],
        risk=RiskAssessment(
            approved=action is not Action.HOLD,
            action=action,
            quantity=37,
            entry_price=100.0,
            stop_price=95.0,
            take_profit_price=110.0,
            position_value=3700.0,
            risk_amount=185.0,
            risk_score=37.0,
            reasons=["size scaled by confidence (0.80×)"],
            checks={"drawdown_ok": True, "volatility_ok": True, "cash_ok": True},
        ),
        summary="Trend +0.71 (conf 0.85): price above SMA50/200; Sentiment +0.20 (lexicon).",
        executed=order is not None and order.is_filled,
        order=order,
    )


def make_snapshot(drawdown_pct: float, *, peak: float = 100_000.0) -> PortfolioSnapshot:
    equity = peak * (1 - drawdown_pct / 100)
    return PortfolioSnapshot(
        initial_capital=peak,
        cash=equity - 18_000.0,
        equity=equity,
        positions_value=18_000.0,
        gross_exposure_pct=18_000.0 / equity * 100,
        unrealized_pnl=-2_000.0,
        total_return=equity - peak,
        total_return_pct=-drawdown_pct,
        peak_equity=peak,
        drawdown_pct=drawdown_pct,
        positions=[
            Position(
                symbol="BEAR",
                quantity=100,
                avg_price=200.0,
                market_price=180.0,
                market_value=18_000.0,
                unrealized_pnl=-2_000.0,
                unrealized_pnl_pct=-10.0,
                stop_price=170.0,
            )
        ],
        updated_at=T0,
    )


@pytest.fixture
def email_settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(**EMAIL_CONFIG)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def manager(
    email_settings: Settings, broker: PaperBroker, clock: Clock, fake_smtp: Any
) -> AlertManager:
    return AlertManager(EmailNotifier(email_settings), broker, email_settings, clock=clock)


@pytest.fixture
def make_manager(
    make_settings: Callable[..., Settings], broker: PaperBroker, clock: Clock, fake_smtp: Any
) -> Callable[..., AlertManager]:
    def factory(**overrides: Any) -> AlertManager:
        settings = make_settings(**{**EMAIL_CONFIG, **overrides})
        return AlertManager(EmailNotifier(settings), broker, settings, clock=clock)

    return factory


# ============================================================================== EmailNotifier


def test_notifier_disabled_without_configuration(
    settings: Settings, fake_smtp: Any, caplog: pytest.LogCaptureFixture
) -> None:
    notifier = EmailNotifier(settings)
    with caplog.at_level(logging.INFO, logger="trading_swarm.alerts.email"):
        assert notifier.send("subject", "body") is False
    assert "email alerts disabled" in caplog.text
    assert fake_smtp.instances == []
    result = notifier.deliver("subject", "body")
    assert result.status is AlertStatus.DISABLED and not result.sent
    assert "SMTP_HOST" in result.detail and "ALERT_EMAIL_TO" in result.detail
    assert notifier.enabled is False


def test_notifier_sends_multipart_over_starttls_with_login(
    email_settings: Settings, fake_smtp: Any
) -> None:
    notifier = EmailNotifier(email_settings)
    assert notifier.send("BUY BULL", "plain body", "<p>html body</p>") is True

    (conn,) = fake_smtp.instances
    assert (conn.host, conn.port, conn.timeout) == ("smtp.test.invalid", 587, 15.0)
    assert conn.ssl is False and conn.started_tls is True
    assert conn.logged_in_as == "bot@example.com"
    assert conn.closed is True

    (message,) = fake_smtp.sent
    assert message.get_content_type() == "multipart/alternative"
    assert message["Subject"] == "BUY BULL"
    assert message["From"] == "MT Swarm <alerts@example.com>"
    assert "ops@example.com" in message["To"] and "pm@example.com" in message["To"]
    assert message["Message-ID"].endswith("@example.com>")  # no DNS lookup for the domain
    assert message["Auto-Submitted"] == "auto-generated"
    assert message["Date"]
    text, html = bodies(message)
    assert text.strip() == "plain body"
    assert html is not None and "<p>html body</p>" in html


def test_notifier_uses_implicit_tls_when_ssl_enabled(
    make_settings: Callable[..., Settings], fake_smtp: Any
) -> None:
    settings = make_settings(**{**EMAIL_CONFIG, "smtp_use_ssl": True, "smtp_port": 465})
    assert settings.smtp_use_tls is False  # SSL wins over STARTTLS
    assert EmailNotifier(settings).send("s", "t") is True
    (conn,) = fake_smtp.instances
    assert conn.ssl is True and conn.started_tls is False and conn.port == 465
    assert EmailNotifier(settings).describe()["security"] == "ssl"


def test_notifier_plain_relay_without_auth_or_tls(
    make_settings: Callable[..., Settings], fake_smtp: Any
) -> None:
    settings = make_settings(
        **{**EMAIL_CONFIG, "smtp_use_tls": False, "smtp_username": "", "smtp_port": 25}
    )
    assert EmailNotifier(settings).send("s", "text only") is True
    (conn,) = fake_smtp.instances
    assert conn.started_tls is False and conn.logged_in_as is None
    (message,) = fake_smtp.sent
    assert message.get_content_type() == "text/plain"


def test_notifier_sanitises_subject_header_injection(
    email_settings: Settings, fake_smtp: Any
) -> None:
    assert EmailNotifier(email_settings).send("Hello\r\nBcc: evil@example.com", "t") is True
    (message,) = fake_smtp.sent
    assert message["Bcc"] is None
    assert message["Subject"] == "Hello Bcc: evil@example.com"
    assert clean_header("x" * 300).endswith("…") and len(clean_header("x" * 300)) == 200


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            smtplib.SMTPAuthenticationError(535, b"5.7.8 Username and Password not accepted"),
            "app password",
        ),
        (
            smtplib.SMTPRecipientsRefused({"ops@example.com": (550, b"no such user")}),
            "recipients refused",
        ),
        (smtplib.SMTPSenderRefused(553, b"not allowed", "alerts@example.com"), "sender"),
        (smtplib.SMTPServerDisconnected("Connection unexpectedly closed"), "SMTP error"),
        (TimeoutError("timed out"), "could not connect"),
        (RuntimeError("boom"), "unexpected error"),
    ],
)
def test_notifier_never_raises_on_delivery_errors(
    email_settings: Settings,
    fake_smtp: Any,
    caplog: pytest.LogCaptureFixture,
    error: BaseException,
    expected: str,
) -> None:
    fake_smtp.fail_with = error
    notifier = EmailNotifier(email_settings)
    with caplog.at_level(logging.DEBUG):
        assert notifier.send("s", "t") is False
        result = notifier.deliver("s", "t")
    assert result.status is AlertStatus.FAILED
    assert expected in result.detail
    assert PASSWORD not in caplog.text and PASSWORD not in result.detail


def test_notifier_reports_connection_failures(
    email_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise ConnectionRefusedError(111, "Connection refused")

    monkeypatch.setattr(smtplib, "SMTP", refuse)
    result = EmailNotifier(email_settings).deliver("s", "t")
    assert result.status is AlertStatus.FAILED
    assert "could not connect to smtp.test.invalid:587" in result.detail


def test_notifier_describe_contains_no_secrets(email_settings: Settings) -> None:
    info = EmailNotifier(email_settings).describe()
    assert info["enabled"] is True and info["security"] == "starttls"
    assert info["recipients"] == 2 and info["missing"] == []
    assert PASSWORD not in repr(info)
    assert isinstance(EmailNotifier(email_settings), Notifier)


# ============================================================================== templates


def test_format_helpers() -> None:
    assert fmt_money(1234.5) == "$1,234.50"
    assert fmt_money(-12.5) == "-$12.50"
    assert fmt_money(7.0, signed=True) == "+$7.00"
    assert fmt_money(None) == fmt_money(math.nan) == "—"
    assert fmt_price(0.12345) == "$0.1235" and fmt_price(187.2) == "$187.20"
    assert fmt_signed(0.62) == "+0.62" and fmt_signed(-0.1) == "-0.10"
    assert fmt_pct(5.123) == "+5.12%" and fmt_pct(5.0, signed=False) == "5.00%"
    assert fmt_confidence(0.812) == "81%"


def test_decision_email_contents_and_escaping(email_settings: Settings) -> None:
    decision = make_decision(rationale="<script>alert('x')</script> & MACD > signal")
    email = render_decision(decision, email_settings)

    assert email.subject == "[MT Swarm] BUY BULL @ $100.00: score +0.62, confidence 80%"
    # Untrusted agent text is escaped in HTML, but readable in the plain-text part.
    assert "<script>" not in email.html
    assert "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt; &amp; MACD &gt; signal" in email.html
    assert "<script>alert('x')</script>" in email.text
    # Agent table, risk numbers (stop/target distances, R multiple) and dashboard links.
    for fragment in ("trend", "sentiment (lexicon)", "+0.71", "85%", "37 shares"):
        assert fragment in email.text and fragment in email.html
    assert "$95.00 (-5.00%)" in email.text and "$110.00 (+10.00%)" in email.text
    assert "2.0R" in email.text and "Capital at risk" in email.text
    assert "3/3 passed" in email.text
    assert "https://swarm.example.com/dash?page=analyze&amp;symbol=BULL" in email.html
    assert "https://swarm.example.com/dash?page=analyze&symbol=BULL" in email.text
    assert "not financial advice" in email.text
    assert_well_formed(email.html)


def test_dashboard_link_rejects_unsafe_urls(make_settings: Callable[..., Settings]) -> None:
    assert dashboard_link(make_settings(dashboard_url="javascript:alert(1)")) is None
    assert dashboard_link(make_settings(dashboard_url="/relative")) is None
    assert dashboard_link(make_settings(dashboard_url="")) is None
    assert dashboard_link(make_settings(dashboard_url="http://localhost:8501")) == (
        "http://localhost:8501/"
    )
    assert dashboard_link(make_settings(dashboard_url="http://h/x?a=1"), symbol="AAPL") == (
        "http://h/x?a=1&symbol=AAPL"
    )
    email = render_decision(make_decision(), make_settings(dashboard_url="javascript:alert(1)"))
    assert "javascript:" not in email.html and "href" not in email.html


def test_all_templates_render_well_formed_html(email_settings: Settings) -> None:
    filled = Order(
        symbol="BULL",
        side=Side.SELL,
        quantity=10,
        status=OrderStatus.FILLED,
        reference_price=94.0,
        fill_price=93.95,
        commission=1.0,
        slippage=0.47,
        realized_pnl=-73.0,
        reason=ExitReason.STOP_LOSS.value,
        filled_at=T0,
    )
    rejected = Order(
        symbol="BULL",
        side=Side.BUY,
        quantity=10,
        status=OrderStatus.REJECTED,
        reference_price=100.0,
        reject_reason="insufficient cash",
    )
    rendered = [
        render_decision(make_decision(Action.SELL, order=filled), email_settings),
        render_order(filled, email_settings, make_decision()),
        render_order(rejected, email_settings),
        render_exit(filled, email_settings, ExitReason.STOP_LOSS),
        render_exit(filled, email_settings, None),
        render_drawdown(make_snapshot(25.0), email_settings),
        render_price_move("BULL", -6.25, email_settings, price=93.75),
        render_test(email_settings, EmailNotifier(email_settings).describe()),
    ]
    for email in rendered:
        assert email.subject.startswith("[MT Swarm] ")
        assert "\n" not in email.subject
        assert email.text.strip() and "Paper trading only" in email.text
        assert_well_formed(email.html)
    assert rendered[2].subject == "[MT Swarm] Order rejected: BUY 10 BULL: insufficient cash"
    assert rendered[3].subject == "[MT Swarm] Stop-loss hit: sold 10 BULL @ $93.95 (P&L -$73.00)"
    assert "Protective exit" in rendered[4].subject


def test_drawdown_email_explains_recovery_level(email_settings: Settings) -> None:
    email = render_drawdown(make_snapshot(25.0), email_settings)
    assert email.subject == ("[MT Swarm] Drawdown 25.0% breached the 20.0% limit: new BUYs halted")
    # BUYs resume above peak × (1 − 20 %) = 80,000, i.e. +6.67 % from 75,000.
    assert "$80,000.00 (+6.67% from here)" in email.text
    assert "BEAR" in email.text and "-$2,000.00 (-10.00%)" in email.text


# ============================================================================== AlertManager


def test_confident_signal_is_sent_and_logged(
    manager: AlertManager, broker: PaperBroker, fake_smtp: Any
) -> None:
    (record,) = manager.on_decision(make_decision())

    assert record.id is not None
    assert (record.rule, record.symbol, record.channel) == (RULE_SIGNAL, "BULL", "email")
    assert record.status is AlertStatus.SENT
    assert record.timestamp == T0
    facts = parse_detail(record.detail)
    assert facts["action"] == "BUY" and facts["score"] == "+0.620"
    assert facts["outcome"].startswith("delivered to 2/2 recipient(s)")
    assert broker.alerts() == [record]

    (message,) = fake_smtp.sent
    assert message["Subject"] == record.subject
    text, html = bodies(message)
    assert "Agent votes" in text and html is not None and "Risk gate" in html


@pytest.mark.parametrize(
    ("action", "confidence", "fires"),
    [
        (Action.BUY, 0.59, False),
        (Action.BUY, 0.60, True),  # threshold is inclusive
        (Action.SELL, 0.95, True),
        (Action.HOLD, 0.99, False),
    ],
)
def test_signal_rule_threshold(
    manager: AlertManager, fake_smtp: Any, action: Action, confidence: float, fires: bool
) -> None:
    records = manager.on_decision(make_decision(action, confidence))
    assert [r.rule for r in records] == ([RULE_SIGNAL] if fires else [])
    assert len(fake_smtp.sent) == int(fires)


def test_cooldown_suppresses_repeats_until_it_expires(
    manager: AlertManager, broker: PaperBroker, clock: Clock, fake_smtp: Any
) -> None:
    decision = make_decision()
    assert manager.on_decision(decision)[0].status is AlertStatus.SENT

    clock.advance(minutes=59)
    (suppressed,) = manager.on_decision(decision)
    assert suppressed.status is AlertStatus.SUPPRESSED
    assert "cooldown: last sent 59 min ago" in parse_detail(suppressed.detail)["outcome"]

    clock.advance(minutes=1)  # exactly ALERT_COOLDOWN_MINUTES later → allowed again
    assert manager.on_decision(decision)[0].status is AlertStatus.SENT
    assert len(fake_smtp.sent) == 2
    assert [a.status for a in broker.alerts(rule=RULE_SIGNAL)] == [
        AlertStatus.SENT,
        AlertStatus.SUPPRESSED,
        AlertStatus.SENT,
    ]


def test_cooldown_is_per_symbol_and_reversals_are_not_suppressed(
    manager: AlertManager, clock: Clock, fake_smtp: Any
) -> None:
    assert manager.on_decision(make_decision(Action.BUY))[0].status is AlertStatus.SENT
    clock.advance(minutes=5)
    other = manager.on_decision(make_decision(Action.BUY, symbol="FLAT"))[0]
    assert other.status is AlertStatus.SENT  # different symbol, own cooldown
    clock.advance(minutes=5)
    reversal = manager.on_decision(make_decision(Action.SELL))[0]
    assert reversal.status is AlertStatus.SENT  # BUY → SELL is news, never suppressed
    clock.advance(minutes=5)
    assert manager.on_decision(make_decision(Action.SELL))[0].status is AlertStatus.SUPPRESSED
    assert len(fake_smtp.sent) == 3


def test_zero_cooldown_disables_dedupe(
    make_manager: Callable[..., AlertManager], fake_smtp: Any
) -> None:
    manager = make_manager(alert_cooldown_minutes=0)
    statuses = [manager.on_decision(make_decision())[0].status for _ in range(3)]
    assert statuses == [AlertStatus.SENT] * 3 and len(fake_smtp.sent) == 3


def test_disabled_channel_is_logged_without_starting_a_cooldown(
    settings: Settings, broker: PaperBroker, clock: Clock, fake_smtp: Any
) -> None:
    manager = AlertManager(EmailNotifier(settings), broker, settings, clock=clock)
    first = manager.on_decision(make_decision())[0]
    second = manager.on_decision(make_decision())[0]
    assert first.status is second.status is AlertStatus.DISABLED
    assert "email alerts disabled" in first.detail
    assert fake_smtp.instances == []
    assert len(broker.alerts(rule=RULE_SIGNAL)) == 2


def test_failed_delivery_is_retried_on_the_next_evaluation(
    manager: AlertManager, clock: Clock, fake_smtp: Any
) -> None:
    fake_smtp.fail_with = smtplib.SMTPServerDisconnected("gone")
    assert manager.on_decision(make_decision())[0].status is AlertStatus.FAILED
    fake_smtp.fail_with = None
    clock.advance(minutes=1)
    assert manager.on_decision(make_decision())[0].status is AlertStatus.SENT


def test_executed_order_triggers_signal_and_order_alerts(
    manager: AlertManager, broker: PaperBroker, fake_smtp: Any
) -> None:
    order = broker.submit_order("BULL", Side.BUY, 10, 100.0, "auto-trade", 95.0, 110.0)
    assert order.is_filled
    decision = make_decision(order=order)

    signal, order_alert = manager.on_decision(decision)
    assert (signal.rule, order_alert.rule) == (RULE_SIGNAL, RULE_ORDER)
    assert order_alert.status is AlertStatus.SENT
    assert order_alert.subject == "[MT Swarm] Order filled: BUY 10 BULL @ $100.05"
    assert parse_detail(order_alert.detail)["order_id"] == order.id
    text, _ = bodies(fake_smtp.sent[1])
    assert "Commission" in text and "Stop-loss attached" in text and "$95.00" in text

    # The same order is never reported twice; the signal falls under its cooldown.
    again = manager.on_decision(decision)
    assert [r.status for r in again] == [AlertStatus.SUPPRESSED, AlertStatus.SUPPRESSED]
    assert "already alerted" in again[1].detail
    assert len(fake_smtp.sent) == 2


def test_low_confidence_trade_and_rejections_still_alert_the_order(
    manager: AlertManager, broker: PaperBroker, fake_smtp: Any
) -> None:
    filled = broker.submit_order("FLAT", Side.BUY, 5, 50.0, "auto-trade")
    (only_order,) = manager.on_decision(make_decision(confidence=0.3, symbol="FLAT", order=filled))
    assert only_order.rule == RULE_ORDER and only_order.status is AlertStatus.SENT

    rejected = broker.submit_order("BEAR", Side.SELL, 5, 50.0, "auto-trade")  # no position
    assert rejected.status is OrderStatus.REJECTED
    (alert,) = manager.on_decision(make_decision(Action.SELL, 0.3, symbol="BEAR", order=rejected))
    assert alert.rule == RULE_ORDER and "rejected" in alert.subject.lower()


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("stop_loss", ExitReason.STOP_LOSS),
        ("take_profit", ExitReason.TAKE_PROFIT),
        ("exit take_profit: 110.20 >= 110.00", ExitReason.TAKE_PROFIT),
        ("stop_loss (gap at open)", ExitReason.STOP_LOSS),
        ("manual sell", None),
        ("stop_lossy", None),
    ],
)
def test_exit_reason_parsing(reason: str, expected: ExitReason | None) -> None:
    order = Order(symbol="BULL", side=Side.SELL, quantity=1, reason=reason)
    assert exit_reason_of(order) is expected


def test_exit_alerts_report_realized_pnl_once(
    manager: AlertManager, broker: PaperBroker, fake_smtp: Any
) -> None:
    broker.submit_order("BULL", Side.BUY, 10, 100.0, "entry", 95.0, 110.0)
    exit_order = broker.submit_order("BULL", Side.SELL, 10, 94.0, ExitReason.STOP_LOSS.value)
    assert exit_order.realized_pnl is not None and exit_order.realized_pnl < 0

    (record,) = manager.on_exit(exit_order)
    assert record.rule == RULE_EXIT and record.status is AlertStatus.SENT
    assert record.subject.startswith("[MT Swarm] Stop-loss hit: sold 10 BULL @ $93.95 (P&L -$")
    assert parse_detail(record.detail)["exit"] == "stop_loss"
    text, html = bodies(fake_smtp.sent[0])
    assert "Realized P&L" in text and html is not None and "stop-loss" in html

    assert manager.on_exit(exit_order)[0].status is AlertStatus.SUPPRESSED
    assert len(fake_smtp.sent) == 1


def test_take_profit_exit_and_ignored_orders(
    manager: AlertManager, broker: PaperBroker, fake_smtp: Any
) -> None:
    broker.submit_order("BULL", Side.BUY, 10, 100.0, "entry", 95.0, 110.0)
    tp = broker.submit_order("BULL", Side.SELL, 10, 111.0, ExitReason.TAKE_PROFIT.value)
    (record,) = manager.on_exit(tp)
    assert record.subject.startswith("[MT Swarm] Take-profit hit")

    rejected = broker.submit_order("BULL", Side.SELL, 10, 111.0, ExitReason.STOP_LOSS.value)
    assert rejected.status is OrderStatus.REJECTED
    assert manager.on_exit(rejected) == []
    buy = broker.submit_order("FLAT", Side.BUY, 1, 10.0, "entry")
    assert manager.on_exit(buy) == []
    assert len(fake_smtp.sent) == 1


def test_drawdown_rule_compares_fractions_and_dedupes(
    manager: AlertManager, broker: PaperBroker, clock: Clock, fake_smtp: Any
) -> None:
    assert manager.on_drawdown(make_snapshot(19.9)) == []
    (record,) = manager.on_drawdown(make_snapshot(20.0))  # at the limit counts as a breach
    assert (record.rule, record.symbol, record.status) == (RULE_DRAWDOWN, None, AlertStatus.SENT)
    assert parse_detail(record.detail)["limit_pct"] == "20.00"
    clock.advance(minutes=30)
    assert manager.on_drawdown(make_snapshot(24.0))[0].status is AlertStatus.SUPPRESSED
    clock.advance(minutes=31)
    assert manager.on_drawdown(make_snapshot(24.0))[0].status is AlertStatus.SENT
    assert len(fake_smtp.sent) == 2
    assert len(broker.alerts(rule=RULE_DRAWDOWN)) == 3


def test_price_move_rule(manager: AlertManager, clock: Clock, fake_smtp: Any) -> None:
    assert manager.on_price_move("BULL", 4.99) == []
    assert manager.on_price_move("BULL", math.nan) == []
    assert manager.on_price_move("BULL", None) == []

    (up,) = manager.on_price_move("bull", 6.2, price=106.2)
    assert (up.rule, up.symbol, up.status) == (RULE_PRICE_MOVE, "BULL", AlertStatus.SENT)
    assert up.subject == "[MT Swarm] BULL up 6.20% on the day ($106.20)"
    clock.advance(minutes=10)
    assert manager.on_price_move("BULL", 7.0)[0].status is AlertStatus.SUPPRESSED
    (down,) = manager.on_price_move("BULL", -5.0)  # direction flip within the cooldown
    assert down.status is AlertStatus.SENT and "down 5.00%" in down.subject
    assert len(fake_smtp.sent) == 2


def test_send_test_outcomes(
    manager: AlertManager,
    settings: Settings,
    broker: PaperBroker,
    fake_smtp: Any,
) -> None:
    sent = manager.send_test()
    assert (sent.rule, sent.status, sent.channel) == (RULE_TEST, AlertStatus.SENT, "email")
    assert sent.detail.startswith("delivered to 2/2 recipient(s) via smtp.test.invalid:587")
    text, _ = bodies(fake_smtp.sent[0])
    assert "smtp.test.invalid:587" in text and PASSWORD not in text

    fake_smtp.fail_with = smtplib.SMTPAuthenticationError(535, b"bad credentials")
    failed = manager.send_test()
    assert failed.status is AlertStatus.FAILED and "authentication failed" in failed.detail

    disabled = AlertManager(EmailNotifier(settings), broker, settings).send_test()
    assert disabled.status is AlertStatus.DISABLED
    assert disabled.detail == (
        "email alerts disabled: set SMTP_HOST, ALERT_EMAIL_FROM, ALERT_EMAIL_TO"
    )
    assert [a.status for a in broker.alerts(rule=RULE_TEST)] == [
        AlertStatus.DISABLED,
        AlertStatus.FAILED,
        AlertStatus.SENT,
    ]


def test_manager_survives_storage_render_and_notifier_failures(
    manager: AlertManager,
    broker: PaperBroker,
    fake_smtp: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken_log(record: Any) -> Any:
        raise RuntimeError("database is locked")

    with monkeypatch.context() as patch:
        patch.setattr(broker, "log_alert", broken_log)
        (record,) = manager.on_decision(make_decision())
    assert record.status is AlertStatus.SENT and record.id is None  # e-mail still went out
    assert "could not persist" in caplog.text

    def broken_template(*args: Any, **kwargs: Any) -> Any:
        raise KeyError("template bug")

    monkeypatch.setattr(manager_module, "render_price_move", broken_template)
    (fallback,) = manager.on_price_move("BULL", 9.0)
    assert fallback.status is AlertStatus.SENT
    assert fallback.subject == "[MT Swarm] price move alert: BULL"
    assert "change_pct: +9.00" in bodies(fake_smtp.sent[-1])[0]

    class ExplodingNotifier:
        channel = "email"
        enabled = True

        def deliver(self, subject: str, text: str, html: str | None = None) -> DeliveryResult:
            raise OSError("socket closed")

        def send(self, subject: str, text: str, html: str | None = None) -> bool:
            raise OSError("socket closed")

    exploding = AlertManager(ExplodingNotifier(), broker, manager.settings)
    (failed,) = exploding.on_drawdown(make_snapshot(30.0))
    assert failed.status is AlertStatus.FAILED and "notifier error: OSError" in failed.detail


def test_rule_methods_never_raise(manager: AlertManager, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(manager, "_dispatch", boom)
    assert manager.on_decision(make_decision()) == []
    assert manager.on_drawdown(make_snapshot(50.0)) == []
    assert manager.on_price_move("BULL", 50.0) == []


def test_manager_accepts_a_send_only_notifier(broker: PaperBroker, settings: Settings) -> None:
    class SendOnly:
        channel = "webhook"

        def __init__(self, enabled: bool, ok: bool) -> None:
            self.enabled, self.ok, self.calls = enabled, ok, 0

        def send(self, subject: str, text: str, html: str | None = None) -> bool:
            self.calls += 1
            return self.ok

    ok = AlertManager(SendOnly(True, True), broker, settings)  # type: ignore[arg-type]
    assert ok.on_price_move("BULL", 8.0)[0].status is AlertStatus.SENT
    assert ok.on_price_move("BEAR", -8.0)[0].channel == "webhook"
    off = AlertManager(SendOnly(False, True), broker, settings)  # type: ignore[arg-type]
    assert off.on_price_move("FLAT", 8.0)[0].status is AlertStatus.DISABLED
    bad = AlertManager(SendOnly(True, False), broker, settings)  # type: ignore[arg-type]
    assert bad.send_test().status is AlertStatus.FAILED


def test_concurrent_evaluations_send_exactly_once(
    manager: AlertManager, broker: PaperBroker, fake_smtp: Any
) -> None:
    decision = make_decision()
    barrier = threading.Barrier(8)

    def evaluate(_: int) -> AlertStatus:
        barrier.wait()
        return manager.on_decision(decision)[0].status

    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(evaluate, range(8)))
    assert statuses.count(AlertStatus.SENT) == 1
    assert statuses.count(AlertStatus.SUPPRESSED) == 7
    assert len(fake_smtp.sent) == 1
    assert len(broker.alerts(limit=None)) == 8


def test_detail_round_trip_and_config_summary(manager: AlertManager, broker: PaperBroker) -> None:
    detail = format_detail(
        {"action": "BUY", "skip": None, "note": "a\nb", "outcome": "delivered 1/2; refused: x@y"}
    )
    assert detail == "action=BUY; note=a b; outcome=delivered 1/2; refused: x@y"
    assert parse_detail(detail) == {
        "action": "BUY",
        "note": "a b",
        "outcome": "delivered 1/2; refused: x@y",
    }
    summary = manager.config_summary()
    assert summary["enabled"] is True and summary["cooldown_minutes"] == 60
    assert summary["drawdown_halt_pct"] == pytest.approx(20.0)
    assert PASSWORD not in repr(summary)
    assert isinstance(broker, AlertLog)
