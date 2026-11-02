"""Cross-module integration: the pieces built in separate modules agree on their contracts.

* static guard: every constant ``logger.*("…%…", args)`` format string is valid
  %-formatting with a matching argument count. The logging module never raises on a bad
  format; it prints a "--- Logging error ---" traceback to stderr and drops the message,
  so ordinary tests cannot catch it.
* the obsolete root-level draft modules are gone and ``import dashboard`` is the package;
* the API's service container shares the coordinator's trade lock and alert manager;
* coordinator + real :class:`AlertManager` + :class:`EmailNotifier` (fake SMTP): signal and
  order e-mails for an auto-traded BUY, then a stop-loss exit e-mail from
  ``scan_watchlist`` whose ``order.reason`` the alert manager recognises.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import dashboard
from trading_swarm.alerts import RULE_EXIT, RULE_ORDER, RULE_SIGNAL, AlertManager, EmailNotifier
from trading_swarm.alerts.manager import exit_reason_of
from trading_swarm.api.deps import build_services
from trading_swarm.broker import PaperBroker
from trading_swarm.config import Settings
from trading_swarm.data import SyntheticProvider
from trading_swarm.models import Action, AlertStatus, ExitReason, OrderStatus, Side
from trading_swarm.swarm import SwarmCoordinator

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = ("trading_swarm", "dashboard", "scripts")

# =============================================================================== log formats

#: One printf-style conversion: ``%[(key)][flags][width][.precision][length]type``.
_DIRECTIVE = re.compile(
    r"%(\([^)]*\))?[#0\- +]*(\*|\d+)?(?:\.(\*|\d+))?[hlL]?([diouxXeEfFgGcrsa%])"
)
_LOG_METHODS = frozenset({"debug", "info", "warning", "error", "exception", "critical", "log"})


def _count_directives(fmt: str) -> int | None:
    """Arguments consumed by ``fmt``, or ``None`` when it contains an invalid directive."""
    count, pos = 0, 0
    while (start := fmt.find("%", pos)) != -1:
        match = _DIRECTIVE.match(fmt, start)
        if match is None:
            return None
        if match.group(4) != "%":
            count += 1 + (match.group(2) == "*") + (match.group(3) == "*")
        pos = match.end()
    return count


def _log_format_problems(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LOG_METHODS
        ):
            continue
        receiver = node.func.value
        name = receiver.id if isinstance(receiver, ast.Name) else getattr(receiver, "attr", "")
        if "log" not in name.lower():
            continue
        args = node.args[1:] if node.func.attr == "log" else node.args
        if not args or not (isinstance(args[0], ast.Constant) and isinstance(args[0].value, str)):
            continue
        if any(isinstance(arg, ast.Starred) for arg in args[1:]):
            continue
        fmt, given = args[0].value, len(args) - 1
        expected = _count_directives(fmt)
        where = f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
        if expected is None:
            yield f"{where}: invalid %-directive in {fmt!r}"
        elif given == 1 and "%(" in fmt:
            continue  # mapping-style format with one dict argument
        elif expected != given:
            yield f"{where}: {expected} directive(s) but {given} argument(s) in {fmt!r}"


def test_directive_counter() -> None:
    assert _count_directives("plain") == 0
    assert _count_directives("%s and %d, 100%%") == 2
    assert _count_directives("%-8s %+.3f %*d") == 4
    assert _count_directives("created paper account with $%,.2f") is None


def test_log_format_strings_are_valid() -> None:
    problems = [
        problem
        for folder in SOURCE_DIRS
        for path in sorted((REPO_ROOT / folder).rglob("*.py"))
        for problem in _log_format_problems(path)
    ]
    assert problems == []


def test_paper_account_log_messages_render(
    make_settings: Callable[..., Settings], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="trading_swarm.broker")
    paper = PaperBroker(make_settings(initial_capital=100_000.0))
    try:
        paper.reset(50_000.0)
    finally:
        paper.close()
    messages = [record.getMessage() for record in caplog.records]
    assert "created paper account with $100,000.00" in messages
    assert "paper account reset to $50,000.00" in messages


# =============================================================================== layout


def test_legacy_draft_modules_are_gone() -> None:
    for name in ("api.py", "swarm.py", "dashboard.py", "agents"):
        assert not (REPO_ROOT / name).exists(), f"obsolete draft {name} is back"
    assert Path(dashboard.__file__).resolve() == REPO_ROOT / "dashboard" / "__init__.py"


# =============================================================================== API wiring


def test_api_services_share_the_coordinator_lock_and_alerts(
    settings: Settings, provider: SyntheticProvider, fake_llm: Any
) -> None:
    services = build_services(settings, provider=provider, llm=fake_llm)
    try:
        coordinator = services.coordinator
        # Manual orders / resets take the very lock auto-trades hold around risk + fill.
        assert services.trade_lock() is coordinator.trade_lock
        assert coordinator.alerts is services.alerts
        assert services.scheduler.coordinator is coordinator
        assert coordinator.broker is services.broker
        assert coordinator.risk.cost_model is services.broker.cost_model
    finally:
        services.close()


# =============================================================================== alerts e2e


@pytest.fixture
def alert_settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings(
        smtp_host="smtp.example.com",
        alert_email_from="swarm@example.com",
        alert_email_to="trader@example.com",
        alert_min_confidence=0.5,
        auto_trade=True,
        watchlist="FLAT",
    )


@pytest.fixture
def alert_broker(alert_settings: Settings) -> Iterator[PaperBroker]:
    paper = PaperBroker(alert_settings)
    try:
        yield paper
    finally:
        paper.close()


def test_coordinator_alerts_end_to_end(
    alert_settings: Settings,
    alert_broker: PaperBroker,
    provider: SyntheticProvider,
    fake_llm: Any,
    fake_smtp: Any,
) -> None:
    alerts = AlertManager(EmailNotifier(alert_settings), alert_broker, alert_settings)
    coordinator = SwarmCoordinator(
        alert_settings, provider, alert_broker, llm=fake_llm, alerts=alerts
    )

    decision = coordinator.analyze("BULL")
    assert decision.action is Action.BUY and decision.executed
    assert decision.order is not None and decision.order.decision_id == decision.id
    logged = {(record.rule, record.status) for record in alert_broker.alerts(limit=None)}
    assert (RULE_SIGNAL, AlertStatus.SENT) in logged
    assert (RULE_ORDER, AlertStatus.SENT) in logged
    subjects = [message["Subject"] for message in fake_smtp.sent]
    assert any("BUY" in subject and "BULL" in subject for subject in subjects)

    # Push the stop above the market: the next scan must close the position first.
    alert_broker.update_stops("BULL", stop=decision.price * 1.5)
    report = coordinator.scan_watchlist()
    (exit_order,) = report.exits
    assert exit_order.symbol == "BULL" and exit_order.side is Side.SELL
    assert exit_order.status is OrderStatus.FILLED
    assert exit_order.reason == ExitReason.STOP_LOSS.value
    assert exit_reason_of(exit_order) is ExitReason.STOP_LOSS
    assert alert_broker.position("BULL") is None

    (exit_alert,) = alert_broker.alerts(limit=None, rule=RULE_EXIT)
    assert exit_alert.status is AlertStatus.SENT and exit_alert.symbol == "BULL"
    assert "stop" in exit_alert.subject.lower()
    assert report.snapshot is not None and report.snapshot.position("BULL") is None
    assert [d.symbol for d in report.decisions] == ["FLAT"]
