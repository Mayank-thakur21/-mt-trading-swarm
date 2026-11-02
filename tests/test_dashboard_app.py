"""Headless tests of the Streamlit dashboard (``streamlit.testing.v1.AppTest``).

``requests.Session.request`` is patched with an in-process fake API that serves payloads
produced by the real engine (coordinator + paper broker + synthetic data), so every page is
rendered against the true response shapes without any network traffic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from trading_swarm import indicators as ind
from trading_swarm.analytics import compute_metrics
from trading_swarm.llm.null_client import NullLLMClient
from trading_swarm.models import AlertRecord, AlertStatus
from trading_swarm.swarm import SwarmCoordinator

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
API_URL = "http://api.test:8000"
TIMEOUT = 60


# =============================================================================== fake API


def _response(status: int, body: Any, headers: dict[str, str] | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = b"" if body is None else json.dumps(body).encode()
    response.headers.update({"Content-Type": "application/json", **(headers or {})})
    response.url = API_URL
    return response


class FakeApi:
    """Routes ``(method, path)`` to canned payloads; records every call.

    A route value may be a JSON payload (→ 200), ``(status, body[, headers])``, an
    exception instance (raised, like a transport failure) or a callable returning either.
    Paths ending in ``/`` match as prefixes (``/quote/`` serves every symbol).
    """

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = dict(routes)
        self.calls: list[dict[str, Any]] = []

    def handler(self, _session: requests.Session, method: str, url: str, **kwargs: Any):
        path = urlsplit(url).path
        self.calls.append({"method": method, "path": path, "url": url, **kwargs})
        route = self.routes.get((method, path))
        if route is None:
            route = next(
                (
                    v
                    for (m, p), v in self.routes.items()
                    if m == method and p.endswith("/") and path.startswith(p)
                ),
                None,
            )
        if route is None:
            return _response(404, {"error": {"code": "not_found", "message": f"no route {path}"}})
        if callable(route) and not isinstance(route, BaseException):
            route = route(method, path, kwargs)
        if isinstance(route, BaseException):
            raise route
        if isinstance(route, tuple):
            return _response(*route)
        return _response(200, route)

    def called(self, method: str, path: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["method"] == method and c["path"] == path]


@pytest.fixture
def engine_payloads(settings, provider, broker) -> dict[str, Any]:
    """Real API-shaped payloads from the engine."""
    coordinator = SwarmCoordinator(settings, provider, broker, llm=NullLLMClient())
    buy = coordinator.analyze("BULL", auto_trade=True)
    scan = coordinator.scan_watchlist()
    broker.log_alert(
        AlertRecord(
            rule="signal",
            symbol="BULL",
            status=AlertStatus.DISABLED,
            subject="BUY BULL",
            detail="email alerts disabled: set SMTP_HOST",
        )
    )
    curve = broker.equity_curve()
    df = provider.history("BULL", period="6mo")
    macd = ind.macd(df["Close"])

    def clean(series: pd.Series) -> list[float | None]:
        return [None if not np.isfinite(v) else float(v) for v in series]

    bt_index = pd.bdate_range("2024-01-02", periods=60, tz="UTC")
    bt_equity = pd.Series(np.linspace(100_000, 112_000, 60), index=bt_index)
    bt_bench = pd.Series(np.linspace(100_000, 118_000, 60), index=bt_index)
    trades = [t.model_dump(mode="json") for t in broker.trades()]
    return {
        "health": {
            "status": "ok",
            "version": "2.0.0",
            "data_provider": "synthetic",
            "llm_provider": "none",
            "llm_model": "",
            "email_enabled": False,
            "scheduler_enabled": False,
            "auto_trade": False,
        },
        "status": {
            "scheduler": {"enabled": False, "running": False, "interval_seconds": 3600.0},
            "alerts": {"channel": "email", "enabled": False, "min_confidence": 0.6},
        },
        "decision": buy.model_dump(mode="json"),
        "scan": {
            "decisions": [d.model_dump(mode="json") for d in scan.decisions],
            "exits": [o.model_dump(mode="json") for o in scan.exits],
            "errors": [e.model_dump(mode="json") for e in scan.errors],
        },
        "portfolio": coordinator.portfolio().model_dump(mode="json"),
        "orders": [o.model_dump(mode="json") for o in broker.orders()],
        "trades": trades,
        "decisions": [d.model_dump(mode="json") for d in broker.decisions()],
        "alerts": [a.model_dump(mode="json") for a in broker.alerts()],
        "equity": {
            "points": [
                {"t": t.isoformat(), "equity": r.equity, "cash": r.cash}
                for t, r in curve.iterrows()
            ],
            "metrics": compute_metrics(curve, broker.trades(limit=None)).model_dump(mode="json"),
        },
        "history": {
            "symbol": "BULL",
            "candles": [
                {
                    "t": t.isoformat(),
                    "o": r.Open,
                    "h": r.High,
                    "l": r.Low,
                    "c": r.Close,
                    "v": r.Volume,
                }
                for t, r in df.iterrows()
            ],
            "indicators": {
                "sma20": clean(ind.sma(df["Close"], 20)),
                "sma50": clean(ind.sma(df["Close"], 50)),
                "rsi": clean(ind.rsi(df["Close"])),
                "macd": clean(macd["macd"]),
                "signal": clean(macd["signal"]),
            },
        },
        "quote": {
            "symbol": "BULL",
            "price": 182.08,
            "change_pct_1d": 0.61,
            "as_of": "2026-09-14T00:00:00Z",
        },
        "backtest": {
            "symbols": ["BULL"],
            "start": "2024-01-02",
            "end": "2024-03-25",
            "initial_capital": 100_000.0,
            "parameters": {"warmup": 200, "use_sentiment": False},
            "equity_curve": [
                {"t": t.isoformat(), "equity": e, "benchmark": b}
                for t, e, b in zip(bt_index, bt_equity, bt_bench, strict=True)
            ],
            "trades": trades,
            "metrics": compute_metrics(bt_equity, trades).model_dump(mode="json"),
            "benchmark_metrics": compute_metrics(bt_bench).model_dump(mode="json"),
        },
    }


@pytest.fixture
def routes(engine_payloads) -> dict[tuple[str, str], Any]:
    p = engine_payloads
    return {
        ("GET", "/health"): p["health"],
        ("GET", "/status"): p["status"],
        ("GET", "/portfolio"): p["portfolio"],
        ("GET", "/equity"): p["equity"],
        ("GET", "/watchlist"): {"symbols": ["BULL", "BEAR", "FLAT"]},
        ("PUT", "/watchlist"): lambda m, path, kw: {"symbols": kw["json"]["symbols"]},
        ("GET", "/quote/"): p["quote"],
        ("GET", "/history/"): p["history"],
        ("GET", "/orders"): p["orders"],
        ("GET", "/trades"): p["trades"],
        ("GET", "/decisions"): p["decisions"],
        ("GET", "/alerts"): p["alerts"],
        ("POST", "/alerts/test"): {
            "sent": False,
            "channel": "email",
            "detail": "email alerts disabled",
        },
        ("POST", "/analyze"): p["decision"],
        ("POST", "/analyze/batch"): {"results": [p["decision"]], "errors": []},
        ("POST", "/backtest"): p["backtest"],
        ("POST", "/scan"): p["scan"],
        ("POST", "/portfolio/reset"): p["portfolio"],
        ("POST", "/orders"): p["orders"][0],
    }


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch, routes) -> Iterator[FakeApi]:
    api = FakeApi(routes)

    def fake_request(session: requests.Session, method: str, url: str, **kwargs: Any):
        # A plain function (not the bound method) so Python binds the session as `self`.
        return api.handler(session, method, url, **kwargs)

    monkeypatch.setattr(requests.Session, "request", fake_request)
    monkeypatch.setenv("API_URL", API_URL)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("DASHBOARD_ALLOW_URL_OVERRIDE", raising=False)
    st.cache_data.clear()  # the GET cache is process-global: isolate tests
    yield api
    st.cache_data.clear()


def run_app(**query: str) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for key, value in query.items():
        at.query_params[key] = value
    return at.run()


def click(at: AppTest, label: str) -> AppTest:
    button = next(b for b in at.button if b.label == label)
    return button.click().run()


def texts(elements: Any) -> str:
    return "\n".join(str(e.value) for e in elements)


def assert_clean(at: AppTest) -> None:
    """No uncaught exception and no error banners."""
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, texts(at.error)


# =============================================================================== tests


def test_api_down_shows_friendly_offline_page(fake_api: FakeApi) -> None:
    fake_api.routes = {("GET", "/health"): requests.exceptions.ConnectionError("refused")}
    at = run_app()
    assert not at.exception
    assert "not reachable" in texts(at.error)
    assert API_URL in texts(at.error)
    assert "uvicorn trading_swarm.api.main:app" in texts(at.markdown)
    assert any(b.label == "Retry" for b in at.button)
    assert not fake_api.called("GET", "/portfolio")  # no page rendering while offline


@pytest.mark.parametrize(
    "page", ["portfolio", "analyze", "backtest", "trades", "alerts", "watchlist"]
)
def test_every_page_renders_cleanly(fake_api: FakeApi, page: str) -> None:
    at = run_app(page=page)
    assert_clean(at)
    assert at.title, "page has a title"
    assert at.query_params["page"] == page  # kept in the URL for shareable links


def test_portfolio_page_shows_account_numbers(fake_api: FakeApi, engine_payloads) -> None:
    at = run_app(page="portfolio")
    assert_clean(at)
    metrics = {m.label: m.value for m in at.metric}
    portfolio = engine_payloads["portfolio"]
    # Six-figure tiles show whole dollars (so they fit at 1280 px); the tooltip is exact.
    assert metrics["Equity"] == f"${portfolio['equity']:,.0f}"
    assert metrics["Cash"] == f"${portfolio['cash']:,.0f}"
    helps = {m.label: m.proto.help for m in at.metric}
    assert f"{portfolio['equity']:,.2f}" in helps["Equity"]
    assert f"{portfolio['cash']:,.2f}" in helps["Cash"]
    assert metrics["Open positions"] == "1"
    assert metrics["CAGR"] == metrics["Sharpe"] == "—"  # < 30 days: annualising is noise
    assert at.dataframe, "positions table rendered"


def test_analyze_runs_the_swarm_and_escapes_api_text(fake_api: FakeApi) -> None:
    at = run_app(page="analyze", symbol="bull")
    assert at.text_input[0].value == "BULL"  # ?symbol= deep link, normalised
    at = click(at, "Analyze")
    assert_clean(at)
    (call,) = fake_api.called("POST", "/analyze")
    assert call["json"] == {"symbol": "BULL", "auto_trade": False}
    assert "Paper order filled: BUY" in texts(at.success)
    assert "\\$" in texts(at.success)  # "$x … $y" must not render as LaTeX
    markdown = texts(at.markdown)
    assert "\\$1,000 at risk / \\$" in markdown  # risk reasons escaped too
    assert "BULL" in markdown
    labels = {m.label for m in at.metric}
    assert {"Quantity", "Stop", "Take profit", "Reward / risk", "Risk score"} <= labels
    assert fake_api.called("GET", "/history/BULL")


def test_analyze_rejects_bad_ticker_locally(fake_api: FakeApi) -> None:
    at = run_app(page="analyze")
    at.text_input[0].set_value("NOT A TICKER")
    at = click(at, "Analyze")
    assert not at.exception
    assert "not a valid ticker" in texts(at.warning)
    assert not fake_api.called("POST", "/analyze")


def test_analyze_upstream_failure_is_explained(fake_api: FakeApi) -> None:
    fake_api.routes[("POST", "/analyze")] = (
        502,
        {"error": {"code": "data_unavailable", "message": "provider down", "request_id": "r-1"}},
    )
    at = click(run_app(page="analyze", symbol="BULL"), "Analyze")
    assert not at.exception
    message = texts(at.error)
    assert "HTTP 502" in message and "provider down" in message
    assert "market-data or LLM provider" in message
    assert "r-1" in message


def test_auth_error_explains_how_to_fix(fake_api: FakeApi) -> None:
    fake_api.routes[("GET", "/portfolio")] = (
        401,
        {"error": {"code": "unauthorized", "message": "invalid API key", "request_id": "x"}},
    )
    at = run_app(page="portfolio")
    assert not at.exception
    assert "HTTP 401" in texts(at.error)
    assert "API_KEY" in texts(at.error)


def test_rate_limit_shows_retry_after(fake_api: FakeApi) -> None:
    fake_api.routes[("GET", "/trades")] = (
        429,
        {"error": {"code": "rate_limited", "message": "too many requests"}},
        {"Retry-After": "12"},
    )
    at = run_app(page="trades")
    assert not at.exception
    assert "Try again in 12 s" in texts(at.warning)


def test_api_key_from_environment_is_sent(fake_api: FakeApi, monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "k-123")
    run_app(page="portfolio")
    portfolio_calls = fake_api.called("GET", "/portfolio")
    assert portfolio_calls and portfolio_calls[0]["headers"]["X-API-Key"] == "k-123"


def test_get_requests_are_cached(fake_api: FakeApi) -> None:
    at = run_app(page="portfolio")
    at.run()  # a rerun inside the TTL hits the cache
    assert len(fake_api.called("GET", "/portfolio")) == 1


def test_backtest_form_runs_and_renders_results(fake_api: FakeApi) -> None:
    at = click(run_app(page="backtest"), "Run backtest")
    assert_clean(at)
    (call,) = fake_api.called("POST", "/backtest")
    assert call["json"]["symbols"] == ["BULL"]
    assert call["json"]["use_sentiment"] is False
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Strategy return"] == "+12.0%"
    assert metrics["Buy & hold return"] == "+18.0%"
    assert any("Results" in s.value for s in at.subheader)


def test_watchlist_scan_and_save(fake_api: FakeApi) -> None:
    at = click(run_app(page="watchlist"), "Scan now")
    assert_clean(at)
    assert fake_api.called("POST", "/scan")
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Analysed"] == "3"
    at.multiselect[0].set_value(["BULL", "MSFT"])
    at = click(at.run(), "Save")
    assert_clean(at)
    (put,) = fake_api.called("PUT", "/watchlist")
    assert put["json"] == {"symbols": ["BULL", "MSFT"]}
    assert "Watchlist saved: BULL, MSFT." in texts(at.success)


def test_alerts_test_email(fake_api: FakeApi) -> None:
    at = click(run_app(page="alerts"), "Send test e-mail")
    assert not at.exception
    assert fake_api.called("POST", "/alerts/test")
    assert "Test e-mail not sent. email alerts disabled" in texts(at.warning)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Disabled"] == "1"


def test_scan_without_a_recorded_snapshot_says_why(
    fake_api: FakeApi, engine_payloads: dict[str, Any]
) -> None:
    fake_api.routes[("POST", "/scan")] = {**engine_payloads["scan"], "snapshot_recorded": False}
    at = click(run_app(page="watchlist"), "Scan now")
    assert_clean(at)
    assert "No equity point was recorded for this scan" in texts(at.caption)


def test_alerts_page_reads_the_configuration_from_status(fake_api: FakeApi) -> None:
    at = run_app(page="alerts")
    assert_clean(at)
    assert fake_api.called("GET", "/status")
    assert any(j.value and "min_confidence" in str(j.value) for j in at.json)


def test_alerts_page_without_status_still_renders(fake_api: FakeApi) -> None:
    fake_api.routes[("GET", "/status")] = (
        401,
        {"error": {"code": "unauthorized", "message": "missing API key", "request_id": "r"}},
    )
    at = run_app(page="alerts")
    assert_clean(at)  # the optional details are just hidden
    assert not at.json


def test_decision_card_shows_exits_the_analysis_made(
    fake_api: FakeApi, engine_payloads: dict[str, Any]
) -> None:
    """A SELL whose position the analysis' own stop check closed is "closed", not
    "vetoed", and every exit the analysis made is listed."""
    decision = dict(engine_payloads["decision"])
    stop_exit = {
        **engine_payloads["orders"][0],
        "side": "SELL",
        "reason": "stop_loss",
        "status": "FILLED",
        "fill_price": 170.5,
        "realized_pnl": -120.25,
    }
    decision.update(
        action="HOLD",
        score=-0.7,
        executed=False,
        order=None,
        exits=[stop_exit],
        risk={**decision["risk"], "approved": False, "action": "HOLD", "quantity": 0},
    )
    fake_api.routes[("POST", "/analyze")] = decision
    at = click(run_app(page="analyze", symbol="BULL"), "Analyze")
    assert_clean(at)
    markdown = texts(at.markdown)
    assert "Closed by stop-loss" in markdown
    assert "vetoed by risk" not in markdown
    info = texts(at.info)
    assert f"Stop-loss hit before this decision: sold {stop_exit['quantity']} BULL" in info
    assert "120.25" in info  # the realised P&L of the exit


def test_unexpected_page_failure_has_no_traceback(fake_api: FakeApi) -> None:
    fake_api.routes[("GET", "/portfolio")] = {"positions": "not-a-list", "equity": "x"}
    at = run_app(page="portfolio")
    assert not at.exception  # caught by the page guard, logged server-side
    assert "Something went wrong" in texts(at.error)


def test_invalid_page_param_falls_back_to_portfolio(fake_api: FakeApi) -> None:
    at = run_app(page="nonsense", symbol="../etc")
    assert_clean(at)
    assert at.title[0].value == "Portfolio"


# =============================================================================== connection


def run_with_session(**state: Any) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def _keys_sent(api: FakeApi) -> set[tuple[str, str | None]]:
    """``(host, X-API-Key)`` of every request made."""
    return {
        (urlsplit(c["url"]).netloc, (c.get("headers") or {}).get("X-API-Key")) for c in api.calls
    }


def test_env_api_key_is_never_sent_to_a_url_typed_in_the_sidebar(
    fake_api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the server's API_KEY was sent to any URL a visitor typed."""
    monkeypatch.setenv("API_KEY", "server-secret")
    monkeypatch.setenv("DASHBOARD_ALLOW_URL_OVERRIDE", "true")
    at = run_with_session(conn_url="http://attacker.test:9")
    client = at.session_state["_api_client"]
    assert client.base_url == "http://attacker.test:9" and client.api_key is None
    assert ("attacker.test:9", "server-secret") not in _keys_sent(fake_api)
    assert all(key is None for _, key in _keys_sent(fake_api))


def test_typed_key_is_used_for_a_typed_url(
    fake_api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_KEY", "server-secret")
    monkeypatch.setenv("DASHBOARD_ALLOW_URL_OVERRIDE", "1")
    run_with_session(conn_url="http://other.test:8000", conn_key="visitor-key")
    assert _keys_sent(fake_api) == {("other.test:8000", "visitor-key")}


def test_env_api_key_still_goes_to_the_configured_url(
    fake_api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("API_KEY", "server-secret")
    monkeypatch.setenv("DASHBOARD_ALLOW_URL_OVERRIDE", "true")
    run_with_session(conn_url=API_URL + "/")  # the same API, spelled slightly differently
    assert _keys_sent(fake_api) == {("api.test:8000", "server-secret")}


def test_url_override_is_disabled_by_default(
    fake_api: FakeApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No server-side requests to visitor-chosen hosts unless the operator opts in."""
    monkeypatch.setenv("API_KEY", "server-secret")
    at = run_with_session(conn_url="http://attacker.test:9")
    assert not at.exception
    assert at.session_state["_api_client"].base_url == API_URL
    assert _keys_sent(fake_api) == {("api.test:8000", "server-secret")}
    assert not [w for w in at.text_input if w.label == "API URL"]  # no URL field
    assert any("DASHBOARD_ALLOW_URL_OVERRIDE" in str(c.value) for c in at.caption)


# =============================================================================== robustness


@pytest.mark.parametrize("capital", [500.0, 5e11])
def test_reset_form_renders_for_any_capital_the_api_accepts(
    fake_api: FakeApi, engine_payloads, capital: float
) -> None:
    """Regression: capital outside the old [1e3, 1e9] widget range crashed the page."""
    fake_api.routes[("GET", "/portfolio")] = {
        **engine_payloads["portfolio"],
        "initial_capital": capital,
    }
    at = run_app(page="portfolio")
    assert_clean(at)
    (field,) = [n for n in at.number_input if n.label == "Starting capital ($)"]
    assert field.value == capital


def test_watchlist_quote_rate_limit_keeps_the_scan_report(fake_api: FakeApi) -> None:
    """Regression: a 429 on one /quote escaped the page and hid the scan just requested."""
    fake_api.routes[("GET", "/quote/")] = (
        429,
        {"error": {"code": "rate_limited", "message": "too many requests"}},
        {"Retry-After": "7"},
    )
    at = run_app(page="watchlist")
    before = len(fake_api.calls)
    at = click(at, "Scan now")
    assert not at.exception and not at.error, texts(at.error)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Analysed"] == "3"  # the scan report is shown
    assert texts(at.warning).count("Rate limit reached") == 1
    assert "Try again in 7 s" in texts(at.warning)
    quote_calls = [c for c in fake_api.calls[before:] if c["path"].startswith("/quote/")]
    assert len(quote_calls) == 1  # the loop stops instead of hammering the limit
    assert any("3 quote(s) not loaded" in str(c.value) for c in at.caption)


def test_watchlist_unknown_symbol_quote_is_listed_not_fatal(fake_api: FakeApi) -> None:
    def quote(method: str, path: str, kwargs: dict[str, Any]) -> Any:
        if path.endswith("/BEAR"):
            return (404, {"error": {"code": "not_found", "message": "no data for BEAR"}})
        return {"symbol": path.rsplit("/", 1)[-1], "price": 10.0, "change_pct_1d": 0.5}

    fake_api.routes[("GET", "/quote/")] = quote
    at = run_app(page="watchlist")
    assert_clean(at)
    assert any("No quote for BEAR: no data for BEAR" in str(c.value) for c in at.caption)
