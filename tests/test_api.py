"""REST API tests (FastAPI TestClient; synthetic market data, fake LLM, no network).

Every app is created with the pinned ``synthetic_provider`` fixture (``BULL`` / ``BEAR`` /
``FLAT`` scenarios, ``NOPE`` = unknown ticker, ``DOWN`` = provider outage) and the bullish
``fake_llm``; the lifespan runs inside ``with TestClient(app)`` so the paper broker is
opened and closed exactly as under uvicorn.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from datetime import UTC, date, datetime, time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import delete

from trading_swarm import __version__
from trading_swarm.analytics import compute_metrics
from trading_swarm.api.deps import MAX_CONCURRENT_BACKTESTS, ServiceContainer
from trading_swarm.api.errors import validation_details, validation_message
from trading_swarm.api.main import create_app
from trading_swarm.api.market import INDICATOR_WARMUP_BARS, lookback_period
from trading_swarm.api.middleware import resolve_request_id
from trading_swarm.api.routes import _daily_curve
from trading_swarm.api.schemas import MAX_BATCH_SYMBOLS
from trading_swarm.api.security import ApiKeyVerifier, SlidingWindowRateLimiter, key_fingerprint
from trading_swarm.broker import PaperBroker
from trading_swarm.broker.db import EquitySnapshotRow
from trading_swarm.data import ProviderUnavailableError, SyntheticProvider, daily_change_pct
from trading_swarm.logging_setup import request_id_ctx
from trading_swarm.models import Decision
from trading_swarm.scheduler import trading_days

ClientFactory = Callable[..., TestClient]
_DEFAULT: Any = object()
HEX32 = re.compile(r"^[0-9a-f]{32}$")


# =============================================================================== fixtures


@pytest.fixture
def make_client(
    make_settings: Callable[..., Any], synthetic_provider: SyntheticProvider, fake_llm: Any
) -> Iterator[ClientFactory]:
    """Factory: ``make_client(**settings_overrides, provider=, llm=, broker=)`` → a
    started :class:`TestClient` (lifespan entered; shut down at teardown)."""
    with ExitStack() as stack:

        def factory(
            *,
            provider: Any = _DEFAULT,
            llm: Any = _DEFAULT,
            broker: PaperBroker | None = None,
            raise_server_exceptions: bool = True,
            **overrides: Any,
        ) -> TestClient:
            app = create_app(
                make_settings(**overrides),
                provider=synthetic_provider if provider is _DEFAULT else provider,
                llm=fake_llm if llm is _DEFAULT else llm,
                broker=broker,
                configure_logs=False,
            )
            client = TestClient(app, raise_server_exceptions=raise_server_exceptions)
            stack.enter_context(client)
            return client

        yield factory


@pytest.fixture
def client(make_client: ClientFactory) -> TestClient:
    return make_client()


@pytest.fixture
def auth_client(make_client: ClientFactory) -> TestClient:
    return make_client(api_keys="k1, k2")


def services_of(client: TestClient) -> ServiceContainer:
    app: FastAPI = client.app  # type: ignore[assignment]
    services = app.state.services
    assert isinstance(services, ServiceContainer)
    return services


def assert_error(response: Any, status: int, code: str) -> dict[str, Any]:
    """Check the error envelope and return its ``error`` object."""
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert error["request_id"] == response.headers["X-Request-ID"]
    return error


def buy(client: TestClient, symbol: str = "BULL", quantity: int = 10, **extra: Any) -> Any:
    return client.post(
        "/orders", json={"symbol": symbol, "side": "BUY", "quantity": quantity, **extra}
    )


# =============================================================================== app & wiring


def test_module_level_app_and_lazy_package_exports() -> None:
    import trading_swarm.api as api_pkg
    from trading_swarm.api import main

    assert isinstance(main.app, FastAPI)
    assert api_pkg.app is main.app
    assert api_pkg.create_app is create_app
    with pytest.raises(AttributeError):
        _ = api_pkg.does_not_exist


def test_lifespan_wires_every_service(client: TestClient, settings: Any) -> None:
    services = services_of(client)
    assert services.provider.name == "synthetic"
    assert services.llm.name == "fake"
    assert services.coordinator.alerts is services.alerts
    assert services.coordinator.broker is services.broker
    assert services.coordinator.sentiment.llm is services.llm
    assert services.coordinator.risk.cost_model is services.broker.cost_model
    assert services.coordinator.execution.provider is services.provider
    assert services.scheduler.coordinator is services.coordinator
    assert services.backtester.llm is services.llm
    assert services.alerts.broker is services.broker
    assert not services.scheduler.is_running  # SCHEDULER_ENABLED defaults to false


def test_requests_before_startup_get_503(make_settings: Callable[..., Any]) -> None:
    app = create_app(make_settings(), configure_logs=False)
    client = TestClient(app)  # no `with`: the lifespan never ran
    assert_error(client.get("/portfolio"), 503, "service_unavailable")
    response = client.get("/health")
    assert_error(response, 503, "service_unavailable")
    assert response.headers["Retry-After"] == "5"


def test_shutdown_stops_scheduler_and_closes_owned_broker(
    make_settings: Callable[..., Any], synthetic_provider: SyntheticProvider, fake_llm: Any
) -> None:
    settings = make_settings(scheduler_enabled=True, watchlist="FLAT")
    app = create_app(settings, provider=synthetic_provider, llm=fake_llm, configure_logs=False)
    with TestClient(app) as client:
        services = services_of(client)
        assert services.scheduler.is_running
        assert client.get("/status").json()["scheduler"]["running"] is True
    assert app.state.services is None
    assert not services.scheduler.is_running


def test_borrowed_broker_is_left_open(
    make_settings: Callable[..., Any],
    synthetic_provider: SyntheticProvider,
    fake_llm: Any,
    broker: PaperBroker,
) -> None:
    app = create_app(
        make_settings(),
        provider=synthetic_provider,
        llm=fake_llm,
        broker=broker,
        configure_logs=False,
    )
    with TestClient(app) as client:
        assert services_of(client).broker is broker
        assert buy(client).status_code == 201
    # Still usable after the app shut down: the fixture owns (and closes) it.
    assert broker.position("BULL") is not None


# =============================================================================== health


def test_health_reports_configuration(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["data_provider"] == "synthetic"
    assert body["llm_provider"] == "fake"
    assert body["llm_model"] == "fake-model"
    assert body["email_enabled"] is False
    assert body["scheduler_enabled"] is False
    assert body["auto_trade"] is False
    assert body["auth_enabled"] is False
    assert body["buy_threshold"] == 0.25
    assert body["sell_threshold"] == -0.25
    assert body["database"] == "ok"
    assert body["uptime_seconds"] >= 0
    assert "scheduler" not in body and "alerts" not in body  # see GET /status
    assert "X-RateLimit-Limit" not in response.headers  # /health is not rate limited


def _health_view(response: Any) -> dict[str, Any]:
    """The health body without the fields that change between calls."""
    return {k: v for k, v in response.json().items() if k not in {"time", "uptime_seconds"}}


def test_health_is_not_an_api_key_oracle(make_client: ClientFactory) -> None:
    """Regression: /health (public, never rate limited) added scheduler / alert details
    only for a valid key, so anyone could test keys at full speed around the limiter."""
    client = make_client(api_keys="s3cret", rate_limit_per_minute=5)
    anonymous = client.get("/health")
    assert anonymous.status_code == 200 and anonymous.json()["auth_enabled"] is True
    guesses = [client.get("/health", headers={"X-API-Key": f"guess{i}"}) for i in range(20)]
    right = client.get("/health", headers={"X-API-Key": "s3cret"})
    assert {r.status_code for r in [*guesses, right]} == {200}
    assert all(_health_view(r) == _health_view(anonymous) for r in [*guesses, right])
    # Key guessing has to go through a protected endpoint, which is throttled per IP.
    codes = [client.get("/status", headers={"X-API-Key": f"g{i}"}).status_code for i in range(7)]
    assert codes == [401] * 5 + [429] * 2


def test_status_reports_scheduler_and_alerts_to_key_holders(auth_client: TestClient) -> None:
    assert_error(auth_client.get("/status"), 401, "unauthorized")
    response = auth_client.get("/status", headers={"X-API-Key": "k2"})
    assert response.status_code == 200
    body = response.json()
    assert body["scheduler"]["enabled"] is False
    assert body["scheduler"]["interval_seconds"] == 3600.0
    assert body["alerts"]["channel"] == "email"
    assert body["alerts"]["notifier"]["missing"] == [
        "SMTP_HOST",
        "ALERT_EMAIL_FROM",
        "ALERT_EMAIL_TO",
    ]


def test_health_database_probe_is_briefly_cached(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unauthenticated flood of /health calls must not queue on the broker's lock."""
    reads = {"n": 0}
    real = PaperBroker.cash

    def counting(self: PaperBroker) -> float:
        reads["n"] += 1
        return real.fget(self)  # type: ignore[attr-defined,no-any-return]

    monkeypatch.setattr(PaperBroker, "cash", property(counting))
    assert all(client.get("/health").status_code == 200 for _ in range(10))
    assert reads["n"] == 1


def test_health_never_leaks_secrets(make_client: ClientFactory) -> None:
    client = make_client(
        smtp_host="smtp.example.com",
        smtp_password="hunter2-smtp",
        alert_email_from="bot@example.com",
        alert_email_to="me@example.com",
        anthropic_api_key="sk-ant-secret",
    )
    for path in ("/health", "/status"):
        text = client.get(path).text
        assert "hunter2-smtp" not in text
        assert "sk-ant-secret" not in text
    assert client.get("/health").json()["email_enabled"] is True


def test_health_degraded_when_database_fails(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(self: PaperBroker) -> float:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(PaperBroker, "cash", property(broken))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == "error"


def test_root_banner(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["version"] == __version__
    assert body["docs"] == "/docs"


# =============================================================================== auth


def test_auth_disabled_allows_anonymous_access(client: TestClient) -> None:
    assert client.get("/portfolio").status_code == 200


def test_missing_api_key_is_401(auth_client: TestClient) -> None:
    response = auth_client.get("/portfolio")
    error = assert_error(response, 401, "unauthorized")
    assert "missing API key" in error["message"]
    assert "X-API-Key" in response.headers["WWW-Authenticate"]


def test_wrong_api_key_is_401(auth_client: TestClient) -> None:
    response = auth_client.get("/portfolio", headers={"X-API-Key": "k3"})
    error = assert_error(response, 401, "unauthorized")
    assert "invalid API key" in error["message"]
    assert "k3" not in response.text


@pytest.mark.parametrize("key", ["k1", "k2"])
def test_every_configured_key_is_accepted(auth_client: TestClient, key: str) -> None:
    response = auth_client.get("/portfolio", headers={"X-API-Key": key})
    assert response.status_code == 200
    assert response.json()["cash"] == 100_000.0


def test_docs_and_openapi_are_public(auth_client: TestClient) -> None:
    assert auth_client.get("/docs").status_code == 200
    assert auth_client.get("/openapi.json").status_code == 200


def test_api_key_verifier_constant_time_membership() -> None:
    verifier = ApiKeyVerifier(["alpha", "beta"])
    assert verifier.enabled and len(verifier) == 2
    assert verifier.verify("alpha") and verifier.verify("beta")
    assert not verifier.verify("alph")
    assert not verifier.verify("alpha ")
    assert not verifier.verify("")
    assert not verifier.verify(None)
    assert "alpha" not in repr(verifier)
    disabled = ApiKeyVerifier([])
    assert not disabled.enabled
    assert not disabled.verify("anything")


def test_key_fingerprint_is_short_and_stable() -> None:
    assert key_fingerprint("k1") == key_fingerprint("k1")
    assert key_fingerprint("k1") != key_fingerprint("k2")
    assert len(key_fingerprint("k1")) == 12
    assert "k1" not in key_fingerprint("k1")


# =============================================================================== rate limiting


def test_rate_limit_returns_429_with_retry_after(make_client: ClientFactory) -> None:
    client = make_client(rate_limit_per_minute=3)
    remaining = [client.get("/portfolio").headers["X-RateLimit-Remaining"] for _ in range(3)]
    assert remaining == ["2", "1", "0"]

    response = client.get("/portfolio")
    error = assert_error(response, 429, "rate_limited")
    retry_after = int(response.headers["Retry-After"])
    assert 1 <= retry_after <= 60
    assert error["retry_after"] == retry_after
    assert response.headers["X-RateLimit-Limit"] == "3"
    assert response.headers["X-RateLimit-Remaining"] == "0"
    # The public health check is never throttled.
    assert client.get("/health").status_code == 200


def test_rate_limit_is_per_api_key(make_client: ClientFactory) -> None:
    client = make_client(api_keys="k1,k2", rate_limit_per_minute=2)
    for _ in range(2):
        assert client.get("/portfolio", headers={"X-API-Key": "k1"}).status_code == 200
    assert client.get("/portfolio", headers={"X-API-Key": "k1"}).status_code == 429
    assert client.get("/portfolio", headers={"X-API-Key": "k2"}).status_code == 200


def test_wrong_keys_are_throttled_per_ip(make_client: ClientFactory) -> None:
    client = make_client(api_keys="k1", rate_limit_per_minute=2)
    statuses = [
        client.get("/portfolio", headers={"X-API-Key": f"guess-{i}"}).status_code for i in range(3)
    ]
    assert statuses == [401, 401, 429]
    # The real key has its own bucket, so brute-forcing does not lock out the owner.
    assert client.get("/portfolio", headers={"X-API-Key": "k1"}).status_code == 200


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


def test_sliding_window_limiter_math() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(2, 60.0, clock=clock)
    assert limiter.hit("a").allowed
    clock.now += 10
    second = limiter.hit("a")
    assert second.allowed and second.remaining == 0
    clock.now += 5
    denied = limiter.hit("a")
    assert not denied.allowed
    assert denied.retry_after == pytest.approx(45.0)  # first hit expires at t0 + 60
    assert limiter.hit("b").allowed  # independent identity

    clock.now += 45  # the first hit is now exactly one window old → expired
    allowed = limiter.hit("a")
    assert allowed.allowed and allowed.remaining == 0
    assert not limiter.hit("a").allowed  # denied requests were not counted, budget still 2


def test_sliding_window_limiter_evicts_idle_and_excess_identities() -> None:
    clock = FakeClock()
    limiter = SlidingWindowRateLimiter(5, 60.0, clock=clock, max_identities=2)
    for identity in ("a", "b", "c"):
        limiter.hit(identity)
    assert limiter.tracked_identities() == 2
    clock.now += 61
    limiter.hit("d")
    assert limiter.tracked_identities() == 1  # a/b/c went idle
    limiter.reset()
    assert limiter.tracked_identities() == 0


def test_sliding_window_limiter_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(0)
    with pytest.raises(ValueError):
        SlidingWindowRateLimiter(10, 0)


# =============================================================================== request ids & errors


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    assert HEX32.fullmatch(client.get("/health").headers["X-Request-ID"])
    assert HEX32.fullmatch(client.get("/missing").headers["X-Request-ID"])


def test_well_formed_incoming_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/quote/NOPE", headers={"X-Request-ID": "dash-req.42"})
    assert response.headers["X-Request-ID"] == "dash-req.42"
    assert response.json()["error"]["request_id"] == "dash-req.42"


@pytest.mark.parametrize("incoming", ["has spaces", "x" * 65, "-leading-dash", "", "<script>"])
def test_malformed_incoming_request_id_is_replaced(client: TestClient, incoming: str) -> None:
    response = client.get("/health", headers={"X-Request-ID": incoming})
    assert HEX32.fullmatch(response.headers["X-Request-ID"])
    assert HEX32.fullmatch(resolve_request_id(incoming))


def test_request_id_reaches_the_threadpool(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str | None] = []
    coordinator = services_of(client).coordinator
    real_analyze = coordinator.analyze

    def spy(symbol: str, auto_trade: bool | None = None) -> Decision:
        seen.append(request_id_ctx.get())
        return real_analyze(symbol, auto_trade=auto_trade)

    monkeypatch.setattr(coordinator, "analyze", spy)
    response = client.post("/analyze", json={"symbol": "FLAT"}, headers={"X-Request-ID": "trace-1"})
    assert response.status_code == 200
    assert seen == ["trace-1"]
    assert request_id_ctx.get() is None  # reset after the request


def test_access_log_line_per_request(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="trading_swarm.api.access")
    client.get("/portfolio", headers={"X-Request-ID": "log-me"})
    client.get("/health")
    client.get("/quote/DOWN")
    records = [r for r in caplog.records if r.name == "trading_swarm.api.access"]
    by_path = {r.http_path: r for r in records}  # type: ignore[attr-defined]
    portfolio = by_path["/portfolio"]
    assert portfolio.levelno == logging.INFO
    assert portfolio.getMessage().startswith("GET /portfolio -> 200 (")
    assert portfolio.http_status == 200  # type: ignore[attr-defined]
    assert by_path["/health"].levelno == logging.DEBUG  # health checks stay quiet
    assert by_path["/quote/DOWN"].levelno == logging.WARNING  # 5xx stand out


def test_unknown_route_and_method_use_the_envelope(client: TestClient) -> None:
    error = assert_error(client.get("/nope"), 404, "not_found")
    assert "/nope" in error["message"]
    assert_error(client.delete("/portfolio"), 405, "method_not_allowed")


def test_unhandled_exception_becomes_500_envelope(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(*args: Any, **kwargs: Any) -> Decision:
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(services_of(client).coordinator, "analyze", explode)
    response = client.post("/analyze", json={"symbol": "BULL"})
    error = assert_error(response, 500, "internal_error")
    assert "secret internal detail" not in response.text
    assert "request id" in error["message"]


def test_malformed_json_is_422(client: TestClient) -> None:
    response = client.post(
        "/analyze", content=b"{not json", headers={"Content-Type": "application/json"}
    )
    assert_error(response, 422, "validation_error")


def test_validation_error_lists_fields(client: TestClient) -> None:
    response = client.post("/analyze", json={"auto_trade": True})
    error = assert_error(response, 422, "validation_error")
    assert error["message"].startswith("symbol:")
    assert error["details"][0]["field"] == "symbol"
    assert "input" not in error["details"][0]


def test_validation_message_helpers_truncate_and_strip_prefixes() -> None:
    errors = [
        {"loc": ("body", "symbols", i), "msg": "Value error, bad", "type": "v"} for i in range(7)
    ]
    details = validation_details(errors)
    assert details[0] == {"field": "symbols.0", "message": "bad", "type": "v"}
    message = validation_message(details)
    assert message.count("bad") == 5 and message.endswith("(+2 more)")
    assert validation_message([]) == "invalid request"


# =============================================================================== openapi / cors / gzip


def test_openapi_documents_every_endpoint_tags_and_security(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()
    expected = {
        ("get", "/health"),
        ("get", "/quote/{symbol}"),
        ("get", "/history/{symbol}"),
        ("post", "/analyze"),
        ("post", "/analyze/batch"),
        ("get", "/portfolio"),
        ("post", "/portfolio/reset"),
        ("post", "/orders"),
        ("get", "/orders"),
        ("get", "/trades"),
        ("get", "/equity"),
        ("get", "/decisions"),
        ("post", "/backtest"),
        ("get", "/alerts"),
        ("post", "/alerts/test"),
        ("get", "/watchlist"),
        ("put", "/watchlist"),
        ("post", "/scan"),
    }
    documented = {(method, path) for path, ops in spec["paths"].items() for method in ops}
    assert expected <= documented
    assert spec["info"]["version"] == __version__
    assert {tag["name"] for tag in spec["tags"]} >= {"analysis", "portfolio", "backtest"}
    scheme = spec["components"]["securitySchemes"]["ApiKey"]
    assert scheme == {**scheme, "type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert "security" not in spec["paths"]["/health"]["get"]
    assert spec["paths"]["/portfolio"]["get"]["security"] == [{"ApiKey": []}]
    for method, path in expected - {("get", "/health")}:
        operation = spec["paths"][path][method]
        assert operation["tags"], path
        assert "ErrorResponse" in str(operation["responses"]["422"]), path
        assert "429" in operation["responses"], path


def test_cors_preflight_and_exposed_headers(make_client: ClientFactory) -> None:
    client = make_client(cors_origins="*")  # explicit opt-in: the default allows no origin
    preflight = client.options(
        "/portfolio",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "X-API-Key",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "*"
    simple = client.get("/portfolio", headers={"Origin": "http://localhost:8501"})
    assert "X-Request-ID" in simple.headers["access-control-expose-headers"]


def test_cors_restricted_origins(make_client: ClientFactory) -> None:
    client = make_client(cors_origins="http://dash.example")
    allowed = client.get("/health", headers={"Origin": "http://dash.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://dash.example"
    denied = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in denied.headers


def test_large_responses_are_gzipped(client: TestClient) -> None:
    response = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"
    assert response.json()["openapi"].startswith("3.")


# =============================================================================== market data


def test_quote(client: TestClient, synthetic_provider: SyntheticProvider) -> None:
    response = client.get("/quote/bull")
    assert response.status_code == 200
    body = response.json()
    history = synthetic_provider.history("BULL", period="5d")
    assert body["symbol"] == "BULL"
    assert body["price"] == pytest.approx(synthetic_provider.latest_price("BULL"), abs=1e-4)
    assert body["previous_close"] == pytest.approx(float(history["Close"].iloc[-2]), abs=1e-4)
    assert body["change_pct_1d"] == pytest.approx(daily_change_pct(history), abs=1e-4)
    assert body["as_of"].startswith("2026-09-14")


def test_quote_falls_back_to_last_close(
    client: TestClient, synthetic_provider: SyntheticProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    def outage(symbol: str) -> float:
        raise ProviderUnavailableError("quote endpoint down", symbol=symbol)

    last_close = float(synthetic_provider.history("FLAT", period="5d")["Close"].iloc[-1])
    monkeypatch.setattr(synthetic_provider, "latest_price", outage)
    body = client.get("/quote/FLAT").json()
    assert body["price"] == pytest.approx(last_close, abs=1e-4)


@pytest.mark.parametrize(
    ("symbol", "status", "code"),
    [("NOPE", 404, "not_found"), ("DOWN", 502, "data_unavailable"), ("$$$", 422, "invalid_symbol")],
)
def test_quote_errors(client: TestClient, symbol: str, status: int, code: str) -> None:
    error = assert_error(client.get(f"/quote/{symbol}"), status, code)
    assert error["symbol"] == symbol


def test_history_candles_and_warm_started_indicators(client: TestClient) -> None:
    response = client.get("/history/BULL", params={"period": "6mo"})
    assert response.status_code == 200
    body = response.json()
    candles = body["candles"]
    assert body["symbol"] == "BULL" and body["period"] == "6mo" and body["interval"] == "1d"
    assert 120 <= len(candles) <= 135
    assert set(candles[0]) == {"t", "o", "h", "l", "c", "v"}
    assert candles[-1]["t"].startswith("2026-09-14")
    assert all(c["l"] <= min(c["o"], c["c"]) and c["h"] >= max(c["o"], c["c"]) for c in candles)

    indicators = body["indicators"]
    assert set(indicators) == {"sma20", "sma50", "rsi", "macd", "signal", "hist"}
    assert all(len(series) == len(candles) for series in indicators.values())
    # Computed on a longer look-back: defined from the very first displayed candle.
    assert all(series[0] is not None for series in indicators.values())
    assert all(0 <= value <= 100 for value in indicators["rsi"])
    for macd, signal, hist in zip(
        indicators["macd"], indicators["signal"], indicators["hist"], strict=True
    ):
        assert hist == pytest.approx(macd - signal, abs=2e-4)
    assert indicators["sma20"][-1] < candles[-1]["c"]  # BULL trades above its SMA20


def test_history_full_range_masks_indicator_warm_up(client: TestClient) -> None:
    body = client.get("/history/FLAT", params={"period": "max"}).json()
    indicators = body["indicators"]
    assert body["candles"][0]["t"].startswith("2021-01-04")
    assert indicators["sma20"][18] is None and indicators["sma20"][19] is not None
    assert indicators["sma50"][48] is None and indicators["sma50"][49] is not None
    assert indicators["macd"][24] is None and indicators["macd"][25] is not None
    assert indicators["signal"][32] is None and indicators["signal"][33] is not None


def test_history_weekly_interval(client: TestClient) -> None:
    body = client.get("/history/BEAR", params={"period": "1y", "interval": "1wk"}).json()
    assert 50 <= len(body["candles"]) <= 54
    assert body["indicators"]["sma50"][0] is not None


@pytest.mark.parametrize(
    ("params", "status", "code"),
    [
        ({"period": "7mo"}, 422, "validation_error"),
        ({"interval": "1h"}, 422, "validation_error"),
    ],
)
def test_history_parameter_validation(
    client: TestClient, params: dict[str, str], status: int, code: str
) -> None:
    assert_error(client.get("/history/BULL", params=params), status, code)


def test_history_unknown_symbol(client: TestClient) -> None:
    assert_error(client.get("/history/NOPE"), 404, "not_found")
    assert_error(client.get("/history/DOWN"), 502, "data_unavailable")


@pytest.mark.parametrize(
    ("period", "interval", "expected"),
    [
        ("1mo", "1d", "6mo"),
        ("6mo", "1d", "1y"),
        ("1y", "1d", "2y"),
        ("ytd", "1d", "2y"),
        ("1y", "1wk", "5y"),
        ("1y", "1mo", "10y"),
        ("5y", "1mo", "max"),
        ("10y", "1d", "max"),
        ("max", "1d", "max"),
    ],
)
def test_lookback_period(period: str, interval: str, expected: str) -> None:
    assert lookback_period(period, interval) == expected
    assert INDICATOR_WARMUP_BARS >= 50 + 26 + 9  # SMA50 plus MACD convergence


# =============================================================================== analysis


def test_analyze_returns_a_persisted_decision(client: TestClient) -> None:
    response = client.post("/analyze", json={"symbol": " bull "})
    assert response.status_code == 200
    decision = Decision.model_validate(response.json())
    assert decision.symbol == "BULL"
    assert decision.action.value == "BUY"
    assert decision.executed is False and decision.order is None
    assert decision.risk.approved and decision.risk.quantity > 0
    assert "Not executed (auto-trade off)" in decision.summary
    sentiment = decision.signal_for("sentiment")
    assert sentiment is not None and sentiment.data["method"] == "fake:fake-model"
    assert decision.signal_for("trend") is not None

    stored = client.get("/decisions").json()
    assert [d["id"] for d in stored] == [decision.id]


def test_analyze_with_auto_trade_executes(client: TestClient) -> None:
    decision = client.post("/analyze", json={"symbol": "BULL", "auto_trade": True}).json()
    assert decision["executed"] is True
    order = decision["order"]
    assert order["status"] == "FILLED" and order["decision_id"] == decision["id"]

    portfolio = client.get("/portfolio").json()
    position = next(p for p in portfolio["positions"] if p["symbol"] == "BULL")
    assert position["quantity"] == decision["risk"]["quantity"]
    assert position["stop_price"] == pytest.approx(decision["risk"]["stop_price"], abs=0.01)
    assert position["take_profit_price"] is not None
    assert [t["order_id"] for t in client.get("/trades").json()] == [order["id"]]


def test_analyze_respects_server_auto_trade_default(make_client: ClientFactory) -> None:
    client = make_client(auto_trade=True)
    assert client.get("/health").json()["auto_trade"] is True
    assert client.post("/analyze", json={"symbol": "BULL"}).json()["executed"] is True
    assert (
        client.post("/analyze", json={"symbol": "BEAR", "auto_trade": False}).json()["executed"]
        is False
    )


@pytest.mark.parametrize(
    ("symbol", "status", "code"),
    [("NOPE", 404, "not_found"), ("DOWN", 502, "data_unavailable"), ("$$$", 404, "not_found")],
)
def test_analyze_errors(client: TestClient, symbol: str, status: int, code: str) -> None:
    assert_error(client.post("/analyze", json={"symbol": symbol}), status, code)


@pytest.mark.parametrize(
    "body",
    [{}, {"symbol": ""}, {"symbol": "X" * 33}, {"symbol": "BULL", "autotrade": True}],
)
def test_analyze_body_validation(client: TestClient, body: dict[str, Any]) -> None:
    assert_error(client.post("/analyze", json=body), 422, "validation_error")


def test_batch_analysis_reports_errors_per_symbol(client: TestClient) -> None:
    response = client.post("/analyze/batch", json={"symbols": ["BULL", "NOPE", "DOWN", "bull"]})
    assert response.status_code == 200
    body = response.json()
    assert [d["symbol"] for d in body["results"]] == ["BULL"]
    assert {(e["symbol"], e["code"]) for e in body["errors"]} == {
        ("NOPE", "not_found"),
        ("DOWN", "data_unavailable"),
    }


@pytest.mark.parametrize("symbols", [[], [f"S{i}" for i in range(MAX_BATCH_SYMBOLS + 1)], "BULL"])
def test_batch_analysis_validation(client: TestClient, symbols: Any) -> None:
    assert_error(client.post("/analyze/batch", json={"symbols": symbols}), 422, "validation_error")


def test_decisions_filters(client: TestClient) -> None:
    for symbol in ("BULL", "FLAT", "BULL"):
        assert client.post("/analyze", json={"symbol": symbol}).status_code == 200
    assert len(client.get("/decisions").json()) == 3
    assert len(client.get("/decisions", params={"limit": 1}).json()) == 1
    flat = client.get("/decisions", params={"symbol": "flat"}).json()
    assert [d["symbol"] for d in flat] == ["FLAT"]
    assert_error(client.get("/decisions", params={"symbol": "$$$"}), 422, "invalid_symbol")
    assert_error(client.get("/decisions", params={"limit": 0}), 422, "validation_error")
    assert_error(client.get("/decisions", params={"limit": 501}), 422, "validation_error")


def test_scan_analyses_watchlist_and_enforces_stops(client: TestClient) -> None:
    assert buy(client, "BULL", 5).status_code == 201
    services_of(client).broker.update_stops("BULL", stop=10_000.0)  # far above the market

    response = client.post("/scan")
    assert response.status_code == 200
    report = response.json()
    assert [d["symbol"] for d in report["decisions"]] == ["BULL", "BEAR", "FLAT"]
    assert report["errors"] == []
    assert len(report["exits"]) == 1
    exit_order = report["exits"][0]
    assert exit_order["side"] == "SELL" and exit_order["status"] == "FILLED"
    assert "stop_loss" in exit_order["reason"]
    assert report["snapshot"]["positions"] == []
    assert report["finished_at"] is not None
    # creation + the manual BUY's fill + the scan
    assert len(client.get("/equity").json()["points"]) == 3


# =============================================================================== portfolio


def test_portfolio_starts_with_initial_capital(client: TestClient) -> None:
    body = client.get("/portfolio").json()
    assert body["cash"] == body["equity"] == body["initial_capital"] == 100_000.0
    assert body["positions"] == []
    assert body["drawdown_pct"] == 0.0


def test_portfolio_marks_positions_to_market(
    client: TestClient, synthetic_provider: SyntheticProvider
) -> None:
    order = buy(client, "BULL", 10).json()
    body = client.get("/portfolio").json()
    position = body["positions"][0]
    assert position["market_price"] == pytest.approx(
        synthetic_provider.latest_price("BULL"), abs=1e-3
    )
    assert body["equity"] == pytest.approx(body["cash"] + body["positions_value"], abs=0.02)
    # Costs (slippage + commission) are the only loss right after the fill.
    assert body["equity"] < 100_000.0
    assert body["cash"] == pytest.approx(
        100_000.0 - order["fill_price"] * 10 - order["commission"], abs=0.02
    )


def test_portfolio_reset(client: TestClient) -> None:
    buy(client)
    client.post("/analyze", json={"symbol": "FLAT"})
    response = client.post("/portfolio/reset", json={"initial_capital": 50_000})
    assert response.status_code == 200
    body = response.json()
    assert body["cash"] == body["equity"] == body["initial_capital"] == 50_000.0
    assert body["positions"] == []
    assert client.get("/orders").json() == []
    assert client.get("/trades").json() == []
    assert len(client.get("/decisions").json()) == 1  # history is kept

    default = client.post("/portfolio/reset")  # body is optional
    assert default.status_code == 200
    assert default.json()["cash"] == 100_000.0
    assert client.post("/portfolio/reset", json={}).json()["cash"] == 100_000.0


@pytest.mark.parametrize(
    "body",
    [{"initial_capital": -1}, {"initial_capital": 0}, {"initial_capital": "lots"}, {"cash": 1}],
)
def test_portfolio_reset_validation(client: TestClient, body: dict[str, Any]) -> None:
    assert_error(client.post("/portfolio/reset", json=body), 422, "validation_error")


def test_equity_curve_and_metrics(client: TestClient) -> None:
    buy(client)
    client.post("/scan")
    response = client.get("/equity")
    assert response.status_code == 200
    assert "NaN" not in response.text and "Infinity" not in response.text
    body = response.json()
    assert len(body["points"]) == 3  # creation, the BUY's fill, the scan
    assert set(body["points"][0]) == {"t", "equity", "cash", "positions_value"}
    assert body["points"][0]["equity"] == 100_000.0
    assert body["points"][-1]["positions_value"] > 0
    metrics = body["metrics"]
    assert metrics["periods"] == 1  # every snapshot reflects the same trading session
    assert metrics["num_trades"] == 0  # nothing closed yet
    assert {"total_return_pct", "sharpe", "max_drawdown_pct", "exposure_pct"} <= set(metrics)
    # Regression: the first session's P&L vanished (total return 0.0 while the account
    # was down the BUY's costs): metrics are measured from the initial capital.
    assert metrics["start_equity"] == 100_000.0
    last = body["points"][-1]["equity"]
    assert last < 100_000.0
    assert metrics["total_return_pct"] == pytest.approx((last / 100_000.0 - 1) * 100, abs=1e-4)
    portfolio = client.get("/portfolio").json()
    assert metrics["total_return_pct"] == pytest.approx(portfolio["total_return_pct"], abs=1e-3)
    assert metrics["max_drawdown_pct"] == pytest.approx(-metrics["total_return_pct"], abs=1e-4)


def test_equity_metrics_include_the_first_sessions_pnl_and_drawdown(client: TestClient) -> None:
    """Regression: the account opened at 100k and closed its first session at 90k, yet
    /equity reported +1.1 % and no drawdown (measured from the first session's close)."""
    broker = services_of(client).broker
    ny = 4  # UTC offset of New York in October (EDT)

    def at(day: int, hour: int) -> datetime:
        return datetime(2026, 10, day, hour + ny, tzinfo=UTC)

    curve = [
        (at(5, 10), 100_000.0),  # Monday 10:00: the account is created (initial capital)
        (at(5, 15), 90_000.0),  # 15:00 the same session: a 10 % loss
        (at(6, 15), 90_500.0),
        (at(7, 15), 91_000.0),
    ]
    with broker.database.session() as session:
        session.execute(delete(EquitySnapshotRow))
        for ts, equity in curve:
            session.add(
                EquitySnapshotRow(timestamp=ts, equity=equity, cash=equity, positions_value=0.0)
            )
    metrics = client.get("/equity").json()["metrics"]
    assert metrics["periods"] == 3  # one point per session; Monday keeps its close
    assert metrics["start_equity"] == 100_000.0
    assert metrics["end_equity"] == 91_000.0
    assert metrics["total_return_pct"] == pytest.approx(-9.0)
    assert metrics["max_drawdown_pct"] == pytest.approx(10.0)
    assert metrics["sharpe"] is not None and metrics["sharpe"] < 0


# =============================================================================== orders


def test_manual_order_fills_at_provider_price(
    client: TestClient, synthetic_provider: SyntheticProvider
) -> None:
    response = buy(client, "bull", 10, stop_price=150.0, take_profit_price=250.0)
    assert response.status_code == 201
    order = response.json()
    price = synthetic_provider.latest_price("BULL")
    assert order["status"] == "FILLED"
    assert order["symbol"] == "BULL" and order["quantity"] == 10
    assert order["reference_price"] == pytest.approx(price, abs=1e-4)
    assert order["fill_price"] == pytest.approx(price * (1 + 5 / 10_000), abs=1e-3)  # 5 bps
    assert order["commission"] == 1.0  # min commission beats 5 bps of ~$1.8k
    assert order["reason"] == "manual order (API)"
    position = client.get("/portfolio").json()["positions"][0]
    assert position["stop_price"] == 150.0 and position["take_profit_price"] == 250.0


def test_round_trip_realises_pnl(client: TestClient) -> None:
    buy(client, "FLAT", 20)
    sell = client.post("/orders", json={"symbol": "FLAT", "side": "sell", "quantity": 20})
    assert sell.status_code == 201
    assert sell.json()["realized_pnl"] < 0  # same price both ways: costs only
    trades = client.get("/trades").json()
    assert [t["side"] for t in trades] == ["SELL", "BUY"]
    assert trades[0]["realized_pnl"] == pytest.approx(sell.json()["realized_pnl"], abs=0.01)
    assert len(client.get("/trades", params={"limit": 1}).json()) == 1
    assert client.get("/trades", params={"symbol": "BULL"}).json() == []


@pytest.mark.parametrize("key", ["price", "limit_price", "fill_price"])
def test_order_rejects_client_supplied_price(client: TestClient, key: str) -> None:
    response = buy(client, **{key: 1.0})
    error = assert_error(response, 422, "validation_error")
    assert key in error["message"]
    assert client.get("/orders").json() == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"quantity": 0},
        {"quantity": -5},
        {"quantity": 1.5},
        {"quantity": "10"},
        {"quantity": True},
        {"side": "HOLD"},
        {"symbol": "$$$"},
        {"stop_price": -1},
        {"take_profit_price": 0},
        {"note": "extra field"},
    ],
)
def test_order_body_validation(client: TestClient, overrides: dict[str, Any]) -> None:
    body = {"symbol": "BULL", "side": "BUY", "quantity": 10, **overrides}
    assert_error(client.post("/orders", json=body), 422, "validation_error")


def test_order_with_stop_above_price_is_invalid(client: TestClient) -> None:
    error = assert_error(buy(client, stop_price=10_000.0), 422, "invalid_order")
    assert "stop-loss" in error["message"]


def test_rejected_orders_are_422_and_recorded(client: TestClient) -> None:
    no_position = client.post("/orders", json={"symbol": "BEAR", "side": "SELL", "quantity": 1})
    error = assert_error(no_position, 422, "order_rejected")
    assert "BEAR" in error["message"]
    assert error["order"]["status"] == "REJECTED"

    too_big = assert_error(buy(client, "BULL", 10_000), 422, "order_rejected")
    assert "cash" in too_big["message"]

    orders = client.get("/orders").json()
    assert [o["status"] for o in orders] == ["REJECTED", "REJECTED"]
    assert len(client.get("/orders", params={"symbol": "BEAR"}).json()) == 1
    assert client.get("/portfolio").json()["cash"] == 100_000.0


def test_order_market_data_errors(client: TestClient) -> None:
    assert_error(buy(client, "NOPE"), 404, "not_found")
    assert_error(buy(client, "DOWN"), 502, "data_unavailable")


def test_list_limits_are_validated(client: TestClient) -> None:
    assert_error(client.get("/orders", params={"limit": 0}), 422, "validation_error")
    assert_error(client.get("/trades", params={"limit": 5001}), 422, "validation_error")
    assert_error(client.get("/alerts", params={"limit": "many"}), 422, "validation_error")


# =============================================================================== backtest


BACKTEST = {"symbols": ["BULL"], "start": "2025-01-02", "end": "2025-03-31", "rebalance": "weekly"}


def test_backtest_happy_path(client: TestClient) -> None:
    response = client.post("/backtest", json={**BACKTEST, "initial_capital": 50_000})
    assert response.status_code == 200
    assert "NaN" not in response.text and "Infinity" not in response.text
    body = response.json()
    assert body["symbols"] == ["BULL"]
    assert body["initial_capital"] == 50_000.0
    assert body["parameters"]["rebalance"] == "weekly"
    assert body["equity_curve"][0]["t"].startswith("2025-01-02")
    assert {"equity", "cash", "drawdown", "benchmark"} <= set(body["equity_curve"][0])
    assert body["metrics"]["periods"] == len(body["equity_curve"])
    assert body["benchmark_metrics"]["total_return_pct"] > 0  # BULL rallied
    assert isinstance(body["trades"], list) and isinstance(body["decisions"], list)
    assert "signal_log" not in body
    # The backtest never touches the live paper account.
    assert client.get("/orders").json() == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"start": "2025-03-31", "end": "2025-01-02"},
        {"start": "2025-01-02", "end": "2025-01-02"},
        {"start": "2010-01-04", "end": "2025-01-02"},
        {"symbols": []},
        {"symbols": ["A", "B", "C", "D", "E", "F"]},
        {"symbols": ["$$$"]},
        {"rebalance": "hourly"},
        {"warmup": -1},
        {"initial_capital": 0},
        {"start": "not-a-date"},
        {"benchmark": "SPY"},
    ],
)
def test_backtest_validation(client: TestClient, overrides: dict[str, Any]) -> None:
    assert_error(client.post("/backtest", json={**BACKTEST, **overrides}), 422, "validation_error")


@pytest.mark.parametrize(
    ("symbols", "status", "code"),
    [(["NOPE"], 404, "not_found"), (["BULL", "DOWN"], 502, "data_unavailable")],
)
def test_backtest_market_data_errors(
    client: TestClient, symbols: list[str], status: int, code: str
) -> None:
    assert_error(client.post("/backtest", json={**BACKTEST, "symbols": symbols}), status, code)


def test_backtest_window_without_bars_is_404(client: TestClient) -> None:
    body = {**BACKTEST, "start": "2027-01-04", "end": "2027-02-01"}  # after the data ends
    assert_error(client.post("/backtest", json=body), 404, "not_found")


def test_backtest_config_errors_from_engine_are_422(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from trading_swarm.backtest import BacktestConfigError

    def reject(*args: Any, **kwargs: Any) -> Any:
        raise BacktestConfigError("warmup too long for this window")

    monkeypatch.setattr(services_of(client).backtester, "run", reject)
    error = assert_error(client.post("/backtest", json=BACKTEST), 422, "invalid_request")
    assert error["message"] == "warmup too long for this window"


def test_backtest_concurrency_cap(client: TestClient) -> None:
    slots = services_of(client).backtest_slots
    for _ in range(MAX_CONCURRENT_BACKTESTS):
        assert slots.acquire(blocking=False)
    try:
        response = client.post("/backtest", json=BACKTEST)
        error = assert_error(response, 503, "busy")
        assert response.headers["Retry-After"] == "15"
        assert error["retry_after"] == 15
    finally:
        for _ in range(MAX_CONCURRENT_BACKTESTS):
            slots.release()
    assert client.post("/backtest", json=BACKTEST).status_code == 200


# =============================================================================== alerts


def test_alert_log_and_disabled_test_email(client: TestClient) -> None:
    assert client.get("/alerts").json() == []
    response = client.post("/alerts/test")
    assert response.status_code == 200
    body = response.json()
    assert body["sent"] is False
    assert body["channel"] == "email"
    assert body["status"] == "disabled"
    assert "SMTP_HOST" in body["detail"]

    log = client.get("/alerts").json()
    assert len(log) == 1
    assert log[0]["rule"] == "test" and log[0]["status"] == "disabled"
    assert client.get("/alerts", params={"rule": "signal"}).json() == []


def test_test_email_is_sent_via_smtp(make_client: ClientFactory, fake_smtp: Any) -> None:
    client = make_client(
        smtp_host="smtp.example.com",
        smtp_username="bot@example.com",
        smtp_password="app-password",
        alert_email_from="bot@example.com",
        alert_email_to="me@example.com, you@example.com",
    )
    body = client.post("/alerts/test").json()
    assert body["sent"] is True and body["status"] == "sent"
    assert len(fake_smtp.sent) == 1
    connection = fake_smtp.instances[0]
    assert connection.host == "smtp.example.com" and connection.started_tls
    assert client.get("/alerts").json()[0]["status"] == "sent"


def test_confident_signal_triggers_alert_on_analyze(
    make_client: ClientFactory, fake_smtp: Any
) -> None:
    client = make_client(
        smtp_host="smtp.example.com",
        alert_email_from="bot@example.com",
        alert_email_to="me@example.com",
        alert_min_confidence=0.5,
    )
    decision = client.post("/analyze", json={"symbol": "BULL"}).json()
    assert decision["action"] == "BUY" and decision["confidence"] >= 0.5
    signals = client.get("/alerts", params={"rule": "signal", "symbol": "BULL"}).json()
    assert [a["status"] for a in signals] == ["sent"]
    assert len(fake_smtp.sent) >= 1


# =============================================================================== watchlist


def test_watchlist_defaults_to_settings(client: TestClient) -> None:
    assert client.get("/watchlist").json() == {"symbols": ["BULL", "BEAR", "FLAT"]}


def test_watchlist_update_persists_and_drives_scans(make_client: ClientFactory) -> None:
    first = make_client()
    response = first.put("/watchlist", json={"symbols": ["flat", " bull ", "FLAT"]})
    assert response.status_code == 200
    assert response.json() == {"symbols": ["FLAT", "BULL"]}
    assert [d["symbol"] for d in first.post("/scan").json()["decisions"]] == ["FLAT", "BULL"]

    # A second app on the same database (a restart) sees the persisted list.
    second = make_client()
    assert second.get("/watchlist").json() == {"symbols": ["FLAT", "BULL"]}


@pytest.mark.parametrize(
    "body",
    [{"symbols": []}, {"symbols": ["$$$"]}, {"symbols": [f"S{i}" for i in range(51)]}, {}],
)
def test_watchlist_validation(client: TestClient, body: dict[str, Any]) -> None:
    assert_error(client.put("/watchlist", json=body), 422, "validation_error")
    assert client.get("/watchlist").json()["symbols"] == ["BULL", "BEAR", "FLAT"]


# =============================================================================== protected surface


PROTECTED = [
    ("get", "/quote/BULL", None),
    ("get", "/history/BULL", None),
    ("post", "/analyze", {"symbol": "BULL"}),
    ("post", "/analyze/batch", {"symbols": ["BULL"]}),
    ("get", "/decisions", None),
    ("post", "/scan", None),
    ("get", "/portfolio", None),
    ("post", "/portfolio/reset", {}),
    ("get", "/equity", None),
    ("post", "/orders", {"symbol": "BULL", "side": "BUY", "quantity": 1}),
    ("get", "/orders", None),
    ("get", "/trades", None),
    ("post", "/backtest", BACKTEST),
    ("get", "/alerts", None),
    ("post", "/alerts/test", None),
    ("get", "/watchlist", None),
    ("put", "/watchlist", {"symbols": ["BULL"]}),
]


@pytest.mark.parametrize(("method", "path", "body"), PROTECTED)
def test_every_endpoint_requires_the_key(
    auth_client: TestClient, method: str, path: str, body: Any
) -> None:
    kwargs = {"json": body} if body is not None else {}
    assert_error(auth_client.request(method, path, **kwargs), 401, "unauthorized")
    authorised = auth_client.request(method, path, headers={"X-API-Key": "k1"}, **kwargs)
    assert 200 <= authorised.status_code < 300, authorised.text
    assert int(authorised.headers["X-RateLimit-Remaining"]) >= 0
    assert not any(isinstance(v, float) and math.isnan(v) for v in _floats(authorised.json()))


def _floats(value: Any) -> Iterator[float]:
    if isinstance(value, float):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _floats(item)
    elif isinstance(value, list):
        for item in value:
            yield from _floats(item)


# =============================================================================== order levels


def test_sell_rejects_protective_levels(client: TestClient) -> None:
    """Regression: a SELL used to echo a stop it never applied to the remaining shares."""
    assert buy(client, "BULL", 5).status_code == 201
    for extra in ({"stop_price": 1.0}, {"take_profit_price": 10_000.0}):
        response = client.post(
            "/orders", json={"symbol": "BULL", "side": "SELL", "quantity": 2, **extra}
        )
        error = assert_error(response, 422, "invalid_order")
        assert "only be set on BUY orders" in error["message"]
    (position,) = client.get("/portfolio").json()["positions"]
    assert position["quantity"] == 5 and position["stop_price"] is None


def test_buy_into_a_held_symbol_reports_the_stop_actually_kept(
    client: TestClient, synthetic_provider: SyntheticProvider
) -> None:
    price = synthetic_provider.latest_price("BULL")
    assert buy(client, "BULL", 5, stop_price=round(price * 0.95, 2)).status_code == 201
    response = buy(client, "BULL", 5, stop_price=round(price * 0.80, 2))
    assert response.status_code == 201
    assert response.json()["stop_price"] == round(price * 0.95, 2)  # not lowered
    (position,) = client.get("/portfolio").json()["positions"]
    assert position["quantity"] == 10 and position["stop_price"] == round(price * 0.95, 2)


def test_buy_into_a_breached_position_is_rejected(
    client: TestClient, synthetic_provider: SyntheticProvider
) -> None:
    assert buy(client, "BULL", 5).status_code == 201
    services_of(client).broker.update_stops("BULL", stop=10_000.0)  # far above the market
    error = assert_error(buy(client, "BULL", 5), 422, "order_rejected")
    assert "stop-loss" in error["message"] and error["order"]["status"] == "REJECTED"


def test_manual_fill_adds_an_equity_point(client: TestClient) -> None:
    assert len(client.get("/equity").json()["points"]) == 1  # account creation
    assert buy(client, "BULL", 5).status_code == 201
    points = client.get("/equity").json()["points"]
    assert len(points) == 2 and points[-1]["positions_value"] > 0


# =============================================================================== cross-site guard

EVIL = "https://evil.example"


def test_default_config_sends_no_cors_headers(client: TestClient) -> None:
    response = client.get("/portfolio", headers={"Origin": EVIL})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
    preflight = client.options(
        "/orders", headers={"Origin": EVIL, "Access-Control-Request-Method": "POST"}
    )
    assert "access-control-allow-origin" not in preflight.headers


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("post", "/portfolio/reset", {}),  # bodiless: a CORS "simple" request
        ("post", "/scan", {}),
        ("post", "/alerts/test", {}),
        (
            "post",
            "/orders",
            {
                "content": '{"symbol": "BULL", "side": "BUY", "quantity": 1}',
                "headers": {"Content-Type": "text/plain"},
            },
        ),
        ("put", "/watchlist", {"json": {"symbols": ["NOPE"]}}),
    ],
)
def test_cross_site_state_changes_are_refused(
    client: TestClient, method: str, path: str, kwargs: dict[str, Any]
) -> None:
    """Regression: with CORS ``*`` and auth off, any web page could reset the account."""
    assert buy(client, "BULL", 3).status_code == 201
    headers = {"Origin": EVIL, **kwargs.pop("headers", {})}
    response = getattr(client, method)(path, headers=headers, **kwargs)
    error = assert_error(response, 403, "forbidden_origin")
    assert EVIL in error["message"]
    portfolio = client.get("/portfolio").json()
    assert [p["symbol"] for p in portfolio["positions"]] == ["BULL"]  # nothing happened
    assert client.get("/watchlist").json()["symbols"] != ["NOPE"]


def test_cross_site_guard_without_origin_uses_fetch_metadata(client: TestClient) -> None:
    response = client.post("/portfolio/reset", headers={"Sec-Fetch-Site": "cross-site"})
    assert_error(response, 403, "forbidden_origin")
    # Same-origin browser requests (the /docs page) and plain clients are fine.
    assert client.post("/portfolio/reset", headers={"Sec-Fetch-Site": "same-origin"}).is_success
    assert client.post("/portfolio/reset", headers={"Origin": "http://testserver"}).is_success
    assert client.post("/portfolio/reset").is_success


def test_cross_site_guard_honours_cors_origins(make_client: ClientFactory) -> None:
    client = make_client(cors_origins="http://dash.example/, http://other.example:8080")
    for origin in ("http://dash.example", "HTTP://Dash.Example:80", "http://other.example:8080"):
        assert client.post("/portfolio/reset", headers={"Origin": origin}).is_success, origin
    assert_error(
        client.post("/portfolio/reset", headers={"Origin": "http://other.example"}),
        403,
        "forbidden_origin",
    )
    assert_error(
        client.post("/portfolio/reset", headers={"Origin": "null"}), 403, "forbidden_origin"
    )
    anyone = make_client(cors_origins="*")
    assert anyone.post("/portfolio/reset", headers={"Origin": EVIL}).is_success


def test_reads_are_not_subject_to_the_cross_site_guard(client: TestClient) -> None:
    assert client.get("/portfolio", headers={"Origin": EVIL}).status_code == 200


# =============================================================================== equity sampling


def _session_curve(days: int = 260, seed: int = 3) -> pd.DataFrame:
    """Equity on consecutive NYSE sessions, snapshotted at 21:00 UTC (after the close)."""
    sessions = trading_days(date(2025, 1, 2), date(2026, 3, 31))[:days]
    rng = np.random.default_rng(seed)
    equity = 100_000 * np.exp(np.cumsum(rng.normal(0.0006, 0.01, len(sessions))))
    index = pd.DatetimeIndex(
        [pd.Timestamp(datetime.combine(d, time(21, 0)), tz="UTC") for d in sessions]
    )
    return pd.DataFrame(
        {"equity": equity, "cash": equity * 0.5, "positions_value": equity * 0.5}, index=index
    )


def test_equity_metrics_ignore_weekend_and_holiday_scans() -> None:
    """Regression: a 24/7 scheduler added zero-return calendar days, so Sharpe and
    volatility were understated by about √(5/7)."""
    sessions = _session_curve()
    hourly_index = pd.date_range(sessions.index[0], sessions.index[-1], freq="1h")
    around_the_clock = sessions.reindex(hourly_index, method="ffill")  # a scan every hour
    reference = compute_metrics(_daily_curve(sessions))
    sampled = compute_metrics(_daily_curve(around_the_clock))
    assert sampled.periods == reference.periods == len(sessions)
    assert sampled.sharpe == pytest.approx(reference.sharpe)
    assert sampled.volatility_pct == pytest.approx(reference.volatility_pct)
    naive = compute_metrics(around_the_clock.groupby(around_the_clock.index.normalize()).last())
    assert naive.volatility_pct < 0.9 * reference.volatility_pct  # what the old code reported


def test_equity_metrics_spread_gaps_between_manual_scans() -> None:
    sessions = _session_curve(days=256)  # 51 weeks + 1: the weekly sample spans it all
    weekly = sessions.iloc[::5]  # someone scans once a week
    daily = _daily_curve(weekly)
    assert len(daily) == len(_daily_curve(sessions))  # every session in the span
    assert daily["equity"].iloc[1] == weekly["equity"].iloc[0]  # carried forward
    reference = compute_metrics(_daily_curve(sessions)).volatility_pct
    sampled = compute_metrics(daily).volatility_pct
    naive = compute_metrics(weekly.groupby(weekly.index.normalize()).last()).volatility_pct
    assert reference is not None and sampled is not None and naive is not None
    assert 0.75 < sampled / reference < 1.33  # same scale (sampling noise only)
    assert naive / reference > 1.8  # ~√5: what one-period-per-snapshot would report
