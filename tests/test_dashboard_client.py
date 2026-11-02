"""Tests for the dashboard's typed API client (no network: a fake ``requests`` session)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import pytest
import requests

from dashboard.client import (
    API_KEY_HEADER,
    DEFAULT_API_URL,
    DEFAULT_LONG_TIMEOUT_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
    ApiAuthError,
    ApiClient,
    ApiConnectionError,
    ApiError,
    ApiNotFoundError,
    ApiRateLimitError,
    ApiResponseError,
    ApiServerError,
    ApiTimeoutError,
    ApiValidationError,
    error_from_response,
    extract_list,
    normalize_symbol,
    normalize_symbols,
    parse_retry_after,
)
from trading_swarm.llm.null_client import NullLLMClient
from trading_swarm.swarm import SwarmCoordinator

BASE = "http://api.test:8000"


# =============================================================================== fakes


def make_response(
    status: int = 200,
    body: Any = None,
    *,
    text: str | None = None,
    headers: dict[str, str] | None = None,
    url: str = f"{BASE}/x",
) -> requests.Response:
    """A real :class:`requests.Response` (so ``.json()``/``.text`` behave as in production)."""
    response = requests.Response()
    response.status_code = status
    if text is not None:
        response._content = text.encode()
    elif body is not None:
        response._content = json.dumps(body).encode()
        response.headers["Content-Type"] = "application/json"
    else:
        response._content = b""
    response.headers.update(headers or {})
    response.url = url
    response.reason = {200: "OK", 204: "No Content", 500: "Internal Server Error"}.get(status, "")
    return response


class FakeSession:
    """Stands in for ``requests.Session``: records calls, replays queued outcomes."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        outcome = self.outcomes.pop(0) if self.outcomes else make_response(200, {})
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(method, url, **kwargs)
        return outcome

    def close(self) -> None:
        self.closed = True

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]


@pytest.fixture
def decision_payload(settings, provider, broker) -> dict[str, Any]:
    """A real Decision, serialised exactly as the API returns it."""
    coordinator = SwarmCoordinator(settings, provider, broker, llm=NullLLMClient())
    return coordinator.analyze("BULL", auto_trade=True).model_dump(mode="json")


@pytest.fixture
def make_client() -> Callable[..., tuple[ApiClient, FakeSession]]:
    def factory(*outcomes: Any, api_key: str | None = None) -> tuple[ApiClient, FakeSession]:
        session = FakeSession(*outcomes)
        client = ApiClient(BASE, api_key, session=session)  # type: ignore[arg-type]
        return client, session

    return factory


# =============================================================================== construction


def test_from_env_defaults() -> None:
    client = ApiClient.from_env({})
    assert client.base_url == DEFAULT_API_URL
    assert client.api_key is None
    assert client.timeout == DEFAULT_TIMEOUT_SECONDS
    assert client.long_timeout == DEFAULT_LONG_TIMEOUT_SECONDS
    client.close()


def test_from_env_reads_variables_and_strips_trailing_slash() -> None:
    env = {
        "API_URL": " https://swarm.example.com/ ",
        "API_KEY": " k1 ",
        "API_TIMEOUT_SECONDS": "5",
        "API_LONG_TIMEOUT_SECONDS": "60",
    }
    with ApiClient.from_env(env) as client:
        assert client.base_url == "https://swarm.example.com"
        assert client.api_key == "k1"
        assert (client.timeout, client.long_timeout) == (5.0, 60.0)


@pytest.mark.parametrize("raw", ["abc", "-3", "0", "nan"])
def test_from_env_ignores_bad_timeouts(raw: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING), ApiClient.from_env({"API_TIMEOUT_SECONDS": raw}) as c:
        assert c.timeout == DEFAULT_TIMEOUT_SECONDS
    assert "API_TIMEOUT_SECONDS" in caplog.text


def test_from_env_overrides_win() -> None:
    with ApiClient.from_env({"API_URL": "http://a:1"}, base_url="http://b:2") as client:
        assert client.base_url == "http://b:2"


@pytest.mark.parametrize("url", ["", "localhost:8000", "ftp://host", "http://"])
def test_rejects_non_http_urls(url: str) -> None:
    with pytest.raises(ValueError, match="http"):
        ApiClient(url)


def test_rejects_non_positive_timeouts() -> None:
    with pytest.raises(ValueError, match="timeouts"):
        ApiClient(BASE, timeout=0)


def test_repr_and_cache_key_never_contain_the_key() -> None:
    secret = "super-secret-key"
    client = ApiClient(BASE, secret, session=FakeSession())  # type: ignore[arg-type]
    assert secret not in repr(client)
    assert "api_key=set" in repr(client)
    assert secret not in client.cache_key
    assert client.cache_key.startswith(BASE)
    other = ApiClient(BASE, "another-key", session=FakeSession())  # type: ignore[arg-type]
    anonymous = ApiClient(BASE, session=FakeSession())  # type: ignore[arg-type]
    assert len({client.cache_key, other.cache_key, anonymous.cache_key}) == 3


def test_owned_session_has_connect_only_retries() -> None:
    client = ApiClient(BASE, retries=3)
    retry = client.session.get_adapter(f"{BASE}/health").max_retries
    assert retry.connect == 3
    assert retry.read is False  # read timeouts surface as ApiTimeoutError, never resent
    assert retry.status == 0  # 404/429/502 are information, not glitches
    client.close()


def test_close_only_closes_owned_sessions() -> None:
    injected = FakeSession()
    ApiClient(BASE, session=injected).close()  # type: ignore[arg-type]
    assert injected.closed is False


# =============================================================================== transport


def test_health_without_key_sends_no_auth_header(make_client) -> None:
    client, session = make_client(make_response(200, {"status": "ok", "version": "2.0.0"}))
    assert client.health()["status"] == "ok"
    call = session.last
    assert call["method"] == "GET"
    assert call["url"] == f"{BASE}/health"
    assert API_KEY_HEADER not in call["headers"]
    assert call["timeout"] == DEFAULT_TIMEOUT_SECONDS


def test_status_is_a_keyed_get(make_client) -> None:
    payload = {"scheduler": {"running": False}, "alerts": {"channel": "email"}}
    client, session = make_client(make_response(200, payload), api_key="k1")
    assert client.status() == payload
    assert session.last["method"] == "GET"
    assert session.last["url"] == f"{BASE}/status"
    assert session.last["headers"][API_KEY_HEADER] == "k1"


def test_api_key_is_sent_as_header(make_client) -> None:
    client, session = make_client(make_response(200, {"symbols": []}), api_key="k1")
    client.watchlist()
    assert session.last["headers"][API_KEY_HEADER] == "k1"


def test_api_key_is_never_logged(make_client, caplog: pytest.LogCaptureFixture) -> None:
    client, _ = make_client(
        make_response(200, {}),
        make_response(500, {"error": {"code": "boom", "message": "x"}}),
        api_key="top-secret",
    )
    with caplog.at_level(logging.DEBUG, logger="dashboard.client"):
        client.portfolio()
        with pytest.raises(ApiServerError):
            client.portfolio()
    assert caplog.records
    assert "top-secret" not in caplog.text


def test_none_params_are_dropped(make_client) -> None:
    client, session = make_client(make_response(200, []))
    client.decisions(limit=5)
    assert session.last["params"] == {"limit": 5}


def test_empty_and_204_bodies_return_none(make_client) -> None:
    client, _ = make_client(make_response(204), make_response(200))
    assert client.request("DELETE", "/x") is None
    assert client.request("GET", "/x") is None


def test_non_json_success_raises_response_error(make_client) -> None:
    client, _ = make_client(
        make_response(200, text="<html>proxy</html>", headers={"X-Request-ID": "r1"})
    )
    with pytest.raises(ApiResponseError) as info:
        client.health()
    assert info.value.code == "invalid_json"
    assert info.value.request_id == "r1"


def test_object_endpoint_rejects_a_list(make_client) -> None:
    client, _ = make_client(make_response(200, [1, 2]))
    with pytest.raises(ApiResponseError, match="JSON object"):
        client.portfolio()


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (requests.exceptions.ConnectionError("refused"), ApiConnectionError),
        (requests.exceptions.ConnectTimeout("slow connect"), ApiConnectionError),
        (requests.exceptions.ReadTimeout("slow answer"), ApiTimeoutError),
        (requests.exceptions.TooManyRedirects("loop"), ApiConnectionError),
    ],
)
def test_transport_failures_map_to_typed_errors(make_client, exc, expected) -> None:
    client, _ = make_client(exc)
    with pytest.raises(expected) as info:
        client.health()
    assert type(info.value) is expected
    assert info.value.status_code is None
    assert BASE in info.value.message


def test_timeout_error_is_a_connection_error() -> None:
    assert issubclass(ApiTimeoutError, ApiConnectionError)


# =============================================================================== error envelope


def test_envelope_401_maps_to_auth_error(make_client) -> None:
    body = {"error": {"code": "unauthorized", "message": "invalid API key", "request_id": "abc"}}
    client, _ = make_client(make_response(401, body))
    with pytest.raises(ApiAuthError) as info:
        client.portfolio()
    err = info.value
    assert (err.status_code, err.code, err.message, err.request_id) == (
        401,
        "unauthorized",
        "invalid API key",
        "abc",
    )
    assert str(err) == "HTTP 401: invalid API key (request id abc)"
    assert "ApiAuthError" in repr(err)


@pytest.mark.parametrize(
    ("status", "cls"),
    [
        (400, ApiValidationError),
        (403, ApiAuthError),
        (404, ApiNotFoundError),
        (409, ApiValidationError),
        (422, ApiValidationError),
        (429, ApiRateLimitError),
        (500, ApiServerError),
        (502, ApiServerError),
        (504, ApiServerError),
        (418, ApiError),
    ],
)
def test_status_codes_map_to_classes(status: int, cls: type[ApiError]) -> None:
    error = error_from_response(make_response(status, {"error": {"message": "m"}}))
    assert type(error) is cls
    assert error.status_code == status


def test_fastapi_validation_detail_is_flattened() -> None:
    body = {
        "detail": [
            {"loc": ["body", "symbols"], "msg": "List should have at most 10 items"},
            {"loc": ["query", "limit"], "msg": "Input should be greater than 0"},
        ]
    }
    error = error_from_response(make_response(422, body))
    assert isinstance(error, ApiValidationError)
    assert error.message == (
        "symbols: List should have at most 10 items; limit: Input should be greater than 0"
    )


def test_plain_string_detail_and_header_request_id() -> None:
    error = error_from_response(
        make_response(404, {"detail": "Not Found"}, headers={"X-Request-ID": "rid-9"})
    )
    assert error.message == "Not Found"
    assert error.request_id == "rid-9"


def test_html_error_page_becomes_a_short_snippet() -> None:
    html = "<html><body><h1>502 Bad Gateway</h1>" + "x" * 1000 + "</body></html>"
    error = error_from_response(make_response(502, text=html))
    assert isinstance(error, ApiServerError)
    assert error.message.startswith("502 Bad Gateway")
    assert "<" not in error.message
    assert len(error.message) <= 300


def test_empty_error_body_falls_back_to_reason() -> None:
    error = error_from_response(make_response(500))
    assert error.message == "Internal Server Error"


def test_rate_limit_reads_retry_after_header(make_client) -> None:
    body = {"error": {"code": "rate_limited", "message": "slow down", "request_id": "r"}}
    client, _ = make_client(make_response(429, body, headers={"Retry-After": "17"}))
    with pytest.raises(ApiRateLimitError) as info:
        client.trades()
    assert info.value.retry_after == 17.0


def test_rate_limit_falls_back_to_envelope_retry_after() -> None:
    body = {"error": {"code": "rate_limited", "message": "slow", "retry_after": 4}}
    assert error_from_response(make_response(429, body)).retry_after == 4.0


def test_parse_retry_after_variants() -> None:
    now = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
    http_date = format_datetime(now + timedelta(seconds=30), usegmt=True)
    assert parse_retry_after(http_date, now=now.timestamp()) == pytest.approx(30.0)
    assert parse_retry_after("2.5") == 2.5
    assert parse_retry_after("-4") == 0.0
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("soon") is None
    assert parse_retry_after("inf") is None


# =============================================================================== symbols


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("aapl", "AAPL"), (" brk.b ", "BRK.B"), ("^gspc", "^GSPC"), ("eurusd=x", "EURUSD=X")],
)
def test_normalize_symbol(raw: str, expected: str) -> None:
    assert normalize_symbol(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "AA PL", "../etc", "A" * 16, "AAPL;DROP"])
def test_normalize_symbol_rejects_garbage(raw: str) -> None:
    with pytest.raises(ApiValidationError) as info:
        normalize_symbol(raw)
    assert info.value.status_code is None
    assert info.value.code == "invalid_symbol"


def test_normalize_symbols_dedupes_and_splits_strings() -> None:
    assert normalize_symbols("aapl, msft,AAPL,,") == ["AAPL", "MSFT"]
    assert normalize_symbols(["nvda", "NVDA", "amd"]) == ["NVDA", "AMD"]
    with pytest.raises(ApiValidationError, match="at least 1"):
        normalize_symbols([])
    with pytest.raises(ApiValidationError, match="at most 2"):
        normalize_symbols(["A", "B", "C"], max_count=2)


def test_invalid_symbol_never_reaches_the_network(make_client) -> None:
    client, session = make_client()
    with pytest.raises(ApiValidationError):
        client.quote("not a ticker")
    with pytest.raises(ApiValidationError):
        client.analyze("")
    assert session.calls == []


# =============================================================================== endpoints


def test_quote_uppercases_and_url_encodes(make_client) -> None:
    client, session = make_client(make_response(200, {"symbol": "^GSPC", "price": 1.0}))
    client.quote("^gspc")
    assert session.last["url"] == f"{BASE}/quote/%5EGSPC"


def test_history_sends_period_and_interval(make_client) -> None:
    client, session = make_client(make_response(200, {"symbol": "AAPL", "candles": []}))
    client.history("aapl", period="1y")
    assert session.last["url"] == f"{BASE}/history/AAPL"
    assert session.last["params"] == {"period": "1y", "interval": "1d"}


def test_analyze_posts_symbol_and_uses_long_timeout(make_client) -> None:
    client, session = make_client(make_response(200, {"symbol": "AAPL"}), make_response(200, {}))
    client.analyze("aapl", auto_trade=True)
    assert session.last["method"] == "POST"
    assert session.last["json"] == {"symbol": "AAPL", "auto_trade": True}
    assert session.last["timeout"] == DEFAULT_LONG_TIMEOUT_SECONDS
    client.analyze("aapl")  # auto_trade omitted → the API's default applies
    assert session.last["json"] == {"symbol": "AAPL"}


def test_analyze_batch_validates_and_fills_defaults(make_client) -> None:
    client, session = make_client(make_response(200, {"results": []}))
    result = client.analyze_batch(["aapl", "msft", "AAPL"], auto_trade=False)
    assert session.last["json"] == {"symbols": ["AAPL", "MSFT"], "auto_trade": False}
    assert result == {"results": [], "errors": []}
    with pytest.raises(ApiValidationError, match="at most 10"):
        client.analyze_batch([f"S{i}" for i in range(11)])


def test_reset_portfolio_body(make_client) -> None:
    client, session = make_client(make_response(200, {}), make_response(200, {}))
    client.reset_portfolio()
    assert session.last["json"] == {}
    client.reset_portfolio(50_000)
    assert session.last["json"] == {"initial_capital": 50_000.0}
    for bad in (0, -5, float("nan"), "lots"):
        with pytest.raises(ApiValidationError):
            client.reset_portfolio(bad)  # type: ignore[arg-type]


def test_place_order_never_sends_a_price(make_client) -> None:
    client, session = make_client(make_response(200, {"status": "FILLED"}))
    client.place_order("aapl", "buy", 10, stop_price=95.0, take_profit_price=None)
    body = session.last["json"]
    assert body == {"symbol": "AAPL", "side": "BUY", "quantity": 10, "stop_price": 95.0}
    assert "price" not in body


@pytest.mark.parametrize(
    ("side", "qty"), [("HOLD", 1), ("BUY", 0), ("BUY", -1), ("BUY", 1.5), ("BUY", True)]
)
def test_place_order_validates_locally(make_client, side: str, qty: Any) -> None:
    client, session = make_client()
    with pytest.raises(ApiValidationError):
        client.place_order("AAPL", side, qty)
    assert session.calls == []


@pytest.mark.parametrize(
    ("method", "key", "path"),
    [
        ("orders", "orders", "/orders"),
        ("trades", "trades", "/trades"),
        ("alerts", "alerts", "/alerts"),
    ],
)
def test_list_endpoints_accept_bare_and_wrapped_lists(make_client, method, key, path) -> None:
    rows = [{"id": "1"}, {"id": "2"}]
    client, session = make_client(make_response(200, rows), make_response(200, {key: rows}))
    assert getattr(client, method)(limit=7) == rows
    assert session.last["url"] == f"{BASE}{path}"
    assert session.last["params"] == {"limit": 7}
    assert getattr(client, method)(limit=7) == rows


def test_decisions_symbol_filter(make_client) -> None:
    client, session = make_client(make_response(200, {"items": []}))
    assert client.decisions(limit=10, symbol="msft") == []
    assert session.last["params"] == {"limit": 10, "symbol": "MSFT"}


def test_extract_list_rejects_scalars() -> None:
    with pytest.raises(ApiResponseError, match="orders"):
        extract_list({"unexpected": 1}, "orders")
    with pytest.raises(ApiResponseError):
        extract_list("nope")


def test_equity_fills_missing_keys(make_client) -> None:
    client, _ = make_client(make_response(200, {}))
    assert client.equity() == {"points": [], "metrics": {}}


def test_backtest_body_and_validation(make_client) -> None:
    client, session = make_client(make_response(200, {"symbols": ["AAPL"]}))
    client.backtest(["aapl"], date(2022, 1, 3), "2023-06-30", initial_capital=25_000)
    assert session.last["json"] == {
        "symbols": ["AAPL"],
        "start": "2022-01-03",
        "end": "2023-06-30",
        "use_sentiment": False,
        "initial_capital": 25_000.0,
    }
    assert session.last["timeout"] == DEFAULT_LONG_TIMEOUT_SECONDS
    with pytest.raises(ApiValidationError, match="before"):
        client.backtest(["AAPL"], "2023-01-01", "2023-01-01")
    with pytest.raises(ApiValidationError, match="at most 5"):
        client.backtest(list("ABCDEF"), "2022-01-01", "2023-01-01")
    with pytest.raises(ApiValidationError, match="YYYY-MM-DD"):
        client.backtest(["AAPL"], "01/02/2022", "2023-01-01")
    assert len(session.calls) == 1


def test_backtest_accepts_datetimes(make_client) -> None:
    client, session = make_client(make_response(200, {}))
    client.backtest(["AAPL"], datetime(2022, 1, 3, 15, 30), datetime(2022, 6, 1))
    assert (session.last["json"]["start"], session.last["json"]["end"]) == (
        "2022-01-03",
        "2022-06-01",
    )


def test_watchlist_get_and_put(make_client) -> None:
    client, session = make_client(
        make_response(200, {"symbols": ["AAPL", "MSFT"]}),
        make_response(200, ["aapl"]),
        make_response(200, {"symbols": ["NVDA", "AMD"]}),
        make_response(204),
    )
    assert client.watchlist() == ["AAPL", "MSFT"]
    assert client.watchlist() == ["AAPL"]
    assert client.set_watchlist(["nvda", "amd"]) == ["NVDA", "AMD"]
    assert session.last["method"] == "PUT"
    assert session.last["json"] == {"symbols": ["NVDA", "AMD"]}
    assert client.set_watchlist(["tsla"]) == ["TSLA"]  # empty reply → what we sent
    with pytest.raises(ApiValidationError):
        client.set_watchlist([])


def test_watchlist_rejects_unexpected_shape(make_client) -> None:
    client, _ = make_client(make_response(200, {"symbols": "AAPL"}))
    with pytest.raises(ApiResponseError):
        client.watchlist()


def test_scan_and_test_alert(make_client) -> None:
    client, session = make_client(
        make_response(200, {"decisions": [{"symbol": "AAPL"}]}),
        make_response(200, {"sent": False, "channel": "email", "detail": "disabled"}),
    )
    scan = client.scan()
    assert scan == {"decisions": [{"symbol": "AAPL"}], "exits": [], "errors": []}
    assert session.last["method"] == "POST"
    assert session.last["url"] == f"{BASE}/scan"
    assert client.send_test_alert()["sent"] is False
    assert session.last["url"] == f"{BASE}/alerts/test"


def test_real_payloads_round_trip(make_client, decision_payload) -> None:
    """The client hands back the API's Decision JSON untouched."""
    client, _ = make_client(make_response(200, decision_payload))
    decision = client.analyze("BULL", auto_trade=True)
    assert decision == decision_payload
    assert decision["risk"]["approved"] is True
