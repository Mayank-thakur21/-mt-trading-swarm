"""Typed HTTP client for the MT Trading Swarm REST API.

The Streamlit dashboard never talks to the trading engine directly: every number it
shows comes through this client, so the dashboard can run in its own container (or on
another machine) and always reflects what the API — and therefore the paper account —
actually holds.

Design
------
* **One method per endpoint**, returning the decoded JSON with a :class:`~typing.TypedDict`
  describing its shape (the API serialises the ``trading_swarm.models`` pydantic models;
  the dashboard deliberately does not import the engine, so it stays a thin client).
* **Typed errors.** Every failure becomes an :class:`ApiError` subclass with a message
  that is safe to show to a user, the HTTP status, the API's machine-readable ``code`` and
  the ``request_id`` from the API's error envelope
  ``{"error": {"code", "message", "request_id"}}`` (FastAPI's ``{"detail": ...}`` shape is
  understood too). Callers map the subclasses to friendly UI messages; nothing here ever
  leaks a traceback or the API key.
* **Robust transport.** Connection errors are retried a couple of times with a short
  backoff (the request never reached the server, so a retry is safe for any method);
  HTTP error statuses are *not* retried — a 502 "data provider down" is information, not a
  glitch. Quick reads use a short timeout, analysis/backtest/scan calls a long one.

Configuration comes from the environment (:meth:`ApiClient.from_env`):

=========================  ==========================  =====================================
variable                   default                     meaning
=========================  ==========================  =====================================
``API_URL``                ``http://localhost:8000``   base URL of the FastAPI service
``API_KEY``                *(empty)*                   sent as ``X-API-Key`` when set
``API_TIMEOUT_SECONDS``    ``15``                      timeout for quick requests
``API_LONG_TIMEOUT_SECONDS`` ``300``                   analyze / batch / scan / backtest
=========================  ==========================  =====================================
"""

from __future__ import annotations

import email.utils
import hashlib
import logging
import math
import os
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, Final, Literal, NotRequired, Self, TypedDict, cast
from urllib.parse import quote, urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

DEFAULT_API_URL: Final = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS: Final = 15.0
#: Analysis may call an LLM and fetch a year of bars per symbol; backtests replay years.
DEFAULT_LONG_TIMEOUT_SECONDS: Final = 300.0
DEFAULT_CONNECT_RETRIES: Final = 2
API_KEY_HEADER: Final = "X-API-Key"
REQUEST_ID_HEADER: Final = "X-Request-ID"

#: Same ticker grammar as the API (``MarketDataProvider.validate_symbol``): upper-case
#: letters, digits and ``. - ^ =`` (``BRK.B``, ``^GSPC``, ``EURUSD=X``), 1-15 characters.
SYMBOL_PATTERN: Final = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
#: Request-size limits enforced by the API; checked locally for instant feedback.
MAX_BATCH_SYMBOLS: Final = 10
MAX_BACKTEST_SYMBOLS: Final = 5
#: Largest initial capital the API accepts (``schemas.MAX_CAPITAL``).
MAX_CAPITAL: Final = 1e12
#: Longest snippet of a non-JSON error body that is quoted back to the user.
_MAX_BODY_SNIPPET: Final = 300

# =============================================================================== payloads
# Shapes of the API's JSON responses (see SWARM_SPEC § API and trading_swarm.models).
# ``*_pct`` fields are percentages (5.0 == 5 %); signals are in [-1, 1], confidences in
# [0, 1]; timestamps are ISO-8601 UTC strings.


class HealthPayload(TypedDict):
    status: str
    version: str
    data_provider: str
    llm_provider: str
    llm_model: str
    email_enabled: bool
    scheduler_enabled: bool
    auto_trade: bool


class QuotePayload(TypedDict):
    symbol: str
    price: float
    change_pct_1d: float | None
    as_of: str


class CandlePayload(TypedDict):
    t: str
    o: float
    h: float
    l: float  # noqa: E741 - the API's compact OHLCV key
    c: float
    v: float


class HistoryPayload(TypedDict):
    symbol: str
    candles: list[CandlePayload]
    #: Series aligned with ``candles`` (``sma20, sma50, rsi, macd, signal``; nulls allowed).
    indicators: dict[str, list[float | None]]


class AgentSignalPayload(TypedDict):
    agent: str
    signal: float
    confidence: float
    rationale: str
    data: dict[str, Any]


class RiskPayload(TypedDict):
    approved: bool
    action: str
    quantity: int
    entry_price: float
    stop_price: float | None
    take_profit_price: float | None
    position_value: float
    risk_amount: float
    risk_score: float
    reasons: list[str]
    checks: dict[str, bool]


class OrderPayload(TypedDict):
    id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    type: str
    status: Literal["FILLED", "REJECTED", "CANCELLED", "PENDING"]
    requested_at: str
    filled_at: str | None
    fill_price: float | None
    commission: float
    slippage: float
    reason: str
    decision_id: str | None
    reject_reason: str | None
    limit_price: NotRequired[float | None]
    reference_price: NotRequired[float | None]
    stop_price: NotRequired[float | None]
    take_profit_price: NotRequired[float | None]
    realized_pnl: NotRequired[float | None]


class DecisionPayload(TypedDict):
    id: str
    symbol: str
    timestamp: str
    price: float
    action: Literal["BUY", "SELL", "HOLD"]
    score: float
    confidence: float
    signals: list[AgentSignalPayload]
    risk: RiskPayload
    summary: str
    executed: bool
    order: OrderPayload | None
    #: Protective exits the analysis made before sizing (any symbol; may be absent in
    #: decisions stored by older API versions).
    exits: NotRequired[list[OrderPayload]]


class TradePayload(TypedDict):
    id: str
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    price: float
    commission: float
    realized_pnl: float | None
    timestamp: str
    reason: str


class PositionPayload(TypedDict):
    symbol: str
    quantity: int
    avg_price: float
    stop_price: float | None
    take_profit_price: float | None
    opened_at: str
    market_price: float | None
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float


class PortfolioPayload(TypedDict):
    initial_capital: float
    cash: float
    equity: float
    positions_value: float
    gross_exposure_pct: float
    realized_pnl: float
    unrealized_pnl: float
    total_return: float
    total_return_pct: float
    peak_equity: float
    drawdown_pct: float
    positions: list[PositionPayload]
    updated_at: str


class EquityPointPayload(TypedDict):
    t: str
    equity: float
    cash: float


class EquityPayload(TypedDict):
    points: list[EquityPointPayload]
    #: ``trading_swarm.analytics.PerformanceMetrics`` (values may be null).
    metrics: dict[str, Any]


class SymbolErrorPayload(TypedDict):
    symbol: str
    error: str
    code: NotRequired[str]


class BatchPayload(TypedDict):
    results: list[DecisionPayload]
    errors: list[SymbolErrorPayload]


class ScanPayload(TypedDict):
    decisions: list[DecisionPayload]
    exits: list[OrderPayload]
    errors: list[SymbolErrorPayload]
    snapshot_recorded: NotRequired[bool]


class StatusPayload(TypedDict):
    """``GET /status``: scheduler state and alerting configuration (needs the API key)."""

    scheduler: dict[str, Any]
    alerts: dict[str, Any]


class AlertPayload(TypedDict):
    id: int | None
    timestamp: str
    rule: str
    symbol: str | None
    channel: str
    status: Literal["sent", "failed", "disabled", "suppressed"]
    subject: str
    detail: str


class AlertTestPayload(TypedDict):
    sent: bool
    channel: str
    detail: str


#: ``BacktestResult.to_dict()``: symbols, start, end, initial_capital, parameters,
#: equity_curve, trades, metrics, benchmark_metrics (and optionally decisions). The
#: equity-curve encoding is normalised by :func:`dashboard.charts.backtest_curve_frame`.
BacktestPayload = dict[str, Any]

# =============================================================================== errors


class ApiError(Exception):
    """Any failure talking to the API.

    Attributes:
        message: Human-readable explanation, safe to display.
        status_code: HTTP status, or ``None`` when no response was received.
        code: Machine-readable error code from the API envelope (``"not_found"`` …).
        request_id: The API's request id, for correlating with its logs.
        url: The URL that was requested (never contains the API key).
        retry_after: Seconds to wait before retrying (rate limiting), when known.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str = "error",
        request_id: str | None = None,
        url: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.url = url
        self.retry_after = retry_after

    def __str__(self) -> str:
        status = f"HTTP {self.status_code}: " if self.status_code else ""
        rid = f" (request id {self.request_id})" if self.request_id else ""
        return f"{status}{self.message}{rid}"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status_code={self.status_code!r}, code={self.code!r}, "
            f"message={self.message!r}, request_id={self.request_id!r})"
        )


class ApiConnectionError(ApiError):
    """The API could not be reached (refused, DNS failure, TLS error, connect timeout)."""


class ApiTimeoutError(ApiConnectionError):
    """The API accepted the connection but did not answer within the timeout."""


class ApiAuthError(ApiError):
    """HTTP 401/403: missing or invalid ``X-API-Key``."""


class ApiRateLimitError(ApiError):
    """HTTP 429: too many requests; :attr:`retry_after` says when to try again."""


class ApiNotFoundError(ApiError):
    """HTTP 404: unknown symbol (no market data) or unknown resource."""


class ApiValidationError(ApiError):
    """The request was rejected as invalid: HTTP 400/409/422, or caught locally.

    Locally detected problems (a malformed ticker, too many symbols) have
    ``status_code=None`` and are raised before any network traffic.
    """


class ApiServerError(ApiError):
    """HTTP 5xx: the API (or its market-data / LLM upstream, for 502/504) failed."""


class ApiResponseError(ApiError):
    """A response arrived but was not the JSON the client expected."""


def _status_error_class(status: int) -> type[ApiError]:
    if status in (401, 403):
        return ApiAuthError
    if status == 404:
        return ApiNotFoundError
    if status == 429:
        return ApiRateLimitError
    if status in (400, 409, 413, 422):
        return ApiValidationError
    if status >= 500:
        return ApiServerError
    return ApiError


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    """Seconds encoded in a ``Retry-After`` header (delta-seconds or an HTTP date)."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        current = time.time() if now is None else now
        seconds = when.timestamp() - current
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def _describe_validation_detail(detail: Any) -> str:
    """Flatten FastAPI's ``detail`` (a string or a list of pydantic errors) to one line."""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, Mapping):
                loc = [str(p) for p in item.get("loc", ()) if p not in ("body", "query", "path")]
                msg = str(item.get("msg", "invalid value"))
                parts.append(f"{'.'.join(loc)}: {msg}" if loc else msg)
            else:
                parts.append(str(item))
        return "; ".join(parts) or "invalid request"
    if isinstance(detail, Mapping):
        return str(detail.get("message") or detail.get("msg") or detail)
    return str(detail)


def _body_snippet(response: requests.Response) -> str:
    text = re.sub(r"<[^>]+>", " ", response.text or "")  # strip HTML error pages
    text = " ".join(text.split())
    if len(text) > _MAX_BODY_SNIPPET:
        text = text[: _MAX_BODY_SNIPPET - 1] + "…"
    return text


def error_from_response(response: requests.Response) -> ApiError:
    """Build the matching :class:`ApiError` subclass for an HTTP error response.

    Understands the API's envelope ``{"error": {"code", "message", "request_id"}}``,
    FastAPI's ``{"detail": ...}`` and plain-text bodies (proxies, crashed workers).
    """
    status = response.status_code
    code = "error"
    message = ""
    request_id = response.headers.get(REQUEST_ID_HEADER)
    retry_after = parse_retry_after(response.headers.get("Retry-After"))
    try:
        body: Any = response.json()
    except ValueError:
        body = None

    if isinstance(body, Mapping):
        envelope = body.get("error")
        if isinstance(envelope, Mapping):
            code = str(envelope.get("code") or code)
            message = _describe_validation_detail(envelope.get("message") or "")
            request_id = str(envelope.get("request_id") or request_id or "") or None
            if retry_after is None:
                raw = envelope.get("retry_after")
                if isinstance(raw, int | float) and not isinstance(raw, bool):
                    retry_after = max(0.0, float(raw))
        elif isinstance(envelope, str):
            message = envelope
        if not message and "detail" in body:
            message = _describe_validation_detail(body["detail"])
        if not message and isinstance(body.get("message"), str):
            message = body["message"]
    if not message:
        message = _body_snippet(response) or response.reason or f"HTTP {status}"

    cls = _status_error_class(status)
    return cls(
        message,
        status_code=status,
        code=code,
        request_id=request_id,
        url=response.url or None,
        retry_after=retry_after,
    )


# =============================================================================== helpers


def normalize_symbol(symbol: str) -> str:
    """Upper-case and validate a ticker (raises :class:`ApiValidationError` if malformed)."""
    if not isinstance(symbol, str):
        raise ApiValidationError("symbol must be a string", code="invalid_symbol")
    cleaned = symbol.strip().upper()
    if not SYMBOL_PATTERN.fullmatch(cleaned):
        shown = cleaned or "(empty)"
        raise ApiValidationError(
            f"'{shown}' is not a valid ticker: use 1-15 letters, digits or . - ^ =",
            code="invalid_symbol",
        )
    return cleaned


def normalize_symbols(
    symbols: Iterable[str], *, min_count: int = 1, max_count: int | None = None
) -> list[str]:
    """Validate, upper-case and de-duplicate tickers (order preserved)."""
    if isinstance(symbols, str):
        symbols = symbols.split(",")
    seen: dict[str, None] = {}
    for raw in symbols:
        if isinstance(raw, str) and not raw.strip():
            continue
        seen.setdefault(normalize_symbol(raw), None)
    result = list(seen)
    if len(result) < min_count:
        noun = "symbol" if min_count == 1 else "symbols"
        raise ApiValidationError(f"enter at least {min_count} {noun}", code="invalid_symbols")
    if max_count is not None and len(result) > max_count:
        raise ApiValidationError(
            f"at most {max_count} symbols per request ({len(result)} given)",
            code="invalid_symbols",
        )
    return result


def _iso_date(value: date | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError as exc:
        raise ApiValidationError(
            f"invalid date {value!r}: use YYYY-MM-DD", code="invalid_date"
        ) from exc


def _positive_number(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiValidationError(f"{name} must be a number", code="invalid_value") from exc
    if not math.isfinite(number) or number <= 0:
        raise ApiValidationError(f"{name} must be greater than 0", code="invalid_value")
    return number


def extract_list(payload: Any, *keys: str) -> list[dict[str, Any]]:
    """Return the list in ``payload``: either the payload itself or ``payload[key]``.

    List endpoints may answer with a bare JSON array or wrap it (``{"orders": [...]}``,
    ``{"items": [...]}``); both are accepted so the dashboard does not break on either.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        for key in (*keys, "items", "results", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    raise ApiResponseError(
        f"expected a list{' under ' + '/'.join(keys) if keys else ''}, "
        f"got {type(payload).__name__}",
        code="unexpected_response",
    )


def _expect_mapping(payload: Any, what: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ApiResponseError(
            f"expected a JSON object for {what}, got {type(payload).__name__}",
            code="unexpected_response",
        )
    return dict(payload)


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("ignoring %s=%r: not a number (using %s)", name, raw, default)
        return default
    if not math.isfinite(value) or value <= 0:
        logger.warning("ignoring %s=%r: must be > 0 (using %s)", name, raw, default)
        return default
    return value


# =============================================================================== client


class ApiClient:
    """Synchronous client for the trading-swarm API.

    Args:
        base_url: Root URL of the API (``http://host:8000``); a trailing slash is fine.
        api_key: Value for the ``X-API-Key`` header (omit when the API has auth disabled).
        timeout: Seconds to wait for quick requests (quotes, lists, portfolio).
        long_timeout: Seconds to wait for analysis, batch, scan and backtest calls.
        retries: How many times to retry a request that could not *connect*.
        session: Optional pre-built :class:`requests.Session` (tests inject fakes); when
            omitted the client owns a session with connection-retry adapters.

    The client is cheap to create and holds one connection pool; call :meth:`close` (or
    use it as a context manager) when done.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        api_key: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        long_timeout: float = DEFAULT_LONG_TIMEOUT_SECONDS,
        retries: int = DEFAULT_CONNECT_RETRIES,
        session: requests.Session | None = None,
    ) -> None:
        url = (base_url or "").strip().rstrip("/")
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError(f"API URL must be an absolute http(s) URL, got {base_url!r}")
        if timeout <= 0 or long_timeout <= 0:
            raise ValueError("timeouts must be positive")
        self.base_url = url
        self.api_key = (api_key or "").strip() or None
        self.timeout = float(timeout)
        self.long_timeout = float(long_timeout)
        self._owns_session = session is None
        self.session = session if session is not None else self._build_session(retries)

    @staticmethod
    def _build_session(retries: int) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=retries,
            connect=retries,
            # The server may already be working on it: never resend after a read error.
            # ``False`` (not 0) re-raises the original ReadTimeout instead of wrapping it
            # in MaxRetryError, which requests would surface as a ConnectionError.
            read=False,
            status=0,  # HTTP errors carry meaning (404 / 429 / 502): surface them
            other=0,
            backoff_factor=0.25,
            allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"Accept": "application/json", "User-Agent": "mt-swarm-dashboard"})
        return session

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, **overrides: Any) -> Self:
        """Build a client from ``API_URL`` / ``API_KEY`` / ``API_*TIMEOUT_SECONDS``."""
        env = os.environ if environ is None else environ
        kwargs: dict[str, Any] = {
            "base_url": env.get("API_URL", "").strip() or DEFAULT_API_URL,
            "api_key": env.get("API_KEY", "").strip() or None,
            "timeout": _env_float(env, "API_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
            "long_timeout": _env_float(
                env, "API_LONG_TIMEOUT_SECONDS", DEFAULT_LONG_TIMEOUT_SECONDS
            ),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Release pooled connections (only when this client created the session)."""
        if self._owns_session:
            self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:  # never show the key itself
        auth = "set" if self.api_key else "none"
        return f"ApiClient(base_url={self.base_url!r}, api_key={auth})"

    @property
    def cache_key(self) -> str:
        """Stable identity for response caches: base URL + a fingerprint of the key.

        Responses depend on which API (and which credentials) answered; the raw key is
        never part of the cache key.
        """
        fingerprint = (
            hashlib.sha256(self.api_key.encode()).hexdigest()[:16] if self.api_key else "-"
        )
        return f"{self.base_url}#{fingerprint}"

    # ------------------------------------------------------------------ transport
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> Any:
        """Send a request and return the decoded JSON body (``None`` for empty bodies).

        Raises:
            ApiError: a subclass describing what went wrong (see module docstring).
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {API_KEY_HEADER: self.api_key} if self.api_key else {}
        clean_params = (
            {k: v for k, v in params.items() if v is not None} if params is not None else None
        )
        started = time.perf_counter()
        try:
            response = self.session.request(
                method,
                url,
                params=clean_params,
                json=json,
                headers=headers,
                timeout=timeout or self.timeout,
            )
        except requests.exceptions.ReadTimeout as exc:
            raise ApiTimeoutError(
                f"the API at {self.base_url} did not answer within {timeout or self.timeout:.0f} s",
                code="timeout",
                url=url,
            ) from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            # ConnectTimeout is both a ConnectionError and a Timeout: unreachable either way.
            raise ApiConnectionError(
                f"cannot reach the API at {self.base_url}", code="unreachable", url=url
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise ApiConnectionError(
                f"request to {self.base_url} failed: {type(exc).__name__}",
                code="request_failed",
                url=url,
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.debug(
            "%s %s -> %s in %.0f ms (rid=%s)",
            method,
            path,
            response.status_code,
            elapsed_ms,
            response.headers.get(REQUEST_ID_HEADER, "-"),
        )
        if response.status_code >= 400:
            error = error_from_response(response)
            log = logger.warning if response.status_code >= 500 else logger.info
            log("%s %s failed: %r", method, path, error)
            raise error
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise ApiResponseError(
                f"the API returned a non-JSON response for {path}",
                status_code=response.status_code,
                code="invalid_json",
                request_id=response.headers.get(REQUEST_ID_HEADER),
                url=url,
            ) from exc

    def _get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return self.request("GET", path, params=params)

    # ------------------------------------------------------------------ system
    def health(self) -> HealthPayload:
        """``GET /health`` — liveness plus the active configuration (public endpoint)."""
        return cast(HealthPayload, _expect_mapping(self._get("/health"), "/health"))

    def status(self) -> StatusPayload:
        """``GET /status`` — scheduler status and alerting configuration (protected)."""
        return cast(StatusPayload, _expect_mapping(self._get("/status"), "/status"))

    # ------------------------------------------------------------------ market data
    def quote(self, symbol: str) -> QuotePayload:
        """``GET /quote/{symbol}`` — latest price and the 1-day change in percent."""
        sym = normalize_symbol(symbol)
        payload = self._get(f"/quote/{quote(sym, safe='')}")
        return cast(QuotePayload, _expect_mapping(payload, "/quote"))

    def history(self, symbol: str, period: str = "6mo", interval: str = "1d") -> HistoryPayload:
        """``GET /history/{symbol}`` — OHLCV candles plus indicator series."""
        sym = normalize_symbol(symbol)
        payload = self._get(
            f"/history/{quote(sym, safe='')}", {"period": period, "interval": interval}
        )
        return cast(HistoryPayload, _expect_mapping(payload, "/history"))

    # ------------------------------------------------------------------ analysis
    def analyze(self, symbol: str, auto_trade: bool | None = None) -> DecisionPayload:
        """``POST /analyze`` — run the swarm on one symbol (paper-trades if ``auto_trade``)."""
        body: dict[str, Any] = {"symbol": normalize_symbol(symbol)}
        if auto_trade is not None:
            body["auto_trade"] = bool(auto_trade)
        payload = self.request("POST", "/analyze", json=body, timeout=self.long_timeout)
        return cast(DecisionPayload, _expect_mapping(payload, "/analyze"))

    def analyze_batch(self, symbols: Iterable[str], auto_trade: bool | None = None) -> BatchPayload:
        """``POST /analyze/batch`` — up to 10 symbols; failures come back per symbol."""
        body: dict[str, Any] = {"symbols": normalize_symbols(symbols, max_count=MAX_BATCH_SYMBOLS)}
        if auto_trade is not None:
            body["auto_trade"] = bool(auto_trade)
        payload = _expect_mapping(
            self.request("POST", "/analyze/batch", json=body, timeout=self.long_timeout),
            "/analyze/batch",
        )
        payload.setdefault("results", [])
        payload.setdefault("errors", [])
        return cast(BatchPayload, payload)

    def decisions(self, limit: int = 50, symbol: str | None = None) -> list[DecisionPayload]:
        """``GET /decisions`` — most recent decisions first (optionally for one symbol)."""
        params: dict[str, Any] = {"limit": int(limit)}
        if symbol:
            params["symbol"] = normalize_symbol(symbol)
        return cast(
            list[DecisionPayload], extract_list(self._get("/decisions", params), "decisions")
        )

    # ------------------------------------------------------------------ portfolio
    def portfolio(self) -> PortfolioPayload:
        """``GET /portfolio`` — account snapshot marked to the latest prices."""
        return cast(PortfolioPayload, _expect_mapping(self._get("/portfolio"), "/portfolio"))

    def reset_portfolio(self, initial_capital: float | None = None) -> PortfolioPayload:
        """``POST /portfolio/reset`` — wipe positions/orders/trades and restart with cash."""
        body: dict[str, Any] = {}
        capital = _positive_number(initial_capital, "initial capital")
        if capital is not None:
            body["initial_capital"] = capital
        payload = self.request("POST", "/portfolio/reset", json=body)
        return cast(PortfolioPayload, _expect_mapping(payload, "/portfolio/reset"))

    def equity(self) -> EquityPayload:
        """``GET /equity`` — equity snapshots and performance metrics of the paper account."""
        payload = _expect_mapping(self._get("/equity"), "/equity")
        payload.setdefault("points", [])
        payload.setdefault("metrics", {})
        return cast(EquityPayload, payload)

    # ------------------------------------------------------------------ orders & trades
    def place_order(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"] | str,
        quantity: int,
        *,
        stop_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> OrderPayload:
        """``POST /orders`` — manual paper order at the API's latest price.

        The fill price is always taken from the market-data provider by the API; the
        client cannot (and does not try to) supply one.
        """
        side_value = str(side).strip().upper()
        if side_value not in ("BUY", "SELL"):
            raise ApiValidationError("side must be BUY or SELL", code="invalid_side")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ApiValidationError(
                "quantity must be a whole number of shares > 0", code="invalid_quantity"
            )
        body: dict[str, Any] = {
            "symbol": normalize_symbol(symbol),
            "side": side_value,
            "quantity": quantity,
        }
        stop = _positive_number(stop_price, "stop price")
        target = _positive_number(take_profit_price, "take-profit price")
        if stop is not None:
            body["stop_price"] = stop
        if target is not None:
            body["take_profit_price"] = target
        payload = self.request("POST", "/orders", json=body)
        return cast(OrderPayload, _expect_mapping(payload, "/orders"))

    def orders(self, limit: int = 50) -> list[OrderPayload]:
        """``GET /orders`` — most recent orders first."""
        payload = self._get("/orders", {"limit": int(limit)})
        return cast(list[OrderPayload], extract_list(payload, "orders"))

    def trades(self, limit: int = 100) -> list[TradePayload]:
        """``GET /trades`` — most recent fills first."""
        payload = self._get("/trades", {"limit": int(limit)})
        return cast(list[TradePayload], extract_list(payload, "trades"))

    # ------------------------------------------------------------------ backtest
    def backtest(
        self,
        symbols: Iterable[str],
        start: date | str,
        end: date | str,
        *,
        initial_capital: float | None = None,
        use_sentiment: bool = False,
    ) -> BacktestPayload:
        """``POST /backtest`` — replay the trend + risk logic over daily bars (1-5 symbols)."""
        start_iso, end_iso = _iso_date(start), _iso_date(end)
        if start_iso >= end_iso:
            raise ApiValidationError("start date must be before end date", code="invalid_dates")
        body: dict[str, Any] = {
            "symbols": normalize_symbols(symbols, max_count=MAX_BACKTEST_SYMBOLS),
            "start": start_iso,
            "end": end_iso,
            "use_sentiment": bool(use_sentiment),
        }
        capital = _positive_number(initial_capital, "initial capital")
        if capital is not None:
            body["initial_capital"] = capital
        payload = self.request("POST", "/backtest", json=body, timeout=self.long_timeout)
        return _expect_mapping(payload, "/backtest")

    # ------------------------------------------------------------------ alerts
    def alerts(self, limit: int = 50) -> list[AlertPayload]:
        """``GET /alerts`` — the alert log (sent, failed, disabled and suppressed)."""
        payload = self._get("/alerts", {"limit": int(limit)})
        return cast(list[AlertPayload], extract_list(payload, "alerts"))

    def send_test_alert(self) -> AlertTestPayload:
        """``POST /alerts/test`` — send a test e-mail through the configured SMTP server."""
        payload = self.request("POST", "/alerts/test", timeout=self.long_timeout)
        return cast(AlertTestPayload, _expect_mapping(payload, "/alerts/test"))

    # ------------------------------------------------------------------ watchlist & scan
    def watchlist(self) -> list[str]:
        """``GET /watchlist`` — the persisted watchlist."""
        return _symbols_from(self._get("/watchlist"))

    def set_watchlist(self, symbols: Sequence[str]) -> list[str]:
        """``PUT /watchlist`` — replace the persisted watchlist; returns the stored list."""
        cleaned = normalize_symbols(symbols, min_count=1)
        payload = self.request("PUT", "/watchlist", json={"symbols": cleaned})
        return _symbols_from(payload) if payload is not None else cleaned

    def scan(self) -> ScanPayload:
        """``POST /scan`` — analyse the watchlist now, enforce stops, snapshot equity."""
        payload = _expect_mapping(self.request("POST", "/scan", timeout=self.long_timeout), "/scan")
        for key in ("decisions", "exits", "errors"):
            payload.setdefault(key, [])
        return cast(ScanPayload, payload)


def _symbols_from(payload: Any) -> list[str]:
    if isinstance(payload, Mapping):
        payload = payload.get("symbols", payload.get("watchlist"))
    if not isinstance(payload, list):
        raise ApiResponseError("expected a list of symbols", code="unexpected_response")
    return [str(s).upper() for s in payload if isinstance(s, str) and s.strip()]


__all__ = [
    "API_KEY_HEADER",
    "DEFAULT_API_URL",
    "DEFAULT_LONG_TIMEOUT_SECONDS",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_BACKTEST_SYMBOLS",
    "MAX_BATCH_SYMBOLS",
    "MAX_CAPITAL",
    "REQUEST_ID_HEADER",
    "AgentSignalPayload",
    "AlertPayload",
    "AlertTestPayload",
    "ApiAuthError",
    "ApiClient",
    "ApiConnectionError",
    "ApiError",
    "ApiNotFoundError",
    "ApiRateLimitError",
    "ApiResponseError",
    "ApiServerError",
    "ApiTimeoutError",
    "ApiValidationError",
    "BacktestPayload",
    "BatchPayload",
    "CandlePayload",
    "DecisionPayload",
    "EquityPayload",
    "EquityPointPayload",
    "HealthPayload",
    "HistoryPayload",
    "OrderPayload",
    "PortfolioPayload",
    "PositionPayload",
    "QuotePayload",
    "RiskPayload",
    "ScanPayload",
    "StatusPayload",
    "SymbolErrorPayload",
    "TradePayload",
    "error_from_response",
    "extract_list",
    "normalize_symbol",
    "normalize_symbols",
    "parse_retry_after",
]
