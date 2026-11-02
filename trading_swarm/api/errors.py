"""The API's error envelope and the mapping from domain exceptions to HTTP statuses.

Every error response — validation failures, auth/rate-limit rejections, unknown routes,
market-data outages and unexpected crashes — has the same JSON shape::

    {
        "error": {
            "code": "not_found",
            "message": "no price data returned for XYZ",
            "request_id": "3f2a…",
        }
    }

``code`` is machine-readable and stable; ``message`` is safe to show to a user (it never
contains a traceback or a secret); ``request_id`` matches the ``X-Request-ID`` response
header and the ``rid=`` field of the server's log lines. Some errors add context keys
(``retry_after`` for 429s, ``details`` for validation errors, ``order`` for rejected
orders).

Domain → HTTP mapping
---------------------
==========================================  ======  =====================
exception                                   status  code
==========================================  ======  =====================
``SymbolNotFound`` (coordinator)            404     ``not_found``
``DataUnavailable`` (coordinator)           502     ``data_unavailable``
``InvalidSymbolError`` (malformed ticker)   422     ``invalid_symbol``
``NoDataError`` (unknown ticker / range)    404     ``not_found``
``ProviderUnavailableError`` / other        502     ``data_unavailable``
``MarketDataError``
``BacktestConfigError``                     422     ``invalid_request``
``OrderValidationError``                    422     ``invalid_order``
request-body / query validation             422     ``validation_error``
==========================================  ======  =====================
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from http import HTTPStatus
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from trading_swarm.backtest import BacktestConfigError
from trading_swarm.broker import OrderValidationError
from trading_swarm.data import (
    InvalidSymbolError,
    MarketDataError,
    NoDataError,
)
from trading_swarm.logging_setup import request_id_ctx
from trading_swarm.models import to_jsonable
from trading_swarm.swarm import DataUnavailable, SwarmError, SymbolNotFound

logger = logging.getLogger(__name__)

#: Default machine-readable code for an HTTP status without a more specific one.
STATUS_CODES: Final[Mapping[int, str]] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "data_unavailable",
    503: "service_unavailable",
}

#: How many individual validation problems are spelled out in ``message``.
_MAX_VALIDATION_MESSAGES: Final = 5
#: Location prefixes FastAPI puts in front of field paths (``("body", "symbols", 0)``).
_LOCATION_PREFIXES: Final = frozenset({"body", "query", "path", "header", "cookie"})


class ApiError(Exception):
    """An error the API reports deliberately (auth, rate limit, rejected order, …).

    Args:
        status_code: HTTP status.
        code: Machine-readable error code (see :data:`STATUS_CODES`).
        message: Human-readable explanation, safe to show to the client.
        headers: Extra response headers (``Retry-After``, ``WWW-Authenticate``).
        extra: Extra keys merged into the ``error`` object (JSON-safe values).
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        headers: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = dict(headers or {})
        self.extra = dict(extra or {})


def current_request_id(request: Request | None = None) -> str | None:
    """The id of the request being handled (set by the request-context middleware)."""
    request_id = request_id_ctx.get()
    if request_id is None and request is not None:
        request_id = getattr(request.state, "request_id", None)
    return request_id


def error_body(
    code: str, message: str, request_id: str | None, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The JSON envelope ``{"error": {"code", "message", "request_id", …extra}}``."""
    error: dict[str, Any] = {"code": code, "message": message, "request_id": request_id}
    if extra:
        for key, value in extra.items():
            error.setdefault(key, to_jsonable(value))
    return {"error": error}


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    request_id: str | None = None,
    headers: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> JSONResponse:
    """A :class:`JSONResponse` carrying the error envelope."""
    return JSONResponse(
        error_body(code, message, request_id, extra),
        status_code=status_code,
        headers=dict(headers) if headers else None,
    )


def _reason_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return f"HTTP {status_code}"


def _strip_value_error(message: str) -> str:
    # pydantic prefixes errors raised by validators with "Value error, ".
    prefix = "Value error, "
    return message[len(prefix) :] if message.startswith(prefix) else message


def validation_details(errors: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Client-facing validation problems: field path, message and pydantic error type.

    The offending ``input`` and pydantic's ``ctx`` are deliberately dropped: they can echo
    large or sensitive request bodies back to the caller and are not always JSON-safe.
    """
    details: list[dict[str, Any]] = []
    for error in errors:
        location = [part for part in error.get("loc", ()) if part not in _LOCATION_PREFIXES]
        details.append(
            {
                "field": ".".join(str(part) for part in location) or None,
                "message": _strip_value_error(str(error.get("msg", "invalid value"))),
                "type": str(error.get("type", "value_error")),
            }
        )
    return details


def validation_message(details: Sequence[Mapping[str, Any]]) -> str:
    """One readable line: ``"symbols: List should have at most 10 items; side: …"``."""
    parts = [
        f"{item['field']}: {item['message']}" if item.get("field") else str(item["message"])
        for item in details[:_MAX_VALIDATION_MESSAGES]
    ]
    remaining = len(details) - _MAX_VALIDATION_MESSAGES
    if remaining > 0:
        parts.append(f"(+{remaining} more)")
    return "; ".join(parts) or "invalid request"


# =============================================================================== handlers


async def _handle_api_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ApiError)
    return error_response(
        exc.status_code,
        exc.code,
        exc.message,
        request_id=current_request_id(request),
        headers=exc.headers,
        extra=exc.extra,
    )


async def _handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    status_code = exc.status_code
    detail = exc.detail
    message = detail if isinstance(detail, str) and detail else _reason_phrase(status_code)
    if status_code == 404 and message == _reason_phrase(404):
        message = f"no route for {request.method} {request.url.path}"
    return error_response(
        status_code,
        STATUS_CODES.get(status_code, "http_error"),
        message,
        request_id=current_request_id(request),
        headers=exc.headers,
    )


async def _handle_validation_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    details = validation_details(exc.errors())
    return error_response(
        422,
        "validation_error",
        validation_message(details),
        request_id=current_request_id(request),
        extra={"details": details},
    )


async def _handle_swarm_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, SwarmError)
    if isinstance(exc, SymbolNotFound):
        status_code, code = 404, "not_found"
    elif isinstance(exc, DataUnavailable):
        status_code, code = 502, "data_unavailable"
        logger.warning("market data unavailable for %s: %s", exc.symbol, exc.message)
    else:  # pragma: no cover - no other SwarmError subclasses exist today
        status_code, code = 500, "internal_error"
    return error_response(
        status_code,
        code,
        exc.message,
        request_id=current_request_id(request),
        extra={"symbol": exc.symbol} if exc.symbol else None,
    )


def market_data_status(exc: MarketDataError) -> tuple[int, str]:
    """HTTP status and code for a market-data failure (see the module docstring)."""
    if isinstance(exc, InvalidSymbolError):
        return 422, "invalid_symbol"
    if isinstance(exc, NoDataError):
        return 404, "not_found"
    return 502, "data_unavailable"


async def _handle_market_data_error(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, MarketDataError)
    status_code, code = market_data_status(exc)
    if status_code >= 500:
        logger.warning("market data unavailable for %s: %s", exc.symbol, exc.message)
    return error_response(
        status_code,
        code,
        exc.message,
        request_id=current_request_id(request),
        extra={"symbol": exc.symbol} if exc.symbol else None,
    )


async def _handle_backtest_config_error(request: Request, exc: Exception) -> JSONResponse:
    return error_response(422, "invalid_request", str(exc), request_id=current_request_id(request))


async def _handle_order_validation_error(request: Request, exc: Exception) -> JSONResponse:
    return error_response(422, "invalid_order", str(exc), request_id=current_request_id(request))


def install_exception_handlers(app: FastAPI) -> None:
    """Register the envelope-producing handlers on ``app``.

    Unexpected exceptions are *not* handled here: the request-context middleware turns
    them into a ``500 internal_error`` envelope (it sits inside Starlette's outermost
    error middleware, so it can still attach the request id).
    """
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_exception)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(SwarmError, _handle_swarm_error)
    app.add_exception_handler(MarketDataError, _handle_market_data_error)
    app.add_exception_handler(BacktestConfigError, _handle_backtest_config_error)
    app.add_exception_handler(OrderValidationError, _handle_order_validation_error)


__all__ = [
    "STATUS_CODES",
    "ApiError",
    "current_request_id",
    "error_body",
    "error_response",
    "install_exception_handlers",
    "market_data_status",
    "validation_details",
    "validation_message",
]
