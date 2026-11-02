"""Request-context middleware: request ids, access logging and the last-resort 500 envelope.

Written as a *pure ASGI* middleware (not ``BaseHTTPMiddleware``) so that:

* the request id stored in :data:`~trading_swarm.logging_setup.request_id_ctx` is visible
  to the endpoint, to its dependencies and to every log line they emit — including code
  running in FastAPI's thread pool (sync endpoints) and in the coordinator's worker
  threads, which copy the context;
* streaming responses are not buffered;
* an unexpected exception can still be answered with the standard error envelope *and*
  the ``X-Request-ID`` header (Starlette's own ``ServerErrorMiddleware`` sits outside
  every user middleware and knows nothing about request ids).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Collection
from typing import Final
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from trading_swarm.api.errors import error_response
from trading_swarm.logging_setup import request_id_ctx

logger = logging.getLogger(__name__)
access_logger = logging.getLogger("trading_swarm.api.access")

REQUEST_ID_HEADER: Final = "X-Request-ID"
#: Client-supplied request ids are honoured only when they look like an id (no spaces,
#: no control characters, bounded length) so they cannot forge or bloat log lines.
_VALID_REQUEST_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:\-]{0,63}$")


def new_request_id() -> str:
    """A fresh 32-hex-character request id."""
    return uuid4().hex


def resolve_request_id(incoming: str | None) -> str:
    """Reuse a well-formed incoming ``X-Request-ID`` (end-to-end tracing), else mint one."""
    if incoming:
        candidate = incoming.strip()
        if _VALID_REQUEST_ID.fullmatch(candidate):
            return candidate
    return new_request_id()


class RequestContextMiddleware:
    """Assigns a request id, logs one access line per request and catches crashes.

    Args:
        app: The wrapped ASGI application.
        quiet_paths: Paths whose successful requests are logged at DEBUG instead of INFO
            (container health checks would otherwise flood the log).
    """

    def __init__(self, app: ASGIApp, *, quiet_paths: Collection[str] = ("/health",)) -> None:
        self.app = app
        self.quiet_paths = frozenset(quiet_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope).get(REQUEST_ID_HEADER))
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_ctx.set(request_id)
        method = scope.get("method", "-")
        path = scope.get("path", "-")
        started = time.perf_counter()
        response_started = False
        status_code = 500

        async def send_with_request_id(message: Message) -> None:
            nonlocal response_started, status_code
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception("unhandled error while serving %s %s", method, path)
            if response_started:  # too late for an error body; let the server drop it
                raise
            response = error_response(
                500,
                "internal_error",
                "internal server error; quote the request id when reporting this",
                request_id=request_id,
            )
            await response(scope, receive, send_with_request_id)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._log_access(scope, method, path, status_code, elapsed_ms)
            request_id_ctx.reset(token)

    def _log_access(
        self, scope: Scope, method: str, path: str, status_code: int, elapsed_ms: float
    ) -> None:
        if status_code >= 500:
            level = logging.WARNING
        elif path in self.quiet_paths and status_code < 400:
            level = logging.DEBUG
        else:
            level = logging.INFO
        if not access_logger.isEnabledFor(level):
            return
        client = scope.get("client")
        client_host = client[0] if client else "-"
        # The query string is not logged: it is rarely useful and could carry identifiers.
        access_logger.log(
            level,
            "%s %s -> %d (%.1f ms)",
            method,
            path,
            status_code,
            elapsed_ms,
            extra={
                "http_method": method,
                "http_path": path,
                "http_status": status_code,
                "duration_ms": round(elapsed_ms, 1),
                "client": client_host,
            },
        )


__all__ = [
    "REQUEST_ID_HEADER",
    "RequestContextMiddleware",
    "new_request_id",
    "resolve_request_id",
]
