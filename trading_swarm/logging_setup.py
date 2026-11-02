"""Logging configuration: human-readable lines for development, JSON lines for production.

Library modules only ever call ``logging.getLogger(__name__)``; the *application* (API
lifespan, CLI, scheduler entry point) calls :func:`configure_logging` exactly once.

A per-request correlation id can be attached to every record emitted while handling a
request by setting :data:`request_id_ctx` (the API's request-id middleware does this); the
:class:`RequestIdFilter` copies it onto each record as ``record.request_id``.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Final

#: Correlation id of the request currently being handled (``None`` outside requests).
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

_HANDLER_MARK: Final = "_trading_swarm_handler"

# Attributes present on every LogRecord; anything else was passed via ``extra=`` and is
# worth serialising.
_STANDARD_ATTRS: Final = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"message", "asctime", "request_id", "taskName"}
)

# Chatty third-party loggers that drown out the swarm's own messages at INFO.
_NOISY_LOGGERS: Final = (
    "yfinance",
    "peewee",
    "urllib3",
    "httpx",
    "httpx2",
    "httpcore",
    "anthropic",
    "openai",
    "multipart",
)


class RequestIdFilter(logging.Filter):
    """Inject the current :data:`request_id_ctx` value into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_ctx.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """Render records as single-line JSON objects (one event per line, UTC timestamps)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id and request_id != "-":
            payload["request_id"] = request_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


_TEXT_FORMAT: Final = "%(asctime)s %(levelname)-7s [%(name)s] [rid=%(request_id)s] %(message)s"


def configure_logging(level: str | int = "INFO", json: bool = False) -> None:
    """Configure the root logger once (idempotent: re-calling replaces our handler).

    Args:
        level: Log level name (``"INFO"``) or number for the application loggers.
        json: Emit JSON lines (for log shippers) instead of human-readable text.
    """
    numeric_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if not isinstance(numeric_level, int):
        raise ValueError(f"unknown log level {level!r}")

    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARK, False):
            root.removeHandler(handler)
            handler.close()

    handler = logging.StreamHandler(stream=sys.stderr)
    setattr(handler, _HANDLER_MARK, True)
    handler.addFilter(RequestIdFilter())
    if json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S%z"))
    root.addHandler(handler)
    root.setLevel(numeric_level)

    quiet_level = max(numeric_level, logging.WARNING)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(quiet_level)


__all__ = ["JsonFormatter", "RequestIdFilter", "configure_logging", "request_id_ctx"]
