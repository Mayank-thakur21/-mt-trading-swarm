"""FastAPI application factory and the module-level ``app`` for uvicorn.

Run the API with::

    uvicorn trading_swarm.api.main:app --host 0.0.0.0 --port 8000 --no-access-log

(``--no-access-log`` because the request-context middleware already writes one access
line per request, tagged with its request id.)

Request pipeline (outermost first)::

    CORS (only when CORS_ORIGINS is set) → request context (X-Request-ID, access log,
         500 envelope) → gzip → exception handlers (error envelope) → router
         → enforce_access (rate limit, cross-site guard, API key) → endpoint (thread pool)

The lifespan builds every long-lived object once — market-data provider, LLM client,
paper broker, agents, coordinator, alert manager, scheduler, backtester (see
:mod:`trading_swarm.api.deps`) — starts the scheduler when ``SCHEDULER_ENABLED=true``,
and on shutdown stops the scheduler (letting a running scan finish) *before* closing the
database.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Final

import anyio.to_thread
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from trading_swarm import __version__
from trading_swarm.api.deps import ServiceContainer, build_services
from trading_swarm.api.errors import install_exception_handlers
from trading_swarm.api.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from trading_swarm.api.routes import (
    TAG_ALERTS,
    TAG_ANALYSIS,
    TAG_BACKTEST,
    TAG_MARKET,
    TAG_ORDERS,
    TAG_PORTFOLIO,
    TAG_SYSTEM,
    TAG_WATCHLIST,
    protected,
    public,
)
from trading_swarm.api.security import (
    API_KEY_HEADER,
    RATE_LIMIT_WINDOW_SECONDS,
    AllowedOrigins,
    ApiKeyVerifier,
    SlidingWindowRateLimiter,
)
from trading_swarm.broker import PaperBroker
from trading_swarm.config import Settings, get_settings
from trading_swarm.data import MarketDataProvider
from trading_swarm.llm import LLMClient
from trading_swarm.logging_setup import configure_logging

logger = logging.getLogger(__name__)

API_TITLE: Final = "MT Trading Swarm API"

API_DESCRIPTION: Final = """
Multi-agent **paper-trading** swarm: a technical *trend* agent and a news *sentiment*
agent (Claude by default, lexicon fallback) vote on each symbol; a *risk* agent sizes the
trade from ATR stops and vetoes unsafe ones; an *execution* agent fills approved trades on
a SQLite-backed paper account. Nothing here places real orders. Not financial advice.

**Authentication.** Send `X-API-Key: <key>` when the server has `API_KEYS` set
(authentication is disabled when it is empty). `/health` is public and answers the
same with or without a key; scheduler and alerting details are at `/status`.

**Rate limiting.** At most `RATE_LIMIT_PER_MINUTE` requests per client in any 60-second
window (per API key, or per IP without one). Excess requests get `429` and a
`Retry-After` header; `X-RateLimit-Limit` / `X-RateLimit-Remaining` report the budget.

**Errors.** Every error uses one envelope —
`{"error": {"code": "...", "message": "...", "request_id": "..."}}` — and every response
carries an `X-Request-ID` header (send your own to correlate logs end to end).
Unknown / unpriceable symbols are `404 not_found`; a market-data outage is
`502 data_unavailable`; malformed input is `422`. A state-changing request that a
browser sends from an origin not listed in `CORS_ORIGINS` is `403 forbidden_origin`
(cross-site request forgery guard; non-browser clients are unaffected).

**Units.** Fields ending in `_pct` are percentages (`5.0` = 5 %); signals are in
[-1, 1] (+1 = maximally bullish); confidences in [0, 1]; timestamps are ISO-8601 UTC.
"""

TAGS_METADATA: Final = [
    {"name": TAG_SYSTEM, "description": "Liveness, configuration and operational status."},
    {
        "name": TAG_MARKET,
        "description": "Quotes and chart data from the configured provider (`DATA_PROVIDER`).",
    },
    {
        "name": TAG_ANALYSIS,
        "description": "Run the agent swarm: consensus of trend + sentiment, risk gate, "
        "optional paper execution; decision history; watchlist scans.",
    },
    {
        "name": TAG_PORTFOLIO,
        "description": "The paper account: snapshot marked to market, reset, equity curve "
        "and performance metrics.",
    },
    {
        "name": TAG_ORDERS,
        "description": "Manual paper orders (always at the provider's latest price), order "
        "and fill history.",
    },
    {
        "name": TAG_BACKTEST,
        "description": "Historical simulation of the same trend/risk logic with next-bar "
        "fills and an equal-weight buy-and-hold benchmark.",
    },
    {"name": TAG_ALERTS, "description": "E-mail alert log and test message."},
    {"name": TAG_WATCHLIST, "description": "Symbols scanned by `POST /scan` and the scheduler."},
]


@dataclass(frozen=True)
class _Overrides:
    """Objects injected by :func:`create_app` instead of being built from settings."""

    provider: MarketDataProvider | None = None
    llm: LLMClient | None = None
    broker: PaperBroker | None = None
    configure_logs: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the services on start-up; stop the scheduler and close the DB on shutdown."""
    settings: Settings = app.state.settings
    overrides: _Overrides = app.state.overrides
    if overrides.configure_logs:
        configure_logging(settings.log_level, json=settings.log_json)

    # Opening the database (and, for Yahoo, nothing else yet) is blocking I/O.
    services: ServiceContainer = await anyio.to_thread.run_sync(
        partial(
            build_services,
            settings,
            provider=overrides.provider,
            llm=overrides.llm,
            broker=overrides.broker,
        )
    )
    try:
        services.scheduler.start()
        if not settings.auth_enabled:
            log = logger.error if settings.is_production else logger.warning
            log(
                "API authentication is DISABLED (API_KEYS is empty): anyone who can reach "
                "this port can trade the paper account"
            )
        logger.info(
            "%s %s ready: data=%s, llm=%s%s, auto_trade=%s, scheduler=%s, email=%s, "
            "auth=%s, rate limit=%d/min",
            API_TITLE,
            __version__,
            getattr(services.provider, "name", settings.data_provider),
            services.llm.name,
            f" ({services.llm.model})" if services.llm.model else "",
            settings.auto_trade,
            "on" if settings.scheduler_enabled else "off",
            "on" if settings.email_enabled else "off",
            "on" if settings.auth_enabled else "off",
            settings.rate_limit_per_minute,
        )
        app.state.services = services
        yield
    finally:
        app.state.services = None
        logger.info("shutting down: stopping the scheduler and closing the database")
        await anyio.to_thread.run_sync(services.close)


def create_app(
    settings: Settings | None = None,
    *,
    provider: MarketDataProvider | None = None,
    llm: LLMClient | None = None,
    broker: PaperBroker | None = None,
    configure_logs: bool = True,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Configuration (default: :func:`~trading_swarm.config.get_settings`).
        provider: Market-data provider to use instead of ``DATA_PROVIDER``'s.
        llm: LLM client to use instead of ``LLM_PROVIDER``'s (e.g. a fake in tests).
        broker: An existing paper broker to use (it is *not* closed on shutdown).
        configure_logs: Configure root logging from ``LOG_LEVEL`` / ``LOG_JSON`` at
            start-up (disable when embedding the app or under pytest).

    Nothing is connected until the lifespan runs (``uvicorn`` start-up, or entering a
    ``TestClient`` context).
    """
    settings = settings if settings is not None else get_settings()
    app = FastAPI(
        title=API_TITLE,
        version=__version__,
        description=API_DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.overrides = _Overrides(provider, llm, broker, configure_logs)
    app.state.services = None
    app.state.api_keys = ApiKeyVerifier(settings.api_key_set)
    app.state.rate_limiter = SlidingWindowRateLimiter(
        settings.rate_limit_per_minute, RATE_LIMIT_WINDOW_SECONDS
    )

    install_exception_handlers(app)
    app.include_router(public)
    app.include_router(protected)

    # add_middleware() wraps the current stack, so the last one added is the outermost.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(RequestContextMiddleware, quiet_paths=("/health",))
    origins = settings.cors_origin_list
    # Browser origins allowed to *change state* (enforce_access's CSRF guard). Same-origin
    # requests (the /docs page) are always allowed.
    app.state.allowed_origins = AllowedOrigins.from_list(origins)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "PUT", "OPTIONS"],
            allow_headers=[API_KEY_HEADER, REQUEST_ID_HEADER, "Content-Type", "Accept"],
            expose_headers=[
                REQUEST_ID_HEADER,
                "Retry-After",
                "X-RateLimit-Limit",
                "X-RateLimit-Remaining",
            ],
            max_age=600,
        )
    if "*" in origins and not settings.auth_enabled:
        logger.warning(
            "CORS_ORIGINS=* with API_KEYS empty: any web page the operator visits can "
            "place paper orders and reset the account through this API"
        )
    return app


#: The application served by ``uvicorn trading_swarm.api.main:app``.
app = create_app()

__all__ = ["API_DESCRIPTION", "API_TITLE", "TAGS_METADATA", "app", "create_app", "lifespan"]
