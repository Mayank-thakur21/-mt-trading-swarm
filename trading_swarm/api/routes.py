"""HTTP endpoints.

Handlers are plain ``def`` functions: they do blocking work (SQLite, market-data
downloads, LLM calls, CPU-bound backtests), so FastAPI runs them in its thread pool and
the event loop stays free. Everything under :data:`protected` goes through
:func:`~trading_swarm.api.security.enforce_access` (rate limit + API key); ``/health`` and
``/`` are public.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Annotated, Any, Final

import pandas as pd
from fastapi import APIRouter, Depends, Path, Query, Request, status
from fastapi.responses import JSONResponse

from trading_swarm import __version__
from trading_swarm.analytics import compute_metrics
from trading_swarm.api.deps import ServiceContainer, ServicesDep, SettingsDep
from trading_swarm.api.errors import ApiError
from trading_swarm.api.market import history_payload, latest_trade_price, quote_payload
from trading_swarm.api.schemas import (
    AlertTestResponse,
    AnalyzeRequest,
    BacktestRequest,
    BatchAnalyzeRequest,
    EquityPoint,
    EquityResponse,
    ErrorResponse,
    HealthResponse,
    HistoryInterval,
    HistoryPeriod,
    HistoryResponse,
    OrderRequest,
    PortfolioResetRequest,
    QuoteResponse,
    StatusResponse,
    WatchlistResponse,
    WatchlistUpdate,
)
from trading_swarm.api.security import enforce_access
from trading_swarm.models import (
    AlertRecord,
    AlertStatus,
    BatchAnalysis,
    Decision,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    ScanReport,
    Trade,
    utc_now,
)
from trading_swarm.scheduler import session_day, trading_days

logger = logging.getLogger(__name__)

#: Reason recorded on orders placed through ``POST /orders``.
MANUAL_ORDER_REASON: Final = "manual order (API)"
#: Seconds a ``/health`` database probe result is reused (see :func:`_database_ok`).
HEALTH_PROBE_TTL_SECONDS: Final = 1.0

TAG_SYSTEM: Final = "system"
TAG_MARKET: Final = "market data"
TAG_ANALYSIS: Final = "analysis"
TAG_PORTFOLIO: Final = "portfolio"
TAG_ORDERS: Final = "orders"
TAG_BACKTEST: Final = "backtest"
TAG_ALERTS: Final = "alerts"
TAG_WATCHLIST: Final = "watchlist"


def _error_doc(description: str) -> dict[str, Any]:
    return {"model": ErrorResponse, "description": description}


_COMMON_ERRORS: Final[dict[int | str, dict[str, Any]]] = {
    401: _error_doc("Missing or invalid `X-API-Key` (only when `API_KEYS` is set)."),
    403: _error_doc(
        "`forbidden_origin`: a browser request from an origin not listed in `CORS_ORIGINS` "
        "tried to change state (cross-site request forgery guard)."
    ),
    422: _error_doc("Invalid request (`validation_error`, `invalid_symbol`, …)."),
    429: _error_doc("Rate limit exceeded; see the `Retry-After` header."),
    503: _error_doc("Service starting up / shutting down, or busy."),
}
_MARKET_ERRORS: Final[dict[int | str, dict[str, Any]]] = {
    404: _error_doc("Unknown symbol or no data (`not_found`)."),
    502: _error_doc("Market-data provider unavailable (`data_unavailable`)."),
}

public = APIRouter()
protected = APIRouter(dependencies=[Depends(enforce_access)], responses=_COMMON_ERRORS)

SymbolPath = Annotated[
    str, Path(min_length=1, max_length=32, description="Ticker, e.g. `AAPL`.", examples=["AAPL"])
]


def _limit(maximum: int) -> Any:
    return Query(ge=1, le=maximum, description=f"Maximum rows (1-{maximum}).")


# =============================================================================== system


@public.get("/", tags=[TAG_SYSTEM], summary="Service banner", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "MT Trading Swarm API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@public.get(
    "/health",
    tags=[TAG_SYSTEM],
    summary="Liveness and configuration",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Database unreachable."}},
)
def health(request: Request, settings: SettingsDep) -> Any:
    """Public health check (not rate limited).

    Reports the active data provider, LLM provider/model and feature flags. The body is
    the same with or without an ``X-API-Key``: an unthrottled endpoint whose answer
    depended on the key would let anyone test keys at full speed, around the rate
    limiter. Scheduler and alerting details are at the protected ``GET /status``.
    Returns ``503`` with ``status: "degraded"`` when the database cannot be read (the
    database probe is cached for :data:`HEALTH_PROBE_TTL_SECONDS`).
    """
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise ApiError(
            503,
            "service_unavailable",
            "the service is starting up or shutting down; retry shortly",
            headers={"Retry-After": "5"},
        )
    database = "ok" if _database_ok(request, services) else "error"
    now = utc_now()
    body = HealthResponse(
        status="ok" if database == "ok" else "degraded",
        version=__version__,
        data_provider=str(getattr(services.provider, "name", settings.data_provider)),
        llm_provider=services.llm.name,
        llm_model=services.llm.model,
        email_enabled=settings.email_enabled,
        scheduler_enabled=settings.scheduler_enabled,
        auto_trade=settings.auto_trade,
        auth_enabled=settings.auth_enabled,
        buy_threshold=settings.buy_threshold,
        sell_threshold=settings.sell_threshold,
        database=database,
        time=now,
        uptime_seconds=round((now - services.started_at).total_seconds(), 3),
    )
    return JSONResponse(
        body.model_dump(mode="json"),
        status_code=status.HTTP_200_OK if database == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE,
    )


def _database_ok(request: Request, services: ServiceContainer) -> bool:
    """Whether the account row can be read, re-probed at most every
    :data:`HEALTH_PROBE_TTL_SECONDS`.

    ``/health`` is public and not rate limited, and reading the account takes the
    broker's lock, which trading also uses. Reusing a fresh result keeps a flood of
    health checks off that lock. The cache lives on the app and is keyed by the service
    container, so a restarted lifespan probes again.
    """
    now = time.monotonic()
    cached = getattr(request.app.state, "health_probe", None)
    if cached is not None:
        owner, checked_at, ok = cached
        if owner is services and now - checked_at < HEALTH_PROBE_TTL_SECONDS:
            return bool(ok)
    try:
        _ = services.broker.cash
        ok = True
    except Exception:
        logger.exception("health check: database unavailable")
        ok = False
    request.app.state.health_probe = (services, now, ok)
    return ok


@protected.get(
    "/status",
    tags=[TAG_SYSTEM],
    summary="Scheduler and alerting status",
    response_model=StatusResponse,
)
def get_status(services: ServicesDep) -> StatusResponse:
    """The background scheduler's state (last and next run, duration, failures, skipped
    runs) and the alerting configuration (SMTP settings in use or missing, thresholds,
    cooldown; never the password). Requires the API key, unlike ``/health``."""
    return StatusResponse(
        scheduler=services.scheduler.status(), alerts=services.alerts.config_summary()
    )


# =============================================================================== market data


@protected.get(
    "/quote/{symbol}",
    tags=[TAG_MARKET],
    summary="Latest price and 1-day change",
    response_model=QuoteResponse,
    responses=_MARKET_ERRORS,
)
def get_quote(symbol: SymbolPath, services: ServicesDep) -> dict[str, Any]:
    """Latest price (the one a market order would use) and the close-to-close change."""
    sym = services.provider.validate_symbol(symbol)
    return quote_payload(services.provider, sym)


@protected.get(
    "/history/{symbol}",
    tags=[TAG_MARKET],
    summary="Candles with SMA20/50, RSI and MACD",
    response_model=HistoryResponse,
    responses=_MARKET_ERRORS,
)
def get_history(
    symbol: SymbolPath,
    services: ServicesDep,
    period: Annotated[HistoryPeriod, Query(description="Trailing window to return.")] = "6mo",
    interval: Annotated[HistoryInterval, Query(description="Bar size.")] = "1d",
) -> dict[str, Any]:
    """Adjusted OHLCV candles for charting, with indicator series of the same length.

    Indicators are computed on a longer look-back and then sliced, so SMA50 and MACD are
    defined from the first displayed candle.
    """
    sym = services.provider.validate_symbol(symbol)
    return history_payload(services.provider, sym, period, interval)


# =============================================================================== analysis


@protected.post(
    "/analyze",
    tags=[TAG_ANALYSIS],
    summary="Run the swarm on one symbol",
    response_model=Decision,
    responses=_MARKET_ERRORS,
)
def analyze(body: AnalyzeRequest, services: ServicesDep) -> Decision:
    """Trend + sentiment votes → consensus → risk gate → (optional) paper trade.

    The decision is persisted (``GET /decisions``) and alert rules are evaluated.
    ``action`` is the final action after the risk gate (a vetoed BUY comes back as HOLD;
    ``summary`` explains why).

    **This endpoint can sell holdings even with ``auto_trade: false``.** Before the risk
    gate it enforces the stop-loss / take-profit of every open position it can price
    (the analysed symbol and every other holding), as a watchlist scan does. Those exit
    orders are returned in ``exits`` and described in ``summary``. A symbol closed this
    way is not bought back in the same call (``risk.checks.reentry_ok = false``).
    ``auto_trade`` only controls the order for the decision's own BUY/SELL.
    """
    return services.coordinator.analyze(body.symbol, auto_trade=body.auto_trade)


@protected.post(
    "/analyze/batch",
    tags=[TAG_ANALYSIS],
    summary="Run the swarm on up to 10 symbols",
    response_model=BatchAnalysis,
)
def analyze_batch(body: BatchAnalyzeRequest, services: ServicesDep) -> BatchAnalysis:
    """Symbols are analysed concurrently (4 at a time). Failures do not fail the request:
    they are listed in ``errors`` with a code (``not_found``, ``data_unavailable``,
    ``error``). As with ``POST /analyze``, breached stop-losses / take-profits of open
    positions are enforced whatever ``auto_trade`` says; each decision lists them in
    ``exits``."""
    return services.coordinator.analyze_many(body.symbols, auto_trade=body.auto_trade)


@protected.get(
    "/decisions",
    tags=[TAG_ANALYSIS],
    summary="Recent decisions",
    response_model=list[Decision],
)
def list_decisions(
    services: ServicesDep,
    limit: Annotated[int, _limit(500)] = 50,
    symbol: Annotated[str | None, Query(max_length=32, description="Filter by ticker.")] = None,
) -> list[Decision]:
    """Most recent first. Includes HOLDs and vetoed proposals (the full audit trail)."""
    return services.broker.decisions(limit=limit, symbol=symbol or None)


@protected.post(
    "/scan",
    tags=[TAG_ANALYSIS],
    summary="Scan the watchlist now",
    response_model=ScanReport,
)
def scan(services: ServicesDep) -> ScanReport:
    """Enforce stop-losses / take-profits with fresh quotes, analyse every watchlist
    symbol (auto-trading per ``AUTO_TRADE``) and record an equity snapshot — the same
    job the scheduler runs. ``exits`` lists every protective exit of the scan. If an
    open position could not be priced, ``snapshot`` shows it at cost and
    ``snapshot_recorded`` is ``false``: no equity point is stored for this scan."""
    return services.coordinator.scan_watchlist()


# =============================================================================== portfolio


@protected.get(
    "/portfolio",
    tags=[TAG_PORTFOLIO],
    summary="Account snapshot",
    response_model=PortfolioSnapshot,
)
def get_portfolio(services: ServicesDep) -> PortfolioSnapshot:
    """Cash, equity, exposure, P&L and positions marked to the latest prices (a position
    whose quote fails is marked at its cost basis)."""
    return services.coordinator.portfolio()


@protected.post(
    "/portfolio/reset",
    tags=[TAG_PORTFOLIO],
    summary="Reset the paper account",
    response_model=PortfolioSnapshot,
)
def reset_portfolio(
    services: ServicesDep, body: PortfolioResetRequest | None = None
) -> PortfolioSnapshot:
    """Delete positions, orders, trades and the equity curve and start again with fresh
    cash. Decisions, the alert log and the watchlist are kept."""
    capital = body.initial_capital if body is not None else None
    with services.trade_lock():
        snapshot = services.broker.reset(capital)
    logger.warning("paper account reset via the API (capital %.2f)", snapshot.initial_capital)
    return snapshot


def _daily_curve(curve: pd.DataFrame) -> pd.DataFrame:
    """One point per NYSE trading session: the sampling that ``periods_per_year=252`` means.

    Snapshots are irregular: one per scan (every ``SCAN_INTERVAL_MINUTES``, around the
    clock unless ``MARKET_HOURS_ONLY``), plus one after each fill. Grouping them by
    calendar day and annualising with 252 would be wrong in two ways. Weekends and
    holidays would add zero-return periods, so a 24/7 scheduler understates volatility and
    Sharpe by about √(5/7). Multi-day gaps between manual scans would pack several
    sessions into one period, which overstates them. So:

    1. Each snapshot is assigned to the session whose prices it reflects
       (:func:`~trading_swarm.scheduler.session_day`). A Saturday snapshot belongs to
       Friday, and so does a Monday snapshot taken before the open.
    2. The last snapshot of each session is kept.
    3. The result is reindexed onto every trading day in the span and forward-filled. A
       session with no snapshot carries the previous value, and the whole move lands on
       the next observed session. The sum of squared returns, and so the annualised
       volatility, is preserved.

    The index is each session's date at midnight UTC.
    """
    if curve.empty:
        return curve
    labels = pd.DatetimeIndex(
        [pd.Timestamp(session_day(ts.to_pydatetime())) for ts in curve.index]
    ).tz_localize("UTC")
    daily = curve.groupby(labels).last()
    sessions = trading_days(daily.index[0].date(), daily.index[-1].date())
    grid = pd.DatetimeIndex([pd.Timestamp(day) for day in sessions]).tz_localize("UTC")
    daily = daily.reindex(grid).ffill()
    daily.index.name = "timestamp"
    return daily


@protected.get(
    "/equity",
    tags=[TAG_PORTFOLIO],
    summary="Equity curve and performance metrics",
    response_model=EquityResponse,
)
def get_equity(services: ServicesDep, settings: SettingsDep) -> EquityResponse:
    """Every recorded snapshot (account creation, resets, scans, fills) plus CAGR,
    Sharpe, Sortino, max drawdown, trade statistics and exposure.

    Snapshots are irregular (intraday scans, weekend scans, gaps between manual scans).
    The metrics therefore use one point per NYSE trading session: the last snapshot
    reflecting each session, forward-filled across sessions without one. That is the
    sampling the 252-periods-per-year annualisation assumes. Returns and drawdowns are
    measured from the account's initial capital, so trading in the first session
    (right after creation or a reset) is included and ``total_return_pct`` matches
    ``GET /portfolio`` at the last snapshot's prices.
    """
    curve = services.broker.equity_curve()
    points = [
        EquityPoint(
            t=ts.to_pydatetime(),
            equity=round(float(row.equity), 2),
            cash=round(float(row.cash), 2),
            positions_value=round(float(row.positions_value), 2),
        )
        for ts, row in zip(curve.index, curve.itertuples(index=False), strict=True)
        if math.isfinite(float(row.equity))
    ]
    metrics = compute_metrics(
        _daily_curve(curve),
        services.broker.trades(limit=None),
        risk_free_rate=settings.risk_free_rate,
        start_equity=_start_equity(services, curve),
    )
    return EquityResponse(points=points, metrics=metrics)


def _start_equity(services: ServiceContainer, curve: pd.DataFrame) -> float | None:
    """The money invested: the account's initial capital (``None`` without a curve).

    :func:`_daily_curve` keeps the *last* snapshot of each session, so the creation /
    reset snapshot (equity = initial capital) is dropped whenever the account trades in
    its first session, and the first session's P&L and drawdown would vanish from the
    metrics. Measuring from the initial capital (as the backtester does) restores them:
    the first period's return is ``E_first_session / initial_capital − 1`` and the
    initial capital is the first peak of the drawdown. The curve always starts with that
    snapshot (account creation and :meth:`~trading_swarm.broker.PaperBroker.reset` write
    it), so this is exact.
    """
    if curve.empty:
        return None
    return float(services.broker.initial_capital)


# =============================================================================== orders


@protected.post(
    "/orders",
    tags=[TAG_ORDERS],
    summary="Place a manual paper order",
    response_model=Order,
    status_code=status.HTTP_201_CREATED,
    responses={
        **_MARKET_ERRORS,
        422: _error_doc(
            "Malformed order (`invalid_order`) or rejected by the broker (`order_rejected`: "
            "insufficient cash, no / insufficient position)."
        ),
    },
)
def place_order(body: OrderRequest, services: ServicesDep) -> Order:
    """Market order at the data provider's latest price ± slippage, with commission.

    The client never supplies the price. Long-only: SELL reduces an existing position
    and takes no `stop_price` / `take_profit_price` (`422 invalid_order`): the
    remaining shares keep their levels. A BUY into a held symbol never lowers the
    position's stop (it keeps the higher one), and the response reports the levels
    actually in force. A BUY is rejected while the position's stop or target is
    already breached. Rejected orders are recorded (``GET /orders``) and reported as
    ``422 order_rejected`` with the broker's reason. A fill adds a point to the equity
    curve.
    """
    price = latest_trade_price(services.provider, body.symbol)
    with services.trade_lock():
        order = services.broker.submit_order(
            body.symbol,
            body.side,
            body.quantity,
            price,
            reason=MANUAL_ORDER_REASON,
            stop=body.stop_price,
            take_profit=body.take_profit_price,
        )
    if order.status is OrderStatus.FILLED:
        services.coordinator.snapshot_after_fill({order.symbol: price})
    if order.status is not OrderStatus.FILLED:
        raise ApiError(
            422,
            "order_rejected",
            order.reject_reason or f"order {order.status.value.lower()}",
            extra={"order": order.model_dump(mode="json")},
        )
    return order


@protected.get(
    "/orders",
    tags=[TAG_ORDERS],
    summary="Recent orders",
    response_model=list[Order],
)
def list_orders(
    services: ServicesDep,
    limit: Annotated[int, _limit(1000)] = 50,
    symbol: Annotated[str | None, Query(max_length=32, description="Filter by ticker.")] = None,
) -> list[Order]:
    """Most recent first, including rejected orders and protective exits."""
    return services.broker.orders(limit=limit, symbol=symbol or None)


@protected.get(
    "/trades",
    tags=[TAG_ORDERS],
    summary="Recent fills",
    response_model=list[Trade],
)
def list_trades(
    services: ServicesDep,
    limit: Annotated[int, _limit(5000)] = 100,
    symbol: Annotated[str | None, Query(max_length=32, description="Filter by ticker.")] = None,
) -> list[Trade]:
    """Most recent first; SELL fills carry the realised P&L (net of both commissions)."""
    return services.broker.trades(limit=limit, symbol=symbol or None)


# =============================================================================== backtest


@protected.post(
    "/backtest",
    tags=[TAG_BACKTEST],
    summary="Backtest the swarm strategy",
    response_model=dict[str, Any],
    responses=_MARKET_ERRORS,
)
def run_backtest(body: BacktestRequest, services: ServicesDep) -> dict[str, Any]:
    """Event-driven replay over daily bars with no look-ahead: decisions at bar *t* use
    data up to *t* only and fill at the next bar's open; stops / targets are checked
    against each bar's low / high (gaps fill at the open). Returns the equity curve
    (with an equal-weight buy-and-hold benchmark), trades, approved decisions and
    metrics for both strategy and benchmark.

    CPU-bound: roughly 3-4 s per symbol-year of daily bars (``rebalance="weekly"`` is
    about 5x faster); at most two backtests run at once (``503 busy`` otherwise).
    """
    with services.backtest_slot():
        result = services.backtester.run(
            body.symbols,
            body.start,
            body.end,
            initial_capital=body.initial_capital,
            use_sentiment=body.use_sentiment,
            warmup=body.warmup,
            rebalance=body.rebalance,
        )
    return result.to_dict()


# =============================================================================== alerts


@protected.get(
    "/alerts",
    tags=[TAG_ALERTS],
    summary="Alert log",
    response_model=list[AlertRecord],
)
def list_alerts(
    services: ServicesDep,
    limit: Annotated[int, _limit(1000)] = 50,
    rule: Annotated[
        str | None,
        Query(max_length=32, description="`signal`, `order`, `exit`, `drawdown`, …"),
    ] = None,
    symbol: Annotated[str | None, Query(max_length=32, description="Filter by ticker.")] = None,
) -> list[AlertRecord]:
    """Every alert that fired, most recent first — sent, failed, disabled or suppressed."""
    return services.broker.alerts(limit=limit, rule=rule or None, symbol=symbol or None)


@protected.post(
    "/alerts/test",
    tags=[TAG_ALERTS],
    summary="Send a test e-mail",
    response_model=AlertTestResponse,
)
def send_test_alert(services: ServicesDep) -> AlertTestResponse:
    """Sends a test message through the configured SMTP server. ``detail`` explains the
    outcome (for example which settings are missing). The attempt is logged."""
    record = services.alerts.send_test()
    return AlertTestResponse(
        sent=record.status is AlertStatus.SENT,
        channel=record.channel,
        status=record.status,
        detail=record.detail,
    )


# =============================================================================== watchlist


@protected.get(
    "/watchlist",
    tags=[TAG_WATCHLIST],
    summary="Current watchlist",
    response_model=WatchlistResponse,
)
def get_watchlist(services: ServicesDep, settings: SettingsDep) -> WatchlistResponse:
    """The persisted watchlist, or ``WATCHLIST`` from the settings until one is saved."""
    return WatchlistResponse(symbols=services.broker.get_watchlist(settings.watchlist_symbols))


@protected.put(
    "/watchlist",
    tags=[TAG_WATCHLIST],
    summary="Replace the watchlist",
    response_model=WatchlistResponse,
)
def put_watchlist(body: WatchlistUpdate, services: ServicesDep) -> WatchlistResponse:
    """Persisted in the database (survives restarts); used by ``POST /scan`` and the
    scheduler."""
    stored = services.broker.set_watchlist(body.symbols)
    logger.info("watchlist updated: %s", ", ".join(stored))
    return WatchlistResponse(symbols=stored)


__all__ = ["HEALTH_PROBE_TTL_SECONDS", "MANUAL_ORDER_REASON", "protected", "public"]
