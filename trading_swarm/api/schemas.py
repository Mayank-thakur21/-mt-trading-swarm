"""Request and response schemas of the REST API.

Domain objects (:class:`~trading_swarm.models.Decision`, ``Order``, ``Trade``,
``PortfolioSnapshot``, …) are returned as-is from :mod:`trading_swarm.models`; this module
holds the request bodies and the API-specific response envelopes.

Request bodies are strict: unknown fields are rejected (``422``) instead of silently
ignored, so a typo such as ``"autotrade"`` or a client trying to dictate ``"price"`` is
reported rather than quietly doing something else.

Units: ``*_pct`` fields are percentages (``5.0`` == 5 %); prices are in the symbol's
quote currency; timestamps are ISO-8601 UTC.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Any, Final, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from trading_swarm.analytics import PerformanceMetrics
from trading_swarm.data import validate_symbol
from trading_swarm.models import AlertStatus, Side

#: Request-size limits (mirrored by the dashboard client).
MAX_BATCH_SYMBOLS: Final = 10
MAX_BACKTEST_SYMBOLS: Final = 5
MAX_WATCHLIST_SYMBOLS: Final = 50
#: Longest backtest window accepted over HTTP (the CLI has no cap). A 5-symbol, 10-year
#: daily backtest takes roughly two to three minutes.
MAX_BACKTEST_YEARS: Final = 10
MAX_BACKTEST_DAYS: Final = 366 * MAX_BACKTEST_YEARS
#: Sanity cap for account sizes (rejects fat-fingered or overflowing values).
MAX_CAPITAL: Final = 1e12
MAX_ORDER_QUANTITY: Final = 10_000_000

HistoryPeriod = Literal["1d", "5d", "1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "10y", "max"]
HistoryInterval = Literal["1d", "1wk", "1mo"]
RebalanceFrequency = Literal["daily", "weekly", "monthly"]


def _checked_symbol(value: str) -> str:
    # validate_symbol raises InvalidSymbolError (a ValueError) → a 422 validation error.
    return validate_symbol(value)


def _unique(symbols: list[str]) -> list[str]:
    return list(dict.fromkeys(symbols))


#: A ticker validated and normalised (upper-case) at the schema level.
Symbol = Annotated[str, AfterValidator(_checked_symbol)]
#: A ticker only trimmed and upper-cased; the coordinator validates it and reports bad
#: tickers per symbol (404 / batch error) rather than failing the whole request.
LooseSymbol = Annotated[
    str, StringConstraints(strip_whitespace=True, to_upper=True, min_length=1, max_length=32)
]
PositivePrice = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Capital = Annotated[float, Field(gt=0, le=MAX_CAPITAL, allow_inf_nan=False)]


class RequestModel(BaseModel):
    """Base for request bodies: unknown fields are an error."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# =============================================================================== requests


class AnalyzeRequest(RequestModel):
    """``POST /analyze``."""

    symbol: LooseSymbol = Field(description="Ticker, e.g. `AAPL`.", examples=["AAPL"])
    auto_trade: bool | None = Field(
        default=None,
        description="Paper-trade an approved BUY/SELL. Defaults to the server's `AUTO_TRADE`.",
    )


class BatchAnalyzeRequest(RequestModel):
    """``POST /analyze/batch``."""

    symbols: list[LooseSymbol] = Field(
        min_length=1,
        max_length=MAX_BATCH_SYMBOLS,
        description=f"1-{MAX_BATCH_SYMBOLS} tickers (duplicates are analysed once).",
        examples=[["AAPL", "MSFT"]],
    )
    auto_trade: bool | None = Field(
        default=None, description="Defaults to the server's `AUTO_TRADE`."
    )

    @field_validator("symbols", mode="after")
    @classmethod
    def _dedupe(cls, symbols: list[str]) -> list[str]:
        return _unique(symbols)


class PortfolioResetRequest(RequestModel):
    """``POST /portfolio/reset`` (the body is optional)."""

    initial_capital: Capital | None = Field(
        default=None,
        description="New starting cash. Defaults to the server's `INITIAL_CAPITAL`.",
        examples=[100_000.0],
    )


class OrderRequest(RequestModel):
    """``POST /orders`` — a manual paper market order.

    There is deliberately no price field: the fill price is the market-data provider's
    latest price ± slippage, never a client-supplied number.
    """

    symbol: Symbol = Field(examples=["AAPL"])
    side: Side = Field(description="`BUY` or `SELL` (long-only: SELL reduces a position).")
    quantity: StrictInt = Field(
        gt=0, le=MAX_ORDER_QUANTITY, description="Whole shares.", examples=[10]
    )
    stop_price: PositivePrice | None = Field(
        default=None,
        description="Protective stop-loss, BUY only (below the fill price). On a held "
        "symbol the position keeps the higher of its current stop and this one.",
    )
    take_profit_price: PositivePrice | None = Field(
        default=None,
        description="Take-profit level, BUY only (above the fill price); replaces the "
        "position's current target.",
    )

    @model_validator(mode="before")
    @classmethod
    def _no_client_price(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("price", "limit_price", "fill_price"):
                if key in data:
                    raise ValueError(
                        f"'{key}' cannot be set: orders fill at the latest market price "
                        "from the data provider"
                    )
        return data

    @field_validator("side", mode="before")
    @classmethod
    def _upper_side(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


class BacktestRequest(RequestModel):
    """``POST /backtest``."""

    symbols: list[Symbol] = Field(
        min_length=1,
        max_length=MAX_BACKTEST_SYMBOLS,
        description=f"1-{MAX_BACKTEST_SYMBOLS} tickers traded from one shared account.",
        examples=[["AAPL", "MSFT"]],
    )
    start: date = Field(description="First simulated date (inclusive).", examples=["2024-01-02"])
    end: date = Field(description="Last simulated date (inclusive).", examples=["2025-12-31"])
    initial_capital: Capital | None = Field(
        default=None, description="Starting cash. Defaults to `INITIAL_CAPITAL`."
    )
    use_sentiment: bool = Field(
        default=False,
        description="Add the point-in-time news-sentiment vote (headlines published by "
        "20:00 UTC on each bar's date). Off by default: trend-only, renormalised weights.",
    )
    rebalance: RebalanceFrequency = Field(
        default="daily",
        description="Decision schedule; `weekly`/`monthly` decide on the first bar of the "
        "period (stops are still checked on every bar).",
    )
    warmup: int = Field(
        default=200, ge=0, le=1000, description="Bars of history required before `start`."
    )

    @field_validator("symbols", mode="after")
    @classmethod
    def _dedupe(cls, symbols: list[str]) -> list[str]:
        return _unique(symbols)

    @model_validator(mode="after")
    def _check_window(self) -> Self:
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        if (self.end - self.start).days > MAX_BACKTEST_DAYS:
            raise ValueError(
                f"the backtest window is limited to {MAX_BACKTEST_YEARS} years over the API "
                "(use scripts/run_backtest.py for longer runs)"
            )
        return self


class WatchlistUpdate(RequestModel):
    """``PUT /watchlist``."""

    symbols: list[Symbol] = Field(
        min_length=1,
        max_length=MAX_WATCHLIST_SYMBOLS,
        description="Tickers scanned by `POST /scan` and the scheduler (duplicates removed).",
        examples=[["AAPL", "MSFT", "NVDA"]],
    )

    @field_validator("symbols", mode="after")
    @classmethod
    def _dedupe(cls, symbols: list[str]) -> list[str]:
        return _unique(symbols)


# =============================================================================== responses


class HealthResponse(BaseModel):
    """``GET /health`` (public). The body is the same for every caller, with or without
    an API key, so the endpoint cannot be used to test keys; scheduler and alerting
    details are served by the protected ``GET /status``."""

    status: Literal["ok", "degraded"]
    version: str
    data_provider: str
    llm_provider: str = Field(description="`anthropic`, `openai` or `none` (lexicon only).")
    llm_model: str
    email_enabled: bool
    scheduler_enabled: bool
    auto_trade: bool
    auth_enabled: bool
    buy_threshold: float
    sell_threshold: float
    database: Literal["ok", "error"]
    time: datetime
    uptime_seconds: float


class StatusResponse(BaseModel):
    """``GET /status`` (protected): operational details kept out of the public health check."""

    scheduler: dict[str, Any] = Field(
        description="Background scan loop: enabled/running, interval, last and next run, "
        "duration, failures and skipped runs."
    )
    alerts: dict[str, Any] = Field(
        description="Alerting configuration: channel, thresholds, cooldown and the SMTP "
        "settings in use or missing (never the password)."
    )


class QuoteResponse(BaseModel):
    """``GET /quote/{symbol}``."""

    symbol: str
    price: float = Field(description="Latest price (what a market order would fill against).")
    previous_close: float | None = Field(description="Close of the prior trading day.")
    change_pct_1d: float | None = Field(
        description="Last close vs the prior close, in percent (5.0 == +5 %)."
    )
    as_of: datetime = Field(description="Trading date (midnight UTC) of the latest bar.")


class Candle(BaseModel):
    """One OHLCV bar (compact keys keep chart payloads small)."""

    t: datetime
    o: float
    h: float
    l: float  # noqa: E741 - compact OHLCV key used by the dashboard
    c: float
    v: float


class HistoryResponse(BaseModel):
    """``GET /history/{symbol}``: candles plus indicator series aligned with them."""

    symbol: str
    period: str
    interval: str
    candles: list[Candle]
    indicators: dict[str, list[float | None]] = Field(
        description="`sma20`, `sma50`, `rsi` (Wilder, 14), `macd`, `signal` (EMA-9 of MACD) "
        "and `hist`; same length as `candles`, `null` during the indicator warm-up."
    )


class EquityPoint(BaseModel):
    t: datetime
    equity: float
    cash: float
    positions_value: float


class EquityResponse(BaseModel):
    """``GET /equity``: recorded equity snapshots and account performance metrics."""

    points: list[EquityPoint]
    metrics: PerformanceMetrics = Field(
        description="Computed on one point per NYSE trading session (the last snapshot "
        "reflecting each session, forward-filled over sessions without one), measured "
        "from the account's initial capital."
    )


class AlertTestResponse(BaseModel):
    """``POST /alerts/test``."""

    sent: bool
    channel: str
    status: AlertStatus
    detail: str


class WatchlistResponse(BaseModel):
    symbols: list[str]


class ErrorDetail(BaseModel):
    """The ``error`` object of every error response (extra keys depend on the error)."""

    model_config = ConfigDict(extra="allow")

    code: str = Field(examples=["not_found"])
    message: str = Field(examples=["no price data returned for XYZ"])
    request_id: str | None = Field(examples=["5f0c2b9e8a7d4c1e9b3a6d2f1e0c9b8a"])


class ErrorResponse(BaseModel):
    """Error envelope: ``{"error": {"code", "message", "request_id", …}}``."""

    error: ErrorDetail


__all__ = [
    "MAX_BACKTEST_DAYS",
    "MAX_BACKTEST_SYMBOLS",
    "MAX_BACKTEST_YEARS",
    "MAX_BATCH_SYMBOLS",
    "MAX_CAPITAL",
    "MAX_ORDER_QUANTITY",
    "MAX_WATCHLIST_SYMBOLS",
    "AlertTestResponse",
    "AnalyzeRequest",
    "BacktestRequest",
    "BatchAnalyzeRequest",
    "Candle",
    "EquityPoint",
    "EquityResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "HistoryInterval",
    "HistoryPeriod",
    "HistoryResponse",
    "OrderRequest",
    "PortfolioResetRequest",
    "QuoteResponse",
    "RebalanceFrequency",
    "StatusResponse",
    "WatchlistResponse",
    "WatchlistUpdate",
]
