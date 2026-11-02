"""Shared domain model (pydantic v2) used by agents, broker, API, backtester and dashboard.

Conventions
-----------
* All timestamps are timezone-aware UTC. Naive datetimes are *interpreted* as UTC (the
  stdlib's ``datetime.utcnow()`` style) and converted; aware ones are converted to UTC.
* ``signal`` values live in ``[-1, 1]`` (+1 = maximally bullish), ``confidence`` values in
  ``[0, 1]``. Tiny floating-point overshoots are clamped; NaN / ±inf are rejected because
  they always indicate a bug upstream.
* Fields ending in ``_pct`` are **percentages** (``5.0`` == 5 %), ready for display.
  Fraction equivalents are exposed as properties where code needs them (for example
  :attr:`PortfolioSnapshot.drawdown`), so risk rules can compare against the fractional
  limits in :class:`~trading_swarm.config.Settings` without unit mistakes.
* Money is stored as floats and rounded to cents only when a model is built for output.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator

# =============================================================================== helpers


def utc_now() -> datetime:
    """Current time as an aware UTC datetime (replacement for the deprecated ``utcnow``)."""
    return datetime.now(UTC)


def new_id() -> str:
    """Random 32-hex-character identifier for orders, trades and decisions."""
    return uuid4().hex


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC datetime (naive input is assumed to be UTC)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _clamp(lo: float, hi: float) -> Callable[[float], float]:
    def validator(value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("value must be a finite number")
        return min(hi, max(lo, value))

    return validator


def to_jsonable(value: Any) -> Any:
    """Recursively convert ``value`` into plain JSON-safe Python objects.

    Agents put indicator values into ``AgentSignal.data``; those frequently are numpy
    scalars (``np.int64`` is *not* an ``int``), ``NaN`` (not valid JSON) or timestamps.
    This keeps decisions serialisable to the API and to the ``decisions`` table.
    """
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, datetime):
        return ensure_utc(value).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [to_jsonable(v) for v in value]
    # numpy / pandas scalars expose .item(); pandas Timestamps expose .isoformat().
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return to_jsonable(item())
        except (TypeError, ValueError):
            pass
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return isoformat()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist())
    return str(value)


UTCDatetime = Annotated[datetime, AfterValidator(ensure_utc)]
#: A directional score in ``[-1, 1]`` (positive = bullish).
SignalValue = Annotated[float, AfterValidator(_clamp(-1.0, 1.0))]
#: A probability-like value in ``[0, 1]``.
UnitInterval = Annotated[float, AfterValidator(_clamp(0.0, 1.0))]
#: A 0-100 score.
Score100 = Annotated[float, AfterValidator(_clamp(0.0, 100.0))]


class SwarmModel(BaseModel):
    """Base class: enums serialise as their values, unknown keys are ignored on load."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# =============================================================================== enums


class Action(StrEnum):
    """What the swarm wants to do with a symbol."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Side(StrEnum):
    """Order side. The paper broker is long-only: SELL only reduces an existing position."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(StrEnum):
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    PENDING = "PENDING"


class ExitReason(StrEnum):
    """Why a protective exit fired (stored in ``Order.reason`` for exit orders)."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class AlertStatus(StrEnum):
    """Outcome of an alert, recorded in the ``alerts_log`` table."""

    SENT = "sent"
    FAILED = "failed"
    DISABLED = "disabled"  # channel not configured; recorded for the audit trail only
    SUPPRESSED = "suppressed"  # deduplicated by the per-(rule, symbol) cooldown


# =============================================================================== agents


class AgentSignal(SwarmModel):
    """One agent's vote.

    ``signal`` is the direction/strength in ``[-1, 1]``; ``confidence`` in ``[0, 1]`` is how
    much the agent trusts its own vote. The coordinator weights each vote by
    ``weight × confidence`` so a confident agent counts more than a guessing one.
    """

    agent: str
    signal: SignalValue = 0.0
    confidence: UnitInterval = 0.0
    rationale: str = ""
    data: dict[str, Any] = Field(default_factory=dict)

    @field_validator("data", mode="after")
    @classmethod
    def _jsonable_data(cls, value: dict[str, Any]) -> dict[str, Any]:
        return to_jsonable(value)

    @classmethod
    def neutral(cls, agent: str, rationale: str, **data: Any) -> AgentSignal:
        """A zero-signal, zero-confidence vote (e.g. "insufficient data")."""
        return cls(agent=agent, signal=0.0, confidence=0.0, rationale=rationale, data=data)


class RiskAssessment(SwarmModel):
    """Output of the risk gate: whether to trade, how many shares, and the protective levels.

    ``action`` may be downgraded (BUY → HOLD) when a rule vetoes the trade; ``reasons``
    explains every veto/adjustment and ``checks`` records each rule's pass/fail.
    """

    approved: bool = False
    action: Action = Action.HOLD
    quantity: int = Field(default=0, ge=0)
    entry_price: float = Field(default=0.0, ge=0)
    stop_price: float | None = None
    take_profit_price: float | None = None
    position_value: float = 0.0
    risk_amount: float = 0.0
    risk_score: Score100 = 0.0
    reasons: list[str] = Field(default_factory=list)
    checks: dict[str, bool] = Field(default_factory=dict)


# =============================================================================== broker


class Order(SwarmModel):
    """A paper order. Market orders fill immediately at ``reference_price`` ± slippage.

    ``slippage`` is the currency cost of slippage (``|fill_price − reference_price| ×
    quantity``); ``commission`` is the fee charged. ``realized_pnl`` is set on filled SELL
    orders (net of commissions on both legs).
    """

    id: str = Field(default_factory=new_id)
    symbol: str
    side: Side
    quantity: int = Field(gt=0)
    type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    reference_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    requested_at: UTCDatetime = Field(default_factory=utc_now)
    filled_at: UTCDatetime | None = None
    fill_price: float | None = None
    commission: float = 0.0
    slippage: float = 0.0
    stop_price: float | None = None
    take_profit_price: float | None = None
    realized_pnl: float | None = None
    reason: str = ""
    decision_id: str | None = None
    reject_reason: str | None = None

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    @property
    def notional(self) -> float:
        """Filled value (``fill_price × quantity``), 0 for unfilled orders."""
        return (self.fill_price or 0.0) * self.quantity if self.is_filled else 0.0


class Trade(SwarmModel):
    """An execution (fill). ``realized_pnl`` is populated for SELL fills only."""

    id: str = Field(default_factory=new_id)
    order_id: str
    symbol: str
    side: Side
    quantity: int = Field(gt=0)
    price: float
    commission: float = 0.0
    realized_pnl: float | None = None
    timestamp: UTCDatetime = Field(default_factory=utc_now)
    reason: str = ""


class Position(SwarmModel):
    """An open long position.

    ``avg_price`` is the commission-inclusive cost basis per share, so
    ``realized_pnl + unrealized_pnl`` always equals the account's total return.
    When no market price is available the position is marked at ``avg_price``.
    """

    symbol: str
    quantity: int = Field(gt=0)
    avg_price: float
    stop_price: float | None = None
    take_profit_price: float | None = None
    opened_at: UTCDatetime = Field(default_factory=utc_now)
    market_price: float | None = None
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0

    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_price


class PortfolioSnapshot(SwarmModel):
    """Account state marked to market.

    * ``equity = cash + positions_value``
    * ``gross_exposure_pct = positions_value / equity × 100``
    * ``drawdown_pct = (peak_equity − equity) / peak_equity × 100`` (≥ 0)
    """

    initial_capital: float
    cash: float
    equity: float
    positions_value: float = 0.0
    gross_exposure_pct: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    peak_equity: float
    drawdown_pct: float = 0.0
    positions: list[Position] = Field(default_factory=list)
    updated_at: UTCDatetime = Field(default_factory=utc_now)

    @property
    def gross_exposure(self) -> float:
        """Gross exposure as a fraction of equity (compare with ``max_gross_exposure``)."""
        return self.gross_exposure_pct / 100.0

    @property
    def drawdown(self) -> float:
        """Drawdown from peak as a fraction (compare with ``max_drawdown_halt``)."""
        return self.drawdown_pct / 100.0

    def position(self, symbol: str) -> Position | None:
        """The open position in ``symbol`` (case-insensitive), if any."""
        wanted = symbol.upper()
        return next((p for p in self.positions if p.symbol == wanted), None)


# =============================================================================== decisions


class Decision(SwarmModel):
    """The swarm's verdict for one symbol at one point in time.

    ``exits`` lists the protective exits (stop-loss / take-profit SELLs) that closed
    positions *before* this decision was sized. An analysis enforces the stops of every
    holding it can price, whether or not auto-trading is on, so it can sell positions,
    including ones in other symbols. Within a watchlist scan, an exit of the analysed
    symbol made by the scan's own exit pass is listed here as well. ``order`` is only the
    order placed for this decision's own action.
    """

    id: str = Field(default_factory=new_id)
    symbol: str
    timestamp: UTCDatetime = Field(default_factory=utc_now)
    price: float
    action: Action
    score: SignalValue = 0.0
    confidence: UnitInterval = 0.0
    signals: list[AgentSignal] = Field(default_factory=list)
    risk: RiskAssessment = Field(default_factory=RiskAssessment)
    summary: str = ""
    executed: bool = False
    order: Order | None = None
    exits: list[Order] = Field(default_factory=list)

    def signal_for(self, agent: str) -> AgentSignal | None:
        """The vote cast by ``agent`` (e.g. ``"trend"``), if present."""
        return next((s for s in self.signals if s.agent == agent), None)

    def exit_for(self, symbol: str | None = None) -> Order | None:
        """The filled protective exit of ``symbol`` (default: this decision's symbol)."""
        wanted = (symbol or self.symbol).upper()
        return next((o for o in self.exits if o.symbol == wanted and o.is_filled), None)


class SymbolError(SwarmModel):
    """A per-symbol failure in a batch operation.

    ``code`` is machine-readable: ``"not_found"`` (unknown symbol / no data),
    ``"data_unavailable"`` (provider down) or ``"error"`` (anything else).
    """

    symbol: str
    error: str
    code: str = "error"


class BatchAnalysis(SwarmModel):
    """Result of analysing several symbols: successes and failures side by side."""

    results: list[Decision] = Field(default_factory=list)
    errors: list[SymbolError] = Field(default_factory=list)


class ScanReport(SwarmModel):
    """Result of one watchlist scan (analysis + protective exits + equity snapshot).

    ``exits`` holds every protective exit of the scan: those of its first exit pass and
    those made inside the per-symbol analyses. ``snapshot`` is the account marked to the
    scan's prices. ``snapshot_recorded`` says whether it was also appended to the equity
    curve. It is ``False`` when some open position had no price (that position is shown
    at cost basis here): a cost-marked point would put a phantom gain or loss into the
    curve and its metrics, so it is not stored.
    """

    decisions: list[Decision] = Field(default_factory=list)
    exits: list[Order] = Field(default_factory=list)
    errors: list[SymbolError] = Field(default_factory=list)
    snapshot: PortfolioSnapshot | None = None
    snapshot_recorded: bool = False
    started_at: UTCDatetime = Field(default_factory=utc_now)
    finished_at: UTCDatetime | None = None


# =============================================================================== news / LLM


class NewsItem(SwarmModel):
    """A news headline about a symbol (untrusted, third-party text)."""

    title: str = Field(min_length=1)
    publisher: str | None = None
    url: str | None = None
    published_at: UTCDatetime | None = None
    summary: str | None = None


SentimentLabel = Literal["bullish", "bearish", "neutral"]

#: Maximum length of :attr:`SentimentJudgement.summary`.
SENTIMENT_SUMMARY_MAX_CHARS = 400


class HeadlineSentiment(SwarmModel):
    """The LLM's reading of one headline, referenced by its 0-based ``index``."""

    index: int = Field(ge=0)
    sentiment: SentimentLabel
    score: SignalValue
    relevance: UnitInterval


class SentimentJudgement(SwarmModel):
    """Structured output of the LLM sentiment call.

    Numeric ranges are enforced here (clamped) rather than in the JSON schema because the
    providers' structured-output modes do not support ``minimum``/``maximum``. ``items``,
    ``overall_score`` and ``confidence`` are required, mirroring the schema, so a payload
    that silently drops a key is rejected rather than defaulted.
    """

    items: list[HeadlineSentiment]
    overall_score: SignalValue
    confidence: UnitInterval
    summary: str = ""

    @field_validator("summary", mode="after")
    @classmethod
    def _truncate_summary(cls, value: str) -> str:
        value = value.strip()
        if len(value) <= SENTIMENT_SUMMARY_MAX_CHARS:
            return value
        return value[: SENTIMENT_SUMMARY_MAX_CHARS - 1].rstrip() + "…"

    def relevance_weighted_score(self) -> float | None:
        """Σ score·relevance / Σ relevance over the items (``None`` if no relevant items)."""
        total_relevance = sum(item.relevance for item in self.items)
        if total_relevance <= 0:
            return None
        weighted = sum(item.score * item.relevance for item in self.items) / total_relevance
        return max(-1.0, min(1.0, weighted))


# =============================================================================== alerts


class AlertRecord(SwarmModel):
    """One row of the ``alerts_log`` table (every alert is logged, sent or not)."""

    id: int | None = None
    timestamp: UTCDatetime = Field(default_factory=utc_now)
    rule: str
    symbol: str | None = None
    channel: str = "email"
    status: AlertStatus
    subject: str = ""
    detail: str = ""


__all__ = [
    "SENTIMENT_SUMMARY_MAX_CHARS",
    "Action",
    "AgentSignal",
    "AlertRecord",
    "AlertStatus",
    "BatchAnalysis",
    "Decision",
    "ExitReason",
    "HeadlineSentiment",
    "NewsItem",
    "Order",
    "OrderStatus",
    "OrderType",
    "PortfolioSnapshot",
    "Position",
    "RiskAssessment",
    "ScanReport",
    "SentimentJudgement",
    "SentimentLabel",
    "Side",
    "SignalValue",
    "SwarmModel",
    "SymbolError",
    "Trade",
    "UTCDatetime",
    "UnitInterval",
    "ensure_utc",
    "new_id",
    "to_jsonable",
    "utc_now",
]
