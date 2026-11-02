"""SQLAlchemy 2.0 (synchronous) engine, sessions and ORM tables for the paper account.

Tables: ``account`` (single row), ``positions``, ``orders``, ``trades``,
``equity_snapshots``, ``decisions`` (JSON payload), ``alerts_log`` and ``settings_kv``.

SQLite specifics: the parent directory of a file database is created on demand; file
databases run in WAL mode with a busy timeout so the API, the scheduler thread and a CLI
can share one file; ``:memory:`` databases use a single shared connection. SQLite has no
time-zone-aware datetime type, so :class:`UTCDateTime` stores naive UTC and re-attaches
``UTC`` on load.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Engine,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import TypeDecorator

from trading_swarm.models import ensure_utc, utc_now

logger = logging.getLogger(__name__)


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes on every backend (naive UTC in storage)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class AccountRow(Base):
    __tablename__ = "account"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cash: Mapped[float] = mapped_column(Float)
    initial_capital: Mapped[float] = mapped_column(Float)
    peak_equity: Mapped[float] = mapped_column(Float)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class PositionRow(Base):
    __tablename__ = "positions"

    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    quantity: Mapped[int] = mapped_column(Integer)
    avg_price: Mapped[float] = mapped_column(Float)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    opened_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class OrderRow(Base):
    __tablename__ = "orders"

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    type: Mapped[str] = mapped_column(String(8), default="MARKET")
    limit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    reference_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(10), index=True)
    requested_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    filled_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    fill_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)
    stop_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    decision_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class TradeRow(Base):
    __tablename__ = "trades"

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    side: Mapped[str] = mapped_column(String(4))
    quantity: Mapped[int] = mapped_column(Integer)
    price: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    realized_pnl: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    reason: Mapped[str] = mapped_column(Text, default="")


class EquitySnapshotRow(Base):
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float, default=0.0)


class DecisionRow(Base):
    __tablename__ = "decisions"

    pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(16), index=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    action: Mapped[str] = mapped_column(String(4))
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    executed: Mapped[bool] = mapped_column(Boolean, default=False)
    payload: Mapped[str] = mapped_column(Text)


class AlertLogRow(Base):
    __tablename__ = "alerts_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)
    rule: Mapped[str] = mapped_column(String(64), index=True)
    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    channel: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), index=True)
    subject: Mapped[str] = mapped_column(Text, default="")
    detail: Mapped[str] = mapped_column(Text, default="")


class SettingKVRow(Base):
    __tablename__ = "settings_kv"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


# =============================================================================== engine


def _is_memory_sqlite(database: str | None, query: Any) -> bool:
    return database in (None, "", ":memory:") or str(query.get("mode", "")) == "memory"


def create_db_engine(url: str, *, echo: bool = False) -> Engine:
    """Create an engine for ``url``; SQLite gets thread-friendly settings (see module doc)."""
    parsed = make_url(url)
    kwargs: dict[str, Any] = {"echo": echo}
    is_sqlite = parsed.get_backend_name() == "sqlite"
    memory = False
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        memory = _is_memory_sqlite(parsed.database, parsed.query)
        if memory:
            kwargs["poolclass"] = StaticPool
        else:
            Path(parsed.database or "").expanduser().parent.mkdir(parents=True, exist_ok=True)
    else:
        kwargs["pool_pre_ping"] = True
    engine = create_engine(url, **kwargs)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA busy_timeout = 30000")
                if not memory:
                    cursor.execute("PRAGMA journal_mode = WAL")
                    cursor.execute("PRAGMA synchronous = NORMAL")
            finally:
                cursor.close()

    return engine


class Database:
    """Engine + session factory; creates the schema on construction."""

    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        self.engine = create_db_engine(url, echo=echo)
        self._sessionmaker = sessionmaker(self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        logger.debug("database ready at %s", make_url(url).render_as_string(hide_password=True))

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A transactional session: commits on success, rolls back on error."""
        session = self._sessionmaker()
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Close all pooled connections."""
        self.engine.dispose()


__all__ = [
    "AccountRow",
    "AlertLogRow",
    "Base",
    "Database",
    "DecisionRow",
    "EquitySnapshotRow",
    "OrderRow",
    "PositionRow",
    "SettingKVRow",
    "TradeRow",
    "UTCDateTime",
    "create_db_engine",
]
