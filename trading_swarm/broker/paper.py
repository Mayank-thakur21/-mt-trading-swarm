"""SQLite-backed paper broker: the persistent, thread-safe paper-trading account.

Replaces the draft's in-memory global ``portfolio`` dict (lost on restart and racy under
concurrent requests). State lives in the database configured by ``DATABASE_URL``; every
public method runs in its own transaction under a re-entrant lock, so the API's thread
pool, the scheduler and the coordinator's worker threads can share one broker.

Besides the account itself the broker persists the swarm's audit trail — decisions,
alert log and small key/value settings (the watchlist) — because they share the same
database and lifecycle.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, Self

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from trading_swarm.broker.accounting import (
    CostModel,
    OrderOutcome,
    OrderRecord,
    PositionState,
    TradeRecord,
    all_priced,
    build_snapshot,
    execute_market_order,
    mark_to_market,
    validate_order_request,
)
from trading_swarm.broker.base import UNSET, Unset
from trading_swarm.broker.db import (
    AccountRow,
    AlertLogRow,
    Database,
    DecisionRow,
    EquitySnapshotRow,
    OrderRow,
    PositionRow,
    SettingKVRow,
    TradeRow,
)
from trading_swarm.config import Settings
from trading_swarm.data.base import validate_symbol
from trading_swarm.models import (
    AlertRecord,
    AlertStatus,
    Decision,
    Order,
    OrderStatus,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
    Trade,
    ensure_utc,
    utc_now,
)

logger = logging.getLogger(__name__)

_ACCOUNT_ID = 1
WATCHLIST_KEY = "watchlist"


# =============================================================================== row mappers


def _position_state(row: PositionRow) -> PositionState:
    return PositionState(
        symbol=row.symbol,
        quantity=row.quantity,
        avg_price=row.avg_price,
        stop_price=row.stop_price,
        take_profit_price=row.take_profit_price,
        opened_at=row.opened_at,
    )


def _order_record(row: OrderRow) -> OrderRecord:
    return OrderRecord(
        id=row.id,
        symbol=row.symbol,
        side=Side(row.side),
        quantity=row.quantity,
        type=OrderType(row.type),
        limit_price=row.limit_price,
        status=OrderStatus(row.status),
        requested_at=row.requested_at,
        reference_price=row.reference_price,
        fill_price=row.fill_price,
        filled_at=row.filled_at,
        commission=row.commission,
        slippage=row.slippage,
        stop_price=row.stop_price,
        take_profit_price=row.take_profit_price,
        realized_pnl=row.realized_pnl,
        reason=row.reason,
        decision_id=row.decision_id,
        reject_reason=row.reject_reason,
    )


def _order_row(record: OrderRecord) -> OrderRow:
    return OrderRow(
        id=record.id,
        symbol=record.symbol,
        side=record.side.value,
        quantity=record.quantity,
        type=record.type.value,
        limit_price=record.limit_price,
        reference_price=record.reference_price,
        status=record.status.value,
        requested_at=record.requested_at,
        filled_at=record.filled_at,
        fill_price=record.fill_price,
        commission=record.commission,
        slippage=record.slippage,
        stop_price=record.stop_price,
        take_profit_price=record.take_profit_price,
        realized_pnl=record.realized_pnl,
        reason=record.reason,
        decision_id=record.decision_id,
        reject_reason=record.reject_reason,
    )


def _trade_record(row: TradeRow) -> TradeRecord:
    return TradeRecord(
        id=row.id,
        order_id=row.order_id,
        symbol=row.symbol,
        side=Side(row.side),
        quantity=row.quantity,
        price=row.price,
        commission=row.commission,
        realized_pnl=row.realized_pnl,
        timestamp=row.timestamp,
        reason=row.reason,
    )


def _alert_record(row: AlertLogRow) -> AlertRecord:
    return AlertRecord(
        id=row.id,
        timestamp=row.timestamp,
        rule=row.rule,
        symbol=row.symbol,
        channel=row.channel,
        status=AlertStatus(row.status),
        subject=row.subject,
        detail=row.detail,
    )


# =============================================================================== broker


class PaperBroker:
    """Persistent long-only paper account (implements :class:`~trading_swarm.broker.base.Broker`).

    Args:
        settings: Supplies ``DATABASE_URL``, ``INITIAL_CAPITAL`` and the cost model.
        database: Share an existing :class:`Database` (the broker then does not dispose it).
        cost_model: Override commission/slippage (defaults to the settings).

    Opening a broker on an existing database resumes that account; the configured
    ``INITIAL_CAPITAL`` only seeds a brand-new account (use :meth:`reset` to change it).
    """

    def __init__(
        self,
        settings: Settings,
        database: Database | None = None,
        *,
        cost_model: CostModel | None = None,
    ) -> None:
        self.settings = settings
        self._owns_db = database is None
        self.database = database if database is not None else Database(settings.database_url)
        self.cost_model = cost_model or CostModel.from_settings(settings)
        self._lock = threading.RLock()
        self._ensure_account()

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        """Release database connections (only if this broker created the database)."""
        if self._owns_db:
            self.database.dispose()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _ensure_account(self) -> None:
        with self._lock, self.database.session() as session:
            if session.get(AccountRow, _ACCOUNT_ID) is None:
                capital = float(self.settings.initial_capital)
                now = utc_now()
                session.add(
                    AccountRow(
                        id=_ACCOUNT_ID,
                        cash=capital,
                        initial_capital=capital,
                        peak_equity=capital,
                        realized_pnl=0.0,
                        created_at=now,
                        updated_at=now,
                    )
                )
                session.add(
                    EquitySnapshotRow(
                        timestamp=now, equity=capital, cash=capital, positions_value=0
                    )
                )
                logger.info("created paper account with $%s", f"{capital:,.2f}")

    @staticmethod
    def _account(session: Session) -> AccountRow:
        account = session.get(AccountRow, _ACCOUNT_ID)
        if account is None:  # pragma: no cover - created in __init__
            raise RuntimeError("paper account row is missing")
        return account

    @staticmethod
    def _states(session: Session) -> list[PositionState]:
        rows = session.scalars(select(PositionRow).order_by(PositionRow.symbol))
        return [_position_state(row) for row in rows]

    # ------------------------------------------------------------------ account info
    @property
    def initial_capital(self) -> float:
        with self._lock, self.database.session() as session:
            return self._account(session).initial_capital

    @property
    def cash(self) -> float:
        with self._lock, self.database.session() as session:
            return self._account(session).cash

    # ------------------------------------------------------------------ orders
    def submit_order(
        self,
        symbol: str,
        side: Side | str,
        quantity: int,
        price: float,
        reason: str = "",
        stop: float | None = None,
        take_profit: float | None = None,
        decision_id: str | None = None,
        *,
        timestamp: datetime | None = None,
    ) -> Order:
        """Fill a market order at ``price`` ± slippage (see :class:`Broker`).

        ``price`` must come from the market-data provider, never from an API client.
        """
        sym = validate_symbol(symbol)
        parsed_side = validate_order_request(side, quantity, price, stop, take_profit)
        with self._lock, self.database.session() as session:
            account = self._account(session)
            row = session.get(PositionRow, sym)
            outcome = execute_market_order(
                symbol=sym,
                side=parsed_side,
                quantity=quantity,
                price=float(price),
                cash=account.cash,
                position=_position_state(row) if row is not None else None,
                costs=self.cost_model,
                reason=reason,
                stop=stop,
                take_profit=take_profit,
                decision_id=decision_id,
                timestamp=timestamp,
            )
            self._persist(session, account, row, outcome)
        order = outcome.order.to_model()
        if order.status is OrderStatus.FILLED:
            logger.info(
                "paper %s %d %s @ %.4f (commission %.2f, slippage %.2f)%s",
                order.side.value,
                order.quantity,
                order.symbol,
                order.fill_price,
                order.commission,
                order.slippage,
                f", realized P&L {order.realized_pnl:+.2f}"
                if order.realized_pnl is not None
                else "",
            )
        else:
            logger.info(
                "paper %s %d %s rejected: %s",
                order.side.value,
                order.quantity,
                order.symbol,
                order.reject_reason,
            )
        return order

    def _persist(
        self,
        session: Session,
        account: AccountRow,
        row: PositionRow | None,
        outcome: OrderOutcome,
    ) -> None:
        session.add(_order_row(outcome.order))
        if not outcome.filled or outcome.trade is None:
            return
        trade = outcome.trade
        session.add(
            TradeRow(
                id=trade.id,
                order_id=trade.order_id,
                symbol=trade.symbol,
                side=trade.side.value,
                quantity=trade.quantity,
                price=trade.price,
                commission=trade.commission,
                realized_pnl=trade.realized_pnl,
                timestamp=trade.timestamp,
                reason=trade.reason,
            )
        )
        state = outcome.position
        now = trade.timestamp
        if state is None:
            if row is not None:
                session.delete(row)
        elif row is None:
            session.add(
                PositionRow(
                    symbol=state.symbol,
                    quantity=state.quantity,
                    avg_price=state.avg_price,
                    stop_price=state.stop_price,
                    take_profit_price=state.take_profit_price,
                    opened_at=state.opened_at,
                    updated_at=now,
                )
            )
        else:
            row.quantity = state.quantity
            row.avg_price = state.avg_price
            row.stop_price = state.stop_price
            row.take_profit_price = state.take_profit_price
            row.updated_at = now
        account.cash = outcome.cash
        account.realized_pnl += outcome.realized_pnl
        account.updated_at = now

    def update_stops(
        self,
        symbol: str,
        stop: float | Unset | None = UNSET,
        take_profit: float | Unset | None = UNSET,
    ) -> Position:
        sym = validate_symbol(symbol)
        with self._lock, self.database.session() as session:
            row = session.get(PositionRow, sym)
            if row is None:
                raise KeyError(f"no open position in {sym}")
            if stop is not UNSET:
                row.stop_price = None if stop is None else float(stop)
            if take_profit is not UNSET:
                row.take_profit_price = None if take_profit is None else float(take_profit)
            row.updated_at = utc_now()
            state = _position_state(row)
        return state.mark(None)

    # ------------------------------------------------------------------ queries
    def portfolio(self, prices: Mapping[str, float] | None = None) -> PortfolioSnapshot:
        """Snapshot marked to ``prices``; raises the stored peak when fully priced."""
        with self._lock, self.database.session() as session:
            account = self._account(session)
            states = self._states(session)
            snapshot = build_snapshot(
                initial_capital=account.initial_capital,
                cash=account.cash,
                positions=states,
                realized_pnl=account.realized_pnl,
                peak_equity=account.peak_equity,
                prices=prices,
            )
            positions_value, _ = mark_to_market(states, prices)
            self._maybe_raise_peak(account, states, prices, account.cash + positions_value)
        return snapshot

    @staticmethod
    def _maybe_raise_peak(
        account: AccountRow,
        states: Sequence[PositionState],
        prices: Mapping[str, float] | None,
        equity: float,
    ) -> None:
        # Only trust equity marked with real prices; avg-price marks are not observations.
        if equity > account.peak_equity and all_priced(states, prices):
            account.peak_equity = equity
            account.updated_at = utc_now()

    def positions(self) -> list[Position]:
        with self._lock, self.database.session() as session:
            return [state.mark(None) for state in self._states(session)]

    def position(self, symbol: str) -> Position | None:
        sym = validate_symbol(symbol)
        with self._lock, self.database.session() as session:
            row = session.get(PositionRow, sym)
            return _position_state(row).mark(None) if row is not None else None

    def orders(self, limit: int | None = 50, symbol: str | None = None) -> list[Order]:
        stmt = select(OrderRow).order_by(OrderRow.requested_at.desc(), OrderRow.pk.desc())
        if symbol:
            stmt = stmt.where(OrderRow.symbol == validate_symbol(symbol))
        if limit is not None:
            stmt = stmt.limit(max(0, limit))
        with self._lock, self.database.session() as session:
            return [_order_record(row).to_model() for row in session.scalars(stmt)]

    def get_order(self, order_id: str) -> Order | None:
        with self._lock, self.database.session() as session:
            row = session.scalar(select(OrderRow).where(OrderRow.id == order_id))
            return _order_record(row).to_model() if row is not None else None

    def trades(self, limit: int | None = 100, symbol: str | None = None) -> list[Trade]:
        stmt = select(TradeRow).order_by(TradeRow.timestamp.desc(), TradeRow.pk.desc())
        if symbol:
            stmt = stmt.where(TradeRow.symbol == validate_symbol(symbol))
        if limit is not None:
            stmt = stmt.limit(max(0, limit))
        with self._lock, self.database.session() as session:
            return [_trade_record(row).to_model() for row in session.scalars(stmt)]

    # ------------------------------------------------------------------ equity curve
    def record_snapshot(
        self, prices: Mapping[str, float] | None = None, *, timestamp: datetime | None = None
    ) -> PortfolioSnapshot:
        """Mark to ``prices``, append a row to ``equity_snapshots`` and return the snapshot."""
        when = ensure_utc(timestamp) if timestamp is not None else utc_now()
        with self._lock, self.database.session() as session:
            account = self._account(session)
            states = self._states(session)
            snapshot = build_snapshot(
                initial_capital=account.initial_capital,
                cash=account.cash,
                positions=states,
                realized_pnl=account.realized_pnl,
                peak_equity=account.peak_equity,
                prices=prices,
                timestamp=when,
            )
            positions_value, _ = mark_to_market(states, prices)
            equity = account.cash + positions_value
            session.add(
                EquitySnapshotRow(
                    timestamp=when,
                    equity=equity,
                    cash=account.cash,
                    positions_value=positions_value,
                )
            )
            self._maybe_raise_peak(account, states, prices, equity)
        return snapshot

    def equity_curve(self) -> pd.DataFrame:
        """Recorded snapshots: UTC ``DatetimeIndex`` with ``equity, cash, positions_value``."""
        stmt = select(EquitySnapshotRow).order_by(EquitySnapshotRow.timestamp, EquitySnapshotRow.id)
        with self._lock, self.database.session() as session:
            rows = [
                (r.timestamp, r.equity, r.cash, r.positions_value) for r in session.scalars(stmt)
            ]
        frame = pd.DataFrame(rows, columns=["timestamp", "equity", "cash", "positions_value"])
        index = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True), name="timestamp")
        frame.index = index
        return frame.astype(float)

    # ------------------------------------------------------------------ reset
    def reset(self, initial_capital: float | None = None) -> PortfolioSnapshot:
        """Wipe positions, orders, trades and the equity curve; start over with fresh cash.

        Decisions, the alert log and settings are kept (they are history, not account
        state).
        """
        capital = float(
            initial_capital if initial_capital is not None else self.settings.initial_capital
        )
        if not capital > 0:
            raise ValueError(f"initial capital must be positive, got {initial_capital!r}")
        now = utc_now()
        with self._lock, self.database.session() as session:
            for table in (PositionRow, OrderRow, TradeRow, EquitySnapshotRow):
                session.execute(delete(table))
            account = self._account(session)
            account.cash = capital
            account.initial_capital = capital
            account.peak_equity = capital
            account.realized_pnl = 0.0
            account.updated_at = now
            session.add(
                EquitySnapshotRow(timestamp=now, equity=capital, cash=capital, positions_value=0)
            )
        logger.info("paper account reset to $%s", f"{capital:,.2f}")
        return self.portfolio()

    # ------------------------------------------------------------------ decisions
    def save_decision(self, decision: Decision) -> None:
        """Insert or update ``decision`` (keyed by ``decision.id``) in ``decisions``."""
        payload = decision.model_dump_json()
        with self._lock, self.database.session() as session:
            row = session.scalar(select(DecisionRow).where(DecisionRow.id == decision.id))
            if row is None:
                row = DecisionRow(id=decision.id)
                session.add(row)
            row.symbol = decision.symbol
            row.timestamp = decision.timestamp
            row.action = decision.action.value
            row.score = decision.score
            row.confidence = decision.confidence
            row.executed = decision.executed
            row.payload = payload

    def decisions(self, limit: int | None = 50, symbol: str | None = None) -> list[Decision]:
        """Most recent decisions first, optionally for one symbol."""
        stmt = select(DecisionRow).order_by(DecisionRow.timestamp.desc(), DecisionRow.pk.desc())
        if symbol:
            stmt = stmt.where(DecisionRow.symbol == validate_symbol(symbol))
        if limit is not None:
            stmt = stmt.limit(max(0, limit))
        with self._lock, self.database.session() as session:
            payloads = [row.payload for row in session.scalars(stmt)]
        decisions: list[Decision] = []
        for payload in payloads:
            try:
                decisions.append(Decision.model_validate_json(payload))
            except ValueError as exc:  # pragma: no cover - corrupted/legacy rows
                logger.warning("skipping unreadable decision row: %s", exc)
        return decisions

    def get_decision(self, decision_id: str) -> Decision | None:
        with self._lock, self.database.session() as session:
            row = session.scalar(select(DecisionRow).where(DecisionRow.id == decision_id))
            payload = row.payload if row is not None else None
        return Decision.model_validate_json(payload) if payload else None

    # ------------------------------------------------------------------ alerts
    def log_alert(self, record: AlertRecord) -> AlertRecord:
        """Append ``record`` to ``alerts_log`` and return it with its database id."""
        with self._lock, self.database.session() as session:
            row = AlertLogRow(
                timestamp=record.timestamp,
                rule=record.rule,
                symbol=record.symbol,
                channel=record.channel,
                status=record.status.value,
                subject=record.subject,
                detail=record.detail,
            )
            session.add(row)
            session.flush()
            return _alert_record(row)

    def alerts(
        self, limit: int | None = 50, rule: str | None = None, symbol: str | None = None
    ) -> list[AlertRecord]:
        """Most recent alert-log entries first."""
        stmt = select(AlertLogRow).order_by(AlertLogRow.timestamp.desc(), AlertLogRow.id.desc())
        if rule:
            stmt = stmt.where(AlertLogRow.rule == rule)
        if symbol:
            stmt = stmt.where(AlertLogRow.symbol == symbol.upper())
        if limit is not None:
            stmt = stmt.limit(max(0, limit))
        with self._lock, self.database.session() as session:
            return [_alert_record(row) for row in session.scalars(stmt)]

    def last_alert_at(
        self,
        rule: str,
        symbol: str | None = None,
        statuses: Iterable[AlertStatus | str] | None = (AlertStatus.SENT,),
    ) -> datetime | None:
        """Timestamp of the latest ``rule`` alert for ``symbol`` (for cooldown dedupe).

        ``statuses`` restricts which outcomes count (default: only alerts actually sent);
        pass ``None`` to consider every logged alert.
        """
        stmt = select(AlertLogRow.timestamp).where(AlertLogRow.rule == rule)
        stmt = stmt.where(
            AlertLogRow.symbol.is_(None) if symbol is None else AlertLogRow.symbol == symbol.upper()
        )
        if statuses is not None:
            stmt = stmt.where(AlertLogRow.status.in_([AlertStatus(s).value for s in statuses]))
        stmt = stmt.order_by(AlertLogRow.timestamp.desc()).limit(1)
        with self._lock, self.database.session() as session:
            return session.scalar(stmt)

    # ------------------------------------------------------------------ settings KV
    def get_setting(self, key: str, default: Any = None) -> Any:
        """JSON value stored under ``key`` in ``settings_kv`` (``default`` if absent)."""
        with self._lock, self.database.session() as session:
            row = session.get(SettingKVRow, key)
            raw = row.value if row is not None else None
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:  # pragma: no cover - corrupted row
            logger.warning("settings_kv[%s] is not valid JSON; using default", key)
            return default

    def set_setting(self, key: str, value: Any) -> None:
        """Store a JSON-serialisable ``value`` under ``key``."""
        raw = json.dumps(value)
        with self._lock, self.database.session() as session:
            row = session.get(SettingKVRow, key)
            if row is None:
                session.add(SettingKVRow(key=key, value=raw, updated_at=utc_now()))
            else:
                row.value = raw
                row.updated_at = utc_now()

    def get_watchlist(self, default: Sequence[str] | None = None) -> list[str]:
        """Persisted watchlist, or ``default`` (e.g. ``settings.watchlist_symbols``)."""
        stored = self.get_setting(WATCHLIST_KEY)
        if isinstance(stored, list) and all(isinstance(s, str) for s in stored):
            return list(stored)
        return list(default) if default is not None else list(self.settings.watchlist_symbols)

    def set_watchlist(self, symbols: Iterable[str]) -> list[str]:
        """Validate, de-duplicate and persist the watchlist; returns the stored list."""
        cleaned: dict[str, None] = {}
        for symbol in symbols:
            cleaned.setdefault(validate_symbol(symbol), None)
        watchlist = list(cleaned)
        self.set_setting(WATCHLIST_KEY, watchlist)
        return watchlist


__all__ = ["WATCHLIST_KEY", "PaperBroker"]
