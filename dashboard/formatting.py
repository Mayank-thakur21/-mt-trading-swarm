"""Pure display helpers for the dashboard: number formats, tables and labels.

Nothing here imports Streamlit; app.py renders what these functions build, and the test
suite checks them directly. Conventions match the API: fields ending in ``_pct`` are
percentages (``5.0`` → "5.00%"), signals live in ``[-1, 1]`` and confidences in
``[0, 1]``. Missing values (``None``, NaN, ±inf) display as an em dash, never "nan".
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import pandas as pd
from pandas.io.formats.style import Styler

from dashboard.charts import LIGHT, ChartTheme

MISSING: Final = "—"

# =============================================================================== numbers


def as_float(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` (for ``None``, NaN, inf, garbage)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fmt_money(value: Any, *, signed: bool = False, decimals: int = 2) -> str:
    """``1234.5`` → ``$1,234.50``; ``-3`` → ``-$3.00``; ``signed`` adds ``+`` for gains."""
    number = as_float(value)
    if number is None:
        return MISSING
    sign = "-" if number < 0 else ("+" if signed and number > 0 else "")
    return f"{sign}${abs(number):,.{decimals}f}"


#: From this absolute amount up, KPI tiles drop the cents (see :func:`fmt_money_kpi`).
KPI_WHOLE_DOLLARS_FROM: Final = 10_000.0


def fmt_money_kpi(value: Any, *, signed: bool = False) -> str:
    """Money for a ``st.metric`` tile: whole dollars from $10,000 up, cents below.

    Four tiles to a row leave about 190 px per value at a 1280 px window, which
    truncates ``$99,980.67`` to ``$99,980.…``; ``$99,981`` fits. Callers put the exact
    amount in the tile's tooltip.
    """
    number = as_float(value)
    whole = number is not None and abs(number) >= KPI_WHOLE_DOLLARS_FROM
    return fmt_money(value, signed=signed, decimals=0 if whole else 2)


def fmt_pct(value: Any, *, signed: bool = False, decimals: int = 2) -> str:
    """A *percentage* value (``5.0`` → ``5.00%``)."""
    number = as_float(value)
    if number is None:
        return MISSING
    return f"{number:+.{decimals}f}%" if signed else f"{number:.{decimals}f}%"


def fmt_fraction_pct(value: Any, *, signed: bool = False, decimals: int = 0) -> str:
    """A *fraction* shown as a percentage (``0.82`` → ``82%``) — for confidences."""
    number = as_float(value)
    if number is None:
        return MISSING
    return fmt_pct(number * 100.0, signed=signed, decimals=decimals)


def fmt_signal(value: Any, decimals: int = 2) -> str:
    """A signed score in ``[-1, 1]`` (``0.7427`` → ``+0.74``)."""
    number = as_float(value)
    return MISSING if number is None else f"{number:+.{decimals}f}"


def fmt_number(value: Any, decimals: int = 2) -> str:
    number = as_float(value)
    return MISSING if number is None else f"{number:,.{decimals}f}"


def fmt_price(value: Any) -> str:
    """Prices: 2 decimals above $1, 4 below (penny stocks, FX)."""
    number = as_float(value)
    if number is None:
        return MISSING
    return f"{number:,.2f}" if abs(number) >= 1 else f"{number:,.4f}"


def parse_timestamp(value: Any) -> datetime | None:
    """ISO-8601 string / datetime → aware UTC datetime (``None`` if unparseable)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    try:
        parsed = pd.Timestamp(str(value))
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
    return parsed.to_pydatetime()


def fmt_timestamp(value: Any, *, seconds: bool = False) -> str:
    """``2026-11-02T17:57:37Z`` → ``2026-11-02 17:57 UTC``."""
    parsed = parse_timestamp(value)
    if parsed is None:
        return MISSING
    return parsed.strftime("%Y-%m-%d %H:%M:%S UTC" if seconds else "%Y-%m-%d %H:%M UTC")


def pct_distance(price: Any, level: Any) -> float | None:
    """Signed distance from ``price`` to ``level`` in percent of ``price``."""
    p, lvl = as_float(price), as_float(level)
    if p is None or lvl is None or p <= 0:
        return None
    return (lvl / p - 1.0) * 100.0


def reward_risk(entry: Any, stop: Any, target: Any) -> float | None:
    """Reward-to-risk multiple ``(target − entry) / (entry − stop)`` of a long trade."""
    e, s, t = as_float(entry), as_float(stop), as_float(target)
    if e is None or s is None or t is None or e <= s:
        return None
    return (t - e) / (e - s)


# =============================================================================== labels

#: Streamlit ``st.badge`` colours per action — the same blue/red/grey direction encoding
#: as the charts (the badge text carries the meaning; colour only reinforces it).
ACTION_BADGE_COLORS: Final = {"BUY": "blue", "SELL": "red", "HOLD": "gray"}
ALERT_STATUS_BADGE_COLORS: Final = {
    "sent": "green",
    "failed": "red",
    "disabled": "gray",
    "suppressed": "orange",
}

#: Human labels for the risk agent's ``checks`` keys (unknown keys are humanised).
CHECK_LABELS: Final = {
    "price_valid": "Valid price",
    "equity_positive": "Positive equity",
    "drawdown_ok": "Drawdown below halt level",
    "volatility_ok": "Volatility within limit",
    "stop_from_atr": "Stop derived from ATR",
    "position_limit_ok": "Below max position size",
    "cash_ok": "Enough cash (incl. costs)",
    "exposure_ok": "Gross-exposure headroom",
    "min_quantity_ok": "At least one share",
    "position_exists": "Position available to sell",
}

#: Display names of the trend agent's components.
COMPONENT_LABELS: Final = {
    "sma20": "Price vs SMA20",
    "sma50": "Price vs SMA50",
    "sma200": "Price vs SMA200",
    "cross": "SMA50/200 cross",
    "macd": "MACD histogram",
    "rsi": "RSI regime",
    "momentum": "20-day momentum",
}


_MARKDOWN_SPECIALS: Final = frozenset("\\`*_[]<>#|~$")


def escape_markdown(text: Any) -> str:
    """Escape text from the API before Streamlit renders it as Markdown.

    Rationales, risk reasons, reject reasons and headlines are data, not markup: an
    unescaped ``$1,000 … $3.73`` pair would render as LaTeX, and third-party headlines
    could inject links or HTML-looking text. Every Markdown-significant character
    (including ``$`` for math and ``[]`` for Streamlit's ``:color[...]`` directives) is
    backslash-escaped, which CommonMark renders as the literal character.
    """
    return "".join(f"\\{ch}" if ch in _MARKDOWN_SPECIALS else ch for ch in str(text or ""))


def humanize(key: str) -> str:
    """``position_limit_ok`` → ``Position limit ok``."""
    text = str(key).replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def action_badge_color(action: Any) -> str:
    return ACTION_BADGE_COLORS.get(str(action or "").upper(), "gray")


def risk_checks(checks: Mapping[str, Any] | None) -> list[tuple[str, bool]]:
    """``RiskAssessment.checks`` → ``[(label, passed)]`` in the order the rules ran."""
    return [(CHECK_LABELS.get(k, humanize(k)), bool(v)) for k, v in (checks or {}).items()]


def proposed_action(score: Any, thresholds: Thresholds) -> str:
    """The action the consensus score proposes before the risk gate (BUY/SELL/HOLD).

    A final HOLD whose score crossed a threshold therefore means "vetoed by risk".
    """
    value = as_float(score)
    if value is None:
        return "HOLD"
    if value >= thresholds.buy:
        return "BUY"
    if value <= thresholds.sell:
        return "SELL"
    return "HOLD"


#: Display names of the protective-exit order reasons (``ExitReason`` values).
EXIT_LABELS: Final = {"stop_loss": "stop-loss", "take_profit": "take-profit"}


def exit_label(order: Mapping[str, Any]) -> str:
    """``"stop-loss"`` / ``"take-profit"`` for a protective exit order payload."""
    reason = str(order.get("reason") or "")
    return EXIT_LABELS.get(reason, reason or "protective exit")


def decision_exits(decision: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The protective exits a decision made before sizing (``[]`` for older payloads)."""
    return [o for o in decision.get("exits") or [] if isinstance(o, Mapping)]


def own_exit(decision: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The filled protective exit of the decision's own symbol, if one closed it."""
    symbol = str(decision.get("symbol") or "").upper()
    for order in decision_exits(decision):
        if str(order.get("symbol") or "").upper() == symbol and order.get("status") == "FILLED":
            return order
    return None


def signal_for(decision: Mapping[str, Any], agent: str) -> Mapping[str, Any] | None:
    """The vote of ``agent`` in a decision payload, if present."""
    for item in decision.get("signals") or []:
        if isinstance(item, Mapping) and item.get("agent") == agent:
            return item
    return None


# =============================================================================== settings


@dataclass(frozen=True, slots=True)
class Thresholds:
    """Consensus cut-offs used to label the gauge (``score ≥ buy`` → BUY, ``≤ sell`` → SELL)."""

    buy: float = 0.25
    sell: float = -0.25


def resolve_thresholds(
    health: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None
) -> Thresholds:
    """Consensus thresholds: from ``/health`` when the API reports them, else from the
    shared environment (``BUY_THRESHOLD`` / ``SELL_THRESHOLD``, same ``.env`` as the API),
    else the defaults. Inconsistent values (sell ≥ buy) fall back to the defaults."""
    env = os.environ if environ is None else environ
    health = health or {}
    raw_nested = health.get("thresholds")
    nested: Mapping[str, Any] = raw_nested if isinstance(raw_nested, Mapping) else {}
    default = Thresholds()

    def first(*candidates: Any, fallback: float) -> float:
        # Explicit None checks: 0.0 is a legitimate threshold, so no ``or`` chaining.
        for candidate in candidates:
            number = as_float(candidate)
            if number is not None:
                return number
        return fallback

    buy = first(
        health.get("buy_threshold"),
        nested.get("buy"),
        env.get("BUY_THRESHOLD"),
        fallback=default.buy,
    )
    sell = first(
        health.get("sell_threshold"),
        nested.get("sell"),
        env.get("SELL_THRESHOLD"),
        fallback=default.sell,
    )
    if not (-1.0 <= sell < buy <= 1.0):
        return default
    return Thresholds(buy=buy, sell=sell)


# =============================================================================== tables


def _frame(rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))


def timestamp_column(values: Iterable[Any]) -> pd.Series:
    """Timestamps as tz-naive UTC datetimes (Streamlit renders them without offsets)."""
    parsed = [parse_timestamp(v) for v in values]
    return pd.Series(
        [p.replace(tzinfo=None) if p is not None else pd.NaT for p in parsed],
        dtype="datetime64[us]",
    )


POSITION_COLUMNS: Final = (
    "Symbol",
    "Qty",
    "Avg cost",
    "Last",
    "Market value",
    "Unrealized P&L",
    "P&L %",
    "Stop",
    "To stop %",
    "Take profit",
    "To target %",
    "Opened (UTC)",
)


def positions_table(portfolio: Mapping[str, Any]) -> pd.DataFrame:
    """Open positions with their protective levels and distances (numeric columns)."""
    rows = []
    for p in portfolio.get("positions") or []:
        last = as_float(p.get("market_price")) or as_float(p.get("avg_price"))
        rows.append(
            {
                "Symbol": p.get("symbol"),
                "Qty": p.get("quantity"),
                "Avg cost": as_float(p.get("avg_price")),
                "Last": last,
                "Market value": as_float(p.get("market_value")),
                "Unrealized P&L": as_float(p.get("unrealized_pnl")),
                "P&L %": as_float(p.get("unrealized_pnl_pct")),
                "Stop": as_float(p.get("stop_price")),
                "To stop %": pct_distance(last, p.get("stop_price")),
                "Take profit": as_float(p.get("take_profit_price")),
                "To target %": pct_distance(last, p.get("take_profit_price")),
                "Opened (UTC)": p.get("opened_at"),
            }
        )
    frame = _frame(rows, POSITION_COLUMNS)
    frame["Opened (UTC)"] = timestamp_column(frame["Opened (UTC)"])
    return frame


ORDER_COLUMNS: Final = (
    "Requested (UTC)",
    "Symbol",
    "Side",
    "Qty",
    "Status",
    "Fill price",
    "Reference",
    "Commission",
    "Slippage",
    "Realized P&L",
    "Stop",
    "Take profit",
    "Reason",
    "Reject reason",
    "Order id",
)


def orders_table(orders: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Requested (UTC)": o.get("requested_at"),
            "Symbol": o.get("symbol"),
            "Side": o.get("side"),
            "Qty": o.get("quantity"),
            "Status": o.get("status"),
            "Fill price": as_float(o.get("fill_price")),
            "Reference": as_float(o.get("reference_price")),
            "Commission": as_float(o.get("commission")),
            "Slippage": as_float(o.get("slippage")),
            "Realized P&L": as_float(o.get("realized_pnl")),
            "Stop": as_float(o.get("stop_price")),
            "Take profit": as_float(o.get("take_profit_price")),
            "Reason": o.get("reason") or "",
            "Reject reason": o.get("reject_reason") or "",
            "Order id": o.get("id"),
        }
        for o in orders
    ]
    frame = _frame(rows, ORDER_COLUMNS)
    frame["Requested (UTC)"] = timestamp_column(frame["Requested (UTC)"])
    return frame


TRADE_COLUMNS: Final = (
    "Time (UTC)",
    "Symbol",
    "Side",
    "Qty",
    "Price",
    "Notional",
    "Commission",
    "Realized P&L",
    "Reason",
    "Order id",
)


def trades_table(trades: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for t in trades:
        price, qty = as_float(t.get("price")), as_float(t.get("quantity"))
        rows.append(
            {
                "Time (UTC)": t.get("timestamp"),
                "Symbol": t.get("symbol"),
                "Side": t.get("side"),
                "Qty": t.get("quantity"),
                "Price": price,
                "Notional": price * qty if price is not None and qty is not None else None,
                "Commission": as_float(t.get("commission")),
                "Realized P&L": as_float(t.get("realized_pnl")),
                "Reason": t.get("reason") or "",
                "Order id": t.get("order_id"),
            }
        )
    frame = _frame(rows, TRADE_COLUMNS)
    frame["Time (UTC)"] = timestamp_column(frame["Time (UTC)"])
    return frame


DECISION_COLUMNS: Final = (
    "Time (UTC)",
    "Symbol",
    "Action",
    "Price",
    "Score",
    "Confidence",
    "Trend",
    "Sentiment",
    "Qty",
    "Executed",
    "Summary",
)


def decisions_table(decisions: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = []
    for d in decisions:
        trend = signal_for(d, "trend") or {}
        sentiment = signal_for(d, "sentiment") or {}
        risk = d.get("risk") or {}
        rows.append(
            {
                "Time (UTC)": d.get("timestamp"),
                "Symbol": d.get("symbol"),
                "Action": d.get("action"),
                "Price": as_float(d.get("price")),
                "Score": as_float(d.get("score")),
                "Confidence": as_float(d.get("confidence")),
                "Trend": as_float(trend.get("signal")),
                "Sentiment": as_float(sentiment.get("signal")),
                "Qty": risk.get("quantity") if risk.get("approved") else 0,
                "Executed": bool(d.get("executed")),
                "Summary": d.get("summary") or "",
            }
        )
    frame = _frame(rows, DECISION_COLUMNS)
    frame["Time (UTC)"] = timestamp_column(frame["Time (UTC)"])
    return frame


ALERT_COLUMNS: Final = ("Time (UTC)", "Rule", "Symbol", "Status", "Channel", "Subject", "Detail")


def alerts_table(alerts: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Time (UTC)": a.get("timestamp"),
            "Rule": a.get("rule"),
            "Symbol": a.get("symbol") or "",
            "Status": a.get("status"),
            "Channel": a.get("channel"),
            "Subject": a.get("subject") or "",
            "Detail": a.get("detail") or "",
        }
        for a in alerts
    ]
    frame = _frame(rows, ALERT_COLUMNS)
    frame["Time (UTC)"] = timestamp_column(frame["Time (UTC)"])
    return frame


def errors_table(errors: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Per-symbol failures of batch / scan calls."""
    rows = [
        {"Symbol": e.get("symbol"), "Code": e.get("code") or "error", "Error": e.get("error")}
        for e in errors
    ]
    return _frame(rows, ("Symbol", "Code", "Error"))


def headlines_table(signal: Mapping[str, Any] | None) -> pd.DataFrame:
    """Per-headline scores of the sentiment agent (LLM or lexicon)."""
    data = (signal or {}).get("data") or {}
    rows = []
    for item in data.get("headlines") or []:
        terms = item.get("terms")
        rows.append(
            {
                "Headline": item.get("title"),
                "Sentiment": item.get("sentiment") or MISSING,
                "Score": as_float(item.get("score")),
                "Relevance": as_float(item.get("relevance")),
                "Matched terms": ", ".join(terms) if isinstance(terms, list) else "",
                "Publisher": item.get("publisher") or "",
                "Published (UTC)": item.get("published_at"),
            }
        )
    frame = _frame(
        rows,
        (
            "Headline",
            "Sentiment",
            "Score",
            "Relevance",
            "Matched terms",
            "Publisher",
            "Published (UTC)",
        ),
    )
    frame["Published (UTC)"] = timestamp_column(frame["Published (UTC)"])
    return frame.dropna(axis="columns", how="all")


def trend_components_table(signal: Mapping[str, Any] | None) -> pd.DataFrame:
    """The trend agent's component scores, weights and contributions to its signal."""
    data = (signal or {}).get("data") or {}
    components = data.get("components") or {}
    weights = data.get("weights") or {}
    rows = []
    for key, score in components.items():
        value, weight = as_float(score), as_float(weights.get(key))
        rows.append(
            {
                "Component": COMPONENT_LABELS.get(key, humanize(key)),
                "Score": value,
                "Weight": weight,
                "Contribution": value * weight
                if value is not None and weight is not None
                else None,
            }
        )
    return _frame(rows, ("Component", "Score", "Weight", "Contribution"))


#: (key, label, decimals, suffix) of the indicator values shown for the trend agent.
INDICATOR_FIELDS: Final = (
    ("close", "Close", 2, ""),
    ("sma20", "SMA20", 2, ""),
    ("sma50", "SMA50", 2, ""),
    ("sma200", "SMA200", 2, ""),
    ("rsi", "RSI 14", 1, ""),
    ("macd", "MACD", 3, ""),
    ("macd_signal", "MACD signal", 3, ""),
    ("macd_hist", "MACD histogram", 3, ""),
    ("atr", "ATR 14", 2, ""),
    ("atr_pct", "ATR % of price", 2, "%"),
    ("adx", "ADX 14", 1, ""),
    ("momentum_20d_pct", "20-day momentum", 2, "%"),
    ("change_1d_pct", "1-day change", 2, "%"),
    ("realized_vol", "Realized vol (annual)", 1, "%"),
)


def indicator_rows(signal: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """``[(label, formatted value)]`` of the trend agent's indicator snapshot."""
    data = (signal or {}).get("data") or {}
    rows = []
    for key, label, decimals, suffix in INDICATOR_FIELDS:
        value = as_float(data.get(key))
        if value is None:
            continue
        if key == "realized_vol":
            value *= 100.0  # annualised fraction → percent
        rows.append((label, f"{value:,.{decimals}f}{suffix}"))
    return rows


#: (key, label, kind) rows of the strategy-vs-benchmark metrics table.
METRIC_ROWS: Final = (
    ("total_return_pct", "Total return", "pct"),
    ("cagr_pct", "CAGR", "pct"),
    ("volatility_pct", "Volatility (annual)", "pct"),
    ("sharpe", "Sharpe ratio", "ratio"),
    ("sortino", "Sortino ratio", "ratio"),
    ("max_drawdown_pct", "Max drawdown", "pct"),
    ("max_drawdown_duration_days", "Longest drawdown (days)", "days"),
    ("calmar", "Calmar ratio", "ratio"),
    ("num_trades", "Closed trades", "int"),
    ("win_rate_pct", "Win rate", "pct"),
    ("profit_factor", "Profit factor", "ratio"),
    ("avg_win", "Average win", "money"),
    ("avg_loss", "Average loss", "money"),
    ("expectancy", "Expectancy / trade", "money"),
    ("exposure_pct", "Time in market", "pct"),
    ("end_equity", "Final equity", "money"),
)


def format_metric(value: Any, kind: str) -> str:
    if kind == "pct":
        return fmt_pct(value)
    if kind == "money":
        return fmt_money(value)
    if kind == "int":
        number = as_float(value)
        return MISSING if number is None else f"{int(number):,}"
    if kind == "days":
        return fmt_number(value, 0)
    return fmt_number(value, 2)


def metrics_table(
    strategy: Mapping[str, Any] | None, benchmark: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    """Side-by-side formatted metrics (strings: this table is for reading, not sorting)."""
    strategy = strategy or {}
    rows = []
    for key, label, kind in METRIC_ROWS:
        if key not in strategy and (benchmark is None or key not in benchmark):
            continue
        row = {"Metric": label, "Strategy": format_metric(strategy.get(key), kind)}
        if benchmark is not None:
            row["Buy & hold"] = format_metric(benchmark.get(key), kind)
        rows.append(row)
    columns = (
        ("Metric", "Strategy", "Buy & hold") if benchmark is not None else ("Metric", "Strategy")
    )
    return _frame(rows, columns)


# =============================================================================== styling


def pnl_color(value: Any, theme: ChartTheme = LIGHT) -> str:
    """CSS for profit/loss text: status good/critical colours (sign is also shown)."""
    number = as_float(value)
    if number is None or number == 0:
        return ""
    return f"color: {theme.good if number > 0 else theme.bad}; font-weight: 600"


def direction_css(value: Any, theme: ChartTheme = LIGHT) -> str:
    """CSS for a directional value: blue when positive, red when negative (as in charts)."""
    number = as_float(value)
    if number is None or number == 0:
        return ""
    return f"color: {theme.up if number > 0 else theme.down}; font-weight: 600"


def style_table(
    frame: pd.DataFrame,
    *,
    money: Sequence[str] = (),
    prices: Sequence[str] = (),
    pct: Sequence[str] = (),
    signed_money: Sequence[str] = (),
    signed_pct: Sequence[str] = (),
    signals: Sequence[str] = (),
    pnl: Sequence[str] = (),
    directional: Sequence[str] = (),
    theme: ChartTheme = LIGHT,
) -> Styler:
    """A :class:`~pandas.io.formats.style.Styler` with number formats and colouring.

    Values stay numeric underneath (sorting in ``st.dataframe`` still works).

    * ``pnl`` columns (money outcomes) use the status pair good/critical (green/red);
    * ``directional`` columns (signals, price moves) use the charts' direction pair
      (blue/red), so a bullish score reads the same in a table as in a chart.

    Coloured columns always carry an explicit ``+``/``-`` sign as well, so the colour
    never carries the meaning alone. Missing values render as an em dash (pass
    ``placeholder`` to ``st.dataframe`` too: Streamlit draws null cells itself).
    """
    formats: dict[str, Any] = {}
    for column in money:
        formats[column] = lambda v: fmt_money(v)
    for column in prices:
        formats[column] = fmt_price
    for column in pct:
        formats[column] = lambda v: fmt_pct(v)
    for column in signed_money:
        formats[column] = lambda v: fmt_money(v, signed=True)
    for column in signed_pct:
        formats[column] = lambda v: fmt_pct(v, signed=True)
    for column in signals:
        formats[column] = fmt_signal
    present = {k: v for k, v in formats.items() if k in frame.columns}
    styler = frame.style.format(present, na_rep=MISSING)
    colored = [c for c in pnl if c in frame.columns]
    if colored:
        styler = styler.map(lambda v: pnl_color(v, theme), subset=colored)
    directed = [c for c in directional if c in frame.columns]
    if directed:
        styler = styler.map(lambda v: direction_css(v, theme), subset=directed)
    return styler.hide(axis="index")


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """UTF-8 CSV for ``st.download_button`` (no index column)."""
    return frame.to_csv(index=False).encode("utf-8")


__all__ = [
    "ACTION_BADGE_COLORS",
    "ALERT_STATUS_BADGE_COLORS",
    "CHECK_LABELS",
    "KPI_WHOLE_DOLLARS_FROM",
    "MISSING",
    "Thresholds",
    "action_badge_color",
    "alerts_table",
    "as_float",
    "decisions_table",
    "direction_css",
    "errors_table",
    "escape_markdown",
    "fmt_fraction_pct",
    "fmt_money",
    "fmt_money_kpi",
    "fmt_number",
    "fmt_pct",
    "fmt_price",
    "fmt_signal",
    "fmt_timestamp",
    "format_metric",
    "headlines_table",
    "humanize",
    "indicator_rows",
    "metrics_table",
    "orders_table",
    "parse_timestamp",
    "pct_distance",
    "pnl_color",
    "positions_table",
    "proposed_action",
    "resolve_thresholds",
    "reward_risk",
    "risk_checks",
    "signal_for",
    "style_table",
    "timestamp_column",
    "to_csv_bytes",
    "trades_table",
    "trend_components_table",
]
