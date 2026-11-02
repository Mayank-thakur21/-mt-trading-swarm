"""MT Trading Swarm — Streamlit dashboard.

Run with ``streamlit run dashboard/app.py`` (the API must be running, by default at
``http://localhost:8000``; configure with ``API_URL`` / ``API_KEY``).

The dashboard is a thin, read-mostly client of the REST API (:mod:`dashboard.client`):

* **Portfolio** — KPIs, equity curve, allocation donut, positions with stops/targets.
* **Analyze** — run the swarm on a symbol: decision card, consensus gauge, per-agent
  votes, risk panel, rationale and a candlestick chart with SMA/RSI/MACD.
* **Backtest** — strategy vs buy-and-hold metrics, equity and drawdown charts, trades.
* **Trades & Orders** — fills, orders and decisions with CSV export; manual paper orders.
* **Alerts** — e-mail configuration status, test e-mail, alert log.
* **Watchlist** — edit the persisted watchlist, live quotes, scan now.

Robustness rules: every API call goes through :func:`api_get` (cached GETs,
``st.cache_data(ttl=30)``) or an explicit mutation followed by :func:`invalidate_cache`;
every :class:`~dashboard.client.ApiError` is rendered as a friendly message (API down,
401, 429, validation, upstream failure) and unexpected exceptions are logged and shown
without a traceback.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``streamlit run dashboard/app.py`` puts dashboard/ (not the repo root) on sys.path;
# make ``import dashboard`` work without requiring ``pip install -e .``.
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Final
from urllib.parse import urlsplit

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import charts
from dashboard import formatting as fmt
from dashboard.client import (
    DEFAULT_API_URL,
    MAX_BACKTEST_SYMBOLS,
    MAX_BATCH_SYMBOLS,
    MAX_CAPITAL,
    ApiAuthError,
    ApiClient,
    ApiConnectionError,
    ApiError,
    ApiNotFoundError,
    ApiRateLimitError,
    ApiResponseError,
    ApiServerError,
    ApiTimeoutError,
    ApiValidationError,
    normalize_symbol,
    normalize_symbols,
)

logger = logging.getLogger("dashboard")
#: Escape API-provided or money-bearing text before it reaches Markdown ($…$ is LaTeX).
md = fmt.escape_markdown

#: Seconds a cached GET stays fresh (spec: ``st.cache_data(ttl=30)``).
GET_CACHE_TTL_SECONDS: Final = 30
#: Client methods that are safe to cache (pure reads).
CACHEABLE_METHODS: Final = frozenset(
    {
        "health",
        "quote",
        "history",
        "portfolio",
        "equity",
        "orders",
        "trades",
        "decisions",
        "alerts",
        "watchlist",
        "status",
    }
)
HISTORY_PERIODS: Final = ("3mo", "6mo", "1y", "2y")
#: Bounds and default of the account-reset capital input (the API allows (0, MAX_CAPITAL]).
MIN_RESET_CAPITAL: Final = 1.0
DEFAULT_RESET_CAPITAL: Final = 100_000.0
PLOTLY_CONFIG: Final = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

# =============================================================================== caching


@st.cache_data(ttl=GET_CACHE_TTL_SECONDS, show_spinner=False, max_entries=512)
def _cached_get(
    cache_key: str,
    method: str,
    args: tuple[Any, ...],
    kwargs: tuple[tuple[str, Any], ...],
    _client: ApiClient,
) -> Any:
    """Run a read-only client method; results are cached per (API, credentials, call).

    ``_client`` is excluded from hashing (leading underscore); ``cache_key`` identifies
    the API URL and a fingerprint of the key instead. Exceptions are never cached, so a
    failed call is retried on the next rerun.
    """
    return getattr(_client, method)(*args, **dict(kwargs))


def api_get(client: ApiClient, method: str, *args: Any, **kwargs: Any) -> Any:
    """Cached call of a read-only :class:`ApiClient` method (``api_get(c, "quote", "AAPL")``)."""
    if method not in CACHEABLE_METHODS:
        raise ValueError(f"{method!r} is not a cacheable read")
    return _cached_get(client.cache_key, method, args, tuple(sorted(kwargs.items())), client)


def invalidate_cache() -> None:
    """Forget every cached response (after a mutation, or on "Refresh")."""
    _cached_get.clear()


# =============================================================================== session


@dataclass(frozen=True, slots=True)
class PageContext:
    """What every page needs: the client, the API's health payload and the chart theme."""

    client: ApiClient
    health: Mapping[str, Any]
    theme: charts.ChartTheme
    thresholds: fmt.Thresholds


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, "").strip() or default


#: Opt-in flag that lets visitors point this dashboard at another API URL.
URL_OVERRIDE_ENV: Final = "DASHBOARD_ALLOW_URL_OVERRIDE"
_TRUTHY: Final = frozenset({"1", "true", "yes", "on"})


def url_override_allowed() -> bool:
    """Whether the sidebar may change the API URL (``DASHBOARD_ALLOW_URL_OVERRIDE``).

    Off by default. The dashboard has no login of its own, and every request it makes
    comes from the dashboard *server*. Letting any visitor type a URL would let them make
    that server request arbitrary hosts (server-side request forgery).
    """
    return _env(URL_OVERRIDE_ENV).lower() in _TRUTHY


def _api_identity(url: str) -> tuple[str, str, str]:
    """Comparable form of an API base URL (scheme, host[:port], path; case-folded)."""
    parts = urlsplit(url.strip())
    return parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/")


def resolve_connection() -> tuple[str, str]:
    """The ``(url, key)`` this browser session talks to.

    * URL: ``API_URL``, or the sidebar value when :func:`url_override_allowed`.
    * Key: a key typed into the sidebar (sent only to the session's URL). Otherwise the
      server's ``API_KEY``, but **only** when the URL is ``API_URL``. Sending the
      server's secret to a URL a visitor typed would hand it to whoever runs that host.
    """
    env_url = _env("API_URL", DEFAULT_API_URL)
    typed_url = (st.session_state.get("conn_url") or "").strip() if url_override_allowed() else ""
    url = typed_url or env_url
    typed_key = (st.session_state.get("conn_key") or "").strip()
    if typed_key:
        return url, typed_key
    if _api_identity(url) == _api_identity(env_url):
        return url, _env("API_KEY")
    return url, ""


def get_client() -> ApiClient:
    """The session's API client, rebuilt when the connection settings change.

    See :func:`resolve_connection`. A key typed into the sidebar overrides the
    environment for this browser session only. The environment's key is never sent to
    the browser, nor to any API URL other than ``API_URL``.
    """
    url, key = resolve_connection()
    signature = (url, hash(key))
    client: ApiClient | None = st.session_state.get("_api_client")
    if client is not None and st.session_state.get("_api_client_sig") == signature:
        return client
    if client is not None:
        client.close()
    try:
        client = ApiClient.from_env(base_url=url, api_key=key or None)
    except ValueError as exc:
        st.session_state["_conn_error"] = str(exc)
        client = ApiClient.from_env(base_url=DEFAULT_API_URL, api_key=key or None)
    else:
        st.session_state.pop("_conn_error", None)
    st.session_state["_api_client"] = client
    st.session_state["_api_client_sig"] = signature
    return client


def flash(message: str, kind: str = "success") -> None:
    """Queue a message to show after the next rerun (mutations rerun the script)."""
    st.session_state.setdefault("_flash", []).append((kind, message))


def show_flashes() -> None:
    for kind, message in st.session_state.pop("_flash", []):
        {"success": st.success, "warning": st.warning, "info": st.info, "error": st.error}.get(
            kind, st.info
        )(message)


# =============================================================================== errors


def render_api_error(error: ApiError, action: str = "complete the request") -> None:
    """Show an :class:`ApiError` as a friendly, actionable message (never a traceback)."""
    rid = f" Request id: `{error.request_id}`." if error.request_id else ""
    message = fmt.escape_markdown(error.message)
    if isinstance(error, ApiTimeoutError):
        st.warning(
            f"The API took too long to {action}. It may still be working (an LLM call or a "
            f"long backtest); try again in a moment.",
            icon=":material/hourglass_top:",
        )
    elif isinstance(error, ApiConnectionError):
        st.error(
            f"Cannot reach the trading-swarm API to {action}: {message}.",
            icon=":material/cloud_off:",
        )
    elif isinstance(error, ApiAuthError):
        st.error(
            f"The API rejected the request (HTTP {error.status_code}): {message}. "
            "Set `API_KEY` for the dashboard, or enter a key under **Connection** in the "
            f"sidebar.{rid}",
            icon=":material/lock:",
        )
    elif isinstance(error, ApiRateLimitError):
        wait = (
            f" Try again in {error.retry_after:.0f} s."
            if error.retry_after is not None
            else " Try again shortly."
        )
        st.warning(f"Rate limit reached while trying to {action}.{wait}", icon=":material/speed:")
    elif isinstance(error, ApiNotFoundError):
        st.warning(f"Not found: {message}.{rid}", icon=":material/search_off:")
    elif isinstance(error, ApiValidationError):
        st.warning(f"Could not {action}: {message}.{rid}", icon=":material/error:")
    elif isinstance(error, ApiServerError):
        upstream = (
            " The market-data or LLM provider may be unavailable."
            if error.status_code in (502, 503, 504)
            else ""
        )
        st.error(
            f"The API failed to {action} (HTTP {error.status_code}): {message}.{upstream}{rid}",
            icon=":material/report:",
        )
    elif isinstance(error, ApiResponseError):
        st.error(
            f"The API sent an unexpected response while trying to {action}: {message}.{rid}",
            icon=":material/report:",
        )
    else:
        st.error(f"Could not {action}: {message}.{rid}", icon=":material/report:")


def safe_get(
    ctx: PageContext, method: str, *args: Any, action: str = "load data", **kwargs: Any
) -> Any | None:
    """:func:`api_get` that renders an error and returns ``None`` instead of raising."""
    try:
        return api_get(ctx.client, method, *args, **kwargs)
    except ApiError as exc:
        render_api_error(exc, action)
        return None


def show_chart(fig: go.Figure, *, key: str | None = None) -> None:
    """Render a figure with its own template (``theme=None``: Streamlit must not restyle)."""
    st.plotly_chart(fig, theme=None, width="stretch", config=PLOTLY_CONFIG, key=key)


def show_table(
    data: pd.DataFrame | Any,
    *,
    column_config: Mapping[str, Any] | None = None,
    height: int | str = "auto",
    key: str | None = None,
) -> None:
    st.dataframe(
        data,
        width="stretch",
        hide_index=True,
        placeholder=fmt.MISSING,
        column_config=dict(column_config or {}),
        height=height,
        key=key,
    )


def download_csv(frame: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    st.download_button(
        label,
        data=fmt.to_csv_bytes(frame),
        file_name=filename,
        mime="text/csv",
        icon=":material/download:",
        disabled=frame.empty,
        key=f"dl_{filename}",
    )


def _today_tag() -> str:
    return datetime.now(UTC).strftime("%Y%m%d")


TIME_COLUMN = st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")

# =============================================================================== sidebar

PAGES: Final[dict[str, str]] = {
    "portfolio": ":material/account_balance_wallet: Portfolio",
    "analyze": ":material/query_stats: Analyze",
    "backtest": ":material/history: Backtest",
    "trades": ":material/receipt_long: Trades & Orders",
    "alerts": ":material/notifications: Alerts",
    "watchlist": ":material/list_alt: Watchlist",
}
#: ``?page=`` aliases accepted from links (alert e-mails use analyze/trades/portfolio/alerts).
PAGE_ALIASES: Final = {"orders": "trades", "decisions": "trades", "scan": "watchlist"}


def apply_query_params() -> None:
    """Pre-select page and symbol from ``?page=…&symbol=…`` once per browser session."""
    if st.session_state.get("_query_applied"):
        return
    st.session_state["_query_applied"] = True
    page = str(st.query_params.get("page", "")).strip().lower()
    page = PAGE_ALIASES.get(page, page)
    if page in PAGES:
        st.session_state["page"] = page
    symbol = str(st.query_params.get("symbol", "")).strip()
    if symbol:
        try:
            st.session_state["analyze_symbol"] = normalize_symbol(symbol)
        except ApiValidationError:
            logger.info("ignoring invalid ?symbol=%r", symbol)


def render_connection_settings(container: Any) -> None:
    env_url = _env("API_URL", DEFAULT_API_URL)
    with container.expander("Connection", icon=":material/settings_ethernet:"):
        if url_override_allowed():
            st.text_input(
                "API URL",
                key="conn_url",
                placeholder=env_url,
                help="Defaults to the API_URL environment variable. The server's API_KEY "
                "is only sent to that URL: enter a key below for any other one.",
            )
        else:
            st.caption(
                f"API: `{env_url}` (set `{URL_OVERRIDE_ENV}=true` on the dashboard "
                "server to change it here)"
            )
        st.text_input(
            "API key",
            key="conn_key",
            type="password",
            placeholder="from API_KEY" if _env("API_KEY") else "not set",
            help="Sent as X-API-Key. Leave empty to use the API_KEY environment variable "
            "(only for the API_URL server).",
        )
        if error := st.session_state.get("_conn_error"):
            st.error(f"Invalid API URL: {fmt.escape_markdown(error)}")


def render_status(container: Any, client: ApiClient, health: Mapping[str, Any] | None) -> None:
    with container:
        if health is None:
            st.badge("API offline", icon=":material/cloud_off:", color="red")
            st.caption(client.base_url)
            return
        status = str(health.get("status", "ok"))
        healthy = status.lower() in ("ok", "healthy")
        st.badge(
            f"API {status} · v{health.get('version', '?')}",
            icon=":material/check_circle:" if healthy else ":material/warning:",
            color="green" if healthy else "orange",
        )
        llm = health.get("llm_provider") or "none"
        model = health.get("llm_model") or ""
        sentiment = f"{llm} ({model})" if model and llm != "none" else "lexicon (no LLM)"
        lines = [
            f"**Data:** {health.get('data_provider', '?')}",
            f"**Sentiment:** {sentiment}",
            f"**Auto-trade:** {'on' if health.get('auto_trade') else 'off'}",
            f"**Scheduler:** {'on' if health.get('scheduler_enabled') else 'off'}",
            f"**E-mail alerts:** {'on' if health.get('email_enabled') else 'off'}",
        ]
        st.caption("  \n".join(lines))


def render_sidebar() -> tuple[str, ApiClient, Mapping[str, Any] | None]:
    """Navigation, API status and connection settings; returns (page, client, health)."""
    client = get_client()  # first: widget values from the last interaction are in state
    sidebar = st.sidebar
    sidebar.title("MT Trading Swarm")
    page = sidebar.radio(
        "Navigate",
        list(PAGES),
        format_func=PAGES.__getitem__,
        key="page",
        label_visibility="collapsed",
    )
    sidebar.divider()
    status_box = sidebar.container()
    if sidebar.button("Refresh data", icon=":material/refresh:", width="stretch"):
        invalidate_cache()
        st.rerun()
    render_connection_settings(sidebar)
    sidebar.caption("Paper trading only — simulated orders, no real money. Not financial advice.")

    health: Mapping[str, Any] | None
    try:
        health = api_get(client, "health")
    except ApiConnectionError:
        health = None
    except ApiError as exc:  # reachable but unhealthy: keep going, pages report errors
        logger.warning("health check failed: %r", exc)
        health = {"status": f"error ({exc.status_code})"}
    render_status(status_box, client, health)
    return page or "portfolio", client, health


def render_offline(client: ApiClient) -> None:
    st.title("MT Trading Swarm")
    st.error(
        f"The trading-swarm API is not reachable at **{client.base_url}**.",
        icon=":material/cloud_off:",
    )
    where = (
        "`API_URL` (or change it under **Connection** in the sidebar)."
        if url_override_allowed()
        else "`API_URL` for the dashboard server."
    )
    st.markdown(
        "Start it from the repository root, then retry:\n\n"
        "```bash\nuvicorn trading_swarm.api.main:app --port 8000\n```\n"
        f"or with Docker: `docker compose up api`. If the API runs elsewhere, set {where}"
    )
    if st.button("Retry", icon=":material/refresh:", type="primary"):
        invalidate_cache()
        st.rerun()


# =============================================================================== portfolio


def _metric_span_days(metrics: Mapping[str, Any]) -> float:
    start, end = fmt.parse_timestamp(metrics.get("start")), fmt.parse_timestamp(metrics.get("end"))
    if start is None or end is None:
        return 0.0
    return (end - start).total_seconds() / 86_400.0


def page_portfolio(ctx: PageContext) -> None:
    st.title("Portfolio")
    portfolio = safe_get(ctx, "portfolio", action="load the portfolio")
    if portfolio is None:
        return
    st.caption(f"Marked to market at {fmt.fmt_timestamp(portfolio.get('updated_at'))}")

    # Tiles show whole dollars from $10,000 up so six-figure values fit four to a row;
    # the exact amount is in each tooltip.
    row1 = st.columns(4)
    row1[0].metric(
        "Equity",
        fmt.fmt_money_kpi(portfolio.get("equity")),
        delta=fmt.fmt_money_kpi(portfolio.get("total_return"), signed=True),
        border=True,
        help=md(
            f"{fmt.fmt_money(portfolio.get('equity'))}: cash + positions. Started with "
            f"{fmt.fmt_money(portfolio.get('initial_capital'))}."
        ),
    )
    equity = fmt.as_float(portfolio.get("equity")) or 0.0
    cash = fmt.as_float(portfolio.get("cash"))
    cash_help = fmt.fmt_money(cash)
    if cash is not None and equity > 0:
        cash_help += f" ({cash / equity:.0%} of equity)"
    row1[1].metric("Cash", fmt.fmt_money_kpi(cash), border=True, help=md(cash_help))
    row1[2].metric(
        "Total return",
        fmt.fmt_pct(portfolio.get("total_return_pct"), signed=True),
        delta=fmt.fmt_money(portfolio.get("total_return"), signed=True),
        border=True,
    )
    row1[3].metric(
        "Drawdown",
        fmt.fmt_pct(portfolio.get("drawdown_pct")),
        border=True,
        help=md(
            f"Below peak equity of {fmt.fmt_money(portfolio.get('peak_equity'))}. New BUYs "
            "halt when this reaches MAX_DRAWDOWN_HALT."
        ),
    )
    row2 = st.columns(4)
    for col, label, key in (
        (row2[0], "Realized P&L", "realized_pnl"),
        (row2[1], "Unrealized P&L", "unrealized_pnl"),
    ):
        col.metric(
            label,
            fmt.fmt_money_kpi(portfolio.get(key), signed=True),
            border=True,
            help=md(fmt.fmt_money(portfolio.get(key), signed=True)),
        )
    row2[2].metric(
        "Gross exposure",
        fmt.fmt_pct(portfolio.get("gross_exposure_pct")),
        border=True,
        help="Positions value / equity.",
    )
    positions = portfolio.get("positions") or []
    row2[3].metric("Open positions", f"{len(positions)}", border=True)

    left, right = st.columns([2, 1], gap="medium")
    with left:
        st.subheader("Equity curve")
        equity_payload = safe_get(ctx, "equity", action="load the equity curve")
        if equity_payload is not None:
            points = charts.equity_frame(equity_payload)
            show_chart(
                charts.equity_curve_figure(
                    points, initial_capital=portfolio.get("initial_capital"), theme=ctx.theme
                ),
                key="equity_curve",
            )
            render_account_metrics(equity_payload.get("metrics") or {})
    with right:
        st.subheader("Allocation")
        show_chart(
            charts.allocation_figure(positions, portfolio.get("cash"), theme=ctx.theme),
            key="allocation",
        )

    st.subheader("Positions")
    table = fmt.positions_table(portfolio)
    if table.empty:
        st.info(
            "No open positions. Analyze a symbol with auto-trade on, place a manual order, "
            "or run a scan.",
            icon=":material/inventory_2:",
        )
    else:
        styled = fmt.style_table(
            table,
            prices=("Avg cost", "Last", "Stop", "Take profit"),
            money=("Market value",),
            signed_money=("Unrealized P&L",),
            signed_pct=("P&L %", "To stop %", "To target %"),
            pnl=("Unrealized P&L", "P&L %"),
            theme=ctx.theme,
        )
        show_table(styled, column_config={"Opened (UTC)": TIME_COLUMN})
        download_csv(table, f"positions_{_today_tag()}.csv")

    render_reset(ctx, portfolio)


def render_account_metrics(metrics: Mapping[str, Any]) -> None:
    """Risk/return statistics of the paper account.

    CAGR and Sharpe annualise whatever span they are given, so a few minutes of
    snapshots would print absurd values (CAGR −100 %, Sharpe −11); they are shown only
    once the account has at least 30 days of history.
    """
    if not metrics:
        return
    long_enough = _metric_span_days(metrics) >= 30
    cols = st.columns(5)
    cols[0].metric(
        "CAGR",
        fmt.fmt_pct(metrics.get("cagr_pct")) if long_enough else fmt.MISSING,
        help=None if long_enough else "Shown once the account has 30+ days of history.",
    )
    cols[1].metric(
        "Sharpe",
        fmt.fmt_number(metrics.get("sharpe")) if long_enough else fmt.MISSING,
        help=None if long_enough else "Annualised ratios need 30+ days of snapshots.",
    )
    cols[2].metric("Max drawdown", fmt.fmt_pct(metrics.get("max_drawdown_pct")))
    cols[3].metric("Win rate", fmt.fmt_pct(metrics.get("win_rate_pct"), decimals=0))
    cols[4].metric("Closed trades", fmt.format_metric(metrics.get("num_trades"), "int"))


def render_reset(ctx: PageContext, portfolio: Mapping[str, Any]) -> None:
    with st.expander("Reset paper account", icon=":material/restart_alt:"):
        st.write(
            "Closes the books: deletes positions, orders, trades and equity snapshots and "
            "restarts with fresh cash. Decisions and the alert log are kept."
        )
        # The API accepts any capital in (0, MAX_CAPITAL]. The default comes from the
        # account and must sit inside the widget's bounds, or Streamlit raises.
        current = fmt.as_float(portfolio.get("initial_capital")) or DEFAULT_RESET_CAPITAL
        capital = st.number_input(
            "Starting capital ($)",
            min_value=MIN_RESET_CAPITAL,
            max_value=MAX_CAPITAL,
            value=min(max(float(current), MIN_RESET_CAPITAL), MAX_CAPITAL),
            step=10_000.0,
            format="%.2f",
        )
        confirmed = st.checkbox("I understand this cannot be undone", key="reset_confirm")
        if st.button(
            "Reset account", type="primary", disabled=not confirmed, icon=":material/delete:"
        ):
            try:
                with st.spinner("Resetting the paper account…"):
                    ctx.client.reset_portfolio(capital)
            except ApiError as exc:
                render_api_error(exc, "reset the account")
                return
            invalidate_cache()
            flash(md(f"Paper account reset with {fmt.fmt_money(capital)}."))
            st.rerun()


# =============================================================================== analyze


def _copy_watchlist_pick() -> None:
    picked = st.session_state.get("analyze_pick")
    if picked:
        st.session_state["analyze_symbol"] = picked
        st.session_state["analyze_pick"] = None


def page_analyze(ctx: PageContext) -> None:
    st.title("Analyze")
    st.caption(
        "Runs the swarm on one symbol: trend and sentiment agents vote, the consensus is "
        "gated by the risk manager, and approved trades are paper-traded when auto-trade is on."
    )
    watchlist = safe_get(ctx, "watchlist", action="load the watchlist") or []
    st.session_state.setdefault("analyze_symbol", watchlist[0] if watchlist else "AAPL")
    st.session_state.setdefault("analyze_auto_trade", bool(ctx.health.get("auto_trade", False)))

    with st.container(border=True):
        cols = st.columns([2, 2, 1.4, 1.2], vertical_alignment="bottom")
        cols[0].text_input("Symbol", key="analyze_symbol", max_chars=15)
        cols[1].selectbox(
            "…or pick from the watchlist",
            options=watchlist,
            index=None,
            key="analyze_pick",
            placeholder="Watchlist",
            on_change=_copy_watchlist_pick,
            disabled=not watchlist,
        )
        cols[2].toggle(
            "Auto-trade",
            key="analyze_auto_trade",
            help="Paper-trade the decision if the risk manager approves it.",
        )
        run = cols[3].button(
            "Analyze", type="primary", icon=":material/play_arrow:", width="stretch"
        )

    if run:
        try:
            symbol = normalize_symbol(st.session_state.get("analyze_symbol", ""))
            auto = bool(st.session_state.get("analyze_auto_trade"))
            with st.spinner(f"Running the swarm on {symbol}…"):
                decision = ctx.client.analyze(symbol, auto_trade=auto)
        except ApiError as exc:
            render_api_error(exc, "analyze the symbol")
        else:
            st.session_state["decision"] = decision
            invalidate_cache()  # decisions (and, with auto-trade, the portfolio) changed

    render_batch(ctx, watchlist)

    decision = st.session_state.get("decision")
    if decision:
        render_decision(ctx, decision)
    elif not run:
        st.info("Enter a ticker and press **Analyze**.", icon=":material/query_stats:")


def _show_batch_decision(results: Sequence[Mapping[str, Any]]) -> None:
    """Callback: open the picked batch result in the decision card, then clear the pick
    (so a later single-symbol analysis is not overwritten on the next rerun)."""
    picked = st.session_state.get("batch_pick")
    match = next((d for d in results if d.get("symbol") == picked), None)
    if match is not None:
        st.session_state["decision"] = match
    st.session_state["batch_pick"] = None


def render_batch(ctx: PageContext, watchlist: Sequence[str]) -> None:
    with st.expander("Analyze several symbols", icon=":material/stacks:"):
        symbols = st.multiselect(
            "Symbols",
            options=list(watchlist),
            default=list(watchlist)[:MAX_BATCH_SYMBOLS],
            max_selections=MAX_BATCH_SYMBOLS,
            accept_new_options=True,
            key="batch_symbols",
        )
        if st.button("Analyze all", icon=":material/play_arrow:", key="batch_run"):
            try:
                cleaned = normalize_symbols(symbols, max_count=MAX_BATCH_SYMBOLS)
                auto = bool(st.session_state.get("analyze_auto_trade"))
                with st.spinner(f"Analyzing {len(cleaned)} symbols…"):
                    st.session_state["batch"] = ctx.client.analyze_batch(cleaned, auto_trade=auto)
            except ApiError as exc:
                render_api_error(exc, "analyze the symbols")
            else:
                invalidate_cache()
        batch = st.session_state.get("batch")
        if not batch:
            return
        results = batch.get("results") or []
        if results:
            table = fmt.decisions_table(results)
            show_table(
                fmt.style_table(
                    table,
                    prices=("Price",),
                    signals=("Score", "Trend", "Sentiment"),
                    directional=("Score", "Trend", "Sentiment"),
                    theme=ctx.theme,
                ),
                column_config={
                    "Time (UTC)": TIME_COLUMN,
                    "Confidence": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="percent"
                    ),
                },
            )
            st.selectbox(
                "Show the full decision for",
                options=[d.get("symbol") for d in results],
                index=None,
                placeholder="Select a symbol",
                key="batch_pick",
                on_change=_show_batch_decision,
                args=(results,),
            )
        if batch.get("errors"):
            st.warning(f"{len(batch['errors'])} symbol(s) failed:")
            show_table(fmt.errors_table(batch["errors"]))


def render_decision(ctx: PageContext, decision: Mapping[str, Any]) -> None:
    """The decision card: verdict, consensus, agent votes, risk, rationale and chart."""
    symbol = str(decision.get("symbol") or "")
    action = str(decision.get("action") or "HOLD").upper()
    risk = decision.get("risk") or {}
    with st.container(border=True):
        head = st.columns([3, 2], vertical_alignment="center")
        with head[0]:
            st.markdown(f"## {symbol} · {fmt.fmt_price(decision.get('price'))}")
            badges = st.container(horizontal=True)
            with badges:
                st.badge(action, color=fmt.action_badge_color(action))
                if decision.get("executed"):
                    st.badge("Executed", icon=":material/check:", color="green")
                elif risk.get("approved") and action != "HOLD":
                    st.badge("Approved, not executed", color="gray")
                proposed = fmt.proposed_action(decision.get("score"), ctx.thresholds)
                closed = fmt.own_exit(decision)
                if action == "HOLD" and proposed == "SELL" and closed is not None:
                    # The analysis' own stop / target check already sold the position.
                    st.badge(
                        f"Closed by {fmt.exit_label(closed)}",
                        icon=":material/logout:",
                        color="orange",
                    )
                elif action == "HOLD" and proposed != "HOLD":
                    st.badge(f"{proposed} vetoed by risk", icon=":material/block:", color="orange")
        head[1].caption(f"Decided {fmt.fmt_timestamp(decision.get('timestamp'), seconds=True)}")

        render_execution(decision)

        cols = st.columns([1, 1.5], gap="medium")
        with cols[0]:
            st.markdown("**Consensus**")
            show_chart(
                charts.consensus_gauge(
                    fmt.as_float(decision.get("score")),
                    buy_threshold=ctx.thresholds.buy,
                    sell_threshold=ctx.thresholds.sell,
                    theme=ctx.theme,
                ),
                key=f"gauge_{decision.get('id')}",
            )
            st.caption(
                f"BUY at ≥ {ctx.thresholds.buy:+.2f}, SELL at ≤ {ctx.thresholds.sell:+.2f}. "
                f"Confidence {fmt.fmt_fraction_pct(decision.get('confidence'))}."
            )
        with cols[1]:
            st.markdown("**Agent votes**")
            show_chart(
                charts.agent_signals_figure(
                    decision.get("signals") or [],
                    consensus=fmt.as_float(decision.get("score")),
                    consensus_confidence=fmt.as_float(decision.get("confidence")),
                    buy_threshold=ctx.thresholds.buy,
                    sell_threshold=ctx.thresholds.sell,
                    theme=ctx.theme,
                ),
                key=f"signals_{decision.get('id')}",
            )
            st.caption("Bar opacity = the agent's own confidence; votes count weight × confidence.")

        st.markdown("**Rationale**")
        st.markdown(fmt.escape_markdown(decision.get("summary") or "No summary."))

    render_risk_panel(risk, action)
    render_agent_details(ctx, decision)
    render_price_chart(ctx, symbol, risk)


def render_exits(decision: Mapping[str, Any]) -> None:
    """Protective exits the analysis made before sizing (it sells even without auto-trade)."""
    for order in fmt.decision_exits(decision):
        label = fmt.exit_label(order).capitalize()
        status = str(order.get("status") or "").upper()
        if status == "FILLED":
            st.info(
                md(
                    f"{label} hit before this decision: sold {order.get('quantity')} "
                    f"{order.get('symbol')} @ {fmt.fmt_price(order.get('fill_price'))} "
                    f"(realized P&L {fmt.fmt_money(order.get('realized_pnl'), signed=True)})."
                ),
                icon=":material/logout:",
            )
        else:
            why = order.get("reject_reason") or "see the orders table"
            st.warning(
                md(f"{label} exit of {order.get('symbol')} {status.lower()}: {why}."),
                icon=":material/block:",
            )


def render_execution(decision: Mapping[str, Any]) -> None:
    render_exits(decision)
    order = decision.get("order")
    risk = decision.get("risk") or {}
    if isinstance(order, Mapping):
        status = str(order.get("status") or "").upper()
        if status == "FILLED":
            st.success(
                md(
                    f"Paper order filled: {order.get('side')} {order.get('quantity')} "
                    f"{order.get('symbol')} @ {fmt.fmt_price(order.get('fill_price'))} "
                    f"(commission {fmt.fmt_money(order.get('commission'))}, slippage "
                    f"{fmt.fmt_money(order.get('slippage'))})."
                ),
                icon=":material/check_circle:",
            )
        else:
            why = order.get("reject_reason") or order.get("reason") or "no reason given"
            st.warning(
                md(f"Order {status.lower() or 'not filled'}: {why}."), icon=":material/block:"
            )
    elif risk.get("approved") and str(decision.get("action")).upper() in ("BUY", "SELL"):
        st.info(
            "Approved by the risk manager but not executed (auto-trade was off). Turn on "
            "auto-trade and analyze again to paper-trade it.",
            icon=":material/info:",
        )


def render_risk_panel(risk: Mapping[str, Any], action: str) -> None:
    with st.container(border=True):
        title, badge = st.columns([3, 1], vertical_alignment="center")
        title.markdown("### Risk manager")
        with badge:
            if risk.get("approved"):
                st.badge("Approved", icon=":material/verified:", color="green")
            else:
                st.badge("Not approved", icon=":material/block:", color="red")
        entry = risk.get("entry_price")
        stop, target = risk.get("stop_price"), risk.get("take_profit_price")
        cols = st.columns(4)
        cols[0].metric("Quantity", f"{int(risk.get('quantity') or 0):,} sh")
        cols[1].metric("Entry", fmt.fmt_price(entry))
        stop_dist = fmt.pct_distance(entry, stop)
        cols[2].metric(
            "Stop",
            fmt.fmt_price(stop),
            delta=fmt.fmt_pct(stop_dist, signed=True) if stop_dist is not None else None,
            delta_color="off",
        )
        target_dist = fmt.pct_distance(entry, target)
        cols[3].metric(
            "Take profit",
            fmt.fmt_price(target),
            delta=fmt.fmt_pct(target_dist, signed=True) if target_dist is not None else None,
            delta_color="off",
        )
        cols = st.columns(4)
        cols[0].metric("Position value", fmt.fmt_money(risk.get("position_value")))
        cols[1].metric(
            "Capital at risk",
            fmt.fmt_money(risk.get("risk_amount")),
            help="Shares × (entry − stop): the loss if the stop is hit.",
        )
        rr = fmt.reward_risk(entry, stop, target)
        cols[2].metric("Reward / risk", f"{rr:.1f}R" if rr is not None else fmt.MISSING)
        score = fmt.as_float(risk.get("risk_score")) or 0.0
        with cols[3]:
            st.metric(
                "Risk score",
                f"{score:.0f} / 100",
                help="35% volatility, 25% exposure, 25% drawdown, 15% concentration — each "
                "as a share of its limit.",
            )
            st.progress(min(max(score / 100.0, 0.0), 1.0))

        checks = fmt.risk_checks(risk.get("checks"))
        if checks:
            st.markdown("**Checks**")
            row = st.container(horizontal=True)
            with row:
                for label, passed in checks:
                    st.badge(
                        label,
                        icon=":material/check:" if passed else ":material/close:",
                        color="green" if passed else "red",
                    )
        reasons = risk.get("reasons") or []
        if reasons:
            st.markdown("**Reasoning**")
            st.markdown("\n".join(f"- {fmt.escape_markdown(reason)}" for reason in reasons))
        elif action == "HOLD":
            st.caption("No trade proposed: the consensus is inside the HOLD band.")


def render_agent_details(ctx: PageContext, decision: Mapping[str, Any]) -> None:
    trend = fmt.signal_for(decision, "trend")
    sentiment = fmt.signal_for(decision, "sentiment")
    if trend:
        with st.expander(
            f"Trend agent · {fmt.fmt_signal(trend.get('signal'))} "
            f"(conf {fmt.fmt_fraction_pct(trend.get('confidence'))})",
            icon=":material/trending_up:",
        ):
            st.markdown(fmt.escape_markdown(trend.get("rationale")))
            left, right = st.columns(2)
            with left:
                rows = fmt.indicator_rows(trend)
                if rows:
                    show_table(pd.DataFrame(rows, columns=["Indicator", "Value"]))
            with right:
                components = fmt.trend_components_table(trend)
                if not components.empty:
                    show_table(
                        fmt.style_table(
                            components, signals=("Score", "Contribution"), theme=ctx.theme
                        ),
                        column_config={"Weight": st.column_config.NumberColumn(format="%.2f")},
                    )
    if sentiment:
        data = sentiment.get("data") or {}
        method = data.get("method") or "?"
        with st.expander(
            f"Sentiment agent · {fmt.fmt_signal(sentiment.get('signal'))} "
            f"(conf {fmt.fmt_fraction_pct(sentiment.get('confidence'))}, {method})",
            icon=":material/newspaper:",
        ):
            st.markdown(fmt.escape_markdown(sentiment.get("rationale")))
            if data.get("llm_fallback_from"):
                st.caption(f"LLM ({data['llm_fallback_from']}) unavailable; used the lexicon.")
            headlines = fmt.headlines_table(sentiment)
            if headlines.empty:
                st.caption("No headlines were available.")
            else:
                show_table(
                    fmt.style_table(headlines, signals=("Score",), theme=ctx.theme),
                    column_config={
                        "Published (UTC)": TIME_COLUMN,
                        "Relevance": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="percent"
                        ),
                    },
                )
                st.caption("Headlines are third-party text, scored as data only.")


def ctx_theme() -> charts.ChartTheme:
    """The chart theme matching Streamlit's current appearance (light when unknown)."""
    return charts.get_theme(st.context.theme.type)


def render_price_chart(ctx: PageContext, symbol: str, risk: Mapping[str, Any]) -> None:
    st.subheader(f"{symbol} price & indicators")
    period = st.segmented_control(
        "Period",
        HISTORY_PERIODS,
        default="6mo",
        required=True,
        key="history_period",
        label_visibility="collapsed",
    )
    history = safe_get(
        ctx, "history", symbol, period=period or "6mo", action=f"load {symbol} history"
    )
    if history is None:
        return
    frame = charts.candles_frame(history)
    approved = bool(risk.get("approved"))
    show_chart(
        charts.price_figure(
            frame,
            symbol=symbol,
            stop_price=risk.get("stop_price") if approved else None,
            take_profit_price=risk.get("take_profit_price") if approved else None,
            entry_price=risk.get("entry_price") if approved else None,
            theme=ctx.theme,
        ),
        key=f"price_{symbol}_{period}",
    )
    with st.expander("Price data", icon=":material/table:"):
        table = frame.reset_index().rename(columns={"t": "Date"})
        table["Date"] = table["Date"].dt.tz_localize(None)
        show_table(table.iloc[::-1], column_config={"Date": st.column_config.DateColumn()})
        download_csv(table, f"{symbol}_{period}_history.csv")


# =============================================================================== backtest


def page_backtest(ctx: PageContext) -> None:
    st.title("Backtest")
    st.caption(
        "Replays the trend + risk logic bar by bar (no look-ahead): signals on the close, "
        "fills at the next open with slippage and commission, stops and targets checked "
        "against the next bar's range. Compared with buying and holding the same symbols."
    )
    watchlist = safe_get(ctx, "watchlist", action="load the watchlist") or []
    today = datetime.now(UTC).date()
    with st.form("backtest_form", border=True):
        symbols = st.multiselect(
            "Symbols",
            options=list(watchlist),
            default=list(watchlist)[:1],
            max_selections=MAX_BACKTEST_SYMBOLS,
            accept_new_options=True,
            help=f"1-{MAX_BACKTEST_SYMBOLS} tickers; type to add one that is not on the watchlist.",
        )
        cols = st.columns(3)
        start = cols[0].date_input("Start", value=today - timedelta(days=730), max_value=today)
        end = cols[1].date_input("End", value=today, max_value=today)
        capital = cols[2].number_input(
            "Initial capital ($)", min_value=1_000.0, max_value=1e9, value=100_000.0, step=10_000.0
        )
        use_sentiment = st.checkbox(
            "Use sentiment",
            value=False,
            help="Off by default: headlines are not point-in-time, so sentiment would leak "
            "today's news into the past. Without it the consensus uses the trend vote alone.",
        )
        submitted = st.form_submit_button(
            "Run backtest", type="primary", icon=":material/play_arrow:"
        )

    if submitted:
        if not isinstance(start, date) or not isinstance(end, date):
            st.warning("Pick a start and an end date.")
        else:
            try:
                with st.spinner("Running the backtest… (a few seconds per symbol-year)"):
                    st.session_state["backtest"] = ctx.client.backtest(
                        symbols,
                        start,
                        end,
                        initial_capital=capital,
                        use_sentiment=use_sentiment,
                    )
            except ApiError as exc:
                render_api_error(exc, "run the backtest")

    result = st.session_state.get("backtest")
    if result:
        render_backtest(ctx, result)


def render_backtest(ctx: PageContext, result: Mapping[str, Any]) -> None:
    symbols = ", ".join(result.get("symbols") or [])
    st.subheader(f"Results · {symbols}")
    st.caption(
        f"{result.get('start', '?')} → {result.get('end', '?')} · initial capital "
        f"{md(fmt.fmt_money(result.get('initial_capital')))}"
    )
    metrics = result.get("metrics") or {}
    benchmark = result.get("benchmark_metrics") or result.get("benchmark") or {}
    if not isinstance(benchmark, Mapping):
        benchmark = {}

    strat_ret, bench_ret = (
        fmt.as_float(metrics.get("total_return_pct")),
        fmt.as_float(benchmark.get("total_return_pct")),
    )
    cols = st.columns(4)
    cols[0].metric(
        "Strategy return",
        fmt.fmt_pct(strat_ret, signed=True, decimals=1),
        delta=(
            f"{strat_ret - bench_ret:+.1f} pp vs B&H"
            if strat_ret is not None and bench_ret is not None
            else None
        ),
        border=True,
        help="Total return of the strategy; the delta is the difference to buy & hold in "
        "percentage points.",
    )
    cols[1].metric(
        "Buy & hold return", fmt.fmt_pct(bench_ret, signed=True, decimals=1), border=True
    )
    bench_sharpe = fmt.as_float(benchmark.get("sharpe"))
    cols[2].metric(
        "Sharpe",
        fmt.fmt_number(metrics.get("sharpe")),
        delta=f"B&H {fmt.fmt_number(bench_sharpe)}" if bench_sharpe is not None else None,
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    bench_dd = fmt.as_float(benchmark.get("max_drawdown_pct"))
    cols[3].metric(
        "Max drawdown",
        fmt.fmt_pct(metrics.get("max_drawdown_pct"), decimals=1),
        delta=f"B&H {fmt.fmt_pct(bench_dd, decimals=1)}" if bench_dd is not None else None,
        delta_color="off",
        delta_arrow="off",
        border=True,
    )

    curve = charts.backtest_curve_frame(result)
    left, right = st.columns([3, 2], gap="medium")
    with left:
        st.markdown("**Equity vs buy & hold**")
        show_chart(
            charts.backtest_equity_figure(
                curve, initial_capital=result.get("initial_capital"), theme=ctx.theme
            ),
            key="bt_equity",
        )
        st.markdown("**Drawdown**")
        show_chart(charts.drawdown_figure(curve, theme=ctx.theme), key="bt_drawdown")
    with right:
        st.markdown("**Metrics**")
        show_table(fmt.metrics_table(metrics, benchmark), height="content")

    trades = result.get("trades") or []
    st.markdown(f"**Trades** ({len(trades)})")
    trades_df = fmt.trades_table(trades)
    if trades_df.empty:
        st.caption("The strategy made no trades in this period.")
    else:
        show_table(
            fmt.style_table(
                trades_df,
                prices=("Price",),
                money=("Notional", "Commission"),
                signed_money=("Realized P&L",),
                pnl=("Realized P&L",),
                theme=ctx.theme,
            ),
            column_config={"Time (UTC)": TIME_COLUMN},
        )
    dl = st.container(horizontal=True)
    with dl:
        tag = "_".join(result.get("symbols") or ["backtest"])
        download_csv(trades_df, f"backtest_{tag}_trades.csv", "Trades CSV")
        curve_csv = curve.reset_index()
        if "t" in curve_csv.columns:
            curve_csv["t"] = curve_csv["t"].dt.tz_localize(None)
        download_csv(curve_csv, f"backtest_{tag}_equity.csv", "Equity curve CSV")
        st.download_button(
            "Full result (JSON)",
            data=json.dumps(result, indent=2, default=str).encode("utf-8"),
            file_name=f"backtest_{tag}.json",
            mime="application/json",
            icon=":material/data_object:",
            key="dl_backtest_json",
        )
    if result.get("parameters"):
        with st.expander("Parameters used", icon=":material/tune:"):
            st.json(result["parameters"])


# =============================================================================== trades


def page_trades(ctx: PageContext) -> None:
    st.title("Trades & Orders")
    cols = st.columns([2, 1, 3], vertical_alignment="bottom")
    symbol_filter = cols[0].text_input("Filter by symbol", key="trades_filter").strip().upper()
    limit = cols[1].selectbox("Rows", (50, 100, 250, 500), index=1, key="trades_limit")

    render_manual_order(ctx)

    tab_trades, tab_orders, tab_decisions = st.tabs(["Trades", "Orders", "Decisions"])

    def matches(row: Mapping[str, Any]) -> bool:
        return not symbol_filter or str(row.get("symbol", "")).upper() == symbol_filter

    with tab_trades:
        trades = safe_get(ctx, "trades", limit=limit, action="load trades")
        if trades is not None:
            table = fmt.trades_table([t for t in trades if matches(t)])
            if table.empty:
                st.info("No trades yet.", icon=":material/receipt_long:")
            else:
                realized = table["Realized P&L"].dropna()
                st.caption(
                    f"{len(table)} fill{'s' if len(table) != 1 else ''} · realized P&L on "
                    "closed trades "
                    f"{md(fmt.fmt_money(realized.sum(), signed=True))} "
                    f"({int((realized > 0).sum())} wins / {int((realized < 0).sum())} losses)"
                )
                show_table(
                    fmt.style_table(
                        table,
                        prices=("Price",),
                        money=("Notional", "Commission"),
                        signed_money=("Realized P&L",),
                        pnl=("Realized P&L",),
                        theme=ctx.theme,
                    ),
                    column_config={"Time (UTC)": TIME_COLUMN},
                )
            download_csv(table, f"trades_{_today_tag()}.csv")
    with tab_orders:
        orders = safe_get(ctx, "orders", limit=limit, action="load orders")
        if orders is not None:
            table = fmt.orders_table([o for o in orders if matches(o)])
            if table.empty:
                st.info("No orders yet.", icon=":material/receipt_long:")
            else:
                show_table(
                    fmt.style_table(
                        table,
                        prices=("Fill price", "Reference", "Stop", "Take profit"),
                        money=("Commission", "Slippage"),
                        signed_money=("Realized P&L",),
                        pnl=("Realized P&L",),
                        theme=ctx.theme,
                    ),
                    column_config={"Requested (UTC)": TIME_COLUMN},
                )
            download_csv(table, f"orders_{_today_tag()}.csv")
    with tab_decisions:
        decisions = safe_get(
            ctx, "decisions", limit=limit, symbol=symbol_filter or None, action="load decisions"
        )
        if decisions is not None:
            table = fmt.decisions_table(decisions)
            if table.empty:
                st.info("No decisions recorded yet.", icon=":material/query_stats:")
            else:
                show_table(
                    fmt.style_table(
                        table,
                        prices=("Price",),
                        signals=("Score", "Trend", "Sentiment"),
                        directional=("Score", "Trend", "Sentiment"),
                        theme=ctx.theme,
                    ),
                    column_config={
                        "Time (UTC)": TIME_COLUMN,
                        "Confidence": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="percent"
                        ),
                        "Summary": st.column_config.TextColumn(width="large"),
                    },
                )
            download_csv(table, f"decisions_{_today_tag()}.csv")


def render_manual_order(ctx: PageContext) -> None:
    with st.expander("Place a manual paper order", icon=":material/add_shopping_cart:"):
        st.caption(
            "Market order at the API's latest quote (the price is never taken from the "
            "browser). Stop / take-profit apply to BUY orders only; leave them at 0 for "
            "none. Buying more of a held symbol never lowers its stop."
        )
        with st.form("manual_order", border=False):
            cols = st.columns([1.5, 1, 1, 1, 1])
            symbol = cols[0].text_input("Symbol", max_chars=15)
            side = cols[1].selectbox("Side", ("BUY", "SELL"))
            quantity = cols[2].number_input("Quantity", min_value=1, value=10, step=1)
            stop = cols[3].number_input("Stop", min_value=0.0, value=0.0, step=1.0)
            target = cols[4].number_input("Take profit", min_value=0.0, value=0.0, step=1.0)
            submitted = st.form_submit_button("Submit order", type="primary")
        if not submitted:
            return
        try:
            order = ctx.client.place_order(
                symbol,
                side,
                int(quantity),
                stop_price=stop or None,
                take_profit_price=target or None,
            )
        except ApiError as exc:
            render_api_error(exc, "place the order")
            return
        invalidate_cache()
        status = str(order.get("status", "")).upper()
        if status == "FILLED":
            st.success(
                md(
                    f"Filled: {order.get('side')} {order.get('quantity')} {order.get('symbol')} "
                    f"@ {fmt.fmt_price(order.get('fill_price'))} (commission "
                    f"{fmt.fmt_money(order.get('commission'))})."
                )
            )
        else:
            reason = fmt.escape_markdown(order.get("reject_reason") or "see the orders table")
            st.warning(f"Order {status.lower()}: {reason}.")


# =============================================================================== alerts


def alert_configuration(ctx: PageContext) -> Mapping[str, Any] | None:
    """The API's alerting configuration from ``GET /status`` (``None`` if unavailable).

    Optional detail: a failure (an older API without ``/status``, a missing key) only
    hides the "Configuration details" popover, it never shows an error banner.
    """
    try:
        status_payload = api_get(ctx.client, "status")
    except ApiError as exc:
        logger.info("alert configuration unavailable: %s", exc)
        return None
    summary = status_payload.get("alerts") if isinstance(status_payload, Mapping) else None
    return summary if isinstance(summary, Mapping) and summary else None


def page_alerts(ctx: PageContext) -> None:
    st.title("Alerts")
    enabled = bool(ctx.health.get("email_enabled"))
    with st.container(border=True):
        cols = st.columns([3, 1], vertical_alignment="center")
        with cols[0]:
            st.markdown("**E-mail alerts**")
            if enabled:
                st.badge("Configured", icon=":material/mark_email_read:", color="green")
                st.caption(
                    "Signals at or above ALERT_MIN_CONFIDENCE, executed orders, stop-loss / "
                    "take-profit exits, drawdown breaches and large daily moves are e-mailed "
                    "(with a per-rule cooldown)."
                )
            else:
                st.badge("Disabled", icon=":material/unsubscribe:", color="gray")
                st.caption(
                    "Set SMTP_HOST, ALERT_EMAIL_FROM and ALERT_EMAIL_TO for the API (Gmail: "
                    "smtp.gmail.com:587 with STARTTLS and an app password). Alerts are still "
                    "logged below."
                )
            summary = alert_configuration(ctx)
            if summary:
                with st.popover("Configuration details", icon=":material/info:"):
                    st.json(summary)
        with cols[1]:
            send = st.button("Send test e-mail", icon=":material/send:", width="stretch")
        if send:
            try:
                with st.spinner("Sending a test e-mail…"):
                    outcome = ctx.client.send_test_alert()
            except ApiError as exc:
                render_api_error(exc, "send the test e-mail")
            else:
                invalidate_cache()
                detail = fmt.escape_markdown(outcome.get("detail"))
                if outcome.get("sent"):
                    st.success(f"Test e-mail sent. {detail}", icon=":material/mark_email_read:")
                else:
                    st.warning(f"Test e-mail not sent. {detail}", icon=":material/mail_lock:")

    st.subheader("Alert log")
    alerts = safe_get(ctx, "alerts", limit=200, action="load the alert log")
    if alerts is None:
        return
    table = fmt.alerts_table(alerts)
    if table.empty:
        st.info("No alerts have been raised yet.", icon=":material/notifications_off:")
        return
    counts = table["Status"].value_counts()
    cols = st.columns(4)
    for col, status in zip(cols, ("sent", "failed", "suppressed", "disabled"), strict=True):
        col.metric(status.capitalize(), int(counts.get(status, 0)))
    statuses = st.multiselect(
        "Status", options=sorted(table["Status"].dropna().unique()), key="alert_status_filter"
    )
    if statuses:
        table = table[table["Status"].isin(statuses)]
    show_table(
        table,
        column_config={
            "Time (UTC)": TIME_COLUMN,
            "Detail": st.column_config.TextColumn(width="large"),
        },
    )
    download_csv(table, f"alerts_{_today_tag()}.csv")


# =============================================================================== watchlist


def page_watchlist(ctx: PageContext) -> None:
    st.title("Watchlist")
    watchlist = safe_get(ctx, "watchlist", action="load the watchlist")
    if watchlist is None:
        return
    with st.container(border=True):
        st.markdown("**Symbols scanned by the swarm**")
        edited = st.multiselect(
            "Watchlist",
            options=list(watchlist),
            default=list(watchlist),
            accept_new_options=True,
            label_visibility="collapsed",
            placeholder="Type a ticker and press Enter",
        )
        cols = st.columns([1, 1, 4])
        save = cols[0].button(
            "Save",
            icon=":material/save:",
            type="primary",
            disabled=[s.upper() for s in edited] == list(watchlist),
        )
        scan = cols[1].button("Scan now", icon=":material/radar:")
        st.caption(
            "A scan analyses every symbol, closes positions whose stop or target was hit and "
            "records an equity snapshot. Trades only execute when AUTO_TRADE is on for the API."
        )
    if save:
        try:
            stored = ctx.client.set_watchlist(edited)
        except ApiError as exc:
            render_api_error(exc, "save the watchlist")
        else:
            invalidate_cache()
            flash(f"Watchlist saved: {', '.join(stored)}.")
            st.rerun()
    if scan:
        try:
            with st.spinner(f"Scanning {len(watchlist)} symbols…"):
                st.session_state["scan"] = ctx.client.scan()
        except ApiError as exc:
            render_api_error(exc, "scan the watchlist")
        else:
            invalidate_cache()

    # The scan the user just asked for comes first: the per-symbol quotes below are many
    # requests against the shared rate limit and must not be able to hide it.
    report = st.session_state.get("scan")
    if report:
        render_scan(ctx, report)
    render_quotes(ctx, watchlist)


#: Errors that will hit every remaining quote too: stop the loop and say so once.
_QUOTE_LOOP_STOPPERS: Final = (ApiRateLimitError, ApiConnectionError, ApiAuthError)


def render_quotes(ctx: PageContext, symbols: Sequence[str]) -> None:
    """One cached ``GET /quote`` per watchlist symbol; failures never break the page.

    A per-symbol failure (unknown ticker, provider error) is listed under the table. A
    rate limit, connection or auth failure would repeat for every other symbol. In that
    case the loop stops, one notice is shown and the quotes already loaded are kept.
    """
    if not symbols:
        st.info("The watchlist is empty: add a ticker above.", icon=":material/list_alt:")
        return
    st.subheader("Quotes")
    rows, failed = [], []
    for k, symbol in enumerate(symbols):
        try:
            q = api_get(ctx.client, "quote", symbol)
        except _QUOTE_LOOP_STOPPERS as exc:
            render_api_error(exc, "load the watchlist quotes")
            skipped = len(symbols) - k
            st.caption(f"{skipped} quote(s) not loaded; use **Refresh data** to retry.")
            break
        except ApiError as exc:
            failed.append(f"{symbol}: {fmt.escape_markdown(exc.message)}")
            continue
        rows.append(
            {
                "Symbol": q.get("symbol", symbol),
                "Price": fmt.as_float(q.get("price")),
                "1-day change %": fmt.as_float(q.get("change_pct_1d")),
                "As of (UTC)": q.get("as_of"),
            }
        )
    if rows:
        table = pd.DataFrame(rows)
        table["As of (UTC)"] = fmt.timestamp_column(table["As of (UTC)"])
        show_table(
            fmt.style_table(
                table,
                prices=("Price",),
                signed_pct=("1-day change %",),
                directional=("1-day change %",),
                theme=ctx.theme,
            ),
            column_config={"As of (UTC)": TIME_COLUMN},
        )
    if failed:
        st.caption("No quote for " + "; ".join(failed))


def render_scan(ctx: PageContext, report: Mapping[str, Any]) -> None:
    st.subheader("Last scan")
    decisions = report.get("decisions") or []
    exits = report.get("exits") or []
    errors = report.get("errors") or []
    executed = sum(1 for d in decisions if d.get("executed"))
    cols = st.columns(4)
    cols[0].metric("Analysed", len(decisions))
    cols[1].metric("Trades executed", executed)
    cols[2].metric("Protective exits", len(exits))
    cols[3].metric("Errors", len(errors))
    if report.get("snapshot_recorded") is False:
        st.caption(
            "No equity point was recorded for this scan: an open position could not be "
            "priced (see the errors below), and a point valued at cost would distort the "
            "equity curve."
        )
    if decisions:
        show_table(
            fmt.style_table(
                fmt.decisions_table(decisions),
                prices=("Price",),
                signals=("Score", "Trend", "Sentiment"),
                directional=("Score", "Trend", "Sentiment"),
                theme=ctx.theme,
            ),
            column_config={
                "Time (UTC)": TIME_COLUMN,
                "Confidence": st.column_config.ProgressColumn(
                    min_value=0.0, max_value=1.0, format="percent"
                ),
            },
        )
    if exits:
        st.markdown("**Exits (stop-loss / take-profit)**")
        show_table(
            fmt.style_table(
                fmt.orders_table(exits),
                prices=("Fill price", "Reference", "Stop", "Take profit"),
                money=("Commission", "Slippage"),
                signed_money=("Realized P&L",),
                pnl=("Realized P&L",),
                theme=ctx.theme,
            ),
            column_config={"Requested (UTC)": TIME_COLUMN},
        )
    if errors:
        st.markdown("**Errors**")
        show_table(fmt.errors_table(errors))


# =============================================================================== main

PAGE_RENDERERS: Final[dict[str, Callable[[PageContext], None]]] = {
    "portfolio": page_portfolio,
    "analyze": page_analyze,
    "backtest": page_backtest,
    "trades": page_trades,
    "alerts": page_alerts,
    "watchlist": page_watchlist,
}


def main() -> None:
    st.set_page_config(
        page_title="MT Trading Swarm",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_query_params()
    page, client, health = render_sidebar()
    if health is None:
        render_offline(client)
        return
    st.query_params["page"] = page
    ctx = PageContext(
        client=client,
        health=health,
        theme=ctx_theme(),
        thresholds=fmt.resolve_thresholds(health),
    )
    show_flashes()
    try:
        PAGE_RENDERERS.get(page, page_portfolio)(ctx)
    except ApiError as exc:  # a page let one through: still no traceback
        render_api_error(exc)
    except Exception:
        logger.exception("dashboard page %r failed", page)
        st.error(
            "Something went wrong while rendering this page. The details were written to "
            "the dashboard log; try **Refresh data** in the sidebar.",
            icon=":material/report:",
        )


# ``streamlit run`` (and AppTest) execute this file as ``__main__``; a plain import (docs,
# linters, ``python -c "import dashboard.app"``) has no side effects.
if __name__ == "__main__":
    main()
