"""Plotly figure builders for the dashboard.

Every function here is pure: it takes API payloads (or DataFrames shaped from them) and
returns a :class:`plotly.graph_objects.Figure`. Nothing imports Streamlit, so the charts
are unit-testable and reusable in notebooks.

Theming
-------
Figures carry their **own** template (:func:`plotly_template`), so they render
identically with ``st.plotly_chart(fig, theme=None)`` — Streamlit's theme cannot recolour
the candles or the signal bars. Two variants exist, :data:`LIGHT` and :data:`DARK`, each
with its own selected colour steps (not an automatic inversion); the app picks one from
``st.context.theme.type``. Backgrounds are opaque, so even if theme detection is briefly
wrong (Streamlit reports the theme late on first load) a chart is still legible — it just
shows as a light card on a dark page or vice versa.

Colour semantics (validated for colour-vision deficiency, see the palette notes below)
--------------------------------------------------------------------------------------
* **Direction / polarity** — bullish, rising candles, positive signals: blue;
  bearish, falling candles, negative signals: red; neutral (HOLD): grey. Blue/red is used
  instead of the green/red trading convention because green↔red collapses for
  deuteranopes (ΔE 6.9-7.2 in our check), while blue↔red stays far apart (ΔE ≥ 19).
* **Identity** — strategy vs buy-and-hold benchmark: categorical slots 1 and 2
  (blue, orange); allocation slices take the categorical slots in a fixed order.
* **Reference lines** (moving averages, MACD, thresholds) use neutral ink so they never
  compete with the direction colours; dashing marks thresholds/levels and the second of
  two same-ink lines.
* **Money outcome** (P&L text in tables) uses the status pair good/critical — see
  :mod:`dashboard.formatting`.

Palette notes: the hexes come from a validated reference palette (light surface
``#fcfcfb`` / dark ``#1a1a19``): blue/red/orange pass the lightness, chroma, CVD
(ΔE ≥ 8) and normal-vision (ΔE ≥ 15) gates in both modes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#: Streamlit's UI face first (already loaded inside the app), then the system sans.
FONT_FAMILY: Final = (
    '"Source Sans Pro", "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif'
)


# =============================================================================== theme


@dataclass(frozen=True, slots=True)
class ChartTheme:
    """Colour tokens for one appearance (light or dark)."""

    name: str
    surface: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    baseline: str
    #: Direction: bullish / up / positive.
    up: str
    #: Direction: bearish / down / negative.
    down: str
    #: HOLD / zero / no signal.
    neutral: str
    #: Status text colours for money outcomes (profit / loss).
    good: str
    bad: str
    #: Accent for single-series indicator panels (RSI).
    accent: str
    #: Cash slice of the allocation donut ("not invested").
    cash: str
    #: Categorical slots in their fixed order (identity encoding).
    series: tuple[str, ...]

    @property
    def is_dark(self) -> bool:
        return self.name == "dark"


LIGHT: Final = ChartTheme(
    name="light",
    surface="#ffffff",
    ink="#0b0b0b",
    ink_secondary="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    baseline="#c3c2b7",
    up="#2a78d6",
    down="#e34948",
    neutral="#898781",
    good="#006300",
    bad="#d03b3b",
    accent="#6250d6",
    cash="#c3c2b7",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"),
)

DARK: Final = ChartTheme(
    name="dark",
    surface="#0e1117",
    ink="#f0efec",
    ink_secondary="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    baseline="#383835",
    up="#3987e5",
    down="#e66767",
    neutral="#898781",
    good="#0ca30c",
    bad="#e66767",
    accent="#9085e9",
    cash="#52514e",
    series=("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"),
)


def get_theme(dark: bool | str | None = False) -> ChartTheme:
    """:data:`DARK` for ``True`` / ``"dark"``, otherwise :data:`LIGHT`."""
    if isinstance(dark, str):
        return DARK if dark.strip().lower() == "dark" else LIGHT
    return DARK if dark else LIGHT


def rgba(hex_color: str, alpha: float) -> str:
    """``#rrggbb`` → ``rgba(r, g, b, alpha)`` (for tints and translucent fills)."""
    value = hex_color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"expected #rrggbb, got {hex_color!r}")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, alpha)):.3f})"


def direction_color(value: float | None, theme: ChartTheme, *, epsilon: float = 1e-9) -> str:
    """Up colour for positive values, down colour for negative, neutral for ~0/None."""
    if value is None or not math.isfinite(value) or abs(value) <= epsilon:
        return theme.neutral
    return theme.up if value > 0 else theme.down


def action_color(action: str | None, theme: ChartTheme) -> str:
    """BUY → up, SELL → down, anything else → neutral."""
    key = (action or "").upper()
    if key == "BUY":
        return theme.up
    if key == "SELL":
        return theme.down
    return theme.neutral


def plotly_template(theme: ChartTheme) -> go.layout.Template:
    """A self-contained template: hairline recessive axes, quiet legend, opaque surface."""
    axis = {
        "gridcolor": theme.grid,
        "gridwidth": 1,
        "griddash": "solid",
        "linecolor": theme.baseline,
        "linewidth": 1,
        "showline": True,
        "zeroline": False,
        "zerolinecolor": theme.baseline,
        "ticks": "",
        "tickfont": {"color": theme.muted, "size": 11},
        "title": {"font": {"color": theme.ink_secondary, "size": 12}},
        "automargin": True,
        "showspikes": False,
    }
    layout = go.Layout(
        font={"family": FONT_FAMILY, "color": theme.ink, "size": 13},
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        colorway=list(theme.series),
        xaxis=axis,
        yaxis=axis,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": theme.ink_secondary, "size": 12},
            "title": {"text": ""},
        },
        hoverlabel={
            "bgcolor": theme.surface,
            "bordercolor": theme.baseline,
            "font": {"color": theme.ink, "family": FONT_FAMILY, "size": 12},
            "align": "left",
        },
        hovermode="x unified",
        margin={"l": 8, "r": 8, "t": 40, "b": 8},
        barcornerradius=4,
        separators=".,",
    )
    return go.layout.Template(layout=layout)


def apply_theme(fig: go.Figure, theme: ChartTheme) -> go.Figure:
    """Attach the template **and** pin the surface/ink colours on the layout itself.

    With ``st.plotly_chart(..., theme=None)`` Streamlit's frontend still fills in
    ``paper_bgcolor``, ``plot_bgcolor`` and ``font`` from the app theme unless the
    figure's *layout* sets them — values that live only in the template are overridden
    (the plot area would turn Streamlit's grey). Setting them explicitly makes the figure
    render identically inside and outside Streamlit.
    """
    fig.update_layout(
        template=plotly_template(theme),
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        font={"family": FONT_FAMILY, "color": theme.ink, "size": 13},
    )
    return fig


def _base_figure(theme: ChartTheme, *, height: int) -> go.Figure:
    fig = apply_theme(go.Figure(), theme)
    fig.update_layout(height=height)
    return fig


def empty_figure(message: str, *, theme: ChartTheme = LIGHT, height: int = 260) -> go.Figure:
    """Placeholder with a centred message (no axes) — never an empty white box."""
    fig = _base_figure(theme, height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": theme.muted, "size": 14},
    )
    fig.update_layout(hovermode=False)
    return fig


# =============================================================================== data shaping


def to_utc_index(values: Iterable[Any]) -> pd.DatetimeIndex:
    """Parse API timestamps (ISO-8601 strings, or epoch seconds / ms) into a UTC index.

    Unparseable entries become ``NaT`` (callers drop them) rather than raising, so one
    malformed row cannot blank a whole chart.
    """
    items = list(values)
    if not items:
        return pd.DatetimeIndex([], tz="UTC")
    if all(isinstance(v, int | float) and not isinstance(v, bool) for v in items):
        numbers = pd.Series(items, dtype="float64")
        unit = "ms" if numbers.abs().max() > 1e11 else "s"
        return pd.DatetimeIndex(pd.to_datetime(numbers, unit=unit, utc=True, errors="coerce"))
    parsed = pd.to_datetime(
        pd.Series([None if v is None else str(v) for v in items], dtype="object"),
        utc=True,
        errors="coerce",
        format="ISO8601",
    )
    return pd.DatetimeIndex(parsed)


def _numeric(values: Iterable[Any], length: int | None = None) -> np.ndarray:
    """Float array with ``None``/garbage → NaN, right-aligned to ``length`` if given."""
    array = pd.to_numeric(pd.Series(list(values), dtype="object"), errors="coerce").to_numpy(
        dtype="float64"
    )
    if length is None or len(array) == length:
        return array
    if len(array) > length:  # more indicator points than candles: keep the latest ones
        return array[-length:]
    padded = np.full(length, np.nan)
    if len(array):
        padded[-len(array) :] = array  # indicators end on the last candle
    return padded


_CANDLE_KEYS: Final = {
    "open": ("o", "open", "Open"),
    "high": ("h", "high", "High"),
    "low": ("l", "low", "Low"),
    "close": ("c", "close", "Close"),
    "volume": ("v", "volume", "Volume"),
}
_TIME_KEYS: Final = ("t", "timestamp", "time", "date", "datetime")
#: Indicator aliases the API might use → canonical column names.
_INDICATOR_ALIASES: Final = {
    "sma20": "sma20",
    "sma_20": "sma20",
    "sma50": "sma50",
    "sma_50": "sma50",
    "rsi": "rsi",
    "rsi14": "rsi",
    "rsi_14": "rsi",
    "macd": "macd",
    "signal": "signal",
    "macd_signal": "signal",
    "hist": "hist",
    "histogram": "hist",
    "macd_hist": "hist",
}


def _first_key(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def candles_frame(history: Mapping[str, Any]) -> pd.DataFrame:
    """``GET /history`` payload → DataFrame indexed by UTC time.

    Columns: ``open, high, low, close, volume`` plus whichever indicator series the API
    supplied (``sma20, sma50, rsi, macd, signal, hist``). Indicator arrays are aligned to
    the candles from the right (they end on the latest bar). ``hist`` is derived as
    ``macd − signal`` when absent. If the API omits the moving averages they are computed
    here (a plain rolling mean — the same definition the engine uses).
    """
    candles = history.get("candles") or []
    rows = [c for c in candles if isinstance(c, Mapping)]
    index = to_utc_index(_first_key(r, _TIME_KEYS) for r in rows)
    data: dict[str, np.ndarray] = {
        column: _numeric(_first_key(r, keys) for r in rows) for column, keys in _CANDLE_KEYS.items()
    }
    frame = pd.DataFrame(data, index=index)
    frame.index.name = "t"

    indicators = history.get("indicators") or {}
    if isinstance(indicators, Mapping):
        for name, values in indicators.items():
            column = _INDICATOR_ALIASES.get(str(name).lower(), str(name).lower())
            if isinstance(values, Sequence) and not isinstance(values, str):
                frame[column] = _numeric(values, len(frame))

    valid = frame.index.notna() & frame[["open", "high", "low", "close"]].notna().all(axis=1)
    frame = frame[valid]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for window in (20, 50):
        column = f"sma{window}"
        if column not in frame.columns and len(frame) >= window:
            frame[column] = frame["close"].rolling(window).mean()
    if "hist" not in frame.columns and {"macd", "signal"} <= set(frame.columns):
        frame["hist"] = frame["macd"] - frame["signal"]
    return frame


def equity_frame(payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """``GET /equity`` payload (or its ``points`` list) → DataFrame[equity, cash] by time."""
    points = payload.get("points", []) if isinstance(payload, Mapping) else payload
    rows = [p for p in points or [] if isinstance(p, Mapping)]
    frame = pd.DataFrame(
        {
            "equity": _numeric(p.get("equity") for p in rows),
            "cash": _numeric(p.get("cash") for p in rows),
        },
        index=to_utc_index(_first_key(p, _TIME_KEYS) for p in rows),
    )
    frame.index.name = "t"
    frame = frame[frame.index.notna() & frame["equity"].notna()]
    return frame[~frame.index.duplicated(keep="last")].sort_index()


_CURVE_ALIASES: Final = {
    "equity": "equity",
    "strategy": "equity",
    "strategy_equity": "equity",
    "cash": "cash",
    "drawdown": "drawdown",
    "benchmark": "benchmark",
    "benchmark_equity": "benchmark",
    "buy_and_hold": "benchmark",
    "buy_hold": "benchmark",
    "positions_value": "positions_value",
}


def backtest_curve_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    """Normalise ``BacktestResult.to_dict()["equity_curve"]`` into a DataFrame.

    Accepted encodings (whatever ``to_dict`` chose): a list of records with a time key
    (``t``/``timestamp``/``date``); pandas ``orient="split"`` (``index``/``columns``/
    ``data``); a dict of columns → lists (plus a time list); a dict of columns →
    ``{timestamp: value}``; or a dict of ``timestamp → {column: value}``. Columns are
    renamed to ``equity, cash, drawdown, benchmark`` where recognisable.
    """
    curve = result.get("equity_curve")
    frame = _curve_to_frame(curve)
    if frame.empty:
        return pd.DataFrame(columns=["equity"], index=pd.DatetimeIndex([], tz="UTC", name="t"))
    frame = frame.rename(columns=lambda c: _CURVE_ALIASES.get(str(c).lower(), str(c).lower()))
    frame = frame.loc[:, ~frame.columns.duplicated()]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[frame.index.notna()]
    if "equity" in frame.columns:
        frame = frame[frame["equity"].notna()]
    frame.index.name = "t"
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def _curve_to_frame(curve: Any) -> pd.DataFrame:
    if isinstance(curve, Sequence) and not isinstance(curve, str):
        rows = [r for r in curve if isinstance(r, Mapping)]
        if not rows:
            return pd.DataFrame()
        index = to_utc_index(_first_key(r, (*_TIME_KEYS, "index")) for r in rows)
        records = [{k: v for k, v in r.items() if k not in (*_TIME_KEYS, "index")} for r in rows]
        return pd.DataFrame(records, index=index)
    if not isinstance(curve, Mapping) or not curve:
        return pd.DataFrame()
    if {"index", "data", "columns"} <= set(curve):  # orient="split"
        return pd.DataFrame(
            curve["data"], columns=curve["columns"], index=to_utc_index(curve["index"])
        )
    time_key = next((k for k in (*_TIME_KEYS, "index") if k in curve), None)
    if time_key is not None and isinstance(curve[time_key], Sequence):
        columns = {
            k: v
            for k, v in curve.items()
            if k != time_key and isinstance(v, Sequence) and not isinstance(v, str)
        }
        index = to_utc_index(curve[time_key])
        return pd.DataFrame({k: _numeric(v, len(index)) for k, v in columns.items()}, index=index)
    first = next(iter(curve.values()))
    if isinstance(first, Mapping):
        keys = list(curve)
        if all(str(k).lower() in _CURVE_ALIASES for k in keys):  # columns → {t: value}
            frame = pd.DataFrame({k: pd.Series(v) for k, v in curve.items()})
        else:  # orient="index": t → {column: value}
            frame = pd.DataFrame.from_dict(curve, orient="index")
        frame.index = to_utc_index(frame.index)
        return frame
    return pd.DataFrame()


def drawdown_pct(equity: pd.Series) -> pd.Series:
    """Drawdown from the running peak in percent (≤ 0; ``-12.5`` = 12.5 % below peak)."""
    series = pd.to_numeric(equity, errors="coerce").astype("float64")
    peak = series.cummax()
    drawdown = (series / peak - 1.0) * 100.0
    return drawdown.where(peak > 0).fillna(0.0).clip(upper=0.0)


def _has_weekend_bars(index: pd.DatetimeIndex) -> bool:
    return bool(len(index)) and bool((index.day_of_week >= 5).any())


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _end_labels(
    fig: go.Figure,
    items: Sequence[tuple[str, pd.Series, str]],
    theme: ChartTheme,
    *,
    fmt: str = "${:,.0f}",
) -> None:
    """Direct-label the last point of each series (marker in series colour, ink text).

    When two end values sit closer than ~7 % of the visible range the labels are pushed
    apart vertically so they never overlap.
    """
    ends = []
    for name, series, color in items:
        clean = series.dropna()
        if clean.empty:
            continue
        ends.append((name, clean.index[-1], float(clean.iloc[-1]), color))
    if not ends:
        return
    values = [v for *_, v, _ in ends]
    lows = [float(s.min()) for _, s, _ in items if not s.dropna().empty]
    highs = [float(s.max()) for _, s, _ in items if not s.dropna().empty]
    span = max(max(highs) - min(lows), 1e-9)
    order = sorted(range(len(ends)), key=lambda i: values[i])
    shifts = [0] * len(ends)
    for rank in range(1, len(order)):
        lower, upper = order[rank - 1], order[rank]
        if (values[upper] - values[lower]) / span < 0.07:
            shifts[lower] -= 9
            shifts[upper] += 9
    for (name, x, y, color), shift in zip(ends, shifts, strict=True):
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                marker={"size": 8, "color": color, "line": {"color": theme.surface, "width": 2}},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_annotation(
            x=x,
            y=y,
            text=f"{name} {fmt.format(y)}",
            showarrow=False,
            xanchor="left",
            xshift=8,
            yshift=shift,
            font={"color": theme.ink_secondary, "size": 12},
        )


# =============================================================================== portfolio


def equity_curve_figure(
    points: pd.DataFrame,
    *,
    initial_capital: float | None = None,
    theme: ChartTheme = LIGHT,
    height: int = 320,
) -> go.Figure:
    """Paper-account equity over time (single series, so no legend box).

    A hairline marks the initial capital, so "above / below where we started" is
    readable at a glance; the last value is labelled directly.
    """
    if points is None or points.empty or "equity" not in points.columns:
        return empty_figure(
            "No equity snapshots yet — run a scan to record one", theme=theme, height=height
        )
    equity = points["equity"].astype("float64")
    color = theme.series[0]
    few = len(equity) < 40
    fig = _base_figure(theme, height=height)
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.to_numpy(),
            name="Equity",
            mode="lines+markers" if few else "lines",
            line={"color": color, "width": 2},
            marker={"size": 8, "color": color, "line": {"color": theme.surface, "width": 2}},
            fill="none",
            hovertemplate="Equity $%{y:,.2f}<extra></extra>",
            showlegend=False,
        )
    )
    capital = _finite_or_none(initial_capital)
    if capital is not None:
        fig.add_hline(
            y=capital,
            line={"color": theme.baseline, "width": 1},
            annotation={
                "text": f"Initial capital ${capital:,.0f}",
                "font": {"color": theme.muted, "size": 11},
            },
            annotation_position="top left",
        )
    _end_labels(fig, [("", equity, color)], theme, fmt="${:,.2f}")
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(margin={"r": 96})
    return fig


def allocation_figure(
    positions: Sequence[Mapping[str, Any]],
    cash: float | None,
    *,
    theme: ChartTheme = LIGHT,
    max_slices: int = 5,
    height: int = 320,
) -> go.Figure:
    """Allocation donut: the largest positions, the rest folded into "Other", plus cash.

    At most ``max_slices`` named positions keep the donut readable (part-to-whole at a
    glance only — the positions table carries the exact numbers). Named slices take the
    categorical slots in alphabetical order, so a symbol's colour does not depend on
    its rank; "Other" is muted and cash is a neutral grey. The centre shows the share of
    equity that is invested.
    """
    values: dict[str, float] = {}
    for position in positions or []:
        symbol = str(position.get("symbol") or "?").upper()
        value = _finite_or_none(position.get("market_value"))
        if value is None:
            qty = _finite_or_none(position.get("quantity")) or 0.0
            price = _finite_or_none(position.get("market_price")) or _finite_or_none(
                position.get("avg_price")
            )
            value = qty * (price or 0.0)
        if value > 0:
            values[symbol] = values.get(symbol, 0.0) + value
    if not values:
        return empty_figure("No open positions — 100% cash", theme=theme, height=height)

    ranked = sorted(values.items(), key=lambda kv: kv[1], reverse=True)
    named = dict(ranked[:max_slices])
    other = sum(v for _, v in ranked[max_slices:])
    cash_value = max(_finite_or_none(cash) or 0.0, 0.0)

    slot_of = {sym: i for i, sym in enumerate(sorted(named))}
    labels: list[str] = []
    slice_values: list[float] = []
    colors: list[str] = []
    for symbol, value in named.items():
        labels.append(symbol)
        slice_values.append(value)
        colors.append(theme.series[slot_of[symbol] % len(theme.series)])
    if other > 0:
        labels.append("Other")
        slice_values.append(other)
        colors.append(theme.muted)
    if cash_value > 0:
        labels.append("Cash")
        slice_values.append(cash_value)
        colors.append(theme.cash)

    total = sum(slice_values)
    invested_pct = (total - cash_value) / total * 100.0 if total > 0 else 0.0
    fig = _base_figure(theme, height=height)
    fig.add_trace(
        go.Pie(
            labels=labels,
            values=slice_values,
            hole=0.62,
            sort=False,
            direction="clockwise",
            rotation=90,
            marker={"colors": colors, "line": {"color": theme.surface, "width": 2}},
            textinfo="label+percent",
            textposition="outside",
            textfont={"color": theme.ink_secondary, "size": 12},
            texttemplate="%{label} %{percent:.0%}",
            hovertemplate="%{label}<br>$%{value:,.2f} (%{percent:.1%})<extra></extra>",
            automargin=True,
        )
    )
    fig.add_annotation(
        text=f"<b>{invested_pct:.0f}%</b><br>invested",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": theme.ink, "size": 15},
    )
    fig.update_layout(
        showlegend=True,
        legend={"orientation": "h", "yanchor": "top", "y": -0.05, "x": 0.5, "xanchor": "center"},
        hovermode="closest",
        margin={"t": 24, "b": 8, "l": 24, "r": 24},
    )
    return fig


# =============================================================================== analysis


def consensus_gauge(
    score: float | None,
    *,
    buy_threshold: float = 0.25,
    sell_threshold: float = -0.25,
    action: str | None = None,
    theme: ChartTheme = LIGHT,
    height: int = 230,
) -> go.Figure:
    """Diverging gauge of the consensus score on ``[-1, 1]``.

    The dial is split into the SELL zone ``[-1, sell]``, the HOLD band and the BUY zone
    ``[buy, 1]`` (tinted with the direction colours); a needle marks the score, coloured by
    the zone it falls in (or by ``action`` when given). A plain
    gauge *bar* would grow from −1, which misreads a diverging quantity, so only the
    needle is drawn.
    """
    value = _finite_or_none(score)
    shown = 0.0 if value is None else max(-1.0, min(1.0, value))
    sell = max(-1.0, min(sell_threshold, 1.0))
    buy = max(sell, min(buy_threshold, 1.0))
    if action:
        needle = action_color(action, theme)
    elif shown >= buy:
        needle = theme.up
    elif shown <= sell:
        needle = theme.down
    else:
        needle = theme.neutral
    fig = _base_figure(theme, height=height)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=shown,
            number={"valueformat": "+.2f", "font": {"color": theme.ink, "size": 34}},
            gauge={
                "shape": "angular",
                "axis": {
                    "range": [-1, 1],
                    "tickvals": [-1, sell, 0, buy, 1],
                    "ticktext": ["-1", f"{sell:+.2f}", "0", f"{buy:+.2f}", "+1"],
                    "tickfont": {"color": theme.muted, "size": 11},
                    "tickcolor": theme.baseline,
                    "ticklen": 4,
                },
                "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
                "bgcolor": theme.surface,
                "borderwidth": 0,
                "steps": [
                    {"range": [-1, sell], "color": rgba(theme.down, 0.22)},
                    {"range": [sell, buy], "color": rgba(theme.neutral, 0.16)},
                    {"range": [buy, 1], "color": rgba(theme.up, 0.22)},
                ],
                "threshold": {
                    "line": {"color": needle, "width": 5},
                    "thickness": 1.0,
                    "value": shown,
                },
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(margin={"t": 24, "b": 8, "l": 24, "r": 24}, hovermode=False)
    return fig


def _wrap(text: str, width: int = 60, max_lines: int = 4) -> str:
    words, lines, current = (text or "").split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = [*lines[: max_lines - 1], lines[max_lines - 1] + " …"]
    return "<br>".join(lines)


def agent_signals_figure(
    signals: Sequence[Mapping[str, Any]],
    *,
    consensus: float | None = None,
    consensus_confidence: float | None = None,
    buy_threshold: float = 0.25,
    sell_threshold: float = -0.25,
    theme: ChartTheme = LIGHT,
) -> go.Figure:
    """Horizontal diverging bars: one per agent vote (plus the consensus) on ``[-1, 1]``.

    Bar colour encodes direction; opacity encodes the agent's confidence (a faint bar is
    a vote the agent itself does not trust) and the confidence is also printed in the
    row label, so it is never colour-only. Dashed rules mark the BUY/SELL thresholds.
    """
    rows: list[tuple[str, float, float, str]] = []
    for item in signals or []:
        name = str(item.get("agent") or "agent").replace("_", " ").title()
        value = _finite_or_none(item.get("signal")) or 0.0
        conf = _finite_or_none(item.get("confidence")) or 0.0
        rows.append((name, value, conf, str(item.get("rationale") or "")))
    if consensus is not None:
        conf = _finite_or_none(consensus_confidence) or 0.0
        rows.append(("Consensus", _finite_or_none(consensus) or 0.0, conf, "weighted vote"))
    if not rows:
        return empty_figure("No agent signals", theme=theme, height=160)

    rows.reverse()  # plotly draws the first category at the bottom
    labels = [f"{name}  ·  {conf:.0%} conf" for name, _, conf, _ in rows]
    values = [value for _, value, _, _ in rows]
    colors = [
        rgba(direction_color(value, theme), 0.35 + 0.65 * max(0.0, min(conf, 1.0)))
        for _, value, conf, _ in rows
    ]
    hover = [f"<b>{name}</b> {value:+.2f}<br>{_wrap(why)}" for name, value, _, why in rows]
    fig = _base_figure(theme, height=64 + 48 * len(rows))
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
            textfont={"color": theme.ink_secondary, "size": 12},
            cliponaxis=False,
            width=0.55,
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False,
        )
    )
    for x, label in ((sell_threshold, "SELL"), (buy_threshold, "BUY")):
        fig.add_vline(
            x=x,
            layer="below",
            line={"color": theme.muted, "width": 1, "dash": "dash"},
            annotation={"text": label, "font": {"color": theme.muted, "size": 10}},
            annotation_position="top",
        )
    fig.add_vline(x=0, layer="below", line={"color": theme.baseline, "width": 1})
    fig.update_xaxes(
        range=[-1.4, 1.4],  # room for the outside value labels at ±1
        tickvals=[-1, -0.5, 0, 0.5, 1],
        ticktext=["-1", "-0.5", "0", "+0.5", "+1"],
        showgrid=False,
    )
    fig.update_yaxes(showgrid=False, showline=False, tickfont={"color": theme.ink_secondary})
    fig.update_layout(hovermode="closest", margin={"t": 28, "r": 16})
    return fig


def price_figure(
    frame: pd.DataFrame,
    *,
    symbol: str = "",
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    entry_price: float | None = None,
    theme: ChartTheme = LIGHT,
    height: int = 640,
) -> go.Figure:
    """Candlesticks with SMA20/SMA50 overlays, plus RSI and MACD panels when available.

    * Price panel — candles in the direction colours; SMA20 solid and SMA50 dashed in
      neutral ink; optional entry / stop / take-profit levels as dotted rules.
    * RSI panel — RSI(14) with the 30/70 oversold/overbought rules.
    * MACD panel — histogram bars coloured by sign, MACD line solid, signal dashed.

    The panels share the time axis but each has its own y-scale (separate subplots, not a
    dual axis). Weekend gaps are removed unless the instrument trades on weekends.
    """
    required = {"open", "high", "low", "close"}
    if frame is None or frame.empty or not required <= set(frame.columns):
        return empty_figure("No price history available", theme=theme, height=height)

    def has(column: str) -> bool:
        return column in frame.columns and bool(frame[column].notna().any())

    panels = ["price"]
    if has("rsi"):
        panels.append("rsi")
    if has("macd"):
        panels.append("macd")
    heights = {"price": 0.6, "rsi": 0.18, "macd": 0.22}
    weights = [heights[p] for p in panels]
    total = sum(weights)
    fig = make_subplots(
        rows=len(panels),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[w / total for w in weights],
    )
    apply_theme(fig, theme)
    fig.update_layout(height=height if len(panels) > 1 else 440)
    x = frame.index

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name=symbol or "Price",
            increasing={"line": {"color": theme.up, "width": 1}, "fillcolor": theme.up},
            decreasing={"line": {"color": theme.down, "width": 1}, "fillcolor": theme.down},
            whiskerwidth=0.3,
        ),
        row=1,
        col=1,
    )
    for column, label, dash in (("sma20", "SMA20", "solid"), ("sma50", "SMA50", "dash")):
        if has(column):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=frame[column],
                    name=label,
                    mode="lines",
                    line={"color": theme.ink_secondary, "width": 1.5, "dash": dash},
                    hovertemplate=f"{label} %{{y:,.2f}}<extra></extra>",
                    connectgaps=False,
                ),
                row=1,
                col=1,
            )
    levels = (
        (entry_price, "Entry", theme.muted),
        (stop_price, "Stop", theme.down),
        (take_profit_price, "Target", theme.up),
    )
    for level, label, color in levels:
        value = _finite_or_none(level)
        if value is None or value <= 0:
            continue
        fig.add_hline(
            y=value,
            line={"color": color, "width": 1, "dash": "dot"},
            annotation={
                "text": f"{label} {value:,.2f}",
                "font": {"color": theme.ink_secondary, "size": 11},
            },
            annotation_position="right",
            row=1,
            col=1,
        )
    fig.update_yaxes(title_text="Price", row=1, col=1)

    row = 1
    if "rsi" in panels:
        row += 1
        fig.add_trace(
            go.Scatter(
                x=x,
                y=frame["rsi"],
                name="RSI 14",
                mode="lines",
                line={"color": theme.accent, "width": 1.5},
                hovertemplate="RSI %{y:.1f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        for level in (30, 70):
            fig.add_hline(
                y=level, line={"color": theme.muted, "width": 1, "dash": "dash"}, row=row, col=1
            )
        fig.update_yaxes(title_text="RSI", range=[0, 100], tickvals=[30, 50, 70], row=row, col=1)
    if "macd" in panels:
        row += 1
        if has("hist"):
            hist = frame["hist"]
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=hist,
                    name="Histogram",
                    marker={
                        "color": [
                            rgba(direction_color(_finite_or_none(v), theme), 0.55) for v in hist
                        ],
                        "line": {"width": 0},
                    },
                    hovertemplate="Hist %{y:+.3f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=frame["macd"],
                name="MACD",
                mode="lines",
                line={"color": theme.ink_secondary, "width": 1.5},
                hovertemplate="MACD %{y:+.3f}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        if has("signal"):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=frame["signal"],
                    name="Signal",
                    mode="lines",
                    line={"color": theme.muted, "width": 1.5, "dash": "dash"},
                    hovertemplate="Signal %{y:+.3f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        fig.add_hline(y=0, line={"color": theme.baseline, "width": 1}, row=row, col=1)
        fig.update_yaxes(title_text="MACD", row=row, col=1)

    fig.update_xaxes(rangeslider_visible=False)
    if not _has_weekend_bars(pd.DatetimeIndex(x)):
        fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    fig.update_layout(bargap=0.15, margin={"r": 96})
    return fig


# =============================================================================== backtest


def backtest_equity_figure(
    curve: pd.DataFrame,
    *,
    initial_capital: float | None = None,
    theme: ChartTheme = LIGHT,
    height: int = 360,
) -> go.Figure:
    """Strategy equity vs the buy-and-hold benchmark on one dollar axis."""
    if curve is None or curve.empty or "equity" not in curve.columns:
        return empty_figure("No equity curve in the backtest result", theme=theme, height=height)
    strategy_color, benchmark_color = theme.series[0], theme.series[1]
    fig = _base_figure(theme, height=height)
    labelled = [("Strategy", curve["equity"], strategy_color)]
    fig.add_trace(
        go.Scatter(
            x=curve.index,
            y=curve["equity"],
            name="Strategy",
            mode="lines",
            line={"color": strategy_color, "width": 2},
            hovertemplate="Strategy $%{y:,.0f}<extra></extra>",
        )
    )
    if "benchmark" in curve.columns and curve["benchmark"].notna().any():
        labelled.append(("Buy & hold", curve["benchmark"], benchmark_color))
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve["benchmark"],
                name="Buy & hold",
                mode="lines",
                line={"color": benchmark_color, "width": 2},
                hovertemplate="Buy & hold $%{y:,.0f}<extra></extra>",
            )
        )
    capital = _finite_or_none(initial_capital)
    if capital is not None:
        fig.add_hline(y=capital, line={"color": theme.baseline, "width": 1})
    _end_labels(fig, labelled, theme)
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    fig.update_layout(margin={"r": 150})
    return fig


def drawdown_figure(
    curve: pd.DataFrame, *, theme: ChartTheme = LIGHT, height: int = 260
) -> go.Figure:
    """Underwater chart: % below the running peak for the strategy and the benchmark.

    Drawdowns are recomputed from the equity columns (so the encoding of any
    ``drawdown`` column in the payload does not matter). The strategy is drawn last (on
    top) and its worst point is labelled.
    """
    if curve is None or curve.empty or "equity" not in curve.columns:
        return empty_figure("No equity curve in the backtest result", theme=theme, height=height)
    strategy_color, benchmark_color = theme.series[0], theme.series[1]
    strategy = drawdown_pct(curve["equity"])
    fig = _base_figure(theme, height=height)
    if "benchmark" in curve.columns and curve["benchmark"].notna().any():
        benchmark = drawdown_pct(curve["benchmark"])
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark,
                name="Buy & hold",
                mode="lines",
                line={"color": benchmark_color, "width": 2},
                hovertemplate="Buy & hold %{y:.2f}%<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=strategy.index,
            y=strategy,
            name="Strategy",
            mode="lines",
            line={"color": strategy_color, "width": 2},
            fill="tozeroy",
            fillcolor=rgba(strategy_color, 0.14),
            hovertemplate="Strategy %{y:.2f}%<extra></extra>",
        )
    )
    if len(strategy) and strategy.min() < 0:
        worst_at = strategy.idxmin()
        fig.add_annotation(
            x=worst_at,
            y=float(strategy.min()),
            text=f"Max {float(strategy.min()):.1f}%",
            showarrow=False,
            yshift=-12,
            font={"color": theme.ink_secondary, "size": 11},
        )
    fig.add_hline(y=0, line={"color": theme.baseline, "width": 1})
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    return fig


__all__ = [
    "DARK",
    "FONT_FAMILY",
    "LIGHT",
    "ChartTheme",
    "action_color",
    "agent_signals_figure",
    "allocation_figure",
    "apply_theme",
    "backtest_curve_frame",
    "backtest_equity_figure",
    "candles_frame",
    "consensus_gauge",
    "direction_color",
    "drawdown_figure",
    "drawdown_pct",
    "empty_figure",
    "equity_curve_figure",
    "equity_frame",
    "get_theme",
    "plotly_template",
    "price_figure",
    "rgba",
    "to_utc_index",
]
