"""Rendering of alert e-mails: subject line, plain-text body and inline-styled HTML body.

Design notes
------------
* **Both bodies carry the same facts.** The plain-text part is not an afterthought: it is
  what text-only clients, screen readers and many phone notification previews show.
* **HTML is e-mail-client safe.** Table layout, inline ``style`` attributes only (Gmail
  strips ``<style>`` blocks in many contexts), no scripts, no external images.
* **Everything dynamic is escaped.** Agent rationales and the decision summary can quote
  news headlines and LLM output, i.e. untrusted third-party text. Every value goes
  through :func:`html.escape` unless it was produced by one of this module's own HTML
  helpers (marked with the private ``_Safe`` string type, in the spirit of MarkupSafe).
* **Finance conventions.** Prices and money are shown to the cent; stop/target distances
  are relative to the entry price; the reward/risk ratio is
  ``(take_profit − entry) / (entry − stop)`` ("R multiple"); the drawdown recovery
  level is ``peak_equity × (1 − MAX_DRAWDOWN_HALT)`` — the equity above which the
  circuit breaker re-allows new BUY orders.
"""

from __future__ import annotations

import html as _html
import math
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from trading_swarm import __version__
from trading_swarm.config import Settings
from trading_swarm.models import (
    Action,
    AgentSignal,
    Decision,
    ExitReason,
    Order,
    OrderStatus,
    PortfolioSnapshot,
    RiskAssessment,
    Side,
    ensure_utc,
)

SUBJECT_PREFIX = "[MT Swarm]"
DISCLAIMER = "Paper trading only: fills are simulated. This is not financial advice."
_RATIONALE_MAX_CHARS = 280
_NA = "—"

# Palette (Tailwind-ish, WCAG AA on white).
GREEN = "#15803d"
RED = "#b91c1c"
SLATE = "#475569"
AMBER = "#b45309"
BLUE = "#1d4ed8"
_TEXT = "#0f172a"
_MUTED = "#64748b"
_BORDER = "#e2e8f0"
_PAGE_BG = "#f1f5f9"
_FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


@dataclass(frozen=True, slots=True)
class RenderedEmail:
    """A ready-to-send alert: ``subject``, ``text`` (plain) and ``html`` bodies."""

    subject: str
    text: str
    html: str


# =============================================================================== formatting


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def fmt_money(value: float | None, *, signed: bool = False) -> str:
    """``1234.5`` → ``"$1,234.50"``; negatives as ``"-$12.00"``; ``signed`` adds ``+``."""
    if value is None or not math.isfinite(value):
        return _NA
    sign = "-" if value < 0 else ("+" if signed and value > 0 else "")
    return f"{sign}${abs(value):,.2f}"


def fmt_price(value: float | None) -> str:
    """A share price: cents for normal prices, 4 decimals for sub-dollar quotes."""
    if value is None or not math.isfinite(value):
        return _NA
    return f"${value:,.4f}" if abs(value) < 1 else f"${value:,.2f}"


def fmt_signed(value: float | None, digits: int = 2) -> str:
    """A signal/score with an explicit sign: ``0.62`` → ``"+0.62"``."""
    if value is None or not math.isfinite(value):
        return _NA
    return f"{value:+.{digits}f}"


def fmt_pct(value: float | None, *, signed: bool = True, digits: int = 2) -> str:
    """A value that is already a percentage: ``5.123`` → ``"+5.12%"``."""
    if value is None or not math.isfinite(value):
        return _NA
    return f"{value:+.{digits}f}%" if signed else f"{value:.{digits}f}%"


def fmt_confidence(value: float | None) -> str:
    """A ``[0, 1]`` confidence as a whole percentage: ``0.812`` → ``"81%"``."""
    if value is None or not math.isfinite(value):
        return _NA
    return f"{value * 100:.0f}%"


def fmt_ts(value: datetime | None) -> str:
    """``2026-09-14 15:30 UTC``."""
    if value is None:
        return _NA
    return ensure_utc(value).strftime("%Y-%m-%d %H:%M UTC")


def _truncate(text: str, limit: int = _RATIONALE_MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _tone(value: float | None, *, dead_zone: float = 0.05) -> str:
    """Green for positive, red for negative, slate around zero."""
    if value is None or not math.isfinite(value):
        return SLATE
    if value > dead_zone:
        return GREEN
    if value < -dead_zone:
        return RED
    return SLATE


def action_color(action: Action | Side | str) -> str:
    """Accent colour for an action/side: BUY green, SELL red, anything else slate."""
    value = str(action).upper()
    if value == "BUY":
        return GREEN
    if value == "SELL":
        return RED
    return SLATE


# =============================================================================== links


def dashboard_link(settings: Settings, **params: str) -> str | None:
    """``DASHBOARD_URL`` with ``params`` merged into its query string.

    Only absolute ``http``/``https`` URLs are linked; anything else (empty, relative,
    ``javascript:`` …) yields ``None`` so a misconfiguration can never produce a
    dangerous link.
    """
    raw = (settings.dashboard_url or "").strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({k: v for k, v in params.items() if v})
    return urlunsplit(parts._replace(path=parts.path or "/", query=urlencode(query)))


def _links(settings: Settings, symbol: str | None, page: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    if symbol:
        target = dashboard_link(settings, page=page, symbol=symbol)
        if target:
            links.append((f"Open {symbol} in the dashboard", target))
    home = dashboard_link(settings)
    if home:
        links.append(("Portfolio overview", dashboard_link(settings, page="portfolio") or home))
        links.append(("Alert log", dashboard_link(settings, page="alerts") or home))
    return links


# =============================================================================== HTML helpers


class _Safe(str):
    """A string of trusted HTML produced by this module (never escaped again)."""

    __slots__ = ()


def _e(value: object) -> str:
    """Escape ``value`` for HTML text/attribute context unless it is already ``_Safe``."""
    if isinstance(value, _Safe):
        return value
    return _html.escape(str(value), quote=True)


def _colored(text: str, color: str, *, bold: bool = True) -> _Safe:
    weight = "font-weight:600;" if bold else ""
    return _Safe(f'<span style="color:{color};{weight}">{_e(text)}</span>')


def _badge(text: str, color: str) -> _Safe:
    return _Safe(
        f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;'
        f"background:{color};color:#ffffff;font-size:12px;font-weight:700;"
        f'letter-spacing:.04em;">{_e(text)}</span>'
    )


def _paragraph(text: str) -> _Safe:
    body = "<br>".join(_e(line) for line in text.strip().splitlines())
    return _Safe(f'<p style="margin:0 0 12px;font-size:14px;line-height:1.55;">{body}</p>')


def _bullets(items: Iterable[str]) -> _Safe:
    lis = "".join(f'<li style="margin:2px 0;">{_e(item)}</li>' for item in items)
    if not lis:
        return _Safe("")
    return _Safe(f'<ul style="margin:8px 0 0;padding-left:20px;font-size:14px;">{lis}</ul>')


def _section(title: str, *parts: str) -> _Safe:
    heading = (
        f'<h2 style="margin:24px 0 8px;padding-bottom:6px;font-size:15px;color:{_TEXT};'
        f'border-bottom:1px solid {_BORDER};">{_e(title)}</h2>'
    )
    return _Safe(heading + "".join(parts))


def _kv_table(rows: Sequence[tuple[str, object]]) -> _Safe:
    body = "".join(
        f'<tr><td style="width:1%;padding:5px 16px 5px 0;color:{_MUTED};white-space:nowrap;'
        f'vertical-align:top;">{_e(label)}</td>'
        f'<td style="padding:5px 0;font-weight:600;color:{_TEXT};">{_e(value)}</td></tr>'
        for label, value in rows
    )
    return _Safe(
        '<table role="presentation" cellpadding="0" cellspacing="0" '
        f'style="width:100%;border-collapse:collapse;font-size:14px;">{body}</table>'
    )


def _data_table(
    headers: Sequence[str], rows: Sequence[Sequence[object]], numeric: Sequence[int] = ()
) -> _Safe:
    def cell(value: object, index: int, tag: str) -> str:
        align = "right" if index in numeric else "left"
        style = (
            f"padding:6px 8px;border-bottom:1px solid {_BORDER};text-align:{align};"
            "vertical-align:top;"
        )
        if tag == "th":
            style += f"font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:{_MUTED};"
        return f'<{tag} style="{style}">{_e(value)}</{tag}>'

    head = "".join(cell(h, i, "th") for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(cell(v, i, "td") for i, v in enumerate(row)) + "</tr>" for row in rows
    )
    return _Safe(
        '<table role="presentation" cellpadding="0" cellspacing="0" '
        f'style="width:100%;border-collapse:collapse;font-size:13px;">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _stats(items: Sequence[tuple[str, str, str]]) -> _Safe:
    """A row of headline figures: ``(label, value, colour)``."""
    width = 100 // max(1, len(items))
    cells = "".join(
        f'<td style="width:{width}%;padding:10px 12px;border:1px solid {_BORDER};'
        f'border-radius:6px;vertical-align:top;">'
        f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;'
        f'color:{_MUTED};">{_e(label)}</div>'
        f'<div style="margin-top:2px;font-size:20px;font-weight:700;color:{color};">'
        f"{_e(value)}</div></td>"
        for label, value, color in items
    )
    return _Safe(
        '<table role="presentation" cellpadding="0" cellspacing="6" '
        f'style="width:100%;border-collapse:separate;"><tr>{cells}</tr></table>'
    )


def _buttons(links: Sequence[tuple[str, str]], accent: str) -> _Safe:
    if not links:
        return _Safe("")
    anchors = "".join(
        f'<a href="{_e(url)}" style="display:inline-block;margin:4px 8px 4px 0;'
        f"padding:9px 14px;border-radius:6px;font-size:13px;font-weight:600;"
        f'text-decoration:none;color:#ffffff;background:{accent if i == 0 else SLATE};">'
        f"{_e(label)}</a>"
        for i, (label, url) in enumerate(links)
    )
    return _Safe(f'<div style="margin-top:24px;">{anchors}</div>')


def _layout(
    *,
    kind: str,
    headline: str,
    subheadline: str,
    accent: str,
    preheader: str,
    body: Sequence[str],
    links: Sequence[tuple[str, str]],
) -> str:
    """Wrap ``body`` sections in the shared header / footer chrome."""
    footer = (
        f"{_e(DISCLAIMER)}<br>Sent by MT Trading Swarm v{_e(__version__)} because this "
        "address is listed in ALERT_EMAIL_TO."
    )
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta name="color-scheme" content="light only">'
        f"<title>{_e(headline)}</title></head>"
        f'<body style="margin:0;padding:0;background:{_PAGE_BG};">'
        # Hidden preheader: the snippet inbox lists show next to the subject.
        f'<div style="display:none;max-height:0;overflow:hidden;opacity:0;">{_e(preheader)}</div>'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="background:{_PAGE_BG};"><tr><td align="center" style="padding:24px 12px;">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        f'style="max-width:640px;background:#ffffff;border:1px solid {_BORDER};'
        f'border-radius:8px;font-family:{_FONT};color:{_TEXT};">'
        f'<tr><td style="padding:18px 24px;background:{accent};color:#ffffff;'
        f'border-radius:8px 8px 0 0;">'
        f'<div style="font-size:11px;text-transform:uppercase;letter-spacing:.1em;'
        f'opacity:.85;">MT Trading Swarm · {_e(kind)}</div>'
        f'<div style="margin-top:4px;font-size:22px;font-weight:700;">{_e(headline)}</div>'
        f'<div style="margin-top:2px;font-size:13px;opacity:.9;">{_e(subheadline)}</div>'
        "</td></tr>"
        f'<tr><td style="padding:8px 24px 24px;">{"".join(body)}{_buttons(links, accent)}</td></tr>'
        f'<tr><td style="padding:14px 24px;border-top:1px solid {_BORDER};font-size:12px;'
        f'line-height:1.5;color:{_MUTED};">{footer}</td></tr>'
        "</table></td></tr></table></body></html>"
    )


# =============================================================================== text helpers


def _text_title(title: str, subtitle: str) -> list[str]:
    return [title, "=" * len(title), subtitle, ""]


def _text_section(title: str) -> list[str]:
    return ["", title, "-" * len(title)]


def _text_kv(rows: Sequence[tuple[str, object]]) -> list[str]:
    if not rows:
        return []
    width = max(len(label) for label, _ in rows)
    return [f"  {label.ljust(width)}  {value}" for label, value in rows]


def _text_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    """Fixed-width table; the last column is left unpadded (it holds free text)."""
    cells = [[str(v) for v in row] for row in rows]
    widths = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)]

    def line(values: Sequence[str]) -> str:
        padded = [v.ljust(widths[i]) for i, v in enumerate(values[:-1])]
        return ("  " + "  ".join([*padded, values[-1]])).rstrip()

    return [line(headers), line(["-" * w for w in widths]), *(line(r) for r in cells)]


def _text_links(links: Sequence[tuple[str, str]]) -> list[str]:
    if not links:
        return []
    return ["", *(f"{label}: {url}" for label, url in links)]


def _text_footer() -> list[str]:
    return ["", "--", DISCLAIMER, f"MT Trading Swarm v{__version__}", ""]


# =============================================================================== shared blocks


def _signal_rows(signals: Sequence[AgentSignal]) -> list[list[str]]:
    rows = []
    for signal in signals:
        agent = signal.agent
        method = signal.data.get("method") if isinstance(signal.data, dict) else None
        if method:
            agent = f"{agent} ({method})"
        rows.append(
            [
                agent,
                fmt_signed(signal.signal),
                fmt_confidence(signal.confidence),
                _truncate(signal.rationale or _NA),
            ]
        )
    return rows


def _signals_html(signals: Sequence[AgentSignal]) -> _Safe:
    rows: list[list[object]] = []
    for signal, plain in zip(signals, _signal_rows(signals), strict=True):
        rows.append([plain[0], _colored(plain[1], _tone(signal.signal)), plain[2], plain[3]])
    return _data_table(["Agent", "Signal", "Confidence", "Rationale"], rows, numeric=(1, 2))


def _distance_pct(level: float | None, reference: float | None) -> float | None:
    """Distance of a protective level from the entry, in percent of the entry price."""
    if level is None or reference is None or not (_finite(level) and _finite(reference)):
        return None
    if reference == 0:
        return None
    return (level / reference - 1.0) * 100.0


def _risk_rows(risk: RiskAssessment) -> list[tuple[str, str]]:
    """The risk gate's numbers: size, protective levels, R multiple, capital at risk."""
    verdict = "Approved" if risk.approved else "Vetoed"
    rows: list[tuple[str, str]] = [("Verdict", f"{verdict} ({risk.action.value})")]
    entry = risk.entry_price or None
    if risk.quantity:
        rows.append(("Quantity", f"{risk.quantity:,} shares"))
    if entry:
        rows.append(("Entry (reference)", fmt_price(entry)))
    if risk.stop_price is not None:
        rows.append(
            (
                "Stop-loss",
                f"{fmt_price(risk.stop_price)} ({fmt_pct(_distance_pct(risk.stop_price, entry))})",
            )
        )
    if risk.take_profit_price is not None:
        distance = fmt_pct(_distance_pct(risk.take_profit_price, entry))
        rows.append(("Take-profit", f"{fmt_price(risk.take_profit_price)} ({distance})"))
    if (
        entry
        and risk.stop_price is not None
        and risk.take_profit_price is not None
        and entry > risk.stop_price
    ):
        reward_risk = (risk.take_profit_price - entry) / (entry - risk.stop_price)
        rows.append(("Reward / risk", f"{reward_risk:.1f}R"))
    if risk.position_value:
        rows.append(("Position value", fmt_money(risk.position_value)))
    if risk.risk_amount:
        rows.append(("Capital at risk", fmt_money(risk.risk_amount)))
    rows.append(("Risk score", f"{risk.risk_score:.0f} / 100"))
    if risk.checks:
        failed = [name for name, ok in risk.checks.items() if not ok]
        summary = f"{len(risk.checks) - len(failed)}/{len(risk.checks)} passed"
        if failed:
            summary += f"; failed: {', '.join(failed)}"
        rows.append(("Checks", summary))
    return rows


def _order_rows(order: Order) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("Status", order.status.value),
        ("Order", f"{order.side.value} {order.quantity:,} {order.symbol} ({order.type.value})"),
    ]
    if order.reference_price is not None:
        rows.append(("Reference price", fmt_price(order.reference_price)))
    if order.fill_price is not None:
        rows.append(("Fill price", fmt_price(order.fill_price)))
        rows.append(("Notional", fmt_money(order.notional)))
    if order.is_filled:
        rows.append(("Commission", fmt_money(order.commission)))
        rows.append(("Slippage cost", fmt_money(order.slippage)))
    if order.realized_pnl is not None:
        rows.append(("Realized P&L", fmt_money(order.realized_pnl, signed=True)))
    if order.stop_price is not None:
        rows.append(("Stop-loss attached", fmt_price(order.stop_price)))
    if order.take_profit_price is not None:
        rows.append(("Take-profit attached", fmt_price(order.take_profit_price)))
    rows.append(("Requested", fmt_ts(order.requested_at)))
    if order.filled_at is not None:
        rows.append(("Filled", fmt_ts(order.filled_at)))
    if order.reason:
        rows.append(("Reason", _truncate(order.reason)))
    if order.reject_reason:
        rows.append(("Rejected because", order.reject_reason))
    rows.append(("Order id", order.id))
    return rows


def _order_html(order: Order) -> _Safe:
    rows: list[tuple[str, object]] = []
    for label, value in _order_rows(order):
        if label == "Realized P&L":
            rows.append((label, _colored(value, _tone(order.realized_pnl, dead_zone=0.0))))
        elif label == "Status":
            color = GREEN if order.is_filled else AMBER
            rows.append((label, _badge(value, color)))
        else:
            rows.append((label, value))
    return _kv_table(rows)


def _decision_blocks(
    decision: Decision, settings: Settings
) -> tuple[list[str], list[str], list[str]]:
    """Summary, agent-signal table and risk panel shared by signal and order e-mails.

    Returns ``(html_parts, text_lines, preheader_bits)``.
    """
    thresholds = (
        f"BUY at score ≥ {settings.buy_threshold:+.2f}, SELL at ≤ {settings.sell_threshold:+.2f}"
    )
    html_parts: list[str] = []
    text: list[str] = []
    if decision.summary:
        html_parts.append(_section("Why", _paragraph(decision.summary)))
        text += [*_text_section("Why"), *_wrap(decision.summary)]
    if decision.signals:
        html_parts.append(
            _section(
                "Agent votes",
                _signals_html(decision.signals),
                _Safe(
                    f'<p style="margin:8px 0 0;font-size:12px;color:{_MUTED};">'
                    f"Consensus score = Σ weight·signal·confidence / Σ weight·confidence; "
                    f"{_e(thresholds)}.</p>"
                ),
            )
        )
        text += [
            *_text_section("Agent votes"),
            *_text_table(["Agent", "Signal", "Conf.", "Rationale"], _signal_rows(decision.signals)),
            f"  ({thresholds})",
        ]
    risk_rows = _risk_rows(decision.risk)
    html_risk_rows: list[tuple[str, object]] = [
        (label, _colored(value, GREEN if decision.risk.approved else RED))
        if label == "Verdict"
        else (label, value)
        for label, value in risk_rows
    ]
    html_parts.append(
        _section("Risk gate", _kv_table(html_risk_rows), _bullets(decision.risk.reasons))
    )
    text += [*_text_section("Risk gate"), *_text_kv(risk_rows)]
    text += [f"  - {reason}" for reason in decision.risk.reasons]
    preheader = [
        f"score {fmt_signed(decision.score)}",
        f"conf {fmt_confidence(decision.confidence)}",
    ]
    return html_parts, text, preheader


def _wrap(text: str, width: int = 78) -> list[str]:
    """Wrap free text for the plain-text body (two-space indent, paragraphs preserved)."""
    lines: list[str] = []
    for paragraph in text.strip().splitlines():
        lines += textwrap.wrap(paragraph, width=width, initial_indent="  ", subsequent_indent="  ")
    return lines


# =============================================================================== renderers


def render_decision(decision: Decision, settings: Settings) -> RenderedEmail:
    """A confident BUY/SELL signal: consensus, agent votes, risk numbers, links."""
    action = decision.action.value
    accent = action_color(decision.action)
    sym = decision.symbol
    subject = (
        f"{SUBJECT_PREFIX} {action} {sym} @ {fmt_price(decision.price)}: score "
        f"{fmt_signed(decision.score)}, confidence {fmt_confidence(decision.confidence)}"
    )
    headline = f"{action} signal · {sym}"
    subheadline = f"{fmt_price(decision.price)} · {fmt_ts(decision.timestamp)}"
    risk = decision.risk
    gate = f"Approved {risk.quantity:,} sh" if risk.approved and risk.quantity else "Approved"
    stats = _stats(
        [
            ("Consensus score", fmt_signed(decision.score), _tone(decision.score)),
            ("Confidence", fmt_confidence(decision.confidence), _TEXT),
            ("Risk gate", gate if risk.approved else "Vetoed", GREEN if risk.approved else RED),
        ]
    )
    blocks, text_blocks, pre = _decision_blocks(decision, settings)
    html_body: list[str] = [stats, *blocks]
    text_lines = [
        *_text_title(headline, subheadline),
        *_text_kv(
            [
                ("Consensus score", fmt_signed(decision.score)),
                ("Confidence", fmt_confidence(decision.confidence)),
                ("Executed", "yes" if decision.executed else "no"),
            ]
        ),
        *text_blocks,
    ]
    if decision.order is not None:
        html_body.append(_section("Order", _order_html(decision.order)))
        text_lines += [*_text_section("Order"), *_text_kv(_order_rows(decision.order))]
    links = _links(settings, sym, "analyze")
    text_lines += [*_text_links(links), *_text_footer()]
    html_doc = _layout(
        kind="Signal",
        headline=headline,
        subheadline=subheadline,
        accent=accent,
        preheader=f"{action} {sym} · " + " · ".join(pre),
        body=html_body,
        links=links,
    )
    return RenderedEmail(subject, "\n".join(text_lines), html_doc)


def render_order(
    order: Order, settings: Settings, decision: Decision | None = None
) -> RenderedEmail:
    """A paper order produced by auto-trading (filled, or rejected by the broker)."""
    sym = order.symbol
    filled = order.status is OrderStatus.FILLED
    verb = "filled" if filled else order.status.value.lower()
    accent = action_color(order.side) if filled else AMBER
    trade = f"{order.side.value} {order.quantity:,} {sym}"
    if filled:
        subject = f"{SUBJECT_PREFIX} Order filled: {trade} @ {fmt_price(order.fill_price)}"
    else:
        why = f": {order.reject_reason}" if order.reject_reason else ""
        subject = f"{SUBJECT_PREFIX} Order {verb}: {trade}{why}"
    headline = f"Order {verb} · {trade}"
    subheadline = fmt_ts(order.filled_at or order.requested_at)
    if filled:
        third = (
            ("Realized P&L", fmt_money(order.realized_pnl, signed=True), _tone(order.realized_pnl))
            if order.realized_pnl is not None
            else ("Commission", fmt_money(order.commission), _TEXT)
        )
        stats = _stats(
            [
                ("Fill price", fmt_price(order.fill_price), _TEXT),
                ("Notional", fmt_money(order.notional), _TEXT),
                third,
            ]
        )
    else:
        stats = _stats(
            [
                ("Status", order.status.value, AMBER),
                ("Quantity", f"{order.quantity:,}", _TEXT),
                ("Reference", fmt_price(order.reference_price), _TEXT),
            ]
        )
    html_body: list[str] = [stats, _section("Order", _order_html(order))]
    text_lines = [
        *_text_title(headline, subheadline),
        *_text_kv(_order_rows(order)),
    ]
    if decision is not None:
        context = (
            f"{decision.action.value} decision: score {fmt_signed(decision.score)}, "
            f"confidence {fmt_confidence(decision.confidence)}"
        )
        blocks, text_blocks, _ = _decision_blocks(decision, settings)
        html_body += [_section("Decision", _paragraph(context)), *blocks]
        text_lines += [*_text_section("Decision"), f"  {context}", *text_blocks]
    links = _links(settings, sym, "trades")
    text_lines += [*_text_links(links), *_text_footer()]
    html_doc = _layout(
        kind="Paper order",
        headline=headline,
        subheadline=subheadline,
        accent=accent,
        preheader=subject.removeprefix(SUBJECT_PREFIX).strip(),
        body=html_body,
        links=links,
    )
    return RenderedEmail(subject, "\n".join(text_lines), html_doc)


_EXIT_LEVEL_NAMES = {ExitReason.STOP_LOSS: "stop-loss", ExitReason.TAKE_PROFIT: "take-profit"}


def exit_label(reason: ExitReason | None) -> str:
    """Human label for an exit reason (``"Stop-loss hit"`` / ``"Take-profit hit"``)."""
    if reason is ExitReason.STOP_LOSS:
        return "Stop-loss hit"
    if reason is ExitReason.TAKE_PROFIT:
        return "Take-profit hit"
    return "Protective exit"


def render_exit(order: Order, settings: Settings, reason: ExitReason | None) -> RenderedEmail:
    """A position closed by its stop-loss or take-profit."""
    sym = order.symbol
    label = exit_label(reason)
    if reason is ExitReason.STOP_LOSS:
        accent = RED
    elif reason is ExitReason.TAKE_PROFIT:
        accent = GREEN
    else:
        accent = SLATE
    pnl = order.realized_pnl
    pnl_text = f" (P&L {fmt_money(pnl, signed=True)})" if pnl is not None else ""
    subject = (
        f"{SUBJECT_PREFIX} {label}: sold {order.quantity:,} {sym} @ "
        f"{fmt_price(order.fill_price)}{pnl_text}"
    )
    headline = f"{label} · {sym}"
    subheadline = fmt_ts(order.filled_at or order.requested_at)
    level = _EXIT_LEVEL_NAMES.get(reason, "protective level") if reason else "protective level"
    note = (
        f"The position was closed automatically because the latest price breached its "
        f"{level}. Realized P&L is net of commissions on both the entry and the exit."
    )
    stats = _stats(
        [
            ("Exit price", fmt_price(order.fill_price), _TEXT),
            ("Quantity", f"{order.quantity:,}", _TEXT),
            ("Realized P&L", fmt_money(pnl, signed=True), _tone(pnl, dead_zone=0.0)),
        ]
    )
    html_body = [stats, _paragraph(note), _section("Exit order", _order_html(order))]
    links = _links(settings, sym, "trades")
    text_lines = [
        *_text_title(headline, subheadline),
        *_wrap(note),
        *_text_section("Exit order"),
        *_text_kv(_order_rows(order)),
        *_text_links(links),
        *_text_footer(),
    ]
    html_doc = _layout(
        kind="Protective exit",
        headline=headline,
        subheadline=subheadline,
        accent=accent,
        preheader=subject.removeprefix(SUBJECT_PREFIX).strip(),
        body=html_body,
        links=links,
    )
    return RenderedEmail(subject, "\n".join(text_lines), html_doc)


def recovery_equity(snapshot: PortfolioSnapshot, max_drawdown_halt: float) -> float:
    """Equity at which drawdown falls back below the halt limit: ``peak × (1 − limit)``."""
    return snapshot.peak_equity * (1.0 - max_drawdown_halt)


def render_drawdown(snapshot: PortfolioSnapshot, settings: Settings) -> RenderedEmail:
    """The account drawdown breached ``MAX_DRAWDOWN_HALT``: the circuit breaker is active."""
    limit_pct = settings.max_drawdown_halt * 100.0
    dd = snapshot.drawdown_pct
    recovery = recovery_equity(snapshot, settings.max_drawdown_halt)
    gain_needed = (recovery / snapshot.equity - 1.0) * 100.0 if snapshot.equity > 0 else None
    subject = (
        f"{SUBJECT_PREFIX} Drawdown {dd:.1f}% breached the {limit_pct:.1f}% limit: new BUYs halted"
    )
    headline = f"Drawdown {dd:.1f}% · circuit breaker active"
    subheadline = f"Equity {fmt_money(snapshot.equity)} · {fmt_ts(snapshot.updated_at)}"
    recovery_text = fmt_money(recovery)
    if gain_needed is not None and gain_needed > 0:
        recovery_text += f" ({fmt_pct(gain_needed)} from here)"
    rows: list[tuple[str, str]] = [
        ("Drawdown from peak", fmt_pct(dd, signed=False)),
        ("Halt limit (MAX_DRAWDOWN_HALT)", fmt_pct(limit_pct, signed=False, digits=1)),
        ("Equity", fmt_money(snapshot.equity)),
        ("Peak equity", fmt_money(snapshot.peak_equity)),
        ("BUYs resume above", recovery_text),
        ("Cash", fmt_money(snapshot.cash)),
        ("Positions value", fmt_money(snapshot.positions_value)),
        ("Gross exposure", fmt_pct(snapshot.gross_exposure_pct, signed=False, digits=1)),
        ("Realized P&L", fmt_money(snapshot.realized_pnl, signed=True)),
        ("Unrealized P&L", fmt_money(snapshot.unrealized_pnl, signed=True)),
        ("Total return", fmt_pct(snapshot.total_return_pct)),
    ]
    note = (
        "The risk manager blocks new BUY orders while the drawdown from peak equity is at or "
        "above the limit. Open positions keep their stop-loss / take-profit protection and "
        "SELL signals are still executed."
    )
    position_headers = ["Symbol", "Qty", "Avg cost", "Last", "Unrealized", "Stop", "Target"]
    position_rows = [
        [
            p.symbol,
            f"{p.quantity:,}",
            fmt_price(p.avg_price),
            fmt_price(p.market_price),
            f"{fmt_money(p.unrealized_pnl, signed=True)} ({fmt_pct(p.unrealized_pnl_pct)})",
            fmt_price(p.stop_price),
            fmt_price(p.take_profit_price),
        ]
        for p in snapshot.positions
    ]
    html_body: list[str] = [
        _stats(
            [
                ("Drawdown", fmt_pct(dd, signed=False, digits=1), RED),
                ("Equity", fmt_money(snapshot.equity), _TEXT),
                ("Peak", fmt_money(snapshot.peak_equity), _TEXT),
            ]
        ),
        _paragraph(note),
        _section("Account", _kv_table(rows)),
    ]
    text_lines = [
        *_text_title(headline, subheadline),
        *_wrap(note),
        *_text_section("Account"),
        *_text_kv(rows),
    ]
    if position_rows:
        html_rows: list[list[object]] = [
            [*row[:4], _colored(row[4], _tone(p.unrealized_pnl, dead_zone=0.0)), *row[5:]]
            for row, p in zip(position_rows, snapshot.positions, strict=True)
        ]
        html_body.append(
            _section(
                "Open positions",
                _data_table(position_headers, html_rows, numeric=(1, 2, 3, 4, 5, 6)),
            )
        )
        text_lines += [
            *_text_section("Open positions"),
            *_text_table(position_headers, position_rows),
        ]
    links = _links(settings, None, "portfolio")
    text_lines += [*_text_links(links), *_text_footer()]
    html_doc = _layout(
        kind="Risk",
        headline=headline,
        subheadline=subheadline,
        accent=RED,
        preheader=subject.removeprefix(SUBJECT_PREFIX).strip(),
        body=html_body,
        links=links,
    )
    return RenderedEmail(subject, "\n".join(text_lines), html_doc)


def render_price_move(
    symbol: str, pct: float, settings: Settings, price: float | None = None
) -> RenderedEmail:
    """A 1-day move of at least ``ALERT_PRICE_MOVE_PCT`` percent (``pct`` is a percentage)."""
    direction = "up" if pct > 0 else "down"
    accent = GREEN if pct > 0 else RED
    subject = f"{SUBJECT_PREFIX} {symbol} {direction} {abs(pct):.2f}% on the day"
    if price is not None:
        subject += f" ({fmt_price(price)})"
    headline = f"{symbol} {direction} {abs(pct):.2f}%"
    threshold = f"±{settings.alert_price_move_pct:.2f}%"
    subheadline = f"1-day move beyond the {threshold} alert threshold"
    rows: list[tuple[str, str]] = [
        ("1-day change", fmt_pct(pct)),
        ("Alert threshold (ALERT_PRICE_MOVE_PCT)", threshold),
    ]
    if price is not None:
        rows.insert(0, ("Last price", fmt_price(price)))
    note = (
        "The 1-day change compares the latest price with the previous session's close. "
        "Large moves often follow news; check the sentiment agent's headlines before acting."
    )
    html_body = [
        _stats(
            [
                ("1-day change", fmt_pct(pct), accent),
                ("Last price", fmt_price(price), _TEXT),
                ("Threshold", threshold, _TEXT),
            ]
        ),
        _paragraph(note),
        _section("Details", _kv_table(rows)),
    ]
    links = _links(settings, symbol, "analyze")
    text_lines = [
        *_text_title(headline, subheadline),
        *_text_kv(rows),
        "",
        *_wrap(note),
        *_text_links(links),
        *_text_footer(),
    ]
    html_doc = _layout(
        kind="Price move",
        headline=headline,
        subheadline=subheadline,
        accent=accent,
        preheader=subject.removeprefix(SUBJECT_PREFIX).strip(),
        body=html_body,
        links=links,
    )
    return RenderedEmail(subject, "\n".join(text_lines), html_doc)


def render_test(settings: Settings, channel_info: Mapping[str, object]) -> RenderedEmail:
    """The ``POST /alerts/test`` message: confirms delivery and echoes the (non-secret) config."""
    subject = f"{SUBJECT_PREFIX} Test alert: e-mail delivery works"
    headline = "Test alert"
    subheadline = "If you can read this, SMTP delivery is configured correctly."
    rows: list[tuple[str, str]] = [
        ("SMTP server", f"{channel_info.get('host', '')}:{channel_info.get('port', '')}"),
        ("Security", str(channel_info.get("security", ""))),
        ("Authentication", "yes" if channel_info.get("auth") else "no"),
        ("Sender", str(channel_info.get("sender", ""))),
        ("Recipients", str(channel_info.get("recipients", ""))),
        ("Min. signal confidence", fmt_confidence(settings.alert_min_confidence)),
        ("Price-move threshold", f"±{settings.alert_price_move_pct:.2f}%"),
        ("Cooldown per rule & symbol", f"{settings.alert_cooldown_minutes} min"),
    ]
    note = (
        "You will receive alerts for confident BUY/SELL signals, executed paper orders, "
        "stop-loss / take-profit exits, drawdown circuit-breaker breaches and large daily "
        "price moves. Repeats of the same alert are suppressed during the cooldown."
    )
    links = _links(settings, None, "alerts")
    html_doc = _layout(
        kind="Test",
        headline=headline,
        subheadline=subheadline,
        accent=BLUE,
        preheader=subheadline,
        body=[_paragraph(note), _section("Configuration", _kv_table(rows))],
        links=links,
    )
    text_lines = [
        *_text_title(headline, subheadline),
        *_wrap(note),
        *_text_section("Configuration"),
        *_text_kv(rows),
        *_text_links(links),
        *_text_footer(),
    ]
    return RenderedEmail(subject, "\n".join(text_lines), html_doc)


__all__ = [
    "AMBER",
    "BLUE",
    "DISCLAIMER",
    "GREEN",
    "RED",
    "SLATE",
    "SUBJECT_PREFIX",
    "RenderedEmail",
    "action_color",
    "dashboard_link",
    "exit_label",
    "fmt_confidence",
    "fmt_money",
    "fmt_pct",
    "fmt_price",
    "fmt_signed",
    "fmt_ts",
    "recovery_equity",
    "render_decision",
    "render_drawdown",
    "render_exit",
    "render_order",
    "render_price_move",
    "render_test",
]
