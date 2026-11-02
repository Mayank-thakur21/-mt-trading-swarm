"""Technical trend agent: votes from price action only.

The agent turns a daily OHLCV history into one directional vote. Each technical
*component* is mapped onto ``[-1, 1]`` (positive = bullish) with a smooth ``tanh``
squashing, so small moves give small votes and the output never flips on a rounding
error. Distances are normalised by volatility — price-vs-average distances in multiples of
ATR, momentum as a z-score of the 20-day return — so the same thresholds work for a $20
stock and a $2,000 one.

Components (weights renormalise over the components that are available)
-----------------------------------------------------------------------
=============  ======  ==================================================================
component      weight  definition
=============  ======  ==================================================================
``sma20``      0.10    ``tanh((close − SMA20) / (2·ATR))``
``sma50``      0.15    ``tanh((close − SMA50) / (3·ATR))``
``sma200``     0.15    ``tanh((close − SMA200) / (5·ATR))`` (needs 200 bars)
``cross``      0.15    ``tanh((SMA50 / SMA200 − 1) / 5 %)`` golden (+) / death (−) cross
``macd``       0.20    ``0.6·tanh(hist / (0.2·ATR)) + 0.4·tanh(Δhist / (0.05·ATR))`` —
                       histogram sign *and* slope (rising = strengthening momentum)
``rsi``        0.10    regime: oversold (< 30) → +0.3…+1, overbought (> 70) → −0.3…−1,
                       otherwise mild trend-following ``0.3·(RSI − 50) / 20``
``momentum``   0.15    ``tanh(z / 2)``, ``z = R₂₀ / (σ_annual·√(20/252))``
=============  ======  ==================================================================

``signal = Σ wᵢ·cᵢ / Σ wᵢ``.

Confidence
----------
``confidence = agreement × (0.5 + 0.5·strength) × history_factor`` where

* ``agreement = |Σ wᵢ·cᵢ| / Σ wᵢ·|cᵢ|`` — 1 when every component points the same way,
  → 0 when they cancel out;
* ``strength = clip((ADX − 15) / 25, 0, 1)`` — Wilder's ADX measures trend strength
  irrespective of direction (≲ 15: no trend, ≳ 40: strong trend); unknown ADX → 0.5;
* ``history_factor`` = 1 with ≥ 200 bars, 0.85 when the long-term components (SMA200 and
  the cross) are unavailable.

Every indicator value goes into ``AgentSignal.data`` (rounded) for the dashboard, together
with the per-component scores and weights so the vote is fully explainable. Scoring uses
full-precision values; rounding only affects what is displayed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final

import pandas as pd

from trading_swarm import indicators as ind
from trading_swarm.config import Settings
from trading_swarm.data.base import daily_change_pct
from trading_swarm.models import AgentSignal

logger = logging.getLogger(__name__)

#: Base weights of the components (renormalised over the available ones).
COMPONENT_WEIGHTS: Final[dict[str, float]] = {
    "sma20": 0.10,
    "sma50": 0.15,
    "sma200": 0.15,
    "cross": 0.15,
    "macd": 0.20,
    "rsi": 0.10,
    "momentum": 0.15,
}

RSI_OVERSOLD: Final = 30.0
RSI_OVERBOUGHT: Final = 70.0
MOMENTUM_WINDOW: Final = 20
#: Histogram slope is measured bar-on-bar (the conventional "ticking up/down" reading).
MACD_SLOPE_BARS: Final = 1
#: Confidence multiplier when fewer than 200 bars exist (no SMA200 / cross components).
SHORT_HISTORY_FACTOR: Final = 0.85
#: Price-distance scale (in ATRs) for each moving average: longer averages sit further away.
_SMA_SCALES: Final = (("sma20", 20, 2.0), ("sma50", 50, 3.0), ("sma200", 200, 5.0))


# =============================================================================== helpers


def _finite(value: Any) -> float | None:
    """``value`` as a float if finite, else ``None``."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _at(series: pd.Series, offset: int = 0) -> float | None:
    """``series.iloc[-1 - offset]`` as a finite float (``None`` if out of range or NaN)."""
    if len(series) <= offset:
        return None
    return _finite(series.iloc[-1 - offset])


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def rsi_component(rsi: float) -> float:
    """Map RSI onto ``[-1, 1]``: contrarian at the extremes, mildly trend-following inside.

    * RSI < 30 (oversold, a bounce is likely): ``+0.3`` rising linearly to ``+1`` at RSI 10;
    * RSI > 70 (overbought, stretched): ``−0.3`` falling linearly to ``−1`` at RSI 90;
    * otherwise ``0.3·(RSI − 50) / 20`` — RSI in the neutral band is a mild continuation
      signal (above 50 = buyers in control).
    """
    if rsi < RSI_OVERSOLD:
        return min(1.0, 0.3 + 0.7 * (RSI_OVERSOLD - rsi) / 20.0)
    if rsi > RSI_OVERBOUGHT:
        return -min(1.0, 0.3 + 0.7 * (rsi - RSI_OVERBOUGHT) / 20.0)
    return 0.3 * (rsi - 50.0) / 20.0


def adx_label(adx: float) -> str:
    """Conventional reading of an ADX value."""
    if adx >= 40:
        return "strong trend"
    if adx >= 25:
        return "trending"
    if adx >= 15:
        return "weak trend"
    return "no trend"


@dataclass(frozen=True)
class _Readings:
    """Full-precision latest indicator values (``None`` = not enough data)."""

    as_of: str
    bars: int
    close: float
    sma20: float | None
    sma50: float | None
    sma200: float | None
    ema12: float | None
    ema26: float | None
    rsi: float | None
    macd: float | None
    macd_signal: float | None
    hist: float | None
    hist_prev: float | None
    atr: float | None
    adx: float | None
    plus_di: float | None
    minus_di: float | None
    percent_b: float | None
    momentum: float | None  # 20-bar return, fraction
    change_1d_pct: float | None
    vol: float | None  # annualised realised volatility, fraction

    @classmethod
    def from_history(cls, frame: pd.DataFrame) -> _Readings:
        close = frame["Close"].astype(float)
        macd = ind.macd(close)
        adx = ind.adx(frame)
        atr = _at(ind.atr(frame))
        last_ts = frame.index[-1]
        return cls(
            as_of=last_ts.isoformat() if hasattr(last_ts, "isoformat") else str(last_ts),
            bars=len(frame),
            close=float(close.iloc[-1]),
            sma20=_at(ind.sma(close, 20)),
            sma50=_at(ind.sma(close, 50)),
            sma200=_at(ind.sma(close, 200)),
            ema12=_at(ind.ema(close, 12)),
            ema26=_at(ind.ema(close, 26)),
            rsi=_at(ind.rsi(close)),
            macd=_at(macd["macd"]),
            macd_signal=_at(macd["signal"]),
            hist=_at(macd["hist"]),
            hist_prev=_at(macd["hist"], MACD_SLOPE_BARS),
            atr=atr if atr is not None and atr > 0 else None,
            adx=_at(adx["adx"]),
            plus_di=_at(adx["plus_di"]),
            minus_di=_at(adx["minus_di"]),
            percent_b=_at(ind.bollinger(close)["percent_b"]),
            momentum=_at(ind.pct_change(close, MOMENTUM_WINDOW)),
            change_1d_pct=_finite(daily_change_pct(frame)),
            vol=_at(ind.realized_volatility(close, MOMENTUM_WINDOW)),
        )

    def display(self) -> dict[str, Any]:
        """Rounded values for ``AgentSignal.data`` (and the dashboard)."""
        return {
            "as_of": self.as_of,
            "bars": self.bars,
            "close": round(self.close, 4),
            "sma20": _round(self.sma20, 4),
            "sma50": _round(self.sma50, 4),
            "sma200": _round(self.sma200, 4),
            "ema12": _round(self.ema12, 4),
            "ema26": _round(self.ema26, 4),
            "rsi": _round(self.rsi, 2),
            "macd": _round(self.macd, 4),
            "macd_signal": _round(self.macd_signal, 4),
            "macd_hist": _round(self.hist, 4),
            "macd_hist_prev": _round(self.hist_prev, 4),
            "atr": _round(self.atr, 4),
            "atr_pct": _round(self.atr / self.close * 100.0, 3) if self.atr else None,
            "adx": _round(self.adx, 2),
            "plus_di": _round(self.plus_di, 2),
            "minus_di": _round(self.minus_di, 2),
            "bollinger_pct_b": _round(self.percent_b, 4),
            "momentum_20d_pct": _round(
                self.momentum * 100.0 if self.momentum is not None else None, 3
            ),
            "change_1d_pct": _round(self.change_1d_pct, 3),
            "realized_vol": _round(self.vol, 4),
        }


# =============================================================================== agent


class TrendAnalyzerAgent:
    """Votes from price action only (see module docstring for the formulas)."""

    name = "trend"
    #: Minimum number of daily bars required for a non-neutral vote.
    MIN_BARS = 60

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def analyze(self, symbol: str, history: pd.DataFrame) -> AgentSignal:
        """Trend vote for ``symbol`` from ``history`` (no look-ahead beyond its last bar).

        ``history`` is the canonical OHLCV frame; every indicator is causal, so slicing the
        frame bar by bar (as the backtester does) reproduces exactly the live vote.
        """
        frame = history.dropna(subset=["Close"]) if history is not None else None
        bars = 0 if frame is None else len(frame)
        if frame is None or bars < self.MIN_BARS:
            logger.debug("trend %s: insufficient data (%d < %d bars)", symbol, bars, self.MIN_BARS)
            return AgentSignal.neutral(
                self.name, "insufficient data", bars=bars, min_bars=self.MIN_BARS
            )

        readings = _Readings.from_history(frame)
        components, notes = self._components(readings)
        weights = {key: COMPONENT_WEIGHTS[key] for key in components}
        total_weight = sum(weights.values())
        weighted = sum(weights[k] * v for k, v in components.items())
        magnitude = sum(weights[k] * abs(v) for k, v in components.items())

        signal = _clip(weighted / total_weight)
        agreement = abs(weighted) / magnitude if magnitude > 0 else 0.0
        strength = 0.5 if readings.adx is None else _clip((readings.adx - 15.0) / 25.0, 0.0, 1.0)
        history_factor = 1.0 if "sma200" in components else SHORT_HISTORY_FACTOR
        confidence = _clip(agreement * (0.5 + 0.5 * strength) * history_factor, 0.0, 1.0)

        if readings.adx is not None:
            notes.append(f"ADX {readings.adx:.0f} ({adx_label(readings.adx)})")
        rationale = "; ".join(notes)
        data = {
            **readings.display(),
            "components": {k: round(v, 4) for k, v in components.items()},
            "weights": {k: round(w / total_weight, 4) for k, w in weights.items()},
            "agreement": round(agreement, 4),
            "trend_strength": round(strength, 4),
        }
        logger.debug("trend %s: %+.3f (conf %.3f): %s", symbol, signal, confidence, rationale)
        return AgentSignal(
            agent=self.name,
            signal=round(signal, 4),
            confidence=round(confidence, 4),
            rationale=rationale,
            data=data,
        )

    # ------------------------------------------------------------------ components
    @staticmethod
    def _components(r: _Readings) -> tuple[dict[str, float], list[str]]:
        """Per-component scores in ``[-1, 1]`` plus the rationale clauses, in fixed order."""
        # Volatility unit for distance normalisation: ATR, or 2 % of price if unknown.
        unit = r.atr if r.atr is not None else 0.02 * r.close
        scores: dict[str, float] = {}
        notes: list[str] = []

        above: list[str] = []
        below: list[str] = []
        for key, window, scale in _SMA_SCALES:
            average = getattr(r, key)
            if average is None:
                continue
            scores[key] = math.tanh((r.close - average) / (scale * unit))
            (above if r.close >= average else below).append(str(window))
        if above:
            notes.append("price above SMA" + "/".join(above))
        if below:
            notes.append(("below SMA" if above else "price below SMA") + "/".join(below))
        if above and below:  # merge "price above SMA20" + "below SMA50/200" into one clause
            notes[-2:] = [f"{notes[-2]}, {notes[-1]}"]

        if r.sma50 is not None and r.sma200 is not None and r.sma200 > 0:
            scores["cross"] = math.tanh((r.sma50 / r.sma200 - 1.0) / 0.05)
            notes.append("golden cross" if r.sma50 >= r.sma200 else "death cross")

        if r.hist is not None:
            level = math.tanh(r.hist / (0.2 * unit))
            slope = (
                0.0 if r.hist_prev is None else math.tanh((r.hist - r.hist_prev) / (0.05 * unit))
            )
            scores["macd"] = _clip(0.6 * level + 0.4 * slope)
            direction = ""
            if r.hist_prev is not None:
                direction = ", rising" if r.hist >= r.hist_prev else ", falling"
            notes.append(f"MACD hist {'positive' if r.hist >= 0 else 'negative'}{direction}")

        if r.rsi is not None:
            scores["rsi"] = rsi_component(r.rsi)
            regime = ""
            if r.rsi < RSI_OVERSOLD:
                regime = " oversold"
            elif r.rsi > RSI_OVERBOUGHT:
                regime = " overbought"
            notes.append(f"RSI {r.rsi:.1f}{regime}")

        if r.momentum is not None:
            # z-score of the 20-day return against the volatility expected over 20 days.
            horizon_sigma = (
                r.vol * math.sqrt(MOMENTUM_WINDOW / ind.TRADING_DAYS_PER_YEAR)
                if r.vol is not None and r.vol > 0
                else 0.05
            )
            scores["momentum"] = math.tanh(r.momentum / horizon_sigma / 2.0)
            notes.append(f"20d momentum {r.momentum * 100:+.1f}%")
        return scores, notes


__all__ = ["COMPONENT_WEIGHTS", "TrendAnalyzerAgent", "adx_label", "rsi_component"]
