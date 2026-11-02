"""
trend_agent.py — Analyzes price trends using technical indicators
"""
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("trend_agent")


def analyze_trend(df: pd.DataFrame, symbol: str) -> dict:
    """
    Analyze price trend using SMA, EMA, RSI, MACD.
    df must have columns: Open, High, Low, Close, Volume
    """
    if df.empty or len(df) < 26:
        return {"symbol": symbol, "trend": "unknown", "reason": "Not enough data"}

    close = df["Close"]

    # ── Moving Averages ───────────────────────────────────────────────────────
    sma_20 = close.rolling(20).mean().iloc[-1]
    sma_50 = close.rolling(min(50, len(close))).mean().iloc[-1]
    ema_12 = close.ewm(span=12).mean().iloc[-1]
    ema_26 = close.ewm(span=26).mean().iloc[-1]
    current = close.iloc[-1]

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_line = ema_12 - ema_26
    signal_line = close.ewm(span=9).mean().iloc[-1]
    macd_hist = macd_line - signal_line

    # ── Determine trend ───────────────────────────────────────────────────────
    bullish_signals = 0
    bearish_signals = 0

    if current > sma_20: bullish_signals += 1
    else: bearish_signals += 1

    if current > sma_50: bullish_signals += 1
    else: bearish_signals += 1

    if macd_hist > 0: bullish_signals += 1
    else: bearish_signals += 1

    if rsi < 30: bullish_signals += 2  # oversold = potential bounce
    elif rsi > 70: bearish_signals += 2  # overbought

    if bullish_signals > bearish_signals:
        trend = "bullish"
        strength = "strong" if bullish_signals >= 4 else "moderate"
    elif bearish_signals > bullish_signals:
        trend = "bearish"
        strength = "strong" if bearish_signals >= 4 else "moderate"
    else:
        trend = "neutral"
        strength = "weak"

    pct_change_1d = ((current - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) > 1 else 0
    pct_change_5d = ((current - close.iloc[-6]) / close.iloc[-6] * 100) if len(close) > 5 else 0

    result = {
        "symbol": symbol,
        "trend": trend,
        "strength": strength,
        "current_price": round(float(current), 4),
        "sma_20": round(float(sma_20), 4),
        "sma_50": round(float(sma_50), 4),
        "rsi": round(float(rsi), 2),
        "macd_histogram": round(float(macd_hist), 4),
        "pct_change_1d": round(float(pct_change_1d), 2),
        "pct_change_5d": round(float(pct_change_5d), 2),
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals,
    }
    logger.info(f"[TrendAgent] {symbol}: {trend} ({strength}) | RSI={rsi:.1f} | Price={current:.2f}")
    return result
