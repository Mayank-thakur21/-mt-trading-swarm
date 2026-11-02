"""
swarm.py — AI Trading Swarm Orchestrator
Coordinates all agents to produce a final trade recommendation.
"""
import logging
from typing import List, Optional
import yfinance as yf
import pandas as pd

from agents.trend_agent import analyze_trend
from agents.risk_agent import assess_risk
from agents.sentiment_agent import analyze_sentiment
from agents.execution_agent import buy, sell, get_portfolio_summary, portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Swarm")


def fetch_price_data(symbol: str, period: str = "3mo", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch historical price data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            logger.warning(f"No data for {symbol}")
            return None
        return df
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None


def run_swarm(symbol: str, auto_trade: bool = False) -> dict:
    """
    Run the full swarm analysis for a symbol.
    Returns combined signals from all agents + final recommendation.
    """
    logger.info(f"[Swarm] Starting analysis for {symbol}")

    # ── Step 1: Fetch price data ──────────────────────────────────────────────
    df = fetch_price_data(symbol)
    if df is None:
        return {"symbol": symbol, "error": "Could not fetch price data"}

    current_price = float(df["Close"].iloc[-1])

    # ── Step 2: Trend Agent ───────────────────────────────────────────────────
    trend = analyze_trend(df, symbol)

    # ── Step 3: Sentiment Agent ───────────────────────────────────────────────
    sentiment = analyze_sentiment(symbol)

    # ── Step 4: Risk Agent ────────────────────────────────────────────────────
    risk = assess_risk(
        symbol=symbol,
        current_price=current_price,
        portfolio_value=get_portfolio_summary().get("total_value", 100_000),
        existing_positions=portfolio["positions"],
        trend_signal=trend["trend"],
        rsi=trend.get("rsi", 50),
    )

    # ── Step 5: Aggregate signals → Final recommendation ─────────────────────
    signals = {
        "trend": trend["trend"],
        "sentiment": sentiment["overall_sentiment"],
        "risk_action": risk["action"],
    }

    bullish_count = sum(1 for v in [trend["trend"], sentiment["overall_sentiment"]] if v == "bullish")
    bearish_count = sum(1 for v in [trend["trend"], sentiment["overall_sentiment"]] if v == "bearish")

    if bullish_count >= 2 and risk["action"] in ("proceed", "reduce"):
        recommendation = "BUY"
        confidence = "high" if bullish_count == 2 else "medium"
    elif bearish_count >= 2:
        recommendation = "SELL"
        confidence = "high" if bearish_count == 2 else "medium"
    elif risk["action"] == "avoid":
        recommendation = "HOLD"
        confidence = "low"
    else:
        recommendation = "HOLD"
        confidence = "medium"

    result = {
        "symbol": symbol,
        "current_price": current_price,
        "recommendation": recommendation,
        "confidence": confidence,
        "signals": signals,
        "trend_analysis": trend,
        "sentiment_analysis": sentiment,
        "risk_assessment": risk,
    }

    # ── Step 6: Auto-execute (paper trade) ───────────────────────────────────
    if auto_trade:
        if recommendation == "BUY" and risk["max_units"] > 0:
            trade_result = buy(
                symbol=symbol,
                units=risk["max_units"],
                price=current_price,
                reason=f"Swarm: {confidence} confidence BUY signal",
            )
            result["trade_executed"] = trade_result
        elif recommendation == "SELL" and symbol in portfolio["positions"]:
            trade_result = sell(
                symbol=symbol,
                units=None,
                price=current_price,
                reason=f"Swarm: {confidence} confidence SELL signal",
            )
            result["trade_executed"] = trade_result
        else:
            result["trade_executed"] = {"action": "HOLD", "reason": recommendation}

    logger.info(f"[Swarm] {symbol}: {recommendation} ({confidence} confidence)")
    return result


def run_swarm_batch(symbols: List[str], auto_trade: bool = False) -> List[dict]:
    """Run swarm analysis for multiple symbols."""
    return [run_swarm(s, auto_trade) for s in symbols]
