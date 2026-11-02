"""
sentiment_agent.py — News sentiment analysis for trading signals
Uses keyword-based scoring (extend with a real news API for production)
"""
import logging
import re
from typing import List

logger = logging.getLogger("sentiment_agent")

BULLISH_KEYWORDS = [
    "surge", "rally", "breakout", "beat", "profit", "growth", "upgrade",
    "buy", "bullish", "record", "high", "strong", "positive", "outperform",
    "exceed", "gain", "rise", "jump", "soar", "boom", "recovery"
]

BEARISH_KEYWORDS = [
    "crash", "drop", "miss", "loss", "downgrade", "sell", "bearish",
    "decline", "fall", "plunge", "weak", "negative", "underperform",
    "layoff", "lawsuit", "fraud", "investigation", "recession", "risk",
    "debt", "bankruptcy", "warning", "concern"
]

# Sample mock headlines (replace with real news API like NewsAPI, Alpha Vantage)
MOCK_HEADLINES = {
    "AAPL": [
        "Apple reports record iPhone sales in Q3",
        "Apple faces antitrust investigation in EU",
        "Apple Vision Pro exceeds analyst expectations",
    ],
    "TSLA": [
        "Tesla cuts prices amid increased competition",
        "Tesla Q3 deliveries beat estimates by 5%",
        "Elon Musk sells $1B in Tesla shares",
    ],
    "MSFT": [
        "Microsoft Azure revenue grows 29% year-over-year",
        "Microsoft Copilot adoption surging in enterprise",
        "Microsoft increases dividend by 10%",
    ],
    "GOOGL": [
        "Google ad revenue rebounds strongly",
        "Google faces new antitrust ruling",
        "Gemini AI usage growing across Google products",
    ],
}


def analyze_sentiment(symbol: str, headlines: List[str] = None) -> dict:
    """Analyze news sentiment for a symbol."""
    if headlines is None:
        headlines = MOCK_HEADLINES.get(symbol, [
            f"{symbol} stock moves on market conditions",
            f"Analysts watch {symbol} closely",
        ])

    scores = []
    analyzed = []
    for headline in headlines:
        text = headline.lower()
        bull = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
        bear = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
        score = (bull - bear)
        scores.append(score)
        analyzed.append({
            "headline": headline,
            "score": score,
            "sentiment": "positive" if score > 0 else "negative" if score < 0 else "neutral"
        })

    avg_score = sum(scores) / len(scores) if scores else 0
    if avg_score > 0.5:
        overall = "bullish"
    elif avg_score < -0.5:
        overall = "bearish"
    else:
        overall = "neutral"

    result = {
        "symbol": symbol,
        "overall_sentiment": overall,
        "sentiment_score": round(avg_score, 2),
        "headlines_analyzed": len(headlines),
        "details": analyzed,
        "source": "mock" if symbol not in MOCK_HEADLINES else "curated",
        "note": "Connect a real news API (NewsAPI, Alpha Vantage) for live sentiment",
    }
    logger.info(f"[SentimentAgent] {symbol}: {overall} (score={avg_score:.2f})")
    return result
