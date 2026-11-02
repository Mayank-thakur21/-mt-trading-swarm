"""MT Trading Swarm — a multi-agent paper-trading system.

A *swarm* of small, single-purpose agents votes on every symbol:

* ``TrendAnalyzerAgent`` reads price action (moving averages, MACD, Wilder RSI, ADX, momentum),
* ``SentimentAgent`` reads news headlines (Claude by default, with a lexicon fallback),
* ``RiskManagerAgent`` sizes the position from ATR stops and vetoes unsafe trades,
* ``ExecutionAgent`` turns approved decisions into paper orders on a SQLite-backed broker.

Nothing here places real orders: every fill is simulated. This is not financial advice.
"""

__version__ = "2.0.0"

__all__ = ["__version__"]
