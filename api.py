"""
api.py — FastAPI REST API for the MT Trading Swarm
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime

from swarm import run_swarm, run_swarm_batch
from agents.execution_agent import get_portfolio_summary, reset_portfolio, buy, sell, portfolio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingAPI")

app = FastAPI(title="MT Trading Swarm API", version="1.0.0",
              description="AI-powered multi-agent trading system (paper trading)")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    symbol: str
    auto_trade: bool = False

class BatchRequest(BaseModel):
    symbols: List[str]
    auto_trade: bool = False

class ManualTradeRequest(BaseModel):
    symbol: str
    action: str  # BUY or SELL
    units: Optional[int] = None
    price: float


@app.get("/")
def root():
    return {"message": "MT Trading Swarm API v1.0", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """Run full swarm analysis for a single symbol."""
    result = run_swarm(req.symbol.upper(), req.auto_trade)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result

@app.post("/analyze/batch")
def analyze_batch(req: BatchRequest):
    """Run swarm analysis for multiple symbols (max 10)."""
    if len(req.symbols) > 10:
        raise HTTPException(status_code=400, detail="Max 10 symbols per batch")
    results = run_swarm_batch([s.upper() for s in req.symbols], req.auto_trade)
    return {"results": results, "count": len(results)}

@app.get("/portfolio")
def get_portfolio():
    """Get current paper trading portfolio."""
    return get_portfolio_summary()

@app.post("/portfolio/reset")
def portfolio_reset(capital: float = 100000.0):
    """Reset paper portfolio to initial state."""
    return reset_portfolio(capital)

@app.post("/trade")
def manual_trade(req: ManualTradeRequest):
    """Manually execute a paper trade."""
    if req.action.upper() == "BUY":
        units = req.units or 1
        return buy(req.symbol.upper(), units, req.price, "Manual trade")
    elif req.action.upper() == "SELL":
        return sell(req.symbol.upper(), req.units, req.price, "Manual trade")
    else:
        raise HTTPException(status_code=400, detail="Action must be BUY or SELL")

@app.get("/trades")
def get_trades(limit: int = 50):
    """Get recent trade history."""
    return {"trades": portfolio["trades"][:limit], "total": len(portfolio["trades"])}

@app.get("/watchlist")
def get_watchlist():
    """Suggested watchlist symbols."""
    return {"symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX"]}
