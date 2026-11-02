"""
execution_agent.py — Paper trading execution engine
Simulates buy/sell orders and tracks portfolio P&L
"""
import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

logger = logging.getLogger("execution_agent")

# ── Paper portfolio ───────────────────────────────────────────────────────────
portfolio = {
    "cash": 100_000.0,   # Starting capital
    "positions": {},     # {symbol: {units, avg_price, value}}
    "trades": [],        # Trade history
    "initial_capital": 100_000.0,
}


def buy(symbol: str, units: int, price: float, reason: str = "") -> dict:
    """Execute a paper buy order."""
    cost = units * price
    commission = round(cost * 0.001, 2)  # 0.1% commission
    total_cost = cost + commission

    if portfolio["cash"] < total_cost:
        return {"success": False, "error": f"Insufficient cash: need ${total_cost:.2f}, have ${portfolio['cash']:.2f}"}

    portfolio["cash"] -= total_cost

    if symbol in portfolio["positions"]:
        pos = portfolio["positions"][symbol]
        total_units = pos["units"] + units
        avg_price = (pos["avg_price"] * pos["units"] + price * units) / total_units
        portfolio["positions"][symbol] = {"units": total_units, "avg_price": avg_price, "value": total_units * price}
    else:
        portfolio["positions"][symbol] = {"units": units, "avg_price": price, "value": units * price}

    trade = {
        "id": str(uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "action": "BUY",
        "symbol": symbol,
        "units": units,
        "price": price,
        "total": round(total_cost, 2),
        "commission": commission,
        "reason": reason,
    }
    portfolio["trades"].insert(0, trade)
    logger.info(f"[ExecutionAgent] BUY {units}x{symbol} @ ${price:.2f} = ${total_cost:.2f}")
    return {"success": True, "trade": trade, "cash_remaining": round(portfolio["cash"], 2)}


def sell(symbol: str, units: Optional[int], price: float, reason: str = "") -> dict:
    """Execute a paper sell order."""
    if symbol not in portfolio["positions"]:
        return {"success": False, "error": f"No position in {symbol}"}

    pos = portfolio["positions"][symbol]
    units = units or pos["units"]  # sell all if not specified
    units = min(units, pos["units"])

    proceeds = units * price
    commission = round(proceeds * 0.001, 2)
    net_proceeds = proceeds - commission
    pnl = round((price - pos["avg_price"]) * units, 2)
    pnl_pct = round((price - pos["avg_price"]) / pos["avg_price"] * 100, 2)

    portfolio["cash"] += net_proceeds
    remaining = pos["units"] - units
    if remaining <= 0:
        del portfolio["positions"][symbol]
    else:
        portfolio["positions"][symbol]["units"] = remaining
        portfolio["positions"][symbol]["value"] = remaining * price

    trade = {
        "id": str(uuid4())[:8],
        "timestamp": datetime.utcnow().isoformat(),
        "action": "SELL",
        "symbol": symbol,
        "units": units,
        "price": price,
        "total": round(net_proceeds, 2),
        "commission": commission,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "reason": reason,
    }
    portfolio["trades"].insert(0, trade)
    logger.info(f"[ExecutionAgent] SELL {units}x{symbol} @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.1f}%)")
    return {"success": True, "trade": trade, "pnl": pnl, "pnl_pct": pnl_pct}


def get_portfolio_summary(current_prices: dict = None) -> dict:
    """Return current portfolio state with unrealized P&L."""
    current_prices = current_prices or {}
    positions_with_pnl = {}
    total_unrealized = 0

    for sym, pos in portfolio["positions"].items():
        cur_price = current_prices.get(sym, pos["avg_price"])
        cur_value = pos["units"] * cur_price
        unrealized = cur_value - (pos["avg_price"] * pos["units"])
        unrealized_pct = (cur_price - pos["avg_price"]) / pos["avg_price"] * 100
        total_unrealized += unrealized
        positions_with_pnl[sym] = {
            **pos,
            "current_price": cur_price,
            "current_value": round(cur_value, 2),
            "unrealized_pnl": round(unrealized, 2),
            "unrealized_pct": round(unrealized_pct, 2),
        }

    total_value = portfolio["cash"] + sum(p["current_value"] for p in positions_with_pnl.values())
    total_return = total_value - portfolio["initial_capital"]
    total_return_pct = total_return / portfolio["initial_capital"] * 100

    realized_pnl = sum(t.get("pnl", 0) for t in portfolio["trades"] if t["action"] == "SELL")

    return {
        "cash": round(portfolio["cash"], 2),
        "total_value": round(total_value, 2),
        "initial_capital": portfolio["initial_capital"],
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "unrealized_pnl": round(total_unrealized, 2),
        "realized_pnl": round(realized_pnl, 2),
        "positions": positions_with_pnl,
        "num_trades": len(portfolio["trades"]),
        "recent_trades": portfolio["trades"][:10],
    }


def reset_portfolio(initial_capital: float = 100_000.0):
    """Reset portfolio to initial state."""
    portfolio["cash"] = initial_capital
    portfolio["positions"] = {}
    portfolio["trades"] = []
    portfolio["initial_capital"] = initial_capital
    return {"message": f"Portfolio reset with ${initial_capital:,.2f}"}
