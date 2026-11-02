"""
risk_agent.py — Portfolio risk management and position sizing
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("risk_agent")

MAX_RISK_PER_TRADE = 0.02   # 2% of portfolio per trade
MAX_PORTFOLIO_RISK = 0.10   # 10% total portfolio risk
STOP_LOSS_PCT = 0.05        # 5% stop loss


def assess_risk(
    symbol: str,
    current_price: float,
    portfolio_value: float,
    existing_positions: dict,
    trend_signal: str,
    rsi: float,
) -> dict:
    """Assess risk and return position sizing recommendation."""

    # ── Portfolio exposure check ──────────────────────────────────────────────
    total_exposure = sum(p["value"] for p in existing_positions.values())
    exposure_pct = total_exposure / portfolio_value if portfolio_value > 0 else 0

    # ── Position sizing (Kelly-lite) ─────────────────────────────────────────
    risk_amount = portfolio_value * MAX_RISK_PER_TRADE
    stop_loss_price = current_price * (1 - STOP_LOSS_PCT)
    risk_per_unit = current_price - stop_loss_price
    max_units = int(risk_amount / risk_per_unit) if risk_per_unit > 0 else 0
    position_value = max_units * current_price

    # ── Risk score (0-100, higher = riskier) ─────────────────────────────────
    risk_score = 50  # base
    if rsi > 75: risk_score += 20
    elif rsi < 25: risk_score -= 10
    if exposure_pct > 0.7: risk_score += 20
    if trend_signal == "bearish": risk_score += 15
    elif trend_signal == "bullish": risk_score -= 10
    if symbol in existing_positions: risk_score += 10  # concentration
    risk_score = max(0, min(100, risk_score))

    # ── Decision ──────────────────────────────────────────────────────────────
    if risk_score >= 75 or exposure_pct >= MAX_PORTFOLIO_RISK:
        action = "avoid"
        reason = "Risk too high"
    elif risk_score >= 50:
        action = "reduce"
        reason = "Moderate risk — reduce position size"
        max_units = max(1, max_units // 2)
        position_value = max_units * current_price
    else:
        action = "proceed"
        reason = "Risk acceptable"

    result = {
        "symbol": symbol,
        "risk_score": risk_score,
        "action": action,
        "reason": reason,
        "max_units": max_units,
        "position_value": round(position_value, 2),
        "stop_loss_price": round(stop_loss_price, 4),
        "portfolio_exposure_pct": round(exposure_pct * 100, 1),
        "risk_per_trade_pct": round(MAX_RISK_PER_TRADE * 100, 1),
    }
    logger.info(f"[RiskAgent] {symbol}: score={risk_score} action={action} units={max_units}")
    return result
