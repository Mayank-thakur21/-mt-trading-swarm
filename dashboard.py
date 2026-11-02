"""
dashboard.py — Streamlit trading dashboard for MT Trading Swarm
"""
import streamlit as st
import requests
import pandas as pd
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MT Trading Swarm", page_icon="📈", layout="wide")

st.markdown("""
<style>
.buy-badge { background:#d4edda; color:#155724; padding:4px 12px; border-radius:20px; font-weight:bold; }
.sell-badge { background:#f8d7da; color:#721c24; padding:4px 12px; border-radius:20px; font-weight:bold; }
.hold-badge { background:#fff3cd; color:#856404; padding:4px 12px; border-radius:20px; font-weight:bold; }
.metric-pos { color: #28a745; }
.metric-neg { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


def api(path, method="GET", data=None):
    try:
        if method == "POST":
            r = requests.post(f"{API_URL}{path}", json=data, timeout=60)
        else:
            r = requests.get(f"{API_URL}{path}", timeout=30)
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 MT Trading Swarm")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Dashboard", "🔍 Analyze", "💼 Portfolio", "📋 Trades", "ℹ️ About"])
    st.markdown("---")
    auto_trade = st.toggle("Auto Paper Trade", value=False)
    if auto_trade:
        st.warning("Auto-trade ON: swarm will execute paper trades")


# ── Dashboard ─────────────────────────────────────────────────────────────────
if "Dashboard" in page:
    st.title("📊 Trading Swarm Dashboard")
    portfolio = api("/portfolio")
    if portfolio:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Portfolio Value", f"${portfolio['total_value']:,.2f}")
        col2.metric("Cash", f"${portfolio['cash']:,.2f}")
        ret_pct = portfolio['total_return_pct']
        col3.metric("Total Return", f"{'+' if ret_pct >= 0 else ''}{ret_pct:.2f}%", delta=f"${portfolio['total_return']:,.2f}")
        col4.metric("Trades Made", portfolio['num_trades'])

        if portfolio["positions"]:
            st.markdown("### 📁 Open Positions")
            rows = []
            for sym, pos in portfolio["positions"].items():
                rows.append({
                    "Symbol": sym,
                    "Units": pos["units"],
                    "Avg Price": f"${pos['avg_price']:.2f}",
                    "Current": f"${pos.get('current_price', pos['avg_price']):.2f}",
                    "Value": f"${pos.get('current_value', pos['value']):,.2f}",
                    "P&L": f"{'+'if pos.get('unrealized_pnl',0)>=0 else ''}{pos.get('unrealized_pnl', 0):.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No open positions. Run an analysis to start trading!")

    st.markdown("### 🚀 Quick Analyze")
    watchlist_data = api("/watchlist")
    if watchlist_data:
        selected = st.multiselect("Pick symbols", watchlist_data["symbols"], default=["AAPL", "MSFT"])
        if st.button("Run Swarm Analysis", type="primary") and selected:
            with st.spinner(f"Analyzing {len(selected)} symbols..."):
                result = api("/analyze/batch", "POST", {"symbols": selected, "auto_trade": auto_trade})
            if result:
                for r in result["results"]:
                    if "error" in r:
                        st.warning(f"{r['symbol']}: {r['error']}")
                        continue
                    rec = r["recommendation"]
                    badge = f'<span class="{rec.lower()}-badge">{rec}</span>'
                    with st.expander(f"{r['symbol']} — {r['current_price']:.2f} {badge}", expanded=False):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Trend", r["trend_analysis"]["trend"].upper())
                        c2.metric("RSI", r["trend_analysis"]["rsi"])
                        c3.metric("Sentiment", r["sentiment_analysis"]["overall_sentiment"].upper())
                        if r.get("trade_executed"):
                            st.success(f"Trade: {r['trade_executed']}")

# ── Analyze Page ──────────────────────────────────────────────────────────────
elif "Analyze" in page:
    st.title("🔍 Deep Analysis")
    symbol = st.text_input("Symbol", value="AAPL").upper()
    if st.button("Analyze", type="primary"):
        with st.spinner(f"Running swarm on {symbol}..."):
            result = api("/analyze", "POST", {"symbol": symbol, "auto_trade": auto_trade})
        if result and "error" not in result:
            rec = result["recommendation"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Price", f"${result['current_price']:.2f}")
            col2.metric("Recommendation", rec)
            col3.metric("Confidence", result["confidence"].upper())

            st.markdown("### Trend Analysis")
            t = result["trend_analysis"]
            cols = st.columns(4)
            cols[0].metric("Trend", t["trend"].upper())
            cols[1].metric("RSI", t["rsi"])
            cols[2].metric("SMA 20", t["sma_20"])
            cols[3].metric("1d Change", f"{t['pct_change_1d']}%")

            st.markdown("### Sentiment")
            s = result["sentiment_analysis"]
            st.metric("Overall", s["overall_sentiment"].upper())
            if s["details"]:
                for h in s["details"]:
                    emoji = "📈" if h["sentiment"] == "positive" else "📉" if h["sentiment"] == "negative" else "➡️"
                    st.write(f"{emoji} {h['headline']}")

            st.markdown("### Risk Assessment")
            r = result["risk_assessment"]
            cols2 = st.columns(4)
            cols2[0].metric("Risk Score", r["risk_score"])
            cols2[1].metric("Action", r["action"].upper())
            cols2[2].metric("Max Units", r["max_units"])
            cols2[3].metric("Stop Loss", f"${r['stop_loss_price']:.2f}")

# ── Portfolio Page ────────────────────────────────────────────────────────────
elif "Portfolio" in page:
    st.title("💼 Portfolio")
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Reset Portfolio"):
            api("/portfolio/reset", "POST")
            st.success("Portfolio reset!")
            st.rerun()
    portfolio = api("/portfolio")
    if portfolio:
        st.metric("Total Value", f"${portfolio['total_value']:,.2f}")
        st.metric("Total Return", f"{portfolio['total_return_pct']:+.2f}% (${portfolio['total_return']:+,.2f})")
        if portfolio["positions"]:
            st.dataframe(pd.DataFrame(portfolio["positions"]).T, use_container_width=True)

# ── Trades Page ───────────────────────────────────────────────────────────────
elif "Trades" in page:
    st.title("📋 Trade History")
    data = api("/trades?limit=100")
    if data and data["trades"]:
        df = pd.DataFrame(data["trades"])
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv, "trades.csv", "text/csv")
    else:
        st.info("No trades yet.")

elif "About" in page:
    st.title("ℹ️ About MT Trading Swarm")
    st.markdown("""
    ### AI Trading Swarm System
    A multi-agent AI system for automated paper trading with real market data.

    **Agents:**
    - 📊 **Trend Agent** — SMA, EMA, RSI, MACD analysis
    - ⚠️ **Risk Agent** — Position sizing, stop-loss, exposure limits
    - 📰 **Sentiment Agent** — News sentiment scoring
    - ⚡ **Execution Agent** — Paper trading simulator

    **Data Source:** Yahoo Finance (yfinance)

    > ⚠️ This is a paper trading simulation only. Not financial advice.
    """)
