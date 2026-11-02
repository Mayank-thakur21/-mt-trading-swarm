# 📈 MT Trading Swarm — AI Multi-Agent Trading System

![CI](https://github.com/Mayank-thakur21/-mt-trading-swarm/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/license-MIT-yellow)

An AI-powered **multi-agent trading swarm** for paper trading with real market data. Four specialized agents work together to analyze trends, assess risk, gauge sentiment, and execute trades.

> ⚠️ **Paper trading only** — no real money involved. Not financial advice.

---

## 🤖 Agents

| Agent | Role |
|-------|------|
| 📊 **Trend Agent** | SMA, EMA, RSI, MACD technical analysis |
| ⚠️ **Risk Agent** | Position sizing, stop-loss, exposure limits |
| 📰 **Sentiment Agent** | News sentiment scoring |
| ⚡ **Execution Agent** | Paper trade simulator with P&L tracking |
| 🧠 **Swarm Orchestrator** | Aggregates all signals → final BUY/SELL/HOLD |

---

## ✨ Features

- Real-time price data via **Yahoo Finance**
- Multi-signal aggregation with confidence scoring
- Paper portfolio with **unrealized P&L tracking**
- REST API + **Streamlit dashboard**
- Docker support for easy deployment
- Batch analysis for multiple symbols

---

## 🚀 Quick Start

### Option 1: Docker
```bash
git clone https://github.com/Mayank-thakur21/-mt-trading-swarm
cd -mt-trading-swarm
docker-compose up --build
```
- Dashboard: http://localhost:8501
- API: http://localhost:8000/docs

### Option 2: Manual
```bash
pip install -r requirements.txt

# Start API
uvicorn api:app --reload

# Start Dashboard (new terminal)
streamlit run dashboard.py
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Analyze a single symbol |
| POST | `/analyze/batch` | Analyze up to 10 symbols |
| GET | `/portfolio` | Get paper portfolio |
| POST | `/portfolio/reset` | Reset portfolio |
| POST | `/trade` | Manual paper trade |
| GET | `/trades` | Trade history |
| GET | `/watchlist` | Suggested symbols |

### Example
```bash
# Analyze AAPL with auto paper trading
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "auto_trade": true}'
```

### Response
```json
{
  "symbol": "AAPL",
  "current_price": 189.42,
  "recommendation": "BUY",
  "confidence": "high",
  "signals": {
    "trend": "bullish",
    "sentiment": "bullish",
    "risk_action": "proceed"
  }
}
```

---

## 🏗️ Project Structure
```
-mt-trading-swarm/
├── agents/
│   ├── trend_agent.py       # Technical analysis (SMA, RSI, MACD)
│   ├── risk_agent.py        # Position sizing & risk management
│   ├── sentiment_agent.py   # News sentiment scoring
│   └── execution_agent.py   # Paper trading engine
├── swarm.py                 # Swarm orchestrator
├── api.py                   # FastAPI REST API
├── dashboard.py             # Streamlit dashboard
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 📄 License
MIT © Mayank Thakur
