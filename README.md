# 🚀 DEMIR AI v8.0 - Professional Cryptocurrency Trading Bot

<div align="center">

[![Python Version](https://img.shields.io/badge/Python-3.11.9-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Railway](https://img.shields.io/badge/Deploy%20on-Railway-purple)](https://railway.app)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com/yourusername/demir-ai-v8)

</div>

<div align="center">
  <h3>
    <a href="#features">Features</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#deployment">Deployment</a> •
    <a href="#api-docs">API</a> •
    <a href="#contributing">Contributing</a>
  </h3>
</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Trading Strategies](#trading-strategies)
- [Risk Management](#risk-management)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

DEMIR AI is a sophisticated cryptocurrency trading bot that operates 24/7, analyzing markets through multiple data layers and executing trades with advanced risk management. Built with Python 3.11.9, it features real-time sentiment analysis from 15 sources, technical indicators, machine learning predictions, and multi-exchange support.

### Key Highlights

- 🔍 **Multi-Layer Analysis**: 15 sentiment sources + technical indicators + ML models
- 🤖 **AI-Powered**: 5 active ML models with ensemble predictions
- 📊 **Real-Time Processing**: WebSocket connections for instant data
- 🔐 **Secure**: Enterprise-grade security with encrypted API keys
- 📈 **Proven Performance**: Backtested on 3 years of historical data
- 🚨 **Risk Management**: Dynamic position sizing and emergency stops

## ✨ Features

### 📊 Data Analysis Layers

#### Sentiment Analysis (15 Active Sources)
- ✅ **CryptoPanic** - Real-time crypto news sentiment
- ✅ **Fear & Greed Index** - Market psychology indicator
- ✅ **BTC Dominance** - Market structure analysis
- ✅ **Exchange Flows** - Whale movement tracking
- ✅ **Funding Rates** - Perpetual market sentiment
- ✅ **Long/Short Ratios** - Positioning analysis
- ✅ **On-Chain Metrics** - Blockchain activity
- ✅ **Order Book Imbalance** - Supply/demand dynamics
- ✅ **Liquidation Cascade** - Risk event detection
- ✅ **And 6 more sources...**

#### Technical Analysis
- 📈 Moving Averages (SMA, EMA, WMA)
- 📊 Oscillators (RSI, MACD, Stochastic)
- 📉 Volatility Indicators (Bollinger Bands, ATR)
- 🎯 Pattern Recognition (Head & Shoulders, Triangles)
- 🕯️ Candlestick Patterns (Doji, Hammer, Engulfing)

#### Machine Learning Models
- 🧠 LSTM (Time-series prediction)
- 🌳 XGBoost (Gradient boosting)
- 🌲 Random Forest (Ensemble learning)
- 📊 Gradient Boosting
- 🔄 KMeans (Market regime clustering)

### 💹 Trading Features

- **Multi-Exchange Support**
  - Binance (Primary)
  - Bybit
  - Coinbase
  
- **Order Types**
  - Market Orders
  - Limit Orders
  - Stop-Loss Orders
  - Take-Profit Orders (3 levels)
  
- **Position Management**
  - Dynamic Position Sizing
  - Trailing Stop-Loss
  - Partial Take-Profits
  - Portfolio Rebalancing

### 🛡️ Risk Management

- **Portfolio Protection**
  - Max Daily Loss: 5%
  - Max Drawdown: 15%
  - Emergency Stop: 20%
  
- **Position Controls**
  - Max Risk per Trade: 2%
  - Max Concurrent Positions: 10
  - Correlation Limits: 0.7
  
- **Circuit Breakers**
  - Automatic pause on high volatility
  - Error rate monitoring
  - Connection failure handling

### 🔔 Alert System

- **Telegram Bot** - Real-time notifications
- **Discord Webhooks** - Team alerts
- **Email Notifications** - Daily summaries
- **Dashboard** - Web interface

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DEMIR AI v8.0                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  Sentiment   │  │  Technical   │  │     ML       │ │
│  │   Analysis   │  │   Analysis   │  │   Models     │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                   │         │
│         └─────────────────┴───────────────────┘         │
│                           │                             │
│                    ┌──────▼───────┐                    │
│                    │    Signal     │                    │
│                    │   Generator   │                    │
│                    └──────┬───────┘                    │
│                           │                             │
│                    ┌──────▼───────┐                    │
│                    │     Risk      │                    │
│                    │  Controller   │                    │
│                    └──────┬───────┘                    │
│                           │                             │
│         ┌─────────────────┴─────────────────┐         │
│         │                                   │         │
│  ┌──────▼───────┐                 ┌────────▼──────┐  │
│  │   Trading    │                 │    Alert      │  │
│  │   Executor   │                 │   Manager     │  │
│  └──────────────┘                 └───────────────┘  │
│                                                       │
└───────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11.9+
- PostgreSQL 15+
- Redis 7+
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/demir-ai-v8.git
cd demir-ai-v8
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

5. **Initialize database**
```bash
python scripts/setup_db.py
```

6. **Run the bot**
```bash
python main.py
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f demir-ai
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Core Settings
ENVIRONMENT=production
VERSION=8.0
DEBUG_MODE=false
ADVISORY_MODE=true  # Set to false for live trading

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/demir_ai

# Exchange APIs
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Alert Services
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Data Providers
ALPHA_VANTAGE_API_KEY=your_key
COINGLASS_API_KEY=your_key
# ... (see .env.example for full list)
```

### Trading Configuration

Edit `config.py` to adjust:

- Trading pairs
- Position sizes
- Risk parameters
- Signal thresholds
- Time intervals

## 🚂 Deployment

### Railway Deployment

1. **Fork this repository**

2. **Create Railway account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

3. **Create new project**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your forked repository

4. **Configure environment variables**
   - Go to Variables tab
   - Add all variables from `.env.example`

5. **Deploy**
   - Railway will automatically deploy
   - Monitor logs for any issues

### Manual VPS Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed VPS setup instructions.

## 📡 API Documentation

### REST API Endpoints

```http
GET /health
GET /metrics
GET /signals
POST /backtest
GET /positions
```

### WebSocket Streams

```javascript
ws://localhost:8000/ws/prices
ws://localhost:8000/ws/signals
ws://localhost:8000/ws/metrics
```

Full API documentation: [docs/API.md](docs/API.md)

## 📈 Trading Strategies

The bot implements multiple strategies:

1. **Trend Following** - Riding momentum
2. **Mean Reversion** - Fade extremes
3. **Breakout Trading** - New highs/lows
4. **Sentiment Driven** - News-based trades
5. **ML Predictions** - AI-powered signals

## 🛡️ Risk Management

### Position Sizing

Uses Kelly Criterion modified for crypto:
```python
position_size = kelly_fraction * account_balance * confidence_score
```

### Stop Loss Strategy

Dynamic stop loss based on:
- ATR (Average True Range)
- Support/Resistance levels
- Signal strength

### Portfolio Allocation

- Maximum 10% per position
- Correlation-based diversification
- Automatic rebalancing

## 📊 Monitoring

### Streamlit Dashboard

Access at `http://localhost:8501`

Features:
- Real-time P&L
- Active positions
- Signal history
- Performance metrics
- System health

### Metrics Tracked

- Win Rate
- Sharpe Ratio
- Maximum Drawdown
- Daily/Monthly Returns
- Risk-adjusted Returns

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=./ --cov-report=html

# Specific module
pytest tests/test_signals.py -v
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linters
black .
flake8 .
mypy .

# Pre-commit hooks
pre-commit install
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational purposes only. Cryptocurrency trading carries substantial risk of loss. 

- Never trade with funds you cannot afford to lose
- Past performance does not guarantee future results
- Always do your own research
- The developers are not responsible for any financial losses

## 🙏 Acknowledgments

- Built with Python and love ❤️
- Inspired by the crypto community
- Special thanks to all contributors

## 📞 Support

- 📧 Email: support@demirai.com
- 💬 Telegram: [@demirai_support](https://t.me/demirai_support)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/demir-ai-v8/issues)
- 📖 Wiki: [Documentation](https://github.com/yourusername/demir-ai-v8/wiki)

---

<div align="center">

**Made with ❤️ by the DEMIR AI Team**

[Website](https://demirai.com) • [Twitter](https://twitter.com/demirai) • [Discord](https://discord.gg/demirai)

</div>
