# Forex & Crypto Signal Analysis Engine

A Python backend that generates trading signals for forex, crypto, and gold pairs by combining technical indicators, a multi-model machine learning ensemble, and news sentiment analysis, served over a Flask REST API.

## Overview

The service pulls OHLC candle data, computes a broad set of technical indicators, feeds them into an ensemble of ML models, and merges the result with sentiment and divergence signals into a single actionable trade recommendation (buy / sell / neutral) with a suggested stop-loss, take-profit, and position size.

## Architecture

```
Data Sources (TwelveData API, yfinance fallback, local DB cache)
        │
        ▼
Feature Engineering (pandas_ta: EMA, RSI, ATR, ADX, MACD, Donchian,
                      Stochastic, MFI, Supertrend)
        │
        ▼
ML Ensemble (Random Forest, Logistic Regression, XGBoost, LSTM)
        │
        ▼
Signal Scoring (trend, momentum, divergence, higher-timeframe
                confirmation, news sentiment)
        │
        ▼
Flask API  →  /analyze  (signal, entry, SL/TP, position size)
```

### Key design decisions

- **Lazy model loading** — ML models (including TensorFlow/Keras for the LSTM) are only loaded on the first request that needs them, so the service starts quickly and can run in a degraded "indicators-only" mode if model files or TensorFlow are unavailable.
- **Data source fallback** — candle data is fetched from a local database cache first, then TwelveData, then yfinance as a last resort, so a single provider outage doesn't take the service down.
- **Ensemble scoring** — each model's output is converted into a normalized score and blended with rule-based signals (trend, RSI extremes, ADX regime, Supertrend, divergence, higher-timeframe alignment, news sentiment) rather than relying on ML alone.
- **Defensive error handling** — indicator, model, and API-fetch failures are caught individually so one missing indicator or a failed model file doesn't crash the whole request.

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Flask app, API routes, signal-scoring logic |
| `train.py` | Model training pipeline |
| `predict.py` | Standalone prediction utility |
| `server.py` | Server entrypoint / process management |
| `database.py` | Candle data persistence layer |
| `import_csv.py` | Bulk-load historical CSV data into the database |
| `utils/` | Shared helper functions |
| `models/` | Serialized trained models (scaler, RF, LR, XGB, LSTM) |
| `data/raw/` | Raw historical price data |
| `*_data.csv` | Historical OHLC datasets per symbol (BTCUSD, EURUSD, GBPUSD, USDJPY, XAUUSD) |

## API

### `GET|POST /analyze`

**Parameters**

| Param | Default | Description |
|---|---|---|
| `symbol` | `EUR/USD` | Trading pair |
| `interval` | `1h` | Candle timeframe |
| `size` | `1000` | Number of candles to fetch |
| `use_htf` | `false` | Enable higher-timeframe trend confirmation |
| `balance` | `1000` | Account balance, used for position sizing |
| `risk` | `1.0` | Risk per trade, in percent |
| `rr` | `1.5` | Reward-to-risk ratio |
| `sl_type` | `static` | `static` (ATR-based) or `dynamic` (Supertrend/EMA-based) stop-loss |

**Example**

```
GET /analyze?symbol=EUR/USD&interval=1h&use_htf=true&balance=5000&risk=1
```

```json
{
  "symbol": "EUR/USD",
  "price": 1.0842,
  "signal": "buy",
  "score": 7.3,
  "setup": { "sl": 1.0812, "tp": 1.0887, "lot_size": 1.67, "risk_amt": 50.0 },
  "indicators": { "rsi": 42.1, "trend": "Uptrend", "adx": 28.4, "regime": "Trending", "..." : "..." }
}
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your own API keys
python main.py
```

Required environment variables (see `.env.example`):

- `TWELVEDATA_API_KEY`
- `ALPHA_VANTAGE_API_KEY`

## Disclaimer

This project is a technical demonstration of a data pipeline, ML ensemble, and signal-generation API. It is **not** financial advice, and the model accuracy has not been independently validated against a held-out backtest — treat any signal it produces as illustrative only.

## Roadmap / known limitations

- No automated test suite yet
- Backtesting and hyperparameter optimization endpoints are currently disabled (`/backtest`, `/optimize`)
- Reported model accuracy is not yet benchmarked; adding a documented backtest with precision/recall per model would strengthen confidence in the ensemble
