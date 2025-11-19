import requests
import numpy as np
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__, template_folder="templates")

API_KEY = "df521019db9f44899bfb172fdce6b454"
BASE_URL = "https://api.twelvedata.com/time_series"


def get_candles(symbol, interval="15min", length=200):
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
        "outputsize": length
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()

    if "values" not in data:
        return None

    values = data["values"]
    closes = np.array([float(v["close"]) for v in values[::-1]])
    return closes


def compute_rsi(prices, period=14):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ma(prices, window=20):
    return np.mean(prices[-window:])


def compute_trend(prices):
    if prices[-1] > prices[-5]:
        return "uptrend"
    elif prices[-1] < prices[-5]:
        return "downtrend"
    return "sideways"


def trading_signal(prices):
    rsi = compute_rsi(prices)
    ma = compute_ma(prices)
    trend = compute_trend(prices)
    last = prices[-1]

    signal = None
    entry = None
    tp = None
    sl = None

    if rsi < 30 and last > ma:
        signal = "buy"
        entry = last
        tp = last + (last * 0.002)
        sl = last - (last * 0.0015)

    elif rsi > 70 and last < ma:
        signal = "sell"
        entry = last
        tp = last - (last * 0.002)
        sl = last + (last * 0.0015)

    return {
        "trend": trend,
        "rsi": rsi,
        "ma": ma,
        "signal": signal,
        "entry": entry,
        "tp": tp,
        "sl": sl
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze")
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "15min")

    prices = get_candles(symbol, interval)
    if prices is None:
        return jsonify({"error": "cannot fetch data"})

    return jsonify(trading_signal(prices))

