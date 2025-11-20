from flask import Flask, request, jsonify, render_template
import requests
import numpy as np

app = Flask(__name__)

# کلید شما از فایل قبلی برداشته شد
API_KEY = "df521019db9f44899bfb172fdce6b454" 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["GET"])
def analyze():
    # دریافت نماد و تایم‌فریم از ورودی کاربر (لینک به فرانت‌اند)
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h") # پیش‌فرض یک ساعته برای دقت بهتر

    # --- Fetch data ---
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=300"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        return jsonify({"error": "no data found", "details": data})

    # پردازش داده‌ها
    try:
        values = {k: np.array([float(v[k]) for v in data["values"]])[::-1]
                  for k in ["close", "high", "low"]}
    except Exception as e:
        return jsonify({"error": "data parsing error"})

    close = values["close"]
    high = values["high"]
    low = values["low"]

    # --------------------------
    #  EMA 20 - EMA 50 (trend)
    # --------------------------
    def ema(series, period):
        k = 2 / (period + 1)
        ema_arr = np.zeros_like(series)
        ema_arr[0] = series[0]
        for i in range(1, len(series)):
            ema_arr[i] = series[i] * k + ema_arr[i - 1] * (1 - k)
        return ema_arr

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)

    trend = "uptrend" if ema20[-1] > ema50[-1] else "downtrend"
    trend_strength = abs((ema20[-1] - ema50[-1]) / ema50[-1])

    # --------------------------
    #  RSI 14
    # --------------------------
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-14:]) # Simplified generic mean for stability
    avg_loss = np.mean(losses[-14:])
    
    if avg_loss == 0:
        rs = 100
    else:
        rs = avg_gain / avg_loss
    
    rsi = 100 - (100 / (1 + rs))

    # RSI signals
    rsi_signal = "neutral"
    if rsi < 30:
        rsi_signal = "oversold"
    elif rsi > 70:
        rsi_signal = "overbought"
    elif rsi > 60:
        rsi_signal = "momentum_bullish"
    elif rsi < 40:
        rsi_signal = "momentum_bearish"

    # --------------------------
    # ATR 14
    # --------------------------
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0] # fix logic error on first element
    
    tr = np.maximum(high - low,
                    np.maximum(abs(high - prev_close),
                               abs(low - prev_close)))
    atr = np.mean(tr[-14:])

    # --------------------------
    # Reversal detection
    # --------------------------
    last = close[-1]

    reversal = "none"
    if rsi < 35 and last > ema20[-1] and ema20[-1] > ema50[-1]:
        reversal = "bullish_reversal"
    if rsi > 65 and last < ema20[-1] and ema20[-1] < ema50[-1]:
        reversal = "bearish_reversal"

    # --------------------------
    # Signal scoring
    # --------------------------
    score = 0

    # Trend weight
    score += 2 if trend == "uptrend" else -2

    # RSI weight
    if rsi_signal == "oversold":
        score += 3 # پتانسیل برگشت قوی
    elif rsi_signal == "overbought":
        score -= 3
    elif rsi_signal == "momentum_bullish" and trend == "uptrend":
        score += 1
    elif rsi_signal == "momentum_bearish" and trend == "downtrend":
        score -= 1

    # Reversal weight
    if reversal == "bullish_reversal":
        score += 3
    if reversal == "bearish_reversal":
        score -= 3

    # Final decision logic (Improved)
    # فقط اگر امتیاز بالا بود بخر، اگر پایین بود بفروش، در غیر این صورت "خنثی"
    final_signal = "neutral"
    if score >= 2:
        final_signal = "buy"
    elif score <= -2:
        final_signal = "sell"

    # --------------------------
    # Entry, SL, TP
    # --------------------------
    entry = float(last)
    sl = None
    tp = None

    if final_signal == "buy":
        sl = entry - (2.0 * atr) # کمی تنگ‌تر برای امنیت
        tp = entry + (3.0 * atr)
    elif final_signal == "sell":
        sl = entry + (2.0 * atr)
        tp = entry - (3.0 * atr)

    # --------------------------
    # Final output
    # --------------------------
    return jsonify({
        "symbol": symbol,
        "price": round(entry, 5),
        "signal": final_signal,
        "score": score,
        "trend": trend,
        "indicators": {
            "rsi": round(float(rsi), 2),
            "atr": round(float(atr), 5),
            "ema20": round(float(ema20[-1]), 5),
            "ema50": round(float(ema50[-1]), 5)
        },
        "setup": {
            "sl": round(sl, 5) if sl else None,
            "tp": round(tp, 5) if tp else None
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
