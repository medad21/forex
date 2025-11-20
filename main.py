from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
import random 

app = Flask(__name__)

# Test API Key for TwelveData (Free tier limits apply)
API_KEY = "df521019db9f44899bfb172fdce6b454"

@app.route("/")
def index():
    return render_template("index.html")

# --- Advanced Analysis Functions ---

def calculate_support_resistance(highs, lows, current_price):
    """
    Calculates Support and Resistance levels based on the last 50 candles (S&R)
    """
    lookback = 50
    if len(highs) < lookback: lookback = len(highs)

    recent_high = np.max(highs[-lookback:])
    recent_low = np.min(lows[-lookback:])
    
    dist_to_resistance = recent_high - current_price
    dist_to_support = current_price - recent_low
    
    return recent_high, recent_low, dist_to_resistance, dist_to_support

def get_market_sentiment():
    """
    Simulates market news sentiment.
    Returns a tuple: (score, {english_text, persian_text})
    """
    # نکته: این بخش در حال حاضر رندوم است. برای واقعی شدن نیاز به اتصال به API اخبار دارید.
    sentiment_score = random.gauss(0, 0.4) 
    
    sentiment_data = {
        "en": "Neutral (No major news impact)",
        "fa": "خنثی (بدون خبر تأثیرگذار)"
    }
    
    if sentiment_score > 0.4:
        sentiment_data = {
            "en": "Bullish Sentiment (Positive News) 🐂",
            "fa": "مثبت (اخبار صعودی بازار) 🐂"
        }
    elif sentiment_score < -0.4:
        sentiment_data = {
            "en": "Bearish Sentiment (Negative News) 🐻",
            "fa": "منفی (اخبار نزولی بازار) 🐻"
        }
        
    return sentiment_score, sentiment_data

# --- Main Analysis Route ---

@app.route("/analyze", methods=["GET"])
def analyze():
    # Get parameters
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")

    # Fetch data from TwelveData
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=100"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        error_message = data.get("message", "Could not fetch data or API limit reached.")
        return jsonify({"error": "no data found", "details": {"message": error_message}})

    try:
        values = data["values"][::-1]
        close = np.array([float(v["close"]) for v in values])
        high = np.array([float(v["high"]) for v in values])
        low = np.array([float(v["low"]) for v in values])
    except Exception as e:
        return jsonify({"error": "data parsing error"})

    # --- Technical Indicators ---

    def ema(series, period):
        k = 2 / (period + 1)
        ema_arr = np.zeros_like(series)
        if len(series) == 0: return ema_arr
        ema_arr[0] = series[0]
        for i in range(1, len(series)):
            ema_arr[i] = series[i] * k + ema_arr[i - 1] * (1 - k)
        return ema_arr

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    
    if len(ema20) < 50:
        trend = "neutral"
    else:
        trend = "uptrend" if ema20[-1] > ema50[-1] else "downtrend"

    # RSI Calculation
    if len(close) < 15:
        rsi = 50.0
        atr = 0.0
    else:
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 100
        rsi = 100 - (100 / (1 + rs))

        # ATR Calculation
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
        atr = np.mean(tr[-14:])

    # Levels and Sentiment
    last_price = close[-1]
    res_level, sup_level, dist_res, dist_sup = calculate_support_resistance(high, low, last_price)
    news_score, news_text_obj = get_market_sentiment()

    # --- Scoring System ---
    score = 0
    
    if trend == "uptrend": score += 2
    elif trend == "downtrend": score -= 2

    if rsi < 30: score += 3 
    elif rsi > 70: score -= 3 
    
    if atr > 0.00001:
        if dist_sup < (atr * 1.5): score += 2
        if dist_res < (atr * 1.5): score -= 2

    if news_score > 0.4: score += 2
    elif news_score < -0.4: score -= 2

    final_signal = "neutral"
    if score >= 4: final_signal = "buy"
    elif score <= -4: final_signal = "sell"

    # --- Risk Management ---
    entry = float(last_price)
    sl = None
    tp = None

    if final_signal == "buy" and atr > 0.00001:
        sl = sup_level - (atr * 0.5) 
        risk = entry - sl
        tp = entry + (risk * 1.5) 
    elif final_signal == "sell" and atr > 0.00001:
        sl = res_level + (atr * 0.5)
        risk = sl - entry
        tp = entry - (risk * 1.5)

    return jsonify({
        "symbol": symbol,
        "price": round(entry, 5), # This fixes the display issue
        "signal": final_signal,
        "score": score,
        "trend": trend,
        "indicators": {
            "rsi": round(float(rsi), 2),
            "atr": round(float(atr), 5),
            "ema20": round(float(ema20[-1]), 5),
            "sentiment": news_text_obj # Sending both FA and EN
        },
        "levels": {
            "support": round(sup_level, 5),
            "resistance": round(res_level, 5)
        },
        "setup": {
            "sl": round(sl, 5) if sl else None,
            "tp": round(tp, 5) if tp else None
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
