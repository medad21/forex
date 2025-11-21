from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
import re

app = Flask(__name__)

# ---------------------------------------------------------
# 🔑 تنظیمات و کلیدهای API (اینجا کلیدهای خود را بگذارید)
# ---------------------------------------------------------
API_KEY_TWELVEDATA = "df521019db9f44899bfb172fdce6b454"  # کلید دیتای تکنیکال (کندل‌ها)
API_KEY_ALPHA = "YOUR_ALPHA_VANTAGE_KEY"              # کلید آلفا ونتیج (برای اخبار دقیق هوش مصنوعی)
API_KEY_FINNHUB = "YOUR_FINNHUB_KEY"                  # کلید فین‌هاب (برای اخبار پشتیبان)
# ---------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

# =========================================================
# 1. بخش تحلیل تکنیکال (الگوها و پیوت‌ها)
# =========================================================

def calculate_pivot_points(high, low, close):
    """محاسبه سطوح حمایت و مقاومت پیوت پوینت"""
    pp = (high + low + close) / 3
    r1 = (2 * pp) - low
    s1 = (2 * pp) - high
    r2 = pp + (high - low)
    s2 = pp - (high - low)
    return pp, r1, s1, r2, s2

def detect_candlestick_pattern(opens, highs, lows, closes):
    """شناسایی ۹ الگوی مهم کندل استیک"""
    if len(closes) < 4: return "Insufficient Data", 0

    O1, H1, L1, C1 = opens[-3], highs[-3], lows[-3], closes[-3]
    O2, H2, L2, C2 = opens[-2], highs[-2], lows[-2], closes[-2]
    O3, H3, L3, C3 = opens[-1], highs[-1], lows[-1], closes[-1]

    body_size = abs(C3 - O3)
    upper_wick = H3 - max(C3, O3)
    lower_wick = min(C3, O3) - L3
    avg_body = np.mean(np.abs(closes[-10:] - opens[-10:]))

    pattern_name = "No Clear Pattern"
    score = 0

    # الگوهای تک کندلی
    if lower_wick > (2 * body_size) and upper_wick < (body_size * 0.5) and body_size > (avg_body * 0.2):
        pattern_name = "Hammer (Bullish) 🔨"
        score = 2
    elif upper_wick > (2 * body_size) and lower_wick < (body_size * 0.5) and body_size > (avg_body * 0.2):
        pattern_name = "Shooting Star (Bearish) 🌠"
        score = -2
    elif body_size < (avg_body * 0.1):
        pattern_name = "Doji (Indecision) ➕"
        score = 0 

    # الگوهای دو کندلی
    elif C3 > O3 and C2 < O2 and C3 > O2 and O3 < C2:
        pattern_name = "Bullish Engulfing 📈"
        score = 3
    elif C3 < O3 and C2 > O2 and C3 < O2 and O3 > C2:
        pattern_name = "Bearish Engulfing 📉"
        score = -3

    # الگوهای سه کندلی
    elif (C1 < O1) and (abs(C2-O2) < avg_body * 0.6) and (C3 > O3) and (C3 > (O1 + C1)/2):
        pattern_name = "Morning Star (Strong Buy) 🌟"
        score = 4
    elif (C1 > O1) and (abs(C2-O2) < avg_body * 0.6) and (C3 < O3) and (C3 < (O1 + C1)/2):
        pattern_name = "Evening Star (Strong Sell) 🌑"
        score = -4
    elif (C1 > O1) and (C2 > O2) and (C3 > O3) and (C2 > C1) and (C3 > C2):
        pattern_name = "3 White Soldiers (Bullish) 💂‍♂️"
        score = 3
    elif (C1 < O1) and (C2 < O2) and (C3 < O3) and (C2 < C1) and (C3 < C2):
        pattern_name = "3 Black Crows (Bearish) 🦅"
        score = -3

    return pattern_name, score

# =========================================================
# 2. بخش تحلیل اخبار (هایبرید)
# =========================================================

def analyze_text_sentiment_basic(headlines):
    """تحلیل ساده متن برای Finnhub"""
    bullish_words = ["rise", "jump", "growth", "bull", "high", "gain", "positive", "recovery", "surge", "up"]
    bearish_words = ["fall", "drop", "crash", "bear", "low", "loss", "negative", "recession", "inflation", "war", "down"]
    
    score = 0
    count = 0
    for title in headlines:
        title_lower = title.lower()
        if any(w in title_lower for w in bullish_words): score += 1
        if any(w in title_lower for w in bearish_words): score -= 1
        count += 1
    
    if count == 0: return 0
    final_score = score / count
    return max(min(final_score * 2, 1), -1) 

def get_market_sentiment_hybrid(symbol):
    """سیستم دوگانه دریافت اخبار: ۱. آلفا ونتیج ۲. فین‌هاب"""
    
    # تلاش اول: Alpha Vantage
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol or "ETH" in symbol: av_symbol = "CRYPTO:BTC"
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=5)
        data = r.json()
        
        if "feed" in data and len(data["feed"]) > 0:
            top_news = data["feed"][0]
            sentiment_score = float(top_news["overall_sentiment_score"])
            label = top_news["overall_sentiment_label"]
            title = top_news["title"]
            return sentiment_score, {
                "en": f"{label}: {title[:60]}...",
                "fa": f"هوش مصنوعی: {label} (منبع: AlphaVantage)"
            }
    except Exception as e:
        print(f"Alpha Vantage Failed: {e}")

    # تلاش دوم: Finnhub
    try:
        category = "forex"
        if "BTC" in symbol: category = "crypto"
        url = f"https://finnhub.io/api/v1/news?category={category}&token={API_KEY_FINNHUB}"
        r = requests.get(url, timeout=5)
        news_list = r.json()
        
        if len(news_list) > 0:
            headlines = [item["headline"] for item in news_list[:3]]
            calculated_score = analyze_text_sentiment_basic(headlines)
            sentiment_text = "Bullish" if calculated_score > 0.2 else "Bearish" if calculated_score < -0.2 else "Neutral"
            return calculated_score, {
                "en": f"{sentiment_text} based on headlines",
                "fa": f"تحلیل تیترها: {sentiment_text} (منبع: Finnhub)"
            }
    except Exception as e:
        print(f"Finnhub Failed: {e}")

    return 0, {"en": "No News Data", "fa": "عدم دسترسی به اخبار"}

# =========================================================
# 3. روت اصلی تحلیل
# =========================================================

@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")

    # 1. دریافت دیتا
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize=100"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        return jsonify({"error": "no data", "details": data})

    try:
        values = data["values"][::-1]
        closes = np.array([float(v["close"]) for v in values])
        highs = np.array([float(v["high"]) for v in values])
        lows = np.array([float(v["low"]) for v in values])
        opens = np.array([float(v["open"]) for v in values])
    except:
        return jsonify({"error": "data parsing error"})

    # 2. محاسبات تکنیکال
    def ema(series, period):
        k = 2 / (period + 1)
        ema_arr = np.zeros_like(series)
        if len(series) == 0: return ema_arr
        ema_arr[0] = series[0]
        for i in range(1, len(series)):
            ema_arr[i] = series[i] * k + ema_arr[i - 1] * (1 - k)
        return ema_arr

    ema20 = ema(closes, 20)
    ema50 = ema(closes, 50)
    trend = "uptrend" if ema20[-1] > ema50[-1] else "downtrend"

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))

    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(abs(highs - prev_close), abs(lows - prev_close)))
    atr = np.mean(tr[-14:])

    # 3. فراخوانی ماژول‌ها
    pp, r1, s1, r2, s2 = calculate_pivot_points(highs[-2], lows[-2], closes[-2])
    pattern_name, pattern_score = detect_candlestick_pattern(opens, highs, lows, closes)
    news_score, news_text_obj = get_market_sentiment_hybrid(symbol)

    # 4. امتیازدهی
    score = 0
    score += 2 if trend == "uptrend" else -2
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    score += pattern_score 
    if news_score > 0.2: score += 2
    elif news_score < -0.2: score -= 2
    
    current_price = closes[-1]
    threshold = atr * 0.5 
    if abs(current_price - s1) < threshold: score += 1
    if abs(current_price - r1) < threshold: score -= 1

    final_signal = "neutral"
    if score >= 4: final_signal = "buy"
    elif score <= -4: final_signal = "sell"

    # 5. مدیریت ریسک
    entry = float(current_price)
    sl, tp = None, None

    if final_signal == "buy" and atr > 0:
        sl_base = min(s1, lows[-1])
        sl = sl_base - (atr * 0.3)
        risk = entry - sl
        tp = entry + (risk * 2)
    elif final_signal == "sell" and atr > 0:
        sl_base = max(r1, highs[-1])
        sl = sl_base + (atr * 0.3)
        risk = sl - entry
        tp = entry - (risk * 2)

    return jsonify({
        "symbol": symbol,
        "price": round(entry, 5),
        "signal": final_signal,
        "score": score,
        "trend": trend,
        "pattern": pattern_name,
        "indicators": {
            "rsi": round(float(rsi), 2),
            "atr": round(float(atr), 5),
            "ema20": round(float(ema20[-1]), 5),
            "sentiment": news_text_obj
        },
        "levels": {"pivot": round(pp, 5), "r1": round(r1, 5), "s1": round(s1, 5)},
        "setup": {"sl": round(sl, 5) if sl else None, "tp": round(tp, 5) if tp else None}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
