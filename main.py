from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np

app = Flask(__name__)

# ---------------------------------------------------------
# 🔑 API KEYS
# ---------------------------------------------------------
API_KEY_TWELVEDATA = "df521019db9f44899bfb172fdce6b454" 
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"              
API_KEY_FINNHUB = "d4gd4r9r01qm5b352il0d4gd4r9r01qm5b352ilg"                  
# ---------------------------------------------------------

TIMEFRAME_MAP = {
    "15min": "1h",
    "1h": "4h",
    "4h": "1day"
}

@app.route("/")
def index():
    return render_template("index.html")

# --- بخش ۱: دریافت کندل‌ها ---
def get_candles(symbol, interval, size=60):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={size}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        if "values" not in data: return None
        
        df = pd.DataFrame(data["values"])
        for c in ['open', 'high', 'low', 'close']: df[c] = pd.to_numeric(df[c])
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except: return None

# --- بخش ۲: دریافت و تحلیل اخبار (بازگردانده شد) ---
def get_market_sentiment(symbol):
    """تحلیل اخبار با استفاده از AlphaVantage یا Finnhub"""
    sentiment_score = 0
    sentiment_text = "اخبار خنثی"
    
    # 1. تلاش با Alpha Vantage (دقیق‌تر)
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol: av_symbol = "CRYPTO:BTC"
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=3)
        data = r.json()
        
        if "feed" in data and len(data["feed"]) > 0:
            item = data["feed"][0]
            score = float(item.get("overall_sentiment_score", 0))
            label = item.get("overall_sentiment_label", "Neutral")
            sentiment_score = score * 2 # ضریب تاثیر
            sentiment_text = f"{label} (AlphaVantage)"
            return sentiment_score, sentiment_text
            
    except: pass

    # 2. تلاش با Finnhub (پشتیبان)
    try:
        category = "crypto" if "BTC" in symbol else "forex"
        url = f"https://finnhub.io/api/v1/news?category={category}&token={API_KEY_FINNHUB}"
        r = requests.get(url, timeout=3)
        news = r.json()
        
        if len(news) > 0:
            # تحلیل ساده روی ۳ تیتر اول
            bull_words = ["rise", "gain", "up", "high", "growth", "bull"]
            bear_words = ["fall", "loss", "down", "low", "crash", "bear"]
            temp_score = 0
            for item in news[:3]:
                headline = item["headline"].lower()
                if any(w in headline for w in bull_words): temp_score += 1
                if any(w in headline for w in bear_words): temp_score -= 1
            
            sentiment_score = temp_score
            sentiment_text = "Bullish (News)" if temp_score > 0 else ("Bearish (News)" if temp_score < 0 else "Neutral")
            
    except: pass
    
    return sentiment_score, sentiment_text

# --- بخش ۳: محاسبات حد سود و ضرر ---
def calculate_smart_sl_tp(entry, signal, atr):
    if not atr or np.isnan(atr): return None, None
    sl_mult, rr = 1.5, 2.0
    if signal == "buy":
        sl = entry - (atr * sl_mult)
        tp = entry + ((entry - sl) * rr)
    else:
        sl = entry + (atr * sl_mult)
        tp = entry - ((sl - entry) * rr)
    return round(sl, 5), round(tp, 5)

# --- روت اصلی تحلیل ---
@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    use_htf = request.args.get("use_htf") == "true"

    # 1. تکنیکال (Technical)
    df = get_candles(symbol, interval, size=60)
    if df is None or df.empty: return jsonify({"error": "API Error"})

    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)

    last = df.iloc[-1]
    price = last['close']
    
    # استخراج امن داده‌ها
    rsi = last.get(next((c for c in df.columns if c.startswith('RSI')), ''), 50)
    ema20 = last.get(next((c for c in df.columns if c.startswith('EMA_20')), ''), price)
    ema50 = last.get(next((c for c in df.columns if c.startswith('EMA_50')), ''), price)
    atr = last.get(next((c for c in df.columns if c.startswith('ATRr')), ''), 0)
    
    trend = "uptrend" if ema20 > ema50 else "downtrend"
    
    # 2. مولتی تایم‌فریم (Multi-Timeframe)
    htf_trend = "neutral"
    htf_status = "غیرفعال"
    
    if use_htf:
        htf_interval = TIMEFRAME_MAP.get(interval)
        if htf_interval:
            df_htf = get_candles(symbol, htf_interval, size=30)
            if df_htf is not None:
                df_htf.ta.ema(length=20, append=True)
                df_htf.ta.ema(length=50, append=True)
                l_htf = df_htf.iloc[-1]
                e20_h = l_htf.get(next((c for c in df_htf.columns if c.startswith('EMA_20')), ''), 0)
                e50_h = l_htf.get(next((c for c in df_htf.columns if c.startswith('EMA_50')), ''), 0)
                
                htf_trend = "uptrend" if e20_h > e50_h else "downtrend"
                htf_status = f"فعال ({htf_interval})"

    # 3. اخبار و سنتیمنت (Fundamental) - اضافه شده
    news_score, news_text = get_market_sentiment(symbol)

    # 4. سیستم امتیازدهی نهایی (Scoring)
    score = 0
    
    # الف) تکنیکال
    score += 2 if trend == "uptrend" else -2
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    
    bb_u = last.get(next((c for c in df.columns if c.startswith('BBU')), ''), price * 10)
    bb_l = last.get(next((c for c in df.columns if c.startswith('BBL')), ''), 0)
    if price > bb_u: score -= 1
    elif price < bb_l: score += 1
    
    # ب) تایم‌فریم بالا
    if use_htf and htf_trend != "neutral":
        if trend == htf_trend: score += 3
        else: score -= 1
        
    # پ) اخبار (تاثیرگذاری روی امتیاز)
    if news_score > 0.5: score += 2
    elif news_score < -0.5: score -= 2

    final_signal = "neutral"
    if score >= 4: final_signal = "buy"
    elif score <= -4: final_signal = "sell"

    sl, tp = calculate_smart_sl_tp(price, final_signal, atr)

    return jsonify({
        "price": price,
        "signal": final_signal,
        "score": round(score, 1),
        "trend": trend,
        "htf_trend": htf_trend,
        "htf_status": htf_status,
        "sentiment": news_text,  # ارسال متن خبر به فرانت
        "setup": {"sl": sl, "tp": tp}
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
