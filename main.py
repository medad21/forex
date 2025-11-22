from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ---------------------------------------------------------
# 🔑 API KEYS
# ---------------------------------------------------------
API_KEY_TWELVEDATA = "df521019db9f44899bfb172fdce6b454" 
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"              
API_KEY_FINNHUB = "d4gd4r9r01qm5b352il0d4gd4r9r01qm5b352ilg"                  
# ---------------------------------------------------------

TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }

@app.route("/")
def index():
    return render_template("index.html")

# دریافت دیتا با حجم بالا (۲۰۰۰ کندل) برای ML
def get_candles(symbol, interval, size=2000):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={size}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "values" not in data: return None
        df = pd.DataFrame(data["values"])
        for c in ['open', 'high', 'low', 'close']: df[c] = pd.to_numeric(df[c])
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except: return None

# --- سطح ۱: تشخیص رژیم بازار (ADX) ---
def check_market_regime(df):
    if 'ADX_14' not in df.columns: df.ta.adx(length=14, append=True)
    last = df.iloc[-1]
    adx_col = next((c for c in df.columns if c.startswith('ADX')), None)
    adx_val = last.get(adx_col, 0)
    regime = "Ranging (رنج)"
    if adx_val > 25: regime = "Trending (رونددار)"
    if adx_val > 50: regime = "Strong Trend (روند قوی)"
    return regime, adx_val

# --- سطح ۲: سطوح حمایت و مقاومت (Donchian) ---
def get_sr_levels(df):
    df.ta.donchian(lower_length=20, upper_length=20, append=True)
    last = df.iloc[-1]
    sup_col = next((c for c in df.columns if c.startswith('DCL')), None)
    res_col = next((c for c in df.columns if c.startswith('DCU')), None)
    support = last.get(sup_col, 0)
    resistance = last.get(res_col, 0)
    return support, resistance

# --- سطح ۳: واگرایی ---
def check_divergence(df):
    if 'RSI_14' not in df.columns: df.ta.rsi(length=14, append=True)
    subset = df.iloc[-15:].reset_index(drop=True)
    price, rsi = subset['close'], subset['RSI_14']
    price_high_idx, price_low_idx = price.idxmax(), price.idxmin()
    curr_price, curr_rsi = price.iloc[-1], rsi.iloc[-1]
    score, msg = 0, "بدون واگرایی"
    if price_high_idx < 14 and curr_price > price[price_high_idx] and curr_rsi < rsi[price_high_idx]: msg, score = "Bearish Div 📉", -3
    if price_low_idx < 14 and curr_price < price[price_low_idx] and curr_rsi > rsi[price_low_idx]: msg, score = "Bullish Div 📈", 3
    return score, msg

# --- سطح ۴: یادگیری ماشین (روی ۲۰۰۰ کندل) ---
def get_ml_prediction(df):
    try:
        df['Returns'] = df['close'].pct_change()
        df['RSI'] = df.ta.rsi(length=14)
        df['ADX'] = df.ta.adx(length=14)[df.ta.adx(length=14).columns[0]]
        df['EMA_Diff'] = df.ta.ema(length=20) - df.ta.ema(length=50)
        df['Volatility'] = df['high'] - df['low']
        
        df = df.dropna()
        if len(df) < 100: return 0, "دیتای ناکافی"

        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        train = df.iloc[:-1]
        last_features = df.iloc[-1][['RSI', 'ADX', 'EMA_Diff', 'Returns', 'Volatility']].to_frame().T
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        X_train = train[['RSI', 'ADX', 'EMA_Diff', 'Returns', 'Volatility']]
        y_train = train['Target']
        
        # بررسی وجود واریانس کافی در دیتا (مهم برای ML)
        if len(np.unique(y_train)) < 2: return 0, "AI: دیتا یکنواخت است"
        
        model.fit(X_train, y_train)
        prob = model.predict_proba(last_features)[0][1]
        
        score, msg = 0, "AI: خنثی"
        if prob > 0.65: score, msg = 3, f"AI: صعود ({int(prob*100)}%) 🚀"
        elif prob < 0.35: score, msg = -3, f"AI: نزول ({int((1-prob)*100)}%) 🔻"
        
        return score, msg
    except Exception as e: return 0, f"AI Error: {str(e)[:15]}..."

# --- تابع اخبار (News) ---
def get_market_sentiment(symbol):
    sentiment_score = 0
    sentiment_text = "اخبار خنثی (بدون رویداد مهم)"
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol: av_symbol = "CRYPTO:BTC"
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=3)
        data = r.json()
        if "feed" in data and len(data["feed"]) > 0:
            label = data["feed"][0].get("overall_sentiment_label", "Neutral")
            if "Bullish" in label: sentiment_text = "🟢 اخبار مثبت (Bullish)"
            elif "Bearish" in label: sentiment_text = "🔴 اخبار منفی (Bearish)"
            sentiment_score = float(data["feed"][0].get("overall_sentiment_score", 0)) * 2
            return sentiment_score, sentiment_text
    except: pass
    return sentiment_score, sentiment_text

# --- مدیریت ریسک ---
def calculate_smart_sl_tp(entry, signal, atr, support, resistance):
    if not atr or np.isnan(atr): return None, None
    
    sl_mult, rr = 1.5, 2.0
    if signal == "buy":
        # استفاده از حمایت برای SL اگر فاصله کمی نزدیک است
        sl_base = support if (entry - support) < (atr * 2.0) and support != 0 else (entry - atr * sl_mult)
        tp = entry + ((entry - sl_base) * rr)
    else:
        # استفاده از مقاومت برای SL
        sl_base = resistance if (resistance - entry) < (atr * 2.0) and resistance != 0 else (entry + atr * sl_mult)
        tp = entry - ((sl_base - entry) * rr)
        
    return round(sl_base, 5), round(tp, 5)

# =========================================================
# MAIN ROUTE
# =========================================================
@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    use_htf = request.args.get("use_htf") == "true"

    # 1. دریافت ۲۰۰۰ کندل
    df = get_candles(symbol, interval, size=2000)
    if df is None or df.empty: return jsonify({"error": "API Error"})

    # محاسبات پایه
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.macd(append=True)

    last = df.iloc[-1]
    price = last['close']
    
    # استخراج داده‌ها
    rsi = last.get(next((c for c in df.columns if c.startswith('RSI')), ''), 50)
    atr = last.get(next((c for c in df.columns if c.startswith('ATRr')), ''), 0)
    
    ema20 = last.get(next((c for c in df.columns if c.startswith('EMA_20')), ''), price)
    ema50 = last.get(next((c for c in df.columns if c.startswith('EMA_50')), ''), price)
    trend = "uptrend" if ema20 > ema50 else "downtrend"
    
    macd_line = last.get(next((c for c in df.columns if c.startswith('MACD_')), ''), 0)
    macd_sig = last.get(next((c for c in df.columns if c.startswith('MACDs_')), ''), 0)
    macd_status = "Bullish 🟢" if macd_line > macd_sig else "Bearish 🔴"
    
    # اجرای ۴ سطح تحلیل
    regime, adx_val = check_market_regime(df)
    support, resistance = get_sr_levels(df)
    div_score, div_msg = check_divergence(df)
    ml_score, ml_msg = get_ml_prediction(df)
    news_score, news_text = get_market_sentiment(symbol)
    
    # تحلیل HTF
    htf_trend = "neutral"
    htf_status = "غیرفعال"
    if use_htf:
        htf_int = TIMEFRAME_MAP.get(interval)
        if htf_int:
            df_h = get_candles(symbol, htf_int, size=100)
            if df_h is not None:
                df_h.ta.ema(length=20, append=True)
                df_h.ta.ema(length=50, append=True)
                l_h = df_h.iloc[-1]
                e20_h = l_h.get(next((c for c in df_h.columns if c.startswith('EMA_20')), ''), 0)
                e50_h = l_h.get(next((c for c in df_h.columns if c.startswith('EMA_50')), ''), 0)
                htf_trend = "uptrend" if e20_h > e50_h else "downtrend"
                htf_status = f"فعال ({htf_int})"

    # === سیستم امتیازدهی پیشرفته ===
    score = 0
    
    # الف) منطق بر اساس رژیم بازار (ADX)
    if adx_val > 25: 
        score += 3 if trend == "uptrend" else -3
        score += 1 if macd_line > macd_sig else -1
    else: 
        score += 1 if trend == "uptrend" else -1
        if rsi < 30: score += 3
        elif rsi > 70: score -= 3
        
    # ب) فیلتر پرایس اکشن (S/R)
    dist_to_res = resistance - price
    dist_to_sup = price - support
    if dist_to_res < (atr * 0.5): score -= 2
    if dist_to_sup < (atr * 0.5): score += 2

    # پ) اضافه کردن امتیازهای پیشرفته
    score += div_score
    score += ml_score
    score += news_score # اخبار نیز وزن داده میشود
    
    # ت) HTF
    if use_htf and htf_trend != "neutral":
        if trend == htf_trend: score += 2
        else: score -= 1

    final_signal = "neutral"
    if score >= 5: final_signal = "buy"
    elif score <= -5: final_signal = "sell"

    sl, tp = calculate_smart_sl_tp(price, final_signal, atr, support, resistance)

    return jsonify({
        "price": price,
        "signal": final_signal,
        "score": round(score, 1),
        "setup": {"sl": sl, "tp": tp},
        "indicators": {
            # موارد درخواستی کاربر
            "trend": "صعودی ↗" if trend == "uptrend" else "نزولی ↘", 
            "rsi": round(rsi, 2),
            "macd": macd_status,
            "news": news_text,
            "htf_status": htf_status,
            "htf_trend": htf_trend,

            # موارد پیشرفته
            "regime": f"{regime} (ADX: {int(adx_val)})",
            "sr_levels": f"S: {support} | R: {resistance}",
            "ai_prediction": ml_msg,
            "divergence": div_msg,
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
