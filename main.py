from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# ---------------------------------------------------------
API_KEY_TWELVEDATA = "df521019db9f44899bfb172fdce6b454" 
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"              
API_KEY_FINNHUB = "d4gd4r9r01qm5b352il0d4gd4r9r01qm5b352ilg"                  
# ---------------------------------------------------------

TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }

@app.route("/")
def index():
    return render_template("index.html")

# دریافت دیتا با حجم بالا برای ML
def get_candles(symbol, interval, size=2000):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={size}"
    try:
        response = requests.get(url, timeout=10) # افزایش تایم‌اوت بخاطر حجم دیتا
        data = response.json()
        if "values" not in data: return None
        df = pd.DataFrame(data["values"])
        for c in ['open', 'high', 'low', 'close']: df[c] = pd.to_numeric(df[c])
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except: return None

# --- سطح ۱: تشخیص رژیم بازار (ADX) ---
def check_market_regime(df):
    """تشخیص اینکه بازار روند دارد یا رنج است"""
    if 'ADX_14' not in df.columns:
        df.ta.adx(length=14, append=True)
    
    last = df.iloc[-1]
    adx_col = next((c for c in df.columns if c.startswith('ADX')), None)
    adx_val = last.get(adx_col, 0)
    
    regime = "Ranging (رنج)"
    if adx_val > 25: regime = "Trending (رونددار)"
    if adx_val > 50: regime = "Strong Trend (روند قوی)"
    
    return regime, adx_val

# --- سطح ۲: سطوح حمایت و مقاومت (Donchian) ---
def get_sr_levels(df):
    """محاسبه حمایت و مقاومت بر اساس کانال دانچیان ۲۰ دوره"""
    df.ta.donchian(lower_length=20, upper_length=20, append=True)
    last = df.iloc[-1]
    
    sup_col = next((c for c in df.columns if c.startswith('DCL')), None)
    res_col = next((c for c in df.columns if c.startswith('DCU')), None)
    
    support = last.get(sup_col, 0)
    resistance = last.get(res_col, 0)
    return support, resistance

# --- سطح ۳: واگرایی ---
def check_divergence(df):
    subset = df.iloc[-15:].reset_index(drop=True)
    price = subset['close']
    rsi = subset['RSI_14']
    
    price_high_idx, price_low_idx = price.idxmax(), price.idxmin()
    curr_price, curr_rsi = price.iloc[-1], rsi.iloc[-1]
    
    score, msg = 0, "بدون واگرایی"
    
    # واگرایی منفی
    if price_high_idx < 14:
        if curr_price > price[price_high_idx] and curr_rsi < rsi[price_high_idx]:
            msg, score = "Bearish Div 📉", -3
            
    # واگرایی مثبت
    if price_low_idx < 14:
        if curr_price < price[price_low_idx] and curr_rsi > rsi[price_low_idx]:
            msg, score = "Bullish Div 📈", 3
            
    return score, msg

# --- سطح ۴: یادگیری ماشین (روی ۲۰۰۰ کندل) ---
def get_ml_prediction(df):
    try:
        # ویژگی‌های بیشتر برای دقت بالاتر
        df['Returns'] = df['close'].pct_change()
        df['RSI'] = df.ta.rsi(length=14)
        df['ADX'] = df.ta.adx(length=14)[df.ta.adx(length=14).columns[0]] # فقط ستون مقدار ADX
        df['EMA_Diff'] = df.ta.ema(length=20) - df.ta.ema(length=50)
        df['Volatility'] = df['high'] - df['low']
        
        df = df.dropna()
        if len(df) < 100: return 0, "دیتای ناکافی"

        # هدف: آیا کندل بعد سبز است؟
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # آموزش روی تمام دیتای موجود (حدود 1900 سطر)
        train = df.iloc[:-1]
        last_features = df.iloc[-1][['RSI', 'ADX', 'EMA_Diff', 'Returns', 'Volatility']].to_frame().T
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(train[['RSI', 'ADX', 'EMA_Diff', 'Returns', 'Volatility']], train['Target'])
        
        prob = model.predict_proba(last_features)[0][1]
        
        score, msg = 0, "AI: خنثی"
        if prob > 0.65: score, msg = 3, f"AI: صعود ({int(prob*100)}%) 🚀"
        elif prob < 0.35: score, msg = -3, f"AI: نزول ({int((1-prob)*100)}%) 🔻"
        
        return score, msg
    except: return 0, "AI Error"

# --- تابع اخبار ---
def get_market_sentiment(symbol):
    # (همان کد قبلی برای خلاصه شدن)
    return 0, "اخبار خنثی"

# --- مدیریت ریسک ---
def calculate_smart_sl_tp(entry, signal, atr, support, resistance):
    if not atr or np.isnan(atr): return None, None
    
    # استفاده هوشمند از سطوح حمایت/مقاومت برای SL
    if signal == "buy":
        # حد ضرر کمی پایین‌تر از حمایت یا محاسبه ATR
        sl_base = support if (entry - support) < (atr * 2) else (entry - atr * 1.5)
        sl = sl_base
        tp = entry + ((entry - sl) * 2)
    else:
        sl_base = resistance if (resistance - entry) < (atr * 2) else (entry + atr * 1.5)
        sl = sl_base
        tp = entry - ((sl - entry) * 2)
        
    return round(sl, 5), round(tp, 5)

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

    # اجرای ۴ سطح تحلیل
    regime, adx_val = check_market_regime(df)       # سطح ۱
    support, resistance = get_sr_levels(df)         # سطح ۲
    div_score, div_msg = check_divergence(df)       # سطح ۳
    ml_score, ml_msg = get_ml_prediction(df)        # سطح ۴
    
    # تحلیل HTF
    htf_trend = "neutral"
    if use_htf:
        htf_int = TIMEFRAME_MAP.get(interval)
        if htf_int:
            df_h = get_candles(symbol, htf_int, size=100)
            if df_h is not None:
                df_h.ta.ema(length=20, append=True)
                df_h.ta.ema(length=50, append=True)
                if df_h.iloc[-1][f'EMA_20'] > df_h.iloc[-1][f'EMA_50']: htf_trend = "uptrend"
                else: htf_trend = "downtrend"

    # === سیستم امتیازدهی پیشرفته ===
    score = 0
    
    # الف) منطق بر اساس رژیم بازار (ADX)
    if adx_val > 25: # بازار رونددار
        score += 3 if trend == "uptrend" else -3
        # در روند، RSI کمتر اهمیت دارد مگر اینکه خیلی افراطی باشد
    else: # بازار رنج
        # در رنج، EMA سیگنال فیک میدهد، پس وزنش را کم میکنیم
        score += 1 if trend == "uptrend" else -1
        # و به اسیلاتورها وزن میدهیم
        if rsi < 30: score += 3
        elif rsi > 70: score -= 3
        
    # ب) فیلتر پرایس اکشن (S/R)
    # اگر سیگنال خرید داریم ولی چسبیده به مقاومتیم -> امتیاز کم کن
    dist_to_res = resistance - price
    dist_to_sup = price - support
    
    if dist_to_res < (atr * 0.5): score -= 2 # خطر مقاومت
    if dist_to_sup < (atr * 0.5): score += 2 # حمایت قوی

    # پ) اضافه کردن امتیازهای پیشرفته
    score += div_score
    score += ml_score
    
    # ت) HTF
    if use_htf and htf_trend != "neutral":
        if trend == htf_trend: score += 2
        else: score -= 1

    # سیگنال نهایی
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
            "regime": f"{regime} (ADX: {int(adx_val)})",
            "sr_levels": f"S: {support} | R: {resistance}",
            "ai_prediction": ml_msg,
            "divergence": div_msg,
            "trend": trend,
            "rsi": round(rsi, 2)
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
