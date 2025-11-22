from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
import os

app = Flask(__name__)

# ---------------------------------------------------------
# 🔑 API KEYS
# ---------------------------------------------------------
# بهتر است از Environment Variables استفاده کنید، اما فعلا همینجا هستند
API_KEY_TWELVEDATA = "df521019db9f44899bfb172fdce6b454" 
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"              
API_KEY_FINNHUB = "d4gd4r9r01qm5b352il0d4gd4r9r01qm5b352ilg"                  
# ---------------------------------------------------------

TIMEFRAME_MAP = {
    "5min": "15min",
    "15min": "1h",
    "1h": "4h",
    "4h": "1day",
    "1day": "1week"
}

@app.route("/")
def index():
    return render_template("index.html")

# =========================================================
#  توابع کمکی و هسته پردازش
# =========================================================

def get_candles(symbol, interval, size=150):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={size}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "values" not in data:
            print("TwelveData Error:", data)
            return None
        
        df = pd.DataFrame(data["values"])
        # تبدیل نوع داده‌ها
        cols = ['open', 'high', 'low', 'close']
        for c in cols:
            df[c] = pd.to_numeric(df[c])
        
        # معکوس کردن دیتا (از قدیم به جدید)
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def apply_technical_analysis(df):
    """محاسبه اندیکاتورها با مدیریت خطا"""
    try:
        # EMA
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        
        # RSI
        df.ta.rsi(length=14, append=True)
        
        # MACD
        df.ta.macd(append=True)
        
        # Bollinger Bands
        df.ta.bbands(length=20, std=2, append=True)
        
        # ATR
        df.ta.atr(length=14, append=True)
    except Exception as e:
        print(f"TA Error: {e}")
    
    return df

def calculate_smart_sl_tp(entry, signal, atr):
    if not atr or np.isnan(atr):
        return None, None
        
    atr_multiplier_sl = 1.5
    risk_reward_ratio = 2.0
    
    if signal == "buy":
        sl = entry - (atr * atr_multiplier_sl)
        risk = entry - sl
        tp = entry + (risk * risk_reward_ratio)
    else: # sell
        sl = entry + (atr * atr_multiplier_sl)
        risk = sl - entry
        tp = entry - (risk * risk_reward_ratio)
        
    return round(sl, 5), round(tp, 5)

# =========================================================
#  روت اصلی تحلیل (اصلاح شده)
# =========================================================

@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    
    # 1. دریافت دیتا
    df = get_candles(symbol, interval)
    if df is None or df.empty:
        return jsonify({"error": "خطا در دریافت دیتای اصلی یا محدودیت API"})
    
    df = apply_technical_analysis(df)
    
    # بررسی وجود دیتا کافی
    if len(df) < 50:
         return jsonify({"error": "دیتای کافی برای محاسبه اندیکاتورها موجود نیست"})

    last_row = df.iloc[-1]
    
    # استخراج امن مقادیر (Safe Extraction)
    price = last_row['close']
    
    # پیدا کردن نام صحیح ستون‌ها به صورت پویا
    # چون ممکن است نام ستون RSI_14 یا RSI باشد
    rsi_col = next((c for c in df.columns if c.startswith('RSI')), None)
    rsi = last_row[rsi_col] if rsi_col else 50

    ema20_col = next((c for c in df.columns if c.startswith('EMA_20')), None)
    ema20 = last_row[ema20_col] if ema20_col else price

    ema50_col = next((c for c in df.columns if c.startswith('EMA_50')), None)
    ema50 = last_row[ema50_col] if ema50_col else price

    atr_col = next((c for c in df.columns if c.startswith('ATRr')), None) # معمولا ATRr_14 است
    atr = last_row[atr_col] if atr_col else 0.001

    # 2. تحلیل روند
    trend = "uptrend" if ema20 > ema50 else "downtrend"
    
    # 3. تحلیل چند زمانی (ساده شده برای جلوگیری از خطای API Limit)
    htf_trend = "neutral"
    htf_data_status = "غیرفعال (صرفه جویی API)"
    
    # 4. امتیازدهی
    score = 0
    if trend == "uptrend": score += 2
    else: score -= 2
    
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    
    # MACD Safe check
    macd_line_col = next((c for c in df.columns if c.startswith('MACD_')), None)
    macd_sig_col = next((c for c in df.columns if c.startswith('MACDs_')), None)
    
    macd_status = "Neutral"
    if macd_line_col and macd_sig_col:
        if last_row[macd_line_col] > last_row[macd_sig_col]:
            score += 1
            macd_status = "Bullish"
        else:
            score -= 1
            macd_status = "Bearish"

    # === فیکس کردن خطای Bollinger Bands ===
    # به جای استفاده از اسم ثابت، دنبال ستونی میگردیم که با BBU_20 شروع شود
    bb_upper_col = next((c for c in df.columns if c.startswith('BBU_20')), None)
    bb_lower_col = next((c for c in df.columns if c.startswith('BBL_20')), None)
    
    bb_status = "Inside"
    
    if bb_upper_col and bb_lower_col:
        bb_upper = last_row[bb_upper_col]
        bb_lower = last_row[bb_lower_col]
        
        if price > bb_upper: 
            bb_status = "Breakout Upper"
            score -= 1
        elif price < bb_lower: 
            bb_status = "Breakout Lower"
            score += 1
    else:
        print("Bollinger Columns not found:", df.columns) # لاگ برای دیباگ آینده

    # سیگنال نهایی
    final_signal = "neutral"
    if score >= 4: final_signal = "buy"
    elif score <= -4: final_signal = "sell"

    sl_val, tp_val = calculate_smart_sl_tp(price, final_signal, atr)

    return jsonify({
        "symbol": symbol,
        "price": round(price, 5),
        "signal": final_signal,
        "score": round(score, 1),
        "trend": trend,
        "htf_trend": htf_trend,
        "htf_status": htf_data_status,
        "indicators": {
            "rsi": round(float(rsi), 2),
            "atr": round(float(atr), 5),
            "macd": macd_status,
            "bb_pos": bb_status
        },
        "setup": {
            "sl": sl_val, 
            "tp": tp_val
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
