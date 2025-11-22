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

# نقشه تایم‌فریم‌ها برای تحلیل چند زمانی (تایم فعلی -> تایم بالاتر)
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
#  توابع کمکی و هسته پردازش (Pandas TA)
# =========================================================

def get_candles(symbol, interval, size=150):
    """دریافت دیتا و تبدیل به DataFrame"""
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={size}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "values" not in data:
            return None
        
        df = pd.DataFrame(data["values"])
        # تبدیل نوع داده‌ها به عددی
        df['open'] = pd.to_numeric(df['open'])
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        
        # معکوس کردن دیتا (از قدیم به جدید برای محاسبات صحیح)
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def apply_technical_analysis(df):
    """محاسبه اندیکاتورها با استفاده از Pandas-TA"""
    # محاسبه EMA
    df.ta.ema(length=20, append=True) # EMA_20
    df.ta.ema(length=50, append=True) # EMA_50
    
    # محاسبه RSI
    df.ta.rsi(length=14, append=True) # RSI_14
    
    # محاسبه MACD
    df.ta.macd(append=True) # MACD_12_26_9
    
    # محاسبه Bollinger Bands
    df.ta.bbands(length=20, std=2, append=True) # BBL_20_2.0, BBU_20_2.0
    
    # محاسبه ATR (برای حد ضرر هوشمند)
    df.ta.atr(length=14, append=True) # ATR_14

    return df

def calculate_smart_sl_tp(entry, signal, atr):
    """(Bonus Algorithm) محاسبه حد سود و ضرر داینامیک"""
    if not atr or np.isnan(atr):
        return None, None
        
    atr_multiplier_sl = 1.5  # حد ضرر ۱.۵ برابر ATR
    risk_reward_ratio = 2.0  # نسبت ریسک به ریوارد ۱ به ۲
    
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
#  روت اصلی تحلیل (ارتقا یافته)
# =========================================================

@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    
    # 1. دریافت دیتای تایم‌فریم اصلی
    df = get_candles(symbol, interval)
    if df is None:
        return jsonify({"error": "خطا در دریافت دیتای اصلی"})
    
    df = apply_technical_analysis(df)
    
    # آخرین ردیف دیتا (کندل جاری/بسته شده اخیر)
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    # استخراج مقادیر اصلی
    price = last_row['close']
    rsi = last_row['RSI_14']
    ema20 = last_row['EMA_20']
    ema50 = last_row['EMA_50']
    atr = last_row['ATRr_14']
    
    # 2. تحلیل روند (تایم اصلی)
    trend = "uptrend" if ema20 > ema50 else "downtrend"
    
    # 3. تحلیل چند زمانی (Multi-Timeframe) - ویژگی "پرو"
    htf_interval = TIMEFRAME_MAP.get(interval)
    htf_trend = "neutral"
    htf_data_status = "آماده نیست"
    
    if htf_interval:
        df_htf = get_candles(symbol, htf_interval, size=50)
        if df_htf is not None:
            df_htf.ta.ema(length=20, append=True)
            df_htf.ta.ema(length=50, append=True)
            last_htf = df_htf.iloc[-1]
            if last_htf['EMA_20'] > last_htf['EMA_50']:
                htf_trend = "uptrend"
            else:
                htf_trend = "downtrend"
            htf_data_status = f"تحلیل شده ({htf_interval})"

    # 4. سیستم امتیازدهی پیشرفته
    score = 0
    
    # الف) امتیاز روند
    if trend == "uptrend": score += 2
    else: score -= 2
    
    # ب) امتیاز همگرایی تایم‌فریم‌ها (تاییدیه قوی)
    if trend == htf_trend:
        score += 3 if trend == "uptrend" else -3
    else:
        # اگر خلاف جهت هم باشند، از قدرت سیگنال کم می‌شود
        score = score / 2 

    # ج) اسیلاتور RSI
    if rsi < 30: score += 2  # اشباع فروش (سیگنال خرید)
    elif rsi > 70: score -= 2 # اشباع خرید (سیگنال فروش)
    
    # د) کراس MACD
    macd_line = last_row['MACD_12_26_9']
    macd_signal = last_row['MACDs_12_26_9']
    if macd_line > macd_signal: score += 1
    elif macd_line < macd_signal: score -= 1

    # ه) وضعیت در بولینگر باند
    bb_upper = last_row['BBU_20_2.0']
    bb_lower = last_row['BBL_20_2.0']
    bb_status = "Inside"
    if price > bb_upper: 
        bb_status = "Breakout Upper"
        score -= 1 # احتمال اصلاح
    elif price < bb_lower: 
        bb_status = "Breakout Lower"
        score += 1 # احتمال اصلاح

    # تعیین سیگنال نهایی
    final_signal = "neutral"
    if score >= 4: final_signal = "buy"
    elif score <= -4: final_signal = "sell"

    # 5. محاسبه مدیریت ریسک (Bonus Algorithm)
    sl_val, tp_val = calculate_smart_sl_tp(price, final_signal, atr)

    # آماده‌سازی پاسخ
    return jsonify({
        "symbol": symbol,
        "price": round(price, 5),
        "signal": final_signal,
        "score": round(score, 1),
        "trend": trend,
        "htf_trend": htf_trend,     # نتیجه تحلیل تایم بالا
        "htf_status": htf_data_status,
        "indicators": {
            "rsi": round(rsi, 2),
            "atr": round(atr, 5),
            "macd": "Bullish" if macd_line > macd_signal else "Bearish",
            "bb_pos": bb_status
        },
        "setup": {
            "sl": sl_val, 
            "tp": tp_val,
            "risk_reward": "1:2"
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
