import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import warnings
from flask import Flask, request, jsonify, render_template

# ✅ ایمپورت ایمن TensorFlow (برای جلوگیری از کرش رم در Railway Free Tier)
tf = None
lstm_model = None
try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully.")
except ImportError:
    print("⚠️ TensorFlow not installed.")
except Exception as e:
    print(f"⚠️ TensorFlow import failed (Low RAM suspected). Error: {e}")

# ✅ ایمپورت ماژول دیتابیس
import database

# ---------------------------------------------------------
# ۱. پیکربندی
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__, template_folder='.')

# کلیدهای API
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "df521019db9f44899bfb172fdce6b454")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# پارامترها
RISK_REWARD_ATR = 1.5
SIGNAL_SCORE_THRESHOLD = 5.0
LSTM_TIME_STEPS = 10
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day", "1day": "1day" }

# متغیرهای سراسری مدل
GLOBAL_MODELS_LOADED = False
rf_model = None
lr_model = None
xgb_model = None
scaler = None
GLOBAL_RF_IMPORTANCES = {"RSI_14": 0.25, "ADX": 0.2, "EMA_Diff_Fast": 0.15} # پیش‌فرض

# ---------------------------------------------------------
# ۲. بارگذاری مدل‌ها
# ---------------------------------------------------------
try:
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        lr_model = joblib.load('models/lr_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        
        # تلاش برای بارگذاری LSTM
        if tf is not None and os.path.exists('models/lstm_model.h5'):
            try:
                lstm_model = tf.keras.models.load_model('models/lstm_model.h5', compile=False)
                print("✅ LSTM Model Loaded.")
            except Exception as lstm_e:
                print(f"⚠️ LSTM Load Failed (Low RAM?): {lstm_e}")
                lstm_model = None
        
        GLOBAL_MODELS_LOADED = True
        print("✅ Core AI Models Loaded.")
    else:
        print("⚠️ Warning: Models not found in 'models/'. AI disabled.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    GLOBAL_MODELS_LOADED = False

# ---------------------------------------------------------
# ۳. توابع کمکی
# ---------------------------------------------------------

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

def get_candles(symbol, interval, size=2000):
    """دریافت کندل‌ها با اولویت دیتابیس"""
    # 1. خواندن از دیتابیس
    df_db = database.get_all_candles(symbol, interval)
    
    # 2. دریافت دیتای جدید از API
    # اگر دیتابیس داریم، فقط جدیدها را می‌گیریم (مثلا 500 تا) وگرنه همه را
    req_size = 500 if not df_db.empty else size
    
    # تبدیل نماد برای API (حذف /)
    api_symbol = symbol.replace("/", "") if "/" in symbol else symbol
    
    url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={req_size}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "values" in data:
            df_new = pd.DataFrame(data["values"])
            cols = ['open', 'high', 'low', 'close', 'volume']
            for c in cols:
                df_new[c] = pd.to_numeric(df_new[c], errors='coerce')
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            df_new = df_new.dropna().iloc[::-1].reset_index(drop=True)
            
            # ذخیره در دیتابیس
            database.save_candles(df_new, symbol, interval)
            
            # ترکیب
            if not df_db.empty:
                df_final = pd.concat([df_db, df_new]).drop_duplicates(subset=['datetime'], keep='last')
                df_final = df_final.sort_values(by='datetime').reset_index(drop=True)
            else:
                df_final = df_new
                
            return df_final.tail(size).reset_index(drop=True)
            
    except Exception as e:
        print(f"⚠️ API Fetch Error: {e}")
    
    # بازگشت دیتابیس در صورت خرابی API
    return df_db.tail(size).reset_index(drop=True) if not df_db.empty else None

def process_data(df):
    if df.empty: return df
    
    # اندیکاتورها
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.donchian(lower_length=20, upper_length=20, append=True)
    
    # پر کردن مقادیر خالی احتمالی برای جلوگیری از خطا
    cols_to_check = ['RSI_14', 'RSI_6', 'ADX_14', 'ATRr_14', 'EMA_20', 'EMA_50', 'EMA_100']
    for col in cols_to_check:
        if col not in df.columns: df[col] = 0
        
    df['DCL'] = df.get('DCL_20_20', df['low'])
    df['DCU'] = df.get('DCU_20_20', df['high'])

    # ویژگی‌های ML
    df['Returns'] = df['close'].pct_change()
    df['Volatility'] = (df['high'] - df['low']) / df['close']
    df['EMA_Diff_Fast'] = (df['EMA_20'] - df['EMA_50']) / df['close']
    df['EMA_Diff_Slow'] = (df['EMA_50'] - df['EMA_100']) / df['close']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()
    
    return df.dropna().reset_index(drop=True)

def get_ml_prediction(df):
    report = {"ensemble_score": 0, "message": "AI: غیرفعال", "individual_results": {}, "ml_score_final": 0}
    
    if not GLOBAL_MODELS_LOADED or len(df) < LSTM_TIME_STEPS + 5:
        return 0, report

    try:
        feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        
        # بررسی وجود ستون‌ها
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            # اگر نام ستون‌ها متفاوت است (مثلا ADX به جای ADX_14)، اینجا اصلاح کنید
            if 'ADX' in df.columns and 'ADX_14' in missing: df['ADX_14'] = df['ADX']
            
        last_row = df.iloc[-1][feature_cols].to_frame().T
        input_scaled = scaler.transform(last_row)
        
        score_sum = 0
        count = 0
        
        # 1. RF
        p_rf = rf_model.predict_proba(input_scaled)[0][1]
        s_rf = (p_rf - 0.5) * 100
        score_sum += s_rf; count += 1
        report["individual_results"]["RF"] = {"prob": round(p_rf*100, 1), "score": round(s_rf, 1)}

        # 2. LR
        p_lr = lr_model.predict_proba(input_scaled)[0][1]
        s_lr = (p_lr - 0.5) * 100
        score_sum += s_lr; count += 1
        report["individual_results"]["LR"] = {"prob": round(p_lr*100, 1), "score": round(s_lr, 1)}

        # 3. XGB
        p_xgb = xgb_model.predict_proba(input_scaled)[0][1]
        s_xgb = (p_xgb - 0.5) * 100
        score_sum += s_xgb; count += 1
        report["individual_results"]["XGB"] = {"prob": round(p_xgb*100, 1), "score": round(s_xgb, 1)}
        
        # 4. LSTM
        if lstm_model:
            last_seq = df.iloc[-LSTM_TIME_STEPS:][feature_cols]
            seq_scaled = scaler.transform(last_seq).reshape(1, LSTM_TIME_STEPS, len(feature_cols))
            p_lstm = float(lstm_model.predict(seq_scaled, verbose=0)[0][0])
            s_lstm = (p_lstm - 0.5) * 100
            score_sum += s_lstm; count += 1
            report["individual_results"]["LSTM"] = {"prob": round(p_lstm*100, 1), "score": round(s_lstm, 1)}
        else:
             report["individual_results"]["LSTM"] = {"prob": 0, "score": 0, "status": "OFF"}

        # نهایی
        final_score = score_sum / count
        report["ensemble_score"] = round(score_sum, 1)
        report["ml_score_final"] = round(np.clip(final_score / 5.0, -10, 10), 1) # نرمال شده بین -10 تا 10
        
        direction = "Bullish 🟢" if final_score > 0 else "Bearish 🔴"
        if abs(final_score) < 5: direction = "Neutral ⚪"
        report["message"] = f"AI: {direction}"
        
        return report["ml_score_final"], report

    except Exception as e:
        print(f"❌ AI Inference Error: {e}")
        report["message"] = "خطای محاسبه AI"
        return 0, report

def calculate_sl_tp(price, signal, atr, dcl, dcu):
    if not atr: return 0, 0
    sl, tp = 0, 0
    if signal == 'buy':
        sl = max(dcl, price - (RISK_REWARD_ATR * atr))
        tp = price + (RISK_REWARD_ATR * (price - sl))
    elif signal == 'sell':
        sl = min(dcu, price + (RISK_REWARD_ATR * atr))
        tp = price - (RISK_REWARD_ATR * (sl - price))
    return round(sl, 5), round(tp, 5)

def get_sentiment(symbol):
    # شبیه‌سازی یا دریافت واقعی اگر API Key دارید
    return 0, "No News"

# ---------------------------------------------------------
# ۴. مسیرهای وب (Flask Routes)
# ---------------------------------------------------------

@app.route("/")
def index():
    # پشتیبانی از نام فایل جدید شما
    if os.path.exists("index (19).html"):
        return render_template("index (19).html")
    # پشتیبانی از نام استاندارد
    if os.path.exists("index.html"):
        return render_template("index.html")
    return "<h1>Please upload index.html</h1>"

# ✅ اصلاح مهم: پشتیبانی از متد POST و دریافت JSON
@app.route("/analyze", methods=["POST"]) 
def analyze():
    try:
        # 1. دریافت اطلاعات از JSON (هماهنگ با index (19).html)
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        symbol = data.get("symbol", "EUR/USD")
        interval = data.get("interval", "1h")
        use_htf = str(data.get("use_htf")).lower() == 'true'
        size = int(data.get("size", 2000))
        
        # 2. دریافت دیتا
        df = get_candles(symbol, interval, size)
        if df is None or len(df) < 50:
            return jsonify({"error": "Not enough data fetched"}), 500
            
        # 3. پردازش
        df = process_data(df)
        last = df.iloc[-1]
        
        # 4. هوش مصنوعی
        ml_score, ml_report = get_ml_prediction(df)
        score = ml_score
        
        # 5. تحلیل کلاسیک (ترکیب با امتیاز AI)
        trend = "Uptrend" if last['EMA_20'] > last['EMA_50'] else "Downtrend"
        rsi = last['RSI_14']
        adx = last['ADX_14']
        
        if trend == "Uptrend": score += 1
        else: score -= 1
        
        if rsi < 30: score += 2
        elif rsi > 70: score -= 2
        
        if adx > 25: score *= 1.2
        
        # سنتیمنت و HTF
        news_score, news_msg = get_sentiment(symbol)
        score += news_score
        
        htf_status = "غیرفعال"
        htf_trend = "N/A"
        if use_htf and interval in TIMEFRAME_MAP:
            htf_int = TIMEFRAME_MAP[interval]
            df_htf = get_candles(symbol, htf_int, 200)
            if df_htf is not None:
                df_htf.ta.ema(length=50, append=True)
                htf_last = df_htf.iloc[-1]
                htf_trend = "Bullish" if htf_last['close'] > htf_last['EMA_50'] else "Bearish"
                htf_status = f"Active: {htf_trend} ({htf_int})"
                if (htf_trend == "Bullish" and trend == "Uptrend") or (htf_trend == "Bearish" and trend == "Downtrend"):
                    score += 2
                else:
                    score -= 2

        # 6. سیگنال نهایی
        signal = "neutral"
        if score >= SIGNAL_SCORE_THRESHOLD: signal = "buy"
        elif score <= -SIGNAL_SCORE_THRESHOLD: signal = "sell"
        
        sl, tp = calculate_sl_tp(last['close'], signal, last['ATR_14'], last['DCL'], last['DCU'])
        
        # 7. پاسخ JSON
        response = {
            "symbol": symbol,
            "price": last['close'],
            "signal": signal,
            "setup": {"sl": sl, "tp": tp},
            "indicators": {
                "rsi": last['RSI_14'],
                "trend": trend,
                "macd": "Bullish" if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else "Bearish",
                "adx": adx,
                "regime": "Trending" if adx > 25 else "Ranging",
                "news": news_msg,
                "htf_status": htf_status,
                "htf_trend": htf_trend,
                "sr_levels": f"S: {round(last['DCL'], 4)} | R: {round(last['DCU'], 4)}",
                "divergence": "N/A",
                "ai_report": {
                    "message": ml_report["message"],
                    "ml_score_final": ml_report["ml_score_final"],
                    "individual_results": ml_report["individual_results"],
                    "accuracy": "N/A",
                    "importances": GLOBAL_RF_IMPORTANCES
                }
            }
        }
        
        return jsonify(convert_to_serializable(response))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # اجرای لوکال
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
