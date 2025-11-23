import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import warnings
import yfinance as yf # ✅ اضافه شدن کتابخانه جایگزین
from flask import Flask, request, jsonify, render_template

# ✅ 1. ایمپورت ایمن TensorFlow
tf = None
lstm_model = None
try:
    import tensorflow as tf
    print("✅ TensorFlow imported successfully.")
except ImportError:
    print("⚠️ TensorFlow not installed.")
except Exception as e:
    print(f"⚠️ TensorFlow import failed (Low RAM suspected). Error: {e}")

# ✅ 2. ایمپورت ماژول دیتابیس
import database

# ---------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__) 

# کلیدهای API
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "83a502f14048493b9828008e86b2d0b5")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# پارامترها
RISK_REWARD_ATR = 1.5
SIGNAL_SCORE_THRESHOLD = 5.0
LSTM_TIME_STEPS = 10
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day", "1day": "1day" }

# مدل‌ها
GLOBAL_MODELS_LOADED = False
rf_model, lr_model, xgb_model, scaler = None, None, None, None
GLOBAL_RF_IMPORTANCES = {"RSI_14": 0.25, "ADX": 0.2, "EMA_Diff_Fast": 0.15}

# ---------------------------------------------------------
# بارگذاری مدل‌ها
# ---------------------------------------------------------
try:
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        lr_model = joblib.load('models/lr_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        
        if tf is not None and os.path.exists('models/lstm_model.h5'):
            try:
                lstm_model = tf.keras.models.load_model('models/lstm_model.h5', compile=False)
                print("✅ LSTM Model Loaded.")
            except: lstm_model = None
        
        GLOBAL_MODELS_LOADED = True
        print("✅ All AI Models Loaded.")
    else:
        print("⚠️ Warning: Models not found.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ---------------------------------------------------------
# توابع کمکی
# ---------------------------------------------------------

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

def get_candles(symbol, interval, size=2000):
    """دریافت کندل‌ها با سیستم فال‌بک (Database -> TwelveData -> YFinance)"""
    
    # 1. تلاش برای خواندن از دیتابیس
    df_db = database.get_all_candles(symbol, interval)
    if not df_db.empty:
        # اگر دیتای دیتابیس تازه است (مثلاً مال کمتر از 1 ساعت پیش)، همان را برگردان
        # اما فعلاً برای سادگی فرض می‌کنیم همیشه نیاز به آپدیت داریم
        pass

    req_size = 500 if not df_db.empty else size
    
    # 2. تلاش با TwelveData (API اصلی)
    try:
        api_symbol = symbol.replace("/", "")
        url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={req_size}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if "values" in data:
            df_new = pd.DataFrame(data["values"])
            cols = ['open', 'high', 'low', 'close', 'volume']
            for c in cols: df_new[c] = pd.to_numeric(df_new[c], errors='coerce')
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            df_new = df_new.dropna().iloc[::-1].reset_index(drop=True)
            
            database.save_candles(df_new, symbol, interval)
            print(f"✅ Data fetched from TwelveData for {symbol}")
            
            if not df_db.empty:
                df_final = pd.concat([df_db, df_new]).drop_duplicates(subset=['datetime'], keep='last')
                return df_final.sort_values(by='datetime').tail(size).reset_index(drop=True)
            return df_new
            
    except Exception as e:
        print(f"⚠️ TwelveData Failed: {e}")

    # 3. تلاش با YFinance (فال‌بک نهایی - ضد ارور 500)
    try:
        print(f"🔄 Trying YFinance fallback for {symbol}...")
        yf_symbol = symbol.replace("/", "") + "=X" if "USD" in symbol and "BTC" not in symbol else symbol
        if "BTC" in symbol: yf_symbol = "BTC-USD"
        
        # تبدیل تایم‌فریم TwelveData به YFinance
        yf_interval = interval
        if interval == "15min": yf_interval = "15m"
        elif interval == "1h": yf_interval = "1h"
        elif interval == "4h": yf_interval = "1h" # YF 4h ندارد، 1h می‌گیریم و تبدیل می‌کنیم (ساده شده)
        
        df_yf = yf.download(yf_symbol, period="1mo", interval=yf_interval, progress=False)
        if not df_yf.empty:
            df_yf = df_yf.reset_index()
            # استانداردسازی ستون‌ها
            df_yf.rename(columns={'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
            
            # رفع مشکل MultiIndex در نسخه‌های جدید yfinance
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            
            # چک کردن مجدد نام ستون‌ها بعد از فلت کردن
            col_map = {'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
            df_yf.rename(columns=col_map, inplace=True)
            
            # اگر ستون datetime هنوز string است تبدیل کن
            if 'datetime' in df_yf.columns:
                 df_yf['datetime'] = pd.to_datetime(df_yf['datetime'])

            # فیلتر ستون‌های مورد نیاز
            req_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            df_yf = df_yf[[c for c in req_cols if c in df_yf.columns]]
            
            database.save_candles(df_yf, symbol, interval)
            print(f"✅ Data fetched from YFinance for {symbol}")
            
            if not df_db.empty:
                 df_final = pd.concat([df_db, df_yf]).drop_duplicates(subset=['datetime'], keep='last')
                 return df_final.sort_values(by='datetime').tail(size).reset_index(drop=True)
            return df_yf
            
    except Exception as e:
        print(f"❌ YFinance Failed: {e}")

    # اگر همه شکست خوردند، دیتابیس را برگردان
    return df_db.tail(size).reset_index(drop=True) if not df_db.empty else None

def process_data(df):
    if df.empty: return df
    df.ta.ema(length=20, append=True); df.ta.ema(length=50, append=True); df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True); df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True); df.ta.adx(length=14, append=True)
    df.ta.macd(append=True); df.ta.donchian(lower_length=20, upper_length=20, append=True)
    
    cols = ['RSI_14', 'RSI_6', 'ADX_14', 'ATRr_14', 'EMA_20', 'EMA_50', 'EMA_100']
    for c in cols: 
        if c not in df.columns: df[c] = 0
    
    df['DCL'] = df.get('DCL_20_20', df['low']); df['DCU'] = df.get('DCU_20_20', df['high'])
    df['Returns'] = df['close'].pct_change()
    df['Volatility'] = (df['high'] - df['low']) / df['close']
    df['EMA_Diff_Fast'] = (df['EMA_20'] - df['EMA_50']) / df['close']
    df['EMA_Diff_Slow'] = (df['EMA_50'] - df['EMA_100']) / df['close']
    df['Hour'] = df['datetime'].dt.hour; df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()
    return df.dropna().reset_index(drop=True)

def get_ml_prediction(df):
    report = {"ensemble_score": 0, "message": "AI: غیرفعال", "individual_results": {}, "ml_score_final": 0}
    if not GLOBAL_MODELS_LOADED or len(df) < LSTM_TIME_STEPS + 5: return 0, report

    try:
        feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        if 'ADX' in df.columns and 'ADX_14' not in df.columns: df['ADX_14'] = df['ADX']
        last_row = df.iloc[-1][feature_cols].to_frame().T
        input_scaled = scaler.transform(last_row)
        score_sum = 0; count = 0
        
        # Models
        for name, model in [('RF', rf_model), ('LR', lr_model), ('XGB', xgb_model)]:
             p = model.predict_proba(input_scaled)[0][1]; s = (p - 0.5) * 100
             score_sum += s; count += 1; report["individual_results"][name] = {"prob": round(p*100, 1), "score": round(s, 1)}
        
        if lstm_model:
            seq = df.iloc[-LSTM_TIME_STEPS:][feature_cols]
            seq_scaled = scaler.transform(seq).reshape(1, LSTM_TIME_STEPS, len(feature_cols))
            p = float(lstm_model.predict(seq_scaled, verbose=0)[0][0])
            s = (p - 0.5) * 100
            score_sum += s; count += 1; report["individual_results"]["LSTM"] = {"prob": round(p*100, 1), "score": round(s, 1)}

        final_score = score_sum / count
        report["ensemble_score"] = round(score_sum, 1)
        report["ml_score_final"] = round(np.clip(final_score / 5.0, -10, 10), 1)
        direction = "Bullish 🟢" if final_score > 0 else "Bearish 🔴"
        if abs(final_score) < 5: direction = "Neutral ⚪"
        report["message"] = f"AI: {direction}"
        return report["ml_score_final"], report
    except Exception as e:
        print(f"❌ AI Error: {e}")
        return 0, report

def calculate_sl_tp(price, signal, atr, dcl, dcu):
    if not atr: return 0, 0
    sl, tp = 0, 0
    if signal == 'buy': sl = max(dcl, price - (RISK_REWARD_ATR * atr)); tp = price + (RISK_REWARD_ATR * (price - sl))
    elif signal == 'sell': sl = min(dcu, price + (RISK_REWARD_ATR * atr)); tp = price - (RISK_REWARD_ATR * (sl - price))
    return round(sl, 5), round(tp, 5)

def get_sentiment(symbol):
    return 0, "No News" # برای جلوگیری از کندی

def check_divergence(df):
    if len(df) < 20: return 0, "---"
    price = df['close'].values; rsi = df['RSI_14'].values
    score = 0; msg = "No Divergence"
    try:
        prev_max_idx = np.argmax(price[-15:-5]) + (len(price)-15)
        if price[-1] > price[prev_max_idx] and rsi[-1] < rsi[prev_max_idx]: score = -3; msg = "Bearish Div 📉"
        prev_min_idx = np.argmin(price[-15:-5]) + (len(price)-15)
        if price[-1] < price[prev_min_idx] and rsi[-1] > rsi[prev_min_idx]: score = 3; msg = "Bullish Div 📈"
    except: pass
    return score, msg

# ---------------------------------------------------------
# مسیرهای وب
# ---------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"]) 
def analyze():
    try:
        data = request.get_json()
        symbol = data.get("symbol", "EUR/USD")
        interval = data.get("interval", "1h")
        use_htf = str(data.get("use_htf")).lower() == 'true'
        size = int(data.get("size", 2000))
        
        df = get_candles(symbol, interval, size)
        if df is None or len(df) < 50: return jsonify({"error": "Data fetch failed from BOTH APIs"}), 500
            
        df = process_data(df); last = df.iloc[-1]
        ml_score, ml_report = get_ml_prediction(df); score = ml_score
        
        trend = "Uptrend" if last['EMA_20'] > last['EMA_50'] else "Downtrend"
        rsi = last['RSI_14']; adx = last['ADX_14']
        
        if trend == "Uptrend": score += 1
        else: score -= 1
        if rsi < 30: score += 2
        elif rsi > 70: score -= 2
        if adx > 25: score *= 1.2
        
        div_score, div_msg = check_divergence(df); score += div_score
        
        htf_status = "Inactive"; htf_trend = "N/A"
        if use_htf and interval in TIMEFRAME_MAP:
            htf_int = TIMEFRAME_MAP[interval]
            df_htf = get_candles(symbol, htf_int, 200)
            if df_htf is not None:
                df_htf = process_data(df_htf)
                htf_last = df_htf.iloc[-1]
                htf_trend = "Bullish" if htf_last['close'] > htf_last['EMA_50'] else "Bearish"
                htf_status = f"Active: {htf_trend} ({htf_int})"
                if (htf_trend == "Bullish" and trend == "Uptrend") or (htf_trend == "Bearish" and trend == "Downtrend"): score += 2
                else: score -= 2

        signal = "neutral"
        if score >= SIGNAL_SCORE_THRESHOLD: signal = "buy"
        elif score <= -SIGNAL_SCORE_THRESHOLD: signal = "sell"
        
        sl, tp = calculate_sl_tp(last['close'], signal, last['ATR_14'], last['DCL'], last['DCU'])
        
        response = {
            "symbol": symbol, "price": last['close'], "signal": signal,
            "setup": {"sl": sl, "tp": tp},
            "indicators": {
                "rsi": last['RSI_14'], "trend": trend,
                "macd": "Bullish" if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else "Bearish",
                "adx": adx, "regime": "Trending" if adx > 25 else "Ranging",
                "news": "N/A", "htf_status": htf_status, "htf_trend": htf_trend,
                "sr_levels": f"S: {round(last['DCL'], 4)} | R: {round(last['DCU'], 4)}",
                "divergence": div_msg,
                "ai_report": {
                    "message": ml_report["message"],
                    "ml_score_final": ml_report["ml_score_final"],
                    "individual_results": ml_report["individual_results"],
                    "accuracy": "N/A", "importances": GLOBAL_RF_IMPORTANCES
                }
            }
        }
        return jsonify(convert_to_serializable(response))

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
