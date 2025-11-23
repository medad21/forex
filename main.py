import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import warnings
import yfinance as yf
import traceback
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
try:
    import database
except ImportError:
    print("⚠️ database.py not found. Saving disabled.")

# ---------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__) 

API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "f24a3dec20104e639d1995e42dc4673c")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

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
    """ دریافت کندل با سیستم فال‌بک قوی (Database -> TwelveData -> YFinance) """
    
    # 1. Database Check
    try:
        df_db = database.get_all_candles(symbol, interval)
    except:
        df_db = pd.DataFrame()

    req_size = 500 if not df_db.empty else size
    
    # 2. TwelveData API
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
            
            try: database.save_candles(df_new, symbol, interval)
            except: pass
            
            if not df_db.empty:
                df_final = pd.concat([df_db, df_new]).drop_duplicates(subset=['datetime'], keep='last')
                return df_final.sort_values(by='datetime').tail(size).reset_index(drop=True)
            return df_new
    except Exception as e:
        print(f"⚠️ TwelveData Error: {e}")

    # 3. YFinance Fallback (ضد ارور 500)
    try:
        print(f"🔄 Trying YFinance fallback for {symbol}...")
        if "BTC" in symbol: yf_symbol = "BTC-USD"
        elif "ETH" in symbol: yf_symbol = "ETH-USD"
        elif "XAU" in symbol: yf_symbol = "GC=F"
        elif "EUR" in symbol: yf_symbol = "EURUSD=X"
        elif "GBP" in symbol: yf_symbol = "GBPUSD=X"
        else: yf_symbol = symbol.replace("/", "") + "=X"
        
        yf_int = "1h"
        if interval == "15min": yf_int = "15m"
        elif interval == "1day": yf_int = "1d"
        
        df_yf = yf.download(yf_symbol, period="1mo", interval=yf_int, progress=False)
        
        if not df_yf.empty:
            df_yf = df_yf.reset_index()
            if isinstance(df_yf.columns, pd.MultiIndex):
                df_yf.columns = df_yf.columns.get_level_values(0)
            
            rename_map = {'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
            df_yf.rename(columns=rename_map, inplace=True)
            
            if 'datetime' in df_yf.columns:
                df_yf['datetime'] = pd.to_datetime(df_yf['datetime']).dt.tz_localize(None)

            req_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            df_yf = df_yf[[c for c in req_cols if c in df_yf.columns]].dropna()

            try: database.save_candles(df_yf, symbol, interval)
            except: pass

            if not df_db.empty:
                 df_final = pd.concat([df_db, df_yf]).drop_duplicates(subset=['datetime'], keep='last')
                 return df_final.sort_values(by='datetime').tail(size).reset_index(drop=True)
            return df_yf
            
    except Exception as e:
        print(f"❌ YFinance Error: {e}")

    return df_db.tail(size).reset_index(drop=True) if not df_db.empty else None

def process_data(df):
    """ محاسبه اندیکاتورها با محافظت در برابر خطا (Fix ATR Error) """
    if df is None or df.empty: return df
    
    try:
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
        
        # ✅ تعمیر نام ستون‌ها
        if 'ATRr_14' in df.columns: df['ATR_14'] = df['ATRr_14']
        if 'ADX_14' not in df.columns and 'ADX' in df.columns: df['ADX_14'] = df['ADX']
        
        # پر کردن ستون‌های حیاتی اگر تولید نشدند (ضد کرش ATR_14)
        required_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'ATR_14', 'EMA_20', 'EMA_50', 'EMA_100']
        for col in required_cols:
            if col not in df.columns: df[col] = 0 
        
        df['DCL'] = df.get('DCL_20_20', df['low'])
        df['DCU'] = df.get('DCU_20_20', df['high'])

        # ویژگی‌ها
        df['Returns'] = df['close'].pct_change()
        df['Volatility'] = (df['high'] - df['low']) / df['close']
        # از .get() استفاده می‌کنیم تا اگر EMAها نبودند کرش نکند
        df['EMA_Diff_Fast'] = (df.get('EMA_20', 0) - df.get('EMA_50', 0)) / df['close']
        df['EMA_Diff_Slow'] = (df.get('EMA_50', 0) - df.get('EMA_100', 0)) / df['close']
        df['Hour'] = df['datetime'].dt.hour
        df['DayOfWeek'] = df['datetime'].dt.dayofweek
        df['HV_20'] = df['Returns'].rolling(20).std()
        
        return df.dropna().reset_index(drop=True)
        
    except Exception as e:
        print(f"⚠️ Error in process_data: {e}")
        traceback.print_exc()
        return df 

def get_ml_prediction(df):
    report = {"ensemble_score": 0, "message": "AI: غیرفعال", "individual_results": {}, "ml_score_final": 0}
    if not GLOBAL_MODELS_LOADED or len(df) < LSTM_TIME_STEPS + 5: return 0, report

    try:
        feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        
        for col in feature_cols:
            if col not in df.columns: df[col] = 0
            
        last_row = df.iloc[-1][feature_cols].to_frame().T
        input_scaled = scaler.transform(last_row)
        score_sum = 0; count = 0
        
        for name, model in [('RF', rf_model), ('LR', lr_model), ('XGB', xgb_model)]:
             if model:
                 p = model.predict_proba(input_scaled)[0][1]; s = (p - 0.5) * 100
                 score_sum += s; count += 1; report["individual_results"][name] = {"prob": round(p*100, 1), "score": round(s, 1)}
        
        if lstm_model:
            seq = df.iloc[-LSTM_TIME_STEPS:][feature_cols]
            seq_scaled = scaler.transform(seq).reshape(1, LSTM_TIME_STEPS, len(feature_cols))
            p = float(lstm_model.predict(seq_scaled, verbose=0)[0][0])
            s = (p - 0.5) * 100
            score_sum += s; count += 1; report["individual_results"]["LSTM"] = {"prob": round(p*100, 1), "score": round(s, 1)}

        if count > 0:
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
    try:
        if not atr or atr == 0: return 0, 0
        sl, tp = 0, 0
        if signal == 'buy': sl = max(dcl, price - (RISK_REWARD_ATR * atr)); tp = price + (RISK_REWARD_ATR * (price - sl))
        elif signal == 'sell': sl = min(dcu, price + (RISK_REWARD_ATR * atr)); tp = price - (RISK_REWARD_ATR * (sl - price))
        return round(sl, 5), round(tp, 5)
    except: return 0, 0

def get_sentiment(symbol):
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol: av_symbol = "CRYPTO:BTC"
        elif "XAU" in symbol: av_symbol = "FOREX:XAUUSD"
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=3)
        data = r.json()
        if "feed" in data and data["feed"]:
            item = data["feed"][0]
            score = float(item.get("overall_sentiment_score", 0))
            label = item.get("overall_sentiment_label", "Neutral")
            return score * 2, f"{label} ({score})"
    except: pass
    return 0, "No News / API Limit"

def check_divergence(df):
    if len(df) < 20: return 0, "---"
    if 'RSI_14' not in df.columns: return 0, "--- (RSI Missing)"

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
        if not data: return jsonify({"error": "No JSON"}), 400

        symbol = data.get("symbol", "EUR/USD")
        interval = data.get("interval", "1h")
        use_htf = str(data.get("use_htf")).lower() == 'true'
        size = int(data.get("size", 2000))
        
        # 1. دریافت دیتا
        df = get_candles(symbol, interval, size)
        
        # اگر دیتا کلا نیامد
        if df is None or len(df) < 50: 
            return jsonify({"error": "Could not fetch data from any source (TwelveData/YF/DB). Check symbol/interval or API keys."}), 500
            
        # 2. پردازش و محاسبه اندیکاتورها
        df = process_data(df)
        
        # 🛑🛑🛑 3. بررسی ایمنی جدید: اگر DataFrame بعد از محاسبات خالی شد 🛑🛑🛑
        if df.empty:
            # این ارور جدید از کرش df.iloc[-1] جلوگیری می‌کند
            return jsonify({"error": "Data not clean. Too few candles available or too many NaN values after indicator calculation."}), 500

        last = df.iloc[-1]
        
        # هوش مصنوعی
        ml_score, ml_report = get_ml_prediction(df)
        score = ml_score
        
        # دسترسی ایمن به ستون‌ها (جهت جلوگیری از ATR_14 KeyError)
        current_atr = last.get('ATR_14', 0)
        current_dcl = last.get('DCL', last['low'])
        current_dcu = last.get('DCU', last['high'])
        
        # تحلیل تکنیکال
        trend = "Uptrend" if last.get('EMA_20', 0) > last.get('EMA_50', 0) else "Downtrend"
        rsi = last.get('RSI_14', 50); adx = last.get('ADX_14', 0)
        
        if trend == "Uptrend": score += 1
        else: score -= 1
        if rsi < 30: score += 2
        elif rsi > 70: score -= 2
        if adx > 25: score *= 1.2
        
        news_score, news_msg = get_sentiment(symbol); score += news_score
        div_score, div_msg = check_divergence(df); score += div_score
        
        htf_status = "Inactive"; htf_trend = "N/A"
        if use_htf and interval in TIMEFRAME_MAP:
            htf_int = TIMEFRAME_MAP[interval]
            df_htf = get_candles(symbol, htf_int, 200)
            if df_htf is not None:
                df_htf = process_data(df_htf)
                htf_last = df_htf.iloc[-1]
                htf_trend = "Bullish" if htf_last.get('EMA_20', 0) > htf_last.get('EMA_50', 0) else "Bearish"
                htf_status = f"Active: {htf_trend} ({htf_int})"
                if (htf_trend == "Bullish" and trend == "Uptrend") or (htf_trend == "Bearish" and trend == "Downtrend"): score += 2
                else: score -= 2

        signal = "neutral"
        if score >= SIGNAL_SCORE_THRESHOLD: signal = "buy"
        elif score <= -SIGNAL_SCORE_THRESHOLD: signal = "sell"
        
        # محاسبه SL/TP با استفاده از ATR ایمن
        sl, tp = calculate_sl_tp(last['close'], signal, current_atr, current_dcl, current_dcu)
        
        response = {
            "symbol": symbol, "price": last['close'], "signal": signal,
            "score": round(score, 1),
            "setup": {"sl": sl, "tp": tp},
            "indicators": {
                "rsi": rsi, "trend": trend,
                "macd": "Bullish" if last.get('MACD_12_26_9', 0) > last.get('MACDs_12_26_9', 0) else "Bearish",
                "adx": adx, "regime": "Trending" if adx > 25 else "Ranging",
                "news": news_msg, "htf_status": htf_status, "htf_trend": htf_trend,
                "sr_levels": f"S: {round(current_dcl, 4)} | R: {round(current_dcu, 4)}",
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
        print("❌ CRITICAL SERVER ERROR:")
        traceback.print_exc()
        # ارور دقیق را برای فرانت می‌فرستیم
        return jsonify({"error": f"Server Logic Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
