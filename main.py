import os
import json
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import joblib 
import sqlite3 # فقط برای سازگاری با database.py اگر از SQLite استفاده شود
from flask import Flask, request, jsonify, render_template

# ✅ ایمپورت ایمن TensorFlow
tf = None
lstm_model = None
try:
    import tensorflow as tf
    # اطمینان حاصل شود که Tensorflow در محیط Railway نصب و قابل استفاده است
    print("✅ TensorFlow imported successfully.")
except ImportError:
    print("⚠️ TensorFlow not installed or failed to import.")
except Exception as e:
    print(f"⚠️ TensorFlow import failed (Low RAM suspected). Error: {e}")

# ✅ ایمپورت ماژول دیتابیس (database.py باید در کنار این فایل باشد)
import database

# ---------------------------------------------------------
# ۱. پیکربندی
# ---------------------------------------------------------
warnings.filterwarnings('ignore')

# 💡 اصلاح مسیر نهایی: استفاده از پوشه پیش‌فرض 'templates' (که 'index.html' در آن قرار دارد)
app = Flask(__name__) 

# کلیدهای API (از Environment Variables بخوانید)
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "df521019db9f44899bfb172fdce6b454")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# پارامترهای استراتژی
RISK_REWARD_ATR = 1.5           
SIGNAL_SCORE_THRESHOLD = 5.0    
LSTM_TIME_STEPS = 10 
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
ML_SCORE_NORMALIZER = 40.0 
TARGET_PERIODS = 5

# متغیرهای سراسری
GLOBAL_RF_IMPORTANCES = {} 
GLOBAL_TEST_ACCURACY = "N/A (Offline Training Required)"
rf_model, lr_model, xgb_model, scaler = None, None, None, None

# ---------------------------------------------------------
# ۲. لود مدل‌ها و ابزارهای مورد نیاز
# ---------------------------------------------------------

def load_models():
    """لود مدل‌های آموزش دیده و StandardScaler."""
    global rf_model, lr_model, xgb_model, lstm_model, scaler, GLOBAL_RF_IMPORTANCES

    try:
        # لود مدل‌های pkl
        rf_model = joblib.load('models/rf_model.pkl')
        lr_model = joblib.load('models/lr_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')

        # لود مدل h5/keras 
        if tf:
            # اطمینان حاصل شود که مسیر ذخیره مدل LSTM شما 'models/lstm_model.h5' باشد.
            lstm_model = tf.keras.models.load_model('models/lstm_model.h5') 
            print("✅ LSTM Model Loaded.")

        # لود Feature Importances
        if hasattr(rf_model, 'feature_importances_') and hasattr(scaler, 'feature_names_in_'):
            feature_names = scaler.feature_names_in_
            importances = dict(zip(feature_names, rf_model.feature_importances_))
            sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
            GLOBAL_RF_IMPORTANCES = {k: round(float(v), 2) for k, v in sorted_importances[:3]}
        
        print("✅ Core AI Models Loaded Successfully.")

    except Exception as e:
        print(f"❌ Error loading models: {e}. Using dummy models.")
        # اگر مدل‌ها لود نشدند، مدل‌ها None باقی می‌مانند و توسط analyze_market مدیریت می‌شوند.


# ---------------------------------------------------------
# ۳. توابع اصلی تحلیل (کامل شده)
# ---------------------------------------------------------

def fetch_data(symbol, interval, size=2000):
    """دریافت داده‌های کندل از TwelveData و ذخیره در DB."""
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={size}&apikey={API_KEY_TWELVEDATA}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'values' not in data or not data['values']:
            print(f"⚠️ TwelveData: No data for {symbol}/{interval}. Response: {data}")
            # بازگشت به دیتابیس در صورت نبود داده جدید
            return database.get_all_candles(symbol, interval)
        
        df = pd.DataFrame(data['values'])
        df = df.astype(float)
        df.rename(columns={'datetime': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.iloc[::-1] # مرتب‌سازی برای جدیدترین داده در انتها

        # ذخیره داده‌های جدید در دیتابیس
        # به دلیل اینکه ستون‌های اندیکاتور در اینجا نیستند، فقط open/high/low/close/volume را ذخیره می‌کنیم
        database.save_candles(df[['open', 'high', 'low', 'close', 'volume']].reset_index(), symbol, interval)
        
        # ترکیب با داده‌های قدیمی ذخیره شده
        existing_df = database.get_all_candles(symbol, interval)
        if existing_df is not None and not existing_df.empty:
            # ترکیب دو دیتافریم و حذف تکراری‌ها بر اساس datetime
            combined_df = pd.concat([existing_df.set_index('datetime'), df[['open', 'high', 'low', 'close', 'volume']]])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
            return combined_df.tail(size)
        
        return df

    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Error: {e}. Falling back to DB only.")
        return database.get_all_candles(symbol, interval)
    except Exception as e:
        print(f"❌ Data Processing Error: {e}")
        return database.get_all_candles(symbol, interval)


def calculate_indicators(df):
    """محاسبه تمام اندیکاتورها و ویژگی‌های لازم."""
    df = df.copy()
    if df.empty: return df

    # اندیکاتورهای اصلی
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    
    # کانال دنچیان (برای محاسبه S/R)
    dc = df.ta.donchian(lower_length=20, upper_length=20, append=True)
    if 'DCL_20' in dc.columns:
        df['DCL'] = dc['DCL_20'] # Donchian Channel Lower
        df['DCU'] = dc['DCU_20'] # Donchian Channel Upper
    else:
        # اگر pandas_ta نام ستون‌ها را تغییر داد
        df['DCL'] = df['close'].rolling(20).min()
        df['DCU'] = df['close'].rolling(20).max()


    # MACD
    macd = df.ta.macd(append=True)
    if 'MACD_12_26_9' in macd.columns:
        df['MACD_12_26_9'] = macd['MACD_12_26_9']
        df['MACDs_12_26_9'] = macd['MACDs_12_26_9']
    else:
        # اگر pandas_ta نام ستون‌ها را تغییر داد، از محاسبه دستی استفاده کنید
        df.ta.macd(append=True, fast=12, slow=26, signal=9) # این ستون‌های پیش‌فرض را اضافه می‌کند

    
    # ساخت ویژگی‌های ML (باید با ویژگی‌های train.py مطابقت داشته باشد)
    df['EMA_Diff_20_50'] = df['EMA_20'] - df['EMA_50']
    df['RSI_Norm'] = df['RSI_14'] / 100
    df['Price_ATR_Ratio'] = (df['close'] - df['DCL']) / df['ATR_14'] # از ستون ATR_14 استفاده می‌کنیم
    df['Vol_HV'] = df['high'] - df['low'] # نوسان داخلی
    df['Vol_Close_Change'] = df['close'].pct_change() # تغییر قیمت
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    
    # حذف NaN
    df = df.dropna()
    return df

def predict_ensemble_signal(df):
    """اجرای مدل‌های Ensemble و محاسبه امتیاز نهایی سیگنال."""
    if df.empty or scaler is None or rf_model is None:
        return {"ensemble_score": 0, "ml_score_final": 0, "individual_results": {}, "message": "No data or models loaded."}
    
    # داده‌ها را برای ML آماده می‌کنیم (فقط ستون‌های مورد استفاده در آموزش)
    try:
        feature_columns = scaler.feature_names_in_
    except AttributeError:
        # اگر feature_names_in_ لود نشده بود، باید لیستی از ستون‌های ویژگی را دستی فراهم کنیم
        # این لیست باید دقیقاً با ویژگی‌های مورد استفاده در train.py شما مطابقت داشته باشد!
        feature_columns = ['open', 'high', 'low', 'close', 'volume', 'Returns', 
                           'EMA_20', 'EMA_50', 'EMA_100', 'RSI_14', 'ATR_14', 
                           'ADX_14', 'DCL', 'DCU', 'MACD_12_26_9', 'MACDs_12_26_9', 
                           'EMA_Diff_20_50', 'RSI_Norm', 'Price_ATR_Ratio', 'Vol_HV', 
                           'Vol_Close_Change', 'Hour', 'DayOfWeek']


    # بررسی می‌کنیم که آیا دیتافریم شامل تمام ستون‌های مورد نیاز هست یا نه
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        return {"ensemble_score": 0, "ml_score_final": 0, "individual_results": {}, 
                "message": f"Missing features for scaling: {', '.join(missing_cols)}"}


    # برای LSTM به اندازه TIME_STEPS + 1 داده نیاز داریم.
    X_latest_full = df[feature_columns].tail(LSTM_TIME_STEPS + 1).copy()
    
    if len(X_latest_full) < LSTM_TIME_STEPS + 1:
        return {"ensemble_score": 0, "ml_score_final": 0, "individual_results": {}, 
                "message": "Not enough data for prediction (Need at least 11 candles)."}

    # داده برای مدل‌های کلاسیک (آخرین کندل)
    X_scalar_raw = X_latest_full.iloc[-1].values.reshape(1, -1)
    X_scaled = scaler.transform(X_scalar_raw)
    
    # ۱. پیش‌بینی مدل‌های کلاسیک (RF, LR, XGB)
    rf_pred_prob = rf_model.predict_proba(X_scaled)[0][1]
    lr_pred_prob = lr_model.predict_proba(X_scaled)[0][1]
    xgb_pred_prob = xgb_model.predict_proba(X_scaled)[0][1]
    
    # ۲. پیش‌بینی مدل LSTM
    lstm_prob = 0.5
    if lstm_model is not None and tf is not None:
        try:
            X_lstm_raw = X_latest_full.values
            X_lstm_scaled = scaler.transform(X_lstm_raw)
            # آماده‌سازی شکل 3D برای LSTM (1 Sample, TIME_STEPS, Features)
            X_lstm_input = X_lstm_scaled[-LSTM_TIME_STEPS:].reshape(1, LSTM_TIME_STEPS, len(feature_columns))
            # استفاده از .predict برای Keras
            lstm_pred_prob = lstm_model.predict(X_lstm_input, verbose=0)[0][0]
            lstm_prob = float(lstm_pred_prob)
        except Exception as e:
            print(f"❌ LSTM Prediction failed: {e}")
            lstm_prob = 0.5 # مقدار خنثی
    
    # تبدیل احتمالات (۰ تا ۱) به امتیاز (Score)
    # امتیاز = (احتمال خرید - احتمال فروش) * ضریب
    def prob_to_score(prob):
        return (prob - 0.5) * ML_SCORE_NORMALIZER

    rf_score = prob_to_score(rf_pred_prob)
    lr_score = prob_to_score(lr_pred_prob)
    xgb_score = prob_to_score(xgb_pred_prob)
    lstm_score = prob_to_score(lstm_prob)

    # 💡 جمع‌آوری امتیاز Ensemble
    ensemble_score = rf_score + lr_score + xgb_score + lstm_score

    # تعیین پیام
    if ensemble_score >= SIGNAL_SCORE_THRESHOLD:
        message = "سیگنال خرید قوی (Strong BUY) بر اساس اجماع AI"
    elif ensemble_score <= -SIGNAL_SCORE_THRESHOLD:
        message = "سیگنال فروش قوی (Strong SELL) بر اساس اجماع AI"
    else:
        message = "خنثی (Neutral) - اجماع مدل‌ها ضعیف است"
        
    return {
        "ensemble_score": round(ensemble_score, 1),
        "ml_score_final": round(ensemble_score * 10 / (ML_SCORE_NORMALIZER * 4), 1), # نرمالایز به رنج -10 تا 10
        "individual_results": {
            "RF": {"prob": round(rf_pred_prob * 100, 1), "score": round(rf_score, 1)},
            "LR": {"prob": round(lr_pred_prob * 100, 1), "score": round(lr_score, 1)},
            "XGB": {"prob": round(xgb_pred_prob * 100, 1), "score": round(xgb_score, 1)},
            "LSTM": {"prob": round(lstm_prob * 100, 1), "score": round(lstm_score, 1)},
        },
        "message": message
    }


def calculate_smart_sl_tp(price, signal, atr_val, support, resistance):
    """محاسبه حد سود و ضرر هوشمند بر اساس ATR و سطوح S/R."""
    if atr_val == 0:
        return 0, 0

    # حد سود/ضرر بر اساس ATR (ریسک/ریوارد ۱.۵)
    atr_sl = atr_val * 1.0 
    atr_tp = atr_val * RISK_REWARD_ATR

    sl, tp = 0, 0

    if signal == "buy":
        # SL: انتخاب بین ATR یا نزدیکترین Support
        sl = min(price - atr_sl, support) 
        # TP: انتخاب بین ATR یا نزدیکترین Resistance
        tp = max(price + atr_tp, resistance) 
    elif signal == "sell":
        # SL: انتخاب بین ATR یا نزدیکترین Resistance
        sl = max(price + atr_sl, resistance) 
        # TP: انتخاب بین ATR یا نزدیکترین Support
        tp = min(price - atr_tp, support)
    else: # Neutral
        return 0, 0

    # رند کردن به ۴ رقم اعشار برای FX
    return round(sl, 4), round(tp, 4)

def convert_to_serializable(obj):
    """تبدیل اشیاء NumPy و Pandas به مقادیر استاندارد Python برای JSON."""
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    elif isinstance(obj, (dict, list)):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return [convert_to_serializable(elem) for elem in obj]
    return obj

# ---------------------------------------------------------
# ۴. مسیردهی Flask
# ---------------------------------------------------------

@app.before_request
def initialize():
    """اجرای یک‌بار برای لود مدل‌ها و دیتابیس در هنگام شروع سرور."""
    # اگر هیچکدام از مدل‌ها لود نشده‌اند، دوباره تلاش کن
    if not any([rf_model, lr_model, xgb_model]):
        load_models()
        # دیتابیس را نیز فقط یک بار در هنگام شروع برنامه مقداردهی اولیه کن
        database.init_db()

@app.route("/", methods=["GET"])
def index():
    """رندر کردن صفحه اصلی."""
    return render_template('index.html')


@app.route("/analyze", methods=["POST"])
def analyze_market():
    """مسیر اصلی برای تحلیل بازار و ارسال سیگنال."""
    
    # ❗❗ چک کردن مدل‌ها و لود مجدد در صورت لزوم
    if not all([rf_model, lr_model, xgb_model, scaler]):
        load_models() 
        if not all([rf_model, lr_model, xgb_model, scaler]):
            return jsonify({
                "error": "AI Models not loaded. Please ensure training was successful (train.py executed) and models folder exists.", 
                "status": 503
            }), 503

    try:
        data = request.get_json(silent=True)
        if not data:
            # اگر JSON فرستاده نشد، از query parameters استفاده کن (برای تست دستی)
            data = request.args
            
        symbol = data.get('symbol', 'EUR/USD').replace('/', '') # حذف اسلش برای TwelveData
        interval = data.get('interval', '1h')
        use_htf = data.get('use_htf', 'true').lower() == 'true'
        size = int(data.get('size', 2000))
        
        # ۱. دریافت داده
        df = fetch_data(symbol, interval, size)
        if df is None or df.empty:
            return jsonify({"error": f"Failed to fetch data for {symbol} on {interval}. Check API Key or Symbol.", "status": 404}), 404

        # ۲. محاسبه اندیکاتورها
        df_ind = calculate_indicators(df)
        if df_ind.empty:
            return jsonify({"error": "Not enough data after calculating indicators (check time frame, size, or data source).", "status": 404}), 404

        last = df_ind.iloc[-1]
        
        # ۳. تحلیل AI
        ml_report = predict_ensemble_signal(df_ind)
        score = ml_report['ensemble_score']

        # ۴. تحلیل High Time Frame (اختیاری)
        htf_trend = "N/A"
        htf_int = "N/A"
        htf_status = "Disabled"

        if use_htf and interval in TIMEFRAME_MAP:
            htf_int = TIMEFRAME_MAP[interval]
            df_htf = fetch_data(symbol, htf_int, size=500)
            if df_htf is not None and not df_htf.empty:
                df_htf_ind = calculate_indicators(df_htf)
                if not df_htf_ind.empty:
                    last_htf = df_htf_ind.iloc[-1]
                    # EMA 20 & 50 Crossover Trend
                    if last_htf['EMA_20'] > last_htf['EMA_50']:
                        htf_trend = "Bullish"
                    elif last_htf['EMA_20'] < last_htf['EMA_50']:
                        htf_trend = "Bearish"
                    else:
                        htf_trend = "Neutral"

                    htf_status = f"Active: {htf_trend} ({htf_int})"

        # ۵. تعیین سیگنال نهایی و مدیریت ریسک
        signal = "neutral"
        if score >= SIGNAL_SCORE_THRESHOLD: signal = "buy"
        elif score <= -SIGNAL_SCORE_THRESHOLD: signal = "sell"
        
        # توجه: از ستون‌های مطمئن استفاده می‌کنیم
        atr_key = 'ATR_14' 
        sl, tp = calculate_smart_sl_tp(last['close'], signal, last[atr_key], last['DCL'], last['DCU'])
        
        # وضعیت‌های اندیکاتورها
        adx_key = 'ADX_14'
        rsi = last['RSI_14']
        adx = last[adx_key]
        
        macd_val = last['MACD_12_26_9']
        macd_sig = last['MACDs_12_26_9']
        macd_status = "Bullish (MACD > Signal)" if macd_val > macd_sig else "Bearish (MACD < Signal)"
        
        trend = "Uptrend (EMA 20 > 50)" if last['EMA_20'] > last['EMA_50'] else "Downtrend (EMA 20 < 50)"
        regime = "Strong Trend" if adx > 25 else "Consolidation/Ranging"
        
        # ۶. فچ اخبار (Dummy)
        news_text = "No real-time news API configured."
        
        # ۷. جمع‌آوری پاسخ نهایی
        response_data = {
            "symbol": symbol.upper(),
            "price": last['close'],
            "score": round(score, 1),
            "signal": signal,
            "setup": {"sl": sl, "tp": tp},
            "indicators": {
                "rsi": round(rsi, 1),
                "trend": trend,
                "macd": macd_status,
                "adx": round(adx, 1),
                "regime": regime,
                "news": news_text,
                "htf_status": htf_status,
                "htf_trend": htf_trend,
                "sr_levels": f"S: {round(last['DCL'],4)} | R: {round(last['DCU'],4)}",
                "divergence": "N/A (Advanced)", 
                "ai_report": {
                    "ensemble_score": ml_report["ensemble_score"],
                    "ml_score_final": ml_report["ml_score_final"],
                    "individual_results": ml_report["individual_results"],
                    "message": ml_report["message"],
                    "accuracy": GLOBAL_TEST_ACCURACY,
                    "importances": GLOBAL_RF_IMPORTANCES,
                }, 
            }
        }
        
        return jsonify(convert_to_serializable(response_data))

    except Exception as e:
        import traceback
        traceback.print_exc()
        # برگرداندن پاسخ JSON حتی در صورت خطا
        return jsonify({"error": f"Internal Error during Analysis: {str(e)}", "status": 500}), 500


@app.route("/backtest", methods=["GET"])
def backtest_route():
    return jsonify({"status": "⚠️ Backtest Disabled on Server"}), 501 

@app.route("/optimize", methods=["GET"])
def optimize_route():
    return jsonify({"status": "⚠️ Optimization Disabled on Server"}), 501 

if __name__ == "__main__":
    load_models()
    database.init_db()
    # در محیط محلی (Local) اجرا می‌شود
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
