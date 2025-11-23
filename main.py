import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import warnings
from flask import Flask, request, jsonify, render_template

# ✅ ایمپورت ایمن TensorFlow (بسیار مهم برای سرورهای کم‌رم Railway)
tf = None
lstm_model = None
try:
    import tensorflow as tf
    # اطمینان از خاموش بودن حالت eager در زمان لازم
    # if tf.executing_eagerly():
    #     tf.compat.v1.disable_eager_execution()
    print("✅ TensorFlow imported successfully.")
except ImportError:
    print("⚠️ TensorFlow not installed or failed to import.")
except Exception as e:
    print(f"⚠️ TensorFlow import failed (Low RAM suspected). Error: {e}")

# ✅ ایمپورت ماژول دیتابیس جدید
import database

# ---------------------------------------------------------
# ۱. پیکربندی
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__, template_folder='.')

# کلیدهای API (در صورت نیاز از Environment Variables بخوانید)
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "df521019db9f44899bfb172fdce6b454")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# پارامترهای استراتژی
RISK_REWARD_ATR = 1.5
SIGNAL_SCORE_THRESHOLD = 5.0
LSTM_TIME_STEPS = 10 
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day", "1day": "1day" } # برای 1day، HTF همان 1day در نظر گرفته می‌شود
ML_SCORE_NORMALIZER = 40.0
GLOBAL_RF_IMPORTANCES = {}
GLOBAL_TEST_ACCURACY = "N/A"

# ---------------------------------------------------------
# ۲. بارگذاری مدل‌ها و ابزارها
# ---------------------------------------------------------

try:
    scaler = joblib.load('models/scaler.pkl')
    rf_model = joblib.load('models/rf_model.pkl')
    lr_model = joblib.load('models/lr_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    
    # 💡 بارگذاری LSTM
    if tf and os.path.exists('models/lstm_model.h5'):
        try:
            # ⚠️ بارگذاری مدل در حالت compile=False برای سرعت بیشتر
            lstm_model = tf.keras.models.load_model('models/lstm_model.h5', compile=False)
            print("✅ All ML models (including LSTM) loaded successfully.")
        except Exception as e:
            print(f"⚠️ Failed to load LSTM model. Error: {e}")
    else:
        print("⚠️ LSTM model file not found or TensorFlow is not available.")
    
    # 💡 بارگذاری Feature Importances (اختیاری)
    if os.path.exists('models/rf_importances.pkl'):
        GLOBAL_RF_IMPORTANCES = joblib.load('models/rf_importances.pkl')
    if os.path.exists('models/test_accuracy.pkl'):
        GLOBAL_TEST_ACCURACY = joblib.load('models/test_accuracy.pkl')

except FileNotFoundError:
    print("❌ Critical: One or more model files (scaler/rf/lr/xgb) not found. Run train.py first!")
    
# ---------------------------------------------------------
# ۳. توابع دریافت داده (استفاده از TwelveData)
# ---------------------------------------------------------

def fetch_candles_twelve(symbol, interval, size=2000):
    """دریافت کندل‌های تاریخی از TwelveData با استفاده از API Key."""
    
    # ⚠️ TwelveData از فرمت BTC/USD استفاده نمی‌کند، باید به BTC/USDT یا BTCUSD تغییر کند
    if '/' in symbol:
        symbol_api = symbol.replace('/', '')
    elif symbol in ['AAPL', 'GOOG']:
        symbol_api = symbol # سهام
    else:
        symbol_api = symbol
    
    url = f"https://api.twelvedata.com/time_series?symbol={symbol_api}&interval={interval}&outputsize={size}&apikey={API_KEY_TWELVEDATA}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status() # بررسی خطاهای HTTP
        data = response.json()
        
        if 'values' not in data or not data['values']:
            if 'message' in data:
                 # اگر پیامی از سمت API آمد (مثلاً محدودیت نرخ یا نماد اشتباه)
                raise Exception(f"TwelveData API Error: {data['message']}")
            else:
                raise Exception(f"No data returned for {symbol_api} ({interval}).")
        
        df = pd.DataFrame(data['values'])
        
        # مرتب‌سازی و تبدیل نوع داده‌ها
        df.rename(columns={'datetime': 'datetime', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        df = df.sort_values('datetime').set_index('datetime')
        
        return df

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching data: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred during fetch: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# ۴. توابع تحلیل (Indicators, ML, Strategy)
# ---------------------------------------------------------

def calculate_indicators(df):
    """محاسبه تمام اندیکاتورهای مورد نیاز برای ML و تحلیل فنی."""
    if df.empty:
        return df
    
    df['Returns'] = df['close'].pct_change()
    
    # اندیکاتورهای اصلی
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.donchian(lower_length=20, upper_length=20, append=True) # برای محاسبه SR
    
    # ویژگی‌های اضافی
    df['Volatility'] = df['high'] - df['low']
    df['DayOfWeek'] = df.index.dayofweek
    df['Hour'] = df.index.hour
    df['EMA_Diff_Fast'] = df['EMA_20'] - df['close']
    df['EMA_Diff_Slow'] = df['EMA_50'] - df['close']
    
    df = df.dropna()
    return df

def get_ml_features(df):
    """استخراج ستون‌های ویژگی مورد نیاز برای مدل‌های ML."""
    # ⚠️ این لیست باید دقیقا با ویژگی‌های استفاده شده در train.py مطابقت داشته باشد!
    # بر اساس محتوای train.py (مثلا train (3).py)
    feature_cols = [
        'close', 'EMA_20', 'EMA_50', 'EMA_100', 'RSI_14', 'RSI_6', 
        'ADX_14', 'ATR_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
        'Volatility', 'DayOfWeek', 'Hour', 'EMA_Diff_Fast', 'EMA_Diff_Slow'
    ]
    
    # فیلتر کردن ستون‌هایی که وجود ندارند (برای جلوگیری از خطا)
    valid_cols = [col for col in feature_cols if col in df.columns]
    
    return df[valid_cols]


def analyze_ml(df):
    """اجرای مدل‌های ML و ترکیب نتایج."""
    if df.empty or len(df) < LSTM_TIME_STEPS:
        return {"ensemble_score": 0, "ml_score_final": 0, "individual_results": {}, "message": "داده کافی نیست."}

    X_features = get_ml_features(df)
    
    # آخرین کندل برای پیش‌بینی
    last_candle_data = X_features.iloc[-1].values.reshape(1, -1)
    
    # استانداردسازی
    last_candle_scaled = scaler.transform(last_candle_data)

    # 1. RF Model (Binary Probability)
    rf_prob = rf_model.predict_proba(last_candle_scaled)[0, 1]
    rf_score = (rf_prob - 0.5) * ML_SCORE_NORMALIZER
    
    # 2. LR Model (Binary Probability)
    lr_prob = lr_model.predict_proba(last_candle_scaled)[0, 1]
    lr_score = (lr_prob - 0.5) * ML_SCORE_NORMALIZER
    
    # 3. XGB Model (Binary Probability)
    xgb_prob = xgb_model.predict_proba(last_candle_scaled)[0, 1]
    xgb_score = (xgb_prob - 0.5) * ML_SCORE_NORMALIZER

    # 4. LSTM Model (Sequence Prediction)
    lstm_score = 0
    if lstm_model:
        # ساخت ورودی 3D برای LSTM (10 گام زمانی قبلی)
        lstm_input_data = X_features.iloc[-LSTM_TIME_STEPS:]
        lstm_input_scaled = scaler.transform(lstm_input_data).reshape(1, LSTM_TIME_STEPS, -1)
        
        lstm_prob = lstm_model.predict(lstm_input_scaled, verbose=0)[0, 0]
        lstm_score = (lstm_prob - 0.5) * ML_SCORE_NORMALIZER
        
    individual_results = {
        "RF": {"prob": round(rf_prob * 100, 1), "score": round(rf_score, 1)},
        "LR": {"prob": round(lr_prob * 100, 1), "score": round(lr_score, 1)},
        "XGB": {"prob": round(xgb_prob * 100, 1), "score": round(xgb_score, 1)},
        "LSTM": {"prob": round(lstm_prob * 100, 1), "score": round(lstm_score, 1)} if lstm_model else {"prob": 50.0, "score": 0.0},
    }
    
    # امتیازدهی اجماع (Ensemble)
    all_scores = [rf_score, lr_score, xgb_score]
    if lstm_model:
        all_scores.append(lstm_score)
        
    ensemble_score = sum(all_scores) / len(all_scores)
    
    # پیام نهایی
    if ensemble_score >= ML_SCORE_NORMALIZER * 0.2: # 8.0
        message = "اجماع قوی AI برای صعود (Strong Buy/Trend)"
    elif ensemble_score >= ML_SCORE_NORMALIZER * 0.05: # 2.0
        message = "اجماع AI برای صعود ضعیف (Weak Buy)"
    elif ensemble_score <= -ML_SCORE_NORMALIZER * 0.2: # -8.0
        message = "اجماع قوی AI برای نزول (Strong Sell/Trend)"
    elif ensemble_score <= -ML_SCORE_NORMALIZER * 0.05: # -2.0
        message = "اجماع AI برای نزول ضعیف (Weak Sell)"
    else:
        message = "بازار در حالت خنثی یا بلاتکلیفی است."
    
    # خروجی نهایی بین -10 تا +10
    ml_score_final = min(max(ensemble_score / 4.0, -10.0), 10.0)
    
    return {
        "ensemble_score": round(ensemble_score, 1),
        "ml_score_final": round(ml_score_final, 1),
        "individual_results": individual_results,
        "message": message
    }

def calculate_smart_sl_tp(price, signal, atr, dcl, dcu):
    """محاسبه SL و TP بر اساس ATR و سطوح SR (Donchian Channel)"""
    
    if signal == 'buy':
        sl = max(dcl, price - (RISK_REWARD_ATR * atr))
        tp = price + (RISK_REWARD_ATR * (price - sl)) # TP بر اساس ریسک واقعی
        
    elif signal == 'sell':
        sl = min(dcu, price + (RISK_REWARD_ATR * atr))
        tp = price - (RISK_REWARD_ATR * (sl - price)) # TP بر اساس ریسک واقعی
        
    else:
        sl = 0.0
        tp = 0.0
        
    return sl, tp

def convert_to_serializable(obj):
    """تبدیل اشیاء NumPy و Pandas به نوع داده قابل ارسال در JSON."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


# ---------------------------------------------------------
# ۵. روترهای Flask
# ---------------------------------------------------------

@app.route("/")
def index():
    """نمایش صفحه اصلی."""
    # 💡 اطمینان از مقداردهی اولیه دیتابیس در هنگام شروع برنامه
    # database.init_db() # ⚠️ بهتر است init_db در هنگام راه‌اندازی Gunicorn اجرا شود نه هر بار لود صفحه
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_route():
    """دریافت ورودی‌ها و اجرای تحلیل AI."""
    try:
        # 💡 خواندن تمام پارامترها از بدنه JSON (که توسط فرانت‌اند جدید ارسال می‌شود)
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received. Check Content-Type header."}), 400

        symbol = data.get("symbol", "EUR/USD")
        interval = data.get("interval", "1h")
        use_htf = data.get("use_htf", "true").lower() == 'true'
        size = int(data.get("size", 2000)) # 💡 خواندن تعداد کندل
        
        # 1. دریافت داده‌های اصلی
        df_main = fetch_candles_twelve(symbol, interval, size=size)
        if df_main.empty:
            return jsonify({"error": "Could not fetch main data. Check symbol/interval/API key."}), 500

        # 2. دریافت داده HTF (اگر فعال باشد)
        df_htf = pd.DataFrame()
        htf_status = "Inactive"
        htf_trend = "N/A"
        
        if use_htf and interval in TIMEFRAME_MAP:
            htf_int = TIMEFRAME_MAP[interval]
            df_htf = fetch_candles_twelve(symbol, htf_int, size=size)
            
            if not df_htf.empty:
                df_htf = calculate_indicators(df_htf)
                if not df_htf.empty:
                    last_htf = df_htf.iloc[-1]
                    # معیار روند HTF (مثلاً EMA 50)
                    htf_trend = "Bullish" if last_htf['close'] > last_htf['EMA_50'] else "Bearish"
                    htf_status = f"Active: {htf_int}"
            else:
                htf_status = f"Fetch Failed for {htf_int}"

        # 3. محاسبه اندیکاتورهای اصلی و اجرای ML
        df_main = calculate_indicators(df_main)
        if df_main.empty:
            return jsonify({"error": "Not enough data after calculating indicators (min 100 candles needed)."}), 500
        
        last = df_main.iloc[-1]
        
        ml_report = analyze_ml(df_main)
        score = ml_report['ml_score_final'] # Score between -10 and +10

        # 4. تحلیل فنی
        rsi_val = last['RSI_14']
        trend = "Uptrend" if last['close'] > last['EMA_20'] and last['close'] > last['EMA_50'] else "Downtrend"
        adx_val = last['ADX_14']
        regime = "Trending" if adx_val > 25 else "Ranging"
        macd_status = "Bullish" if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else "Bearish"
        
        # تعیین سیگنال نهایی
        signal = "neutral"
        if score >= SIGNAL_SCORE_THRESHOLD: 
            signal = "buy"
        elif score <= -SIGNAL_SCORE_THRESHOLD: 
            signal = "sell"
        
        # مدیریت ریسک و سطوح SR
        sl, tp = calculate_smart_sl_tp(last['close'], signal, last['ATR_14'], last['DCL_20'], last['DCU_20'])
        support = last['DCL_20']
        resistance = last['DCU_20']
        
        # 5. ساخت پاسخ (News و Divergence به صورت موقت خالی)
        news_text = "No real-time news data available."
        div_msg = "N/A (Advanced Analysis Required)"

        response_data = {
            "symbol": symbol,
            "price": last['close'],
            "score": round(score, 1),
            "signal": signal,
            "setup": {"sl": round(sl, 5), "tp": round(tp, 5)},
            "indicators": {
                "rsi": round(rsi_val, 1),
                "trend": trend,
                "macd": macd_status,
                "news": news_text, 
                "htf_status": htf_status,
                "htf_trend": htf_trend,
                "regime": f"{regime} (ADX: {int(adx_val)})",
                "sr_levels": f"S: {round(support, 5)} | R: {round(resistance, 5)}",
                "divergence": div_msg,
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
        return jsonify({"error": f"Internal Error during Analysis: {str(e)}", "status": 500}), 500

@app.route("/backtest", methods=["GET"])
def backtest_route():
    return jsonify({"status": "⚠️ Backtest Disabled on Server"}), 501 

@app.route("/optimize", methods=["GET"])
def optimize_route():
    return jsonify({"status": "⚠️ Optimization Disabled on Server"}), 501 

if __name__ == "__main__":
    database.init_db()
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)))
