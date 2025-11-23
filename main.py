import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import warnings
import time
from flask import Flask, request, jsonify, render_template

# ✅ ایمپورت ایمن TensorFlow (بسیار مهم برای سرورهای کم‌رم Railway)
# این کار مطمئن می‌شود که اگر TensorFlow نصب نشد، برنامه حداقل با مدل‌های Scikit-learn کار کند.
tf = None
lstm_model = None
try:
    # ابتدا سعی می‌کنیم TensorFlow را ایمپورت کنیم
    import tensorflow as tf
    # تنظیمات برای جلوگیری از هدر رفتن منابع در هنگام پیش‌بینی
    tf.config.set_visible_devices([], 'GPU')
    print("✅ TensorFlow imported successfully.")
except ImportError:
    print("⚠️ TensorFlow not installed or failed to import.")
except Exception as e:
    # این اغلب به دلیل کمبود RAM در محیط‌های ابری مانند Railway رخ می‌دهد
    print(f"⚠️ TensorFlow import failed (Low RAM suspected). Error: {e}")

# ✅ ایمپورت ماژول دیتابیس جدید
import database

# ---------------------------------------------------------
# ۱. پیکربندی و راه‌اندازی اولیه
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__, template_folder='.')

# 🔑 کلیدهای API (بهتر است از متغیرهای محیطی بخوانید)
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "YOUR_TWELVEDATA_KEY") 
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "YOUR_ALPHA_VANTAGE_KEY")

# 📊 پارامترهای استراتژی
RISK_REWARD_ATR = 1.5           
SIGNAL_SCORE_THRESHOLD = 5.0    
LSTM_TIME_STEPS = 10 
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
ML_SCORE_NORMALIZER = 40.0 # برای نرمال‌سازی مجموع امتیاز مدل‌ها
DEFAULT_SYMBOL = "EUR/USD"
DEFAULT_INTERVAL = "1h"

# متغیرهای سراسری (اطلاعات پیش‌فرض برای UI قبل از تحلیل واقعی)
GLOBAL_RF_IMPORTANCES = {}
GLOBAL_TEST_ACCURACY = 0.85 

# ---------------------------------------------------------
# ۲. بارگذاری مدل‌های ماشین لرنینگ و Scaler
# ---------------------------------------------------------

SCALER, RF_MODEL, LR_MODEL, XGB_MODEL, LSTM_MODEL = None, None, None, None, None
MODEL_LOAD_SUCCESS = False

def load_models():
    """بارگذاری تمام مدل‌ها و Scaler از دیسک."""
    global SCALER, RF_MODEL, LR_MODEL, XGB_MODEL, LSTM_MODEL, MODEL_LOAD_SUCCESS

    if not os.path.isdir('models'):
        print(\"❌ ERROR: 'models' directory not found. Run 'python train.py' first.\")
        return

    try:
        # بارگذاری Scaler (مهم: باید از Scaler آموزش دیده در train.py استفاده شود)
        SCALER = joblib.load('models/scaler.pkl')
        
        # بارگذاری مدل‌های Scikit-learn و XGBoost
        RF_MODEL = joblib.load('models/rf_model.pkl')
        LR_MODEL = joblib.load('models/lr_model.pkl')
        XGB_MODEL = joblib.load('models/xgb_model.pkl')
        
        # بارگذاری مدل LSTM (اگر TensorFlow با موفقیت ایمپورت شده باشد)
        if tf is not None:
            LSTM_MODEL = tf.keras.models.load_model('models/lstm_model.h5')
        
        MODEL_LOAD_SUCCESS = True
        print("✅ All ML Models and Scaler loaded successfully.")
    except Exception as e:
        print(f"❌ ERROR loading models. Run 'python train.py' first. Error: {e}")
        MODEL_LOAD_SUCCESS = False

# ---------------------------------------------------------
# ۳. توابع کمکی اصلی
# ---------------------------------------------------------

def calculate_indicators_for_prediction(df):
    """
    محاسبه دقیقاً همان اندیکاتورهایی که برای آموزش مدل‌ها در train.py استفاده شد.
    این تابع داده‌ها را برای پیش‌بینی آماده می‌کند.
    """
    df = df.copy()
    
    # اصلاح نام ستون‌ها (اگر لازم باشد)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    # 1. محاسبه اندیکاتورها
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True) # شامل ADX_14, DMP_14, DMN_14
    
    # 2. ساخت ویژگی‌های ترکیبی
    df['Volatility'] = (df['high'] - df['low']) / df['close']
    df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
    df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_100']
    
    # 3. محاسبه MACD (شامل MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
    df.ta.macd(append=True)
    
    # 4. کانال دانچین (برای سطوح SR)
    df.ta.donchian(length=20, append=True) # شامل DCL و DCU
    
    # ستون‌های ویژگی نهایی (باید دقیقاً با train.py یکسان باشد)
    feature_cols = [
        'close', 'Returns', 'EMA_20', 'EMA_50', 'EMA_100', 'RSI_14', 'RSI_6', 'ATR', 'ADX_14', 
        'DMP_14', 'DMN_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
        'Volatility', 'EMA_Diff_Fast', 'EMA_Diff_Slow'
    ]
    
    # حذف سطرهایی که اندیکاتورهای آن‌ها هنوز محاسبه نشده است
    return df.dropna(subset=feature_cols)

def fetch_and_save_data_from_twelvedata(symbol, interval, output_size=5000):
    """فچ کردن کندل‌های جدید و ذخیره آن‌ها در دیتابیس."""
    print(f"⏳ Fetching data for {symbol}/{interval}...")
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={output_size}&apikey={API_KEY_TWELVEDATA}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == 'error':
            return pd.DataFrame(), f"API Error: {data.get('message', 'Unknown')}"

        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df.rename(columns={'datetime': 'datetime', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
            df = df.iloc[::-1].reset_index(drop=True) # معکوس کردن ترتیب و ایندکس جدید
            
            # تبدیل ستون‌های قیمتی به اعداد اعشاری
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # ذخیره در دیتابیس
            database.save_candles(df, symbol, interval)
            print(f"✅ Successfully fetched and saved {len(df)} candles.")
            return df, None
        else:
            return pd.DataFrame(), "API returned no 'values'."

    except requests.exceptions.RequestException as e:
        return pd.DataFrame(), f"Network/API Request Error: {e}"
    except Exception as e:
        return pd.DataFrame(), f"An unexpected error occurred during fetch: {e}"

def get_news(symbol):
    """فچ کردن تیترهای خبری مرتبط."""
    query = symbol.replace("/", " ")
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=finance&sort=RELEVANCE&keywords={query}&limit=3&apikey={API_KEY_ALPHA}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        feed = data.get('feed', [])
        if feed:
            latest_title = feed[0].get('title', 'No recent news title.')
            return f"🔥 News: {latest_title}"
        return "No significant recent news found."
    except Exception:
        return "Could not fetch news (API limit or error)."

def calculate_ensemble_ml_score(df_scaled, htf_trend_score):
    """پیش‌بینی توسط مدل‌های آموزش دیده و محاسبه امتیاز نهایی."""
    if not MODEL_LOAD_SUCCESS:
        return {"message": "❌ ML models failed to load. Analysis skipped.", "ensemble_score": 0.0, "ml_score_final": 0.0, "individual_results": {}}

    last_candle_features = df_scaled[-1].reshape(1, -1)
    
    results = {}
    total_score = 0
    
    # 1. Random Forest (RF)
    rf_prob = RF_MODEL.predict_proba(last_candle_features)[0]
    rf_pred = 1 if rf_prob[1] > 0.5 else -1
    rf_score = (rf_prob[1] - rf_prob[0]) * 10
    total_score += rf_score
    results["RF"] = {"prob": round(rf_prob[1] * 100, 1), "score": round(rf_score, 1), "pred": rf_pred}
    
    # 2. Logistic Regression (LR)
    lr_prob = LR_MODEL.predict_proba(last_candle_features)[0]
    lr_pred = 1 if lr_prob[1] > 0.5 else -1
    lr_score = (lr_prob[1] - lr_prob[0]) * 10
    total_score += lr_score
    results["LR"] = {"prob": round(lr_prob[1] * 100, 1), "score": round(lr_score, 1), "pred": lr_pred}

    # 3. XGBoost (XGB)
    xgb_prob = XGB_MODEL.predict_proba(last_candle_features)[0]
    xgb_pred = 1 if xgb_prob[1] > 0.5 else -1
    xgb_score = (xgb_prob[1] - xgb_prob[0]) * 10
    total_score += xgb_score * 1.5 # وزن بیشتر برای XGB
    results["XGB"] = {"prob": round(xgb_prob[1] * 100, 1), "score": round(xgb_score, 1), "pred": xgb_pred}

    # 4. LSTM (اگر موجود باشد)
    lstm_score = 0
    if LSTM_MODEL is not None and tf is not None and len(df_scaled) >= LSTM_TIME_STEPS:
        # آماده‌سازی داده 3D برای LSTM
        lstm_input = df_scaled[-LSTM_TIME_STEPS:].reshape(1, LSTM_TIME_STEPS, df_scaled.shape[1])
        lstm_prob = LSTM_MODEL.predict(lstm_input, verbose=0)[0][0]
        lstm_pred = 1 if lstm_prob > 0.5 else -1
        lstm_score = (lstm_prob - 0.5) * 20 # امتیاز بین -10 تا 10
        total_score += lstm_score * 1.5 # وزن بیشتر برای LSTM
        results["LSTM"] = {"prob": round(lstm_prob * 100, 1), "score": round(lstm_score, 1), "pred": lstm_pred}
    else:
        results["LSTM"] = {"prob": 0.0, "score": 0.0, "pred": 0}

    # 5. HTF Trend (امتیاز تایم فریم بالا)
    total_score += htf_trend_score * 10 # وزن دهی قوی

    # امتیاز نهایی (نرمال شده برای تناسب با دامنه کلی سیگنال)
    ml_score_final = total_score / ML_SCORE_NORMALIZER
    
    # پیام نهایی
    if ml_score_final > 1.5: message = "HIGH CONFIDENCE BUY (قوی)"
    elif ml_score_final > 0.5: message = "LOW CONFIDENCE BUY (احتیاط)"
    elif ml_score_final < -1.5: message = "HIGH CONFIDENCE SELL (قوی)"
    elif ml_score_final < -0.5: message = "LOW CONFIDENCE SELL (احتیاط)"
    else: message = "NEUTRAL (خنثی)"
        
    return {
        "message": message,
        "ensemble_score": round(total_score, 1), # مجموع امتیازات خام
        "ml_score_final": round(ml_score_final, 1), # امتیاز نرمال شده
        "individual_results": results
    }

def calculate_smart_sl_tp(price, signal, atr, support, resistance):
    """محاسبه حد سود و ضرر بر اساس ATR و سطوح SR."""
    if signal == "buy":
        # حد ضرر: کمی زیر ATR یا سطح حمایت (هر کدام که امن تر است)
        sl_atr = price - atr * RISK_REWARD_ATR
        sl = min(sl_atr, support * 0.9999) # انتخاب SL نزدیک‌تر و محافظه‌کارانه‌تر
        # حد سود: ۱.۵ برابر ریسک یا سطح مقاومت
        tp_atr = price + atr * RISK_REWARD_ATR 
        tp = resistance # حد سود روی سطح مقاومت
        return round(sl, 4), round(tp, 4)
    elif signal == "sell":
        # حد ضرر: کمی بالای ATR یا سطح مقاومت
        sl_atr = price + atr * RISK_REWARD_ATR
        sl = max(sl_atr, resistance * 1.0001)
        # حد سود: ۱.۵ برابر ریسک یا سطح حمایت
        tp_atr = price - atr * RISK_REWARD_ATR 
        tp = support # حد سود روی سطح حمایت
        return round(sl, 4), round(tp, 4)
    else:
        return 0.0, 0.0

def convert_to_serializable(obj):
    """تبدیل اشیاء NumPy و Pandas به انواع پایتون استاندارد برای jsonify."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.iloc[-1]
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    return obj

# ---------------------------------------------------------
# ۴. مسیرهای API و هسته برنامه
# ---------------------------------------------------------

@app.route("/")
def index_route():
    """نمایش فایل HTML فرانت‌اند."""
    return render_template('index.html')

@app.before_request
def check_models_loaded():
    """بررسی می‌کند که آیا مدل‌ها قبلا بارگذاری شده‌اند یا خیر."""
    if not MODEL_LOAD_SUCCESS and not request.path.startswith('/static'):
        # برای تضمین، اگر مدل‌ها هنوز بارگذاری نشده‌اند، دوباره سعی می‌کنیم
        if SCALER is None:
            load_models()
        # همچنین دیتابیس را نیز راه‌اندازی می‌کنیم
        database.init_db()


@app.route("/analyze", methods=["GET"])
def analyze_route():
    """مسیر اصلی تحلیل که داده‌ها را فچ کرده، تحلیل می‌کند و گزارش AI را برمی‌گرداند."""
    
    if not MODEL_LOAD_SUCCESS:
         # این پیام در حالت واقعی نباید دیده شود چون در before_request مدل‌ها بارگذاری می‌شوند
         return jsonify({"error": "ML Models not loaded. Please ensure train.py was run successfully.", "status": 503}), 503

    symbol = request.args.get("symbol", DEFAULT_SYMBOL)
    interval = request.args.get("interval", DEFAULT_INTERVAL)
    
    try:
        # ۱. فچ کردن و به‌روزرسانی داده
        
        # ابتدا از دیتابیس می‌خوانیم
        df_db = database.get_all_candles(symbol, interval)
        
        # اگر دیتای کافی نبود یا دیتابیس خالی بود، فچ آنلاین می‌کنیم
        if df_db.empty or len(df_db) < 500:
            print("⚠️ Insufficient data in DB. Fetching online...")
            df_online, fetch_error = fetch_and_save_data_from_twelvedata(symbol, interval)
            if fetch_error:
                 return jsonify({"error": f"Data Fetch Error: {fetch_error}", "status": 500}), 500
            df = df_online
        else:
            df = df_db.copy()
            # همیشه چند کندل آخر را برای اطمینان از تازگی به‌روزرسانی می‌کنیم
            fetch_and_save_data_from_twelvedata(symbol, interval, output_size=50) 
            
            # دوباره از دیتابیس می‌خوانیم تا کندل‌های جدید اضافه شده را در نظر بگیریم
            df = database.get_all_candles(symbol, interval)
            
        if df.empty or len(df) < 200:
            return jsonify({"error": "Not enough historical data for analysis.", "status": 500}), 500
        
        # ۲. آماده‌سازی و تحلیل داده
        
        df_indicators = calculate_indicators_for_prediction(df)
        if df_indicators.empty:
            return jsonify({"error": "Data is too short or indicators could not be calculated.", "status": 500}), 500
            
        last = df_indicators.iloc[-1]
        
        # ویژگی‌ها برای مدل ML (باید دقیقاً با train.py یکسان باشد)
        feature_cols = [
            'close', 'Returns', 'EMA_20', 'EMA_50', 'EMA_100', 'RSI_14', 'RSI_6', 'ATR', 'ADX_14', 
            'DMP_14', 'DMN_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9', 
            'Volatility', 'EMA_Diff_Fast', 'EMA_Diff_Slow'
        ]

        data_to_scale = df_indicators[feature_cols].values
        
        # اسکالر را روی کل داده اجرا می‌کنیم و فقط سطر آخر را برای پیش‌بینی برمی‌داریم
        df_scaled = SCALER.transform(data_to_scale)

        # ۳. تحلیل تایم فریم بالا (HTF)
        htf_interval = TIMEFRAME_MAP.get(interval)
        htf_trend_score = 0
        htf_text = "N/A"
        
        if htf_interval:
            df_htf = database.get_all_candles(symbol, htf_interval)
            if df_htf.empty or len(df_htf) < 200:
                 # اگر دیتای HTF در DB نبود، یک بار فچ می‌کنیم
                df_htf_online, _ = fetch_and_save_data_from_twelvedata(symbol, htf_interval, output_size=500)
                df_htf = df_htf_online

            if not df_htf.empty and len(df_htf) > 100:
                df_htf_ind = calculate_indicators_for_prediction(df_htf)
                if not df_htf_ind.empty:
                    last_htf = df_htf_ind.iloc[-1]
                    htf_trend = "Bullish" if last_htf['EMA_20'] > last_htf['EMA_50'] else "Bearish"
                    htf_trend_score = 1 if htf_trend == "Bullish" else -1
                    htf_adx = int(last_htf['ADX_14'])
                    htf_text = f"{htf_trend} ({htf_interval}, ADX: {htf_adx})"
                    
        # ۴. اجرای مدل‌های ML
        ml_report = calculate_ensemble_ml_score(df_scaled, htf_trend_score)
        
        # ۵. فچ کردن خبر
        news_text = get_news(symbol)

        # ۶. محاسبه سیگنال نهایی
        # یک امتیاز کلی بر اساس ترکیب اندیکاتورهای کلاسیک و امتیاز ML
        
        rsi_score = 1 if last['RSI_14'] > 60 else (-1 if last['RSI_14'] < 40 else 0)
        macd_score = 1 if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else (-1 if last['MACD_12_26_9'] < last['MACDs_12_26_9'] else 0)
        adx_score = 0.5 if last['ADX_14'] > 25 else 0 # ترند بودن مقداری امتیاز مثبت دارد
        
        classic_score = (rsi_score * 2) + macd_score + adx_score + (htf_trend_score * 1) # وزن دهی
        score = (classic_score * 5) + (ml_report['ml_score_final'] * 10) # وزن دهی قوی به ML

        trend = "صعودی" if last['EMA_20'] > last['EMA_50'] else "نزولی"
        rsi = last['RSI_14']
        adx = last['ADX_14']
        regime = "Trending" if adx > 25 else "Ranging"
        
        support = last['DCL_20']
        resistance = last['DCU_20']
        atr = last['ATR_14']
        
        # سیگنال نهایی
        signal = "خنثی"
        if score >= SIGNAL_SCORE_THRESHOLD: signal = "خرید"
        elif score <= -SIGNAL_SCORE_THRESHOLD: signal = "فروش"
        
        sl, tp = calculate_smart_sl_tp(last['close'], signal, atr, support, resistance)
        
        # ۷. ساختن پاسخ نهایی
        response_data = {
            "symbol": symbol,
            "interval": interval,
            "timestamp": last.name.strftime('%Y-%m-%d %H:%M:%S') if hasattr(last.name, 'strftime') else str(last.name),
            "price": round(last['close'], 5),
            "score": round(score, 1),
            "signal": signal,
            "setup": {"sl": sl, "tp": tp},
            "indicators": {
                "rsi": round(rsi, 1),
                "trend": trend,
                "adx": round(adx, 1),
                "atr": round(atr, 5),
                "macd": "صعودی" if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else "نزولی",
                "news": news_text, 
                "htf_status": htf_text,
                "regime": f"{regime} (ADX: {int(adx)})",
                "sr_levels": f"S: {round(support, 5)} | R: {round(resistance, 5)}",
                "divergence": "---", # (قابل توسعه)
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

# ---------------------------------------------------------
# ۵. اجرای برنامه (فقط برای تست محلی)
# ---------------------------------------------------------

if __name__ == "__main__":
    # در محیط لوکال، مدل‌ها را مستقیماً بارگذاری می‌کنیم
    load_models()
    database.init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)
