import os
import json
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import joblib
import traceback
import gc
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# تنظیمات پایه
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__)

# کلیدهای API: این کلیدها ابتدا از متغیرهای محیطی خوانده می‌شوند، 
# اما اگر متغیر محیطی تنظیم نشده باشد، از مقدار پیش‌فرض داخل کد استفاده می‌شود.
# لطفاً مقادیر پیش‌فرض (YOUR_API_KEY_HERE) را با کلیدهای واقعی خود جایگزین کنید.
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "YOUR_TWELVEDATA_API_KEY_HERE") 
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "YOUR_ALPHA_VANTAGE_API_KEY_HERE")

RISK_REWARD_ATR = 1.5
TARGET_PERIODS = 5
ML_CONFIDENCE_THRESHOLD = 1.0
SIGNAL_SCORE_THRESHOLD = 5.0
LSTM_TIME_STEPS = 10
TIMEFRAME_MAP = {
    "5min": "15min",
    "15min": "1h",
    "30min": "1h",
    "1h": "4h",
    "4h": "1day",
    "1day": "1week",
    "1week": "1month",
    "1month": "1month"
}
ML_SCORE_NORMALIZER = 4.0

GLOBAL_TEST_ACCURACY = {}
GLOBAL_RF_IMPORTANCES = {}
GLOBAL_ML_MODELS = {}

# ---------------------------------------------------------
# توابع کمکی
# ---------------------------------------------------------

def convert_to_serializable(obj):
    """تبدیل اشیاء غیراستاندارد به قالبی قابل سریال‌سازی (JSON)."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj

def check_api_keys():
    """بررسی می‌کند که آیا کلیدهای API تنظیم شده‌اند یا خیر."""
    # اگر کلیدها همچنان مقادیر پیش‌فرض را داشته باشند، خطا می‌دهد
    if API_KEY_TWELVEDATA == "YOUR_TWELVEDATA_API_KEY_HERE" or API_KEY_ALPHA == "YOUR_ALPHA_VANTAGE_API_KEY_HERE":
        return False, "لطفاً کلیدهای API را در فایل (main.py) با کلیدهای واقعی جایگزین کنید."
    if not API_KEY_TWELVEDATA or not API_KEY_ALPHA:
        return False, "لطفاً کلیدهای API را از طریق متغیرهای محیطی یا مستقیماً در فایل تنظیم کنید."
    return True, ""

# ---------------------------------------------------------
# توابع بارگیری داده
# ---------------------------------------------------------

def fetch_data_twelve_data(symbol, interval, outputsize=1000):
    """بارگیری داده‌های کندل از Twelve Data."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY_TWELVEDATA,
        "outputsize": outputsize,
        "format": "json"
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if 'values' not in data or not data['values']:
            return None, f"Twelve Data: داده‌ای برای نماد {symbol} و بازه زمانی {interval} یافت نشد یا خطا: {data.get('message', 'نامشخص')}"

        df = pd.DataFrame(data['values'])
        df = df.rename(columns={'datetime': 'time', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna()
        return df, None
    except requests.exceptions.HTTPError as e:
        return None, f"Twelve Data HTTP Error: {e}"
    except Exception as e:
        return None, f"Twelve Data General Error: {str(e)}"

# ---------------------------------------------------------
# توابع تحلیل و اندیکاتور
# ---------------------------------------------------------

def calculate_indicators(df):
    """محاسبه مجموعه‌ای از اندیکاتورها."""
    # اندیکاتورهای مومنتوم
    df.ta.rsi(append=True)
    df.ta.stoch(append=True)
    df.ta.macd(append=True)

    # اندیکاتورهای نوسان
    df.ta.atr(append=True)
    df.ta.bbands(append=True)
    df.ta.donchian(append=True) # کانال دانچین

    # اندیکاتورهای حجم
    df.ta.vwap(append=True)
    df.ta.obv(append=True)

    # اندیکاتورهای روند
    df.ta.adx(append=True)
    df.ta.ema(length=20, append=True)
    df.ta.sma(length=50, append=True)
    
    # اندیکاتورهای اضافی
    df.ta.regime(append=True) # شناسایی حالت بازار (روند یا رنج)

    df = df.dropna()
    return df

def detect_divergence(df):
    """شناسایی واگرایی‌های ساده بر اساس RSI و قیمت."""
    if len(df) < 20:
        return "Not Enough Data"
    
    # واگرایی معمولی صعودی (Bullish Regular Divergence)
    # قیمت کف پایین‌تر (Lower Low)، RSI کف بالاتر (Higher Low)
    
    # از چند کندل آخر برای تعیین کف‌ها استفاده کنید
    price_lows = df['Low'].iloc[-5:]
    rsi_lows = df['RSI_14'].iloc[-5:]
    
    if price_lows.min() < price_lows.iloc[-2] and rsi_lows.idxmin() > rsi_lows.index[-2]:
        return "Bullish Regular Divergence"

    # واگرایی معمولی نزولی (Bearish Regular Divergence)
    # قیمت سقف بالاتر (Higher High)، RSI سقف پایین‌تر (Lower High)
    price_highs = df['High'].iloc[-5:]
    rsi_highs = df['RSI_14'].iloc[-5:]

    if price_highs.max() > price_highs.iloc[-2] and rsi_highs.idxmax() < rsi_highs.index[-2]:
        return "Bearish Regular Divergence"

    return "No Clear Divergence"

# ---------------------------------------------------------
# توابع مدل یادگیری ماشین
# ---------------------------------------------------------

def generate_features_for_ml(df):
    """ایجاد ویژگی‌ها برای مدل ML از اندیکاتورها."""
    features = pd.DataFrame(index=df.index)
    
    # استفاده از مقادیر اندیکاتورهای محاسبه شده
    features['RSI'] = df['RSI_14']
    features['MACD'] = df['MACDh_12_26_9']
    features['Stoch_K'] = df['STOCHk_14_3_3']
    features['ADX'] = df['ADX_14']
    features['BB_Width'] = (df['BBU_5_2.0'] - df['BBL_5_2.0']) / df['BB_5_2.0']
    features['Close_to_EMA'] = (df['Close'] - df['EMA_20']) / df['Close']
    features['Close_to_SMA'] = (df['Close'] - df['SMA_50']) / df['Close']
    features['Donchian_Mid'] = (df['DCH_20'] + df['DCL_20']) / 2
    features['Close_to_Donchian'] = (df['Close'] - features['Donchian_Mid']) / df['Close']
    features['Regime'] = df['Regime'] # استفاده مستقیم از اندیکاتور Regime
    
    # ویژگی‌های تغییرات اخیر (Rate of Change)
    features['RSI_ROC'] = df['RSI_14'].diff()
    features['Close_ROC'] = df['Close'].diff()
    features['Volume_ROC'] = df['Volume'].diff()

    # حذف سطر‌هایی که NaN دارند
    features = features.dropna()
    return features

def load_and_predict(features, symbol, timeframe):
    """بارگیری مدل ML و تولید پیش‌بینی."""
    global GLOBAL_ML_MODELS
    
    model_key = f"{symbol}_{timeframe}"
    if model_key not in GLOBAL_ML_MODELS:
        # شبیه‌سازی بارگیری مدل: در محیط واقعی، مدل را از دیسک بارگیری کنید (joblib.load)
        # در این دمو، فقط پیش‌بینی‌های تصادفی تولید می‌کنیم
        # این بخش باید با منطق واقعی بارگیری مدل جایگزین شود
        GLOBAL_ML_MODELS[model_key] = "MockModel" # شبیه‌سازی بارگیری موفق

    if GLOBAL_ML_MODELS[model_key] == "MockModel":
        # شبیه‌سازی تولید گزارش ML
        import random
        random.seed(int(time.time()))
        
        # امتیاز نهایی (بین -100 تا 100)
        ensemble_score = round(random.uniform(-100, 100), 2)
        
        # پیام توصیه‌ای
        if ensemble_score > 50:
            message = "قوی صعودی - احتمال بالا برای رشد."
        elif ensemble_score > 10:
            message = "صعودی - آماده برای حرکت."
        elif ensemble_score < -50:
            message = "قوی نزولی - احتمال بالا برای کاهش."
        elif ensemble_score < -10:
            message = "نزولی - احتیاط لازم است."
        else:
            message = "خنثی - بازار در حالت تثبیت."

        # نتایج مدل‌های فردی (شبیه‌سازی)
        individual_results = {
            "LSTM": {"score": round(random.uniform(-10, 10), 1), "prob": random.randint(50, 99)},
            "RandomForest": {"score": round(random.uniform(-10, 10), 1), "prob": random.randint(50, 99)},
            "SVC": {"score": round(random.uniform(-10, 10), 1), "prob": random.randint(50, 99)},
            "CNN": {"score": round(random.uniform(-10, 10), 1), "prob": random.randint(50, 99)},
        }
        
        # شبیه‌سازی دقت و اهمیت ویژگی‌ها
        global GLOBAL_TEST_ACCURACY, GLOBAL_RF_IMPORTANCES
        if not GLOBAL_TEST_ACCURACY:
             GLOBAL_TEST_ACCURACY = {"LSTM": 75, "RF": 82, "SVC": 78, "CNN": 80}
             GLOBAL_RF_IMPORTANCES = {"RSI": 0.2, "MACD": 0.15, "Close_to_EMA": 0.3, "ADX": 0.05, "BB_Width": 0.1, "Others": 0.2}

        return {
            "message": message,
            "ensemble_score": ensemble_score,
            "individual_results": individual_results,
            "accuracy": GLOBAL_TEST_ACCURACY,
            "importances": GLOBAL_RF_IMPORTANCES
        }

    return {"message": "Model not ready.", "ensemble_score": 0, "individual_results": {}, "accuracy": {}, "importances": {}}


def check_higher_timeframe(symbol, current_timeframe):
    """بررسی روند در تایم فریم بالاتر."""
    htf = TIMEFRAME_MAP.get(current_timeframe)
    if not htf:
        return "N/A", "N/A"

    df_htf, error = fetch_data_twelve_data(symbol, htf)
    if error:
        return "Error", f"HTF Data Error: {error}"

    df_htf = calculate_indicators(df_htf)
    if df_htf.empty:
        return "Error", "HTF Indicators failed."

    last = df_htf.iloc[-1]
    
    # تعیین روند بر اساس EMA و ADX
    ema_col = f'EMA_{20}'
    adx_col = f'ADX_14'
    dipi = f'DIp_14'
    dimi = f'DIm_14'

    trend = "Neutral"
    status = f"ADX: {round(last[adx_col], 2)}"

    if last[ema_col] is not None and last['Close'] > last[ema_col]:
        trend = "Bullish"
    elif last[ema_col] is not None and last['Close'] < last[ema_col]:
        trend = "Bearish"

    if last[adx_col] > 25:
        if last[dipi] > last[dimi]:
            trend = "Strong Bullish"
            status = f"Strong Trend (ADX>25, +DI > -DI)"
        else:
            trend = "Strong Bearish"
            status = f"Strong Trend (ADX>25, -DI > +DI)"
    
    return trend, status

# ---------------------------------------------------------
# روترها (Routes)
# ---------------------------------------------------------

@app.route("/")
def index():
    """نمایش صفحه اصلی (واسط کاربری)."""
    return render_template('index.html')

@app.route("/analyze", methods=["POST"])
def analyze_route():
    """نقطه پایانی برای دریافت تحلیل."""
    
    is_ready, error_msg = check_api_keys()
    if not is_ready:
        return jsonify({"error": error_msg}), 400

    try:
        data = request.get_json()
        symbol = data.get("symbol", "BTC/USD")
        timeframe = data.get("timeframe", "1h")

        # 1. دریافت داده
        df, error = fetch_data_twelve_data(symbol, timeframe)
        if error:
            return jsonify({"error": error}), 400
        
        # 2. محاسبه اندیکاتورها
        df = calculate_indicators(df)
        if df.empty:
            return jsonify({"error": "Failed to calculate indicators (Not enough clean data)."}), 400

        # 3. بررسی تایم فریم بالاتر
        htf_trend, htf_status = check_higher_timeframe(symbol, timeframe)
        
        # 4. آماده‌سازی ویژگی‌ها و پیش‌بینی ML
        features = generate_features_for_ml(df)
        if features.empty:
             return jsonify({"error": "Failed to generate ML features (Not enough clean data)."}), 400

        ml_report = load_and_predict(features, symbol, timeframe)
        
        # 5. تحلیل و جمع‌بندی
        last = df.iloc[-1]
        div_msg = detect_divergence(df)

        response = {
            "symbol": symbol,
            "timeframe": timeframe,
            "last_close": round(last['Close'], 2),
            "indicators": {
                "htf_trend": htf_trend,
                "htf_status": htf_status,
                "rsi": round(last.get('RSI_14', 0), 2),
                "adx": f"ADX: {round(last.get('ADX_14', 0), 2)} | +DI: {round(last.get('DIp_14', 0), 2)} | -DI: {round(last.get('DIm_14', 0), 2)}",
                "macd": f"MACD: {round(last.get('MACD_12_26_9', 0), 4)} | H: {round(last.get('MACDh_12_26_9', 0), 4)}",
                "stoch": f"K: {round(last.get('STOCHk_14_3_3', 0), 2)} | D: {round(last.get('STOCHd_14_3_3', 0), 2)}",
                "bbands": f"U: {round(last.get('BBU_5_2.0', 0), 2)} | M: {round(last.get('BB_5_2.0', 0), 2)} | L: {round(last.get('BBL_5_2.0', 0), 2)}",
                "atr": round(last.get('ATR_14', 0), 4),
                "regime": f"Regime: {last.get('Regime', 'N/A')}",
                "donchian": f"L: {round(last.get('DCL', 0), 4)} | R: {round(last.get('DCU', 0), 4)}",
                "divergence": div_msg,
                "ai_report": {
                    "message": ml_report.get("message"),
                    "ensemble_score": ml_report.get("ensemble_score"), # تغییر نام برای وضوح
                    "individual_results": ml_report.get("individual_results"),
                    "accuracy": GLOBAL_TEST_ACCURACY,
                    "importances": GLOBAL_RF_IMPORTANCES
                }
            }
        }

        # پاکسازی حافظه‌های موقت
        try:
            del df
            if 'df_htf' in locals():
                del df_htf
        except Exception:
            pass
        gc.collect()

        return jsonify(convert_to_serializable(response))

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

@app.route("/backtest", methods=["GET"])
def backtest_route():
    return jsonify({"status": "⚠️ Backtest Disabled on Server"}), 501

@app.route("/optimize", methods=["GET"])
def optimize_route():
    return jsonify({"status": "⚠️ Optimization Disabled on Server"}), 501

# ---------------------------------------------------------
# entrypoint
# ---------------------------------------------------------
if __name__ == '__main__':
    # این خط را در محیطی که از متغیر محیطی برای پورت استفاده می‌شود، حفظ کنید
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
