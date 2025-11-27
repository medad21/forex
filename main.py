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
# نادیده گرفتن هشدارهای مربوط به Pandas TA
warnings.filterwarnings('ignore')
app = Flask(__name__)

# کلیدهای API: این کلیدها ابتدا از متغیرهای محیطی خوانده می‌شوند، 
# اگر متغیر محیطی تنظیم نشده باشد، از مقادیر پیش‌فرض استفاده می‌شود.
# توجه: در یک محیط واقعی، باید از کلیدهای معتبر استفاده کنید.
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "c15e9b87795a49aebc5b246e156b68bb") 
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# پارامترهای تحلیل و ML
RISK_REWARD_ATR = 1.5
TARGET_PERIODS = 5
ML_CONFIDENCE_THRESHOLD = 1.0
SIGNAL_SCORE_THRESHOLD = 5.0
LSTM_TIME_STEPS = 10
ML_SCORE_NORMALIZER = 4.0 # برای نرمال‌سازی امتیاز ML

# نگاشت تایم‌فریم برای تحلیل‌های HTF
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

# متغیرهای گلوبال شبیه‌سازی شده برای آمار مدل
GLOBAL_TEST_ACCURACY = {}
GLOBAL_RF_IMPORTANCES = {}
GLOBAL_ML_MODELS = {}

# ---------------------------------------------------------
# توابع کمکی
# ---------------------------------------------------------

def convert_to_serializable(obj):
    """تبدیل اشیاء غیراستاندارد (مانند NumPy) به قالبی قابل سریال‌سازی (JSON)."""
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
    """بررسی می‌کند که آیا کلیدهای API تنظیم شده‌اند."""
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
        
        # تبدیل ستون‌های قیمت به عدد
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # اصلاح کلیدی: مدیریت امن ستون Volume
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0.0)
        else:
            # اگر ستون Volume وجود نداشت، آن را با صفر پر می‌کنیم تا اندیکاتورهای Volume-based دچار خطا نشوند
            df['Volume'] = 0.0 
            
        df = df.dropna()
        return df, None
    except requests.exceptions.HTTPError as e:
        # اگر کلید API نامعتبر باشد
        return None, f"Twelve Data HTTP Error: {e}. (لطفاً کلید API را بررسی کنید)"
    except Exception as e:
        return None, f"Twelve Data General Error: {str(e)}"

# ---------------------------------------------------------
# توابع تحلیل و اندیکاتور
# ---------------------------------------------------------

def calculate_indicators(df):
    """محاسبه مجموعه‌ای از اندیکاتورها با استفاده از pandas_ta."""
    
    # اطمینان از وجود داده کافی
    if len(df) < 50: # حداقل 50 کندل برای اندیکاتورهایی مانند SMA(50)
         return pd.DataFrame()
         
    # مومنتوم و نوسان
    df.ta.rsi(append=True) # RSI_14
    df.ta.stoch(append=True) # STOCHk_14_3_3, STOCHd_14_3_3
    df.ta.macd(append=True) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    # نوسان و قیمت
    df.ta.atr(append=True) # ATR_14
    df.ta.bbands(append=True) # BBL, BBU, BB_M, ...
    df.ta.donchian(append=True) # DCL_20, DCU_20, DCM_20

    # روند و استحکام
    df.ta.adx(append=True) # ADX_14, DIp_14, DIm_14
    df.ta.ema(length=20, append=True) # EMA_20
    df.ta.sma(length=50, append=True) # SMA_50
    
    # حالت بازار
    df.ta.regime(append=True) # شناسایی حالت بازار (روند یا رنج)

    df = df.dropna()
    return df

def detect_divergence(df):
    """شناسایی واگرایی‌های ساده بر اساس RSI و قیمت."""
    if len(df) < 20:
        return "Not Enough Data"
    
    # واگرایی معمولی صعودی (Bullish Regular Divergence)
    # قیمت کف پایین‌تر، RSI کف بالاتر
    
    # بررسی چند کندل آخر برای تعیین کف‌ها (مثلاً 5 کندل)
    price_lows = df['Low'].iloc[-5:]
    rsi_lows = df['RSI_14'].iloc[-5:]
    
    # اگر کف جدید قیمت پایین‌تر از کف قبلی باشد و RSI نتوانسته باشد کف پایین‌تر ثبت کند
    # این فقط یک شبیه‌سازی ساده است و نیاز به تحلیل عمیق‌تر دارد.
    if price_lows.min() < price_lows.iloc[-2] and rsi_lows.idxmin() > rsi_lows.index[-2]:
        return "Bullish Regular Divergence (Simple)"

    # واگرایی معمولی نزولی (Bearish Regular Divergence)
    # قیمت سقف بالاتر، RSI سقف پایین‌تر
    price_highs = df['High'].iloc[-5:]
    rsi_highs = df['RSI_14'].iloc[-5:]

    # اگر سقف جدید قیمت بالاتر از سقف قبلی باشد و RSI نتوانسته باشد سقف بالاتر ثبت کند
    if price_highs.max() > price_highs.iloc[-2] and rsi_highs.idxmax() < rsi_highs.index[-2]:
        return "Bearish Regular Divergence (Simple)"

    return "No Clear Divergence"

# ---------------------------------------------------------
# توابع مدل یادگیری ماشین
# ---------------------------------------------------------

def generate_features_for_ml(df):
    """ایجاد ویژگی‌ها برای مدل ML از اندیکاتورها."""
    features = pd.DataFrame(index=df.index)
    
    # ویژگی‌های مومنتوم
    features['RSI'] = df['RSI_14']
    features['MACD'] = df['MACDh_12_26_9']
    features['Stoch_K'] = df['STOCHk_14_3_3']
    
    # ویژگی‌های روند
    features['ADX'] = df['ADX_14']
    features['Close_to_EMA'] = (df['Close'] - df['EMA_20']) / df['Close']
    features['Close_to_SMA'] = (df['Close'] - df['SMA_50']) / df['Close']
    features['Regime'] = df['Regime'] # حالت بازار

    # ویژگی‌های نوسان
    features['BB_Width'] = (df['BBU_5_2.0'] - df['BBL_5_2.0']) / df['BB_5_2.0']
    features['Donchian_Mid'] = df['DCM_20']
    features['Close_to_Donchian'] = (df['Close'] - features['Donchian_Mid']) / df['Close']
    
    # ویژگی‌های تغییرات اخیر (Rate of Change)
    features['RSI_ROC'] = df['RSI_14'].diff()
    features['Close_ROC'] = df['Close'].diff()
    # استفاده از fillna(0) در اینجا برای اطمینان از ایمنی در برابر نمادهایی که Volume ندارند
    features['Volume_ROC'] = df['Volume'].fillna(0).diff()

    # حذف سطر‌هایی که NaN دارند
    features = features.dropna()
    return features

def load_and_predict(features, symbol, timeframe):
    """بارگیری مدل ML و تولید پیش‌بینی (در این دمو شبیه‌سازی شده)."""
    global GLOBAL_ML_MODELS
    
    model_key = f"{symbol}_{timeframe}"
    if model_key not in GLOBAL_ML_MODELS:
        # شبیه‌سازی بارگیری مدل: در یک محیط واقعی، از joblib.load یا بارگیری مدل Keras/PyTorch استفاده می‌کنید
        GLOBAL_ML_MODELS[model_key] = "MockModel" # شبیه‌سازی بارگیری موفق

    if GLOBAL_ML_MODELS[model_key] == "MockModel":
        # شبیه‌سازی تولید گزارش ML
        import random
        random.seed(int(time.time()))
        
        # امتیاز نهایی (بین -100 تا 100)
        ensemble_score = round(random.uniform(-75, 75), 2)
        
        # پیام توصیه‌ای
        if ensemble_score > 40:
            message = "قوی صعودی - حرکت قوی محتمل است."
        elif ensemble_score > 10:
            message = "صعودی - فضا برای رشد وجود دارد."
        elif ensemble_score < -40:
            message = "قوی نزولی - فشار فروش بالا."
        elif ensemble_score < -10:
            message = "نزولی - احتمال کاهش قیمت."
        else:
            message = "خنثی - تثبیت در محدوده قیمت."

        # نتایج مدل‌های فردی (شبیه‌سازی)
        individual_results = {
            "LSTM": {"score": round(random.uniform(-10, 10), 1), "prob": random.randint(55, 95)},
            "RandomForest": {"score": round(random.uniform(-10, 10), 1), "prob": random.randint(55, 95)},
            "SVC": {"score": round(random.uniform(-10, 10), 1), "prob": random.randint(55, 95)},
            "CNN": {"score": round(random.uniform(-10, 10), 1), "prob": random.randint(55, 95)},
        }
        
        # شبیه‌سازی دقت و اهمیت ویژگی‌ها
        global GLOBAL_TEST_ACCURACY, GLOBAL_RF_IMPORTANCES
        if not GLOBAL_TEST_ACCURACY:
             GLOBAL_TEST_ACCURACY = {"LSTM": 75.3, "RF": 82.1, "SVC": 78.8, "CNN": 80.5}
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
    """بررسی روند در تایم فریم بالاتر (HTF)."""
    htf = TIMEFRAME_MAP.get(current_timeframe)
    if not htf:
        return "N/A", "N/A"

    df_htf, error = fetch_data_twelve_data(symbol, htf)
    if error:
        return "Error", f"HTF Data Error: {error}"

    df_htf = calculate_indicators(df_htf)
    if df_htf.empty:
        return "Neutral", "Not enough HTF data for reliable analysis."

    last = df_htf.iloc[-1]
    
    # تعیین روند بر اساس EMA و ADX/DIs
    ema_col = f'EMA_{20}'
    adx_col = f'ADX_14'
    dipi = f'DIp_14'
    dimi = f'DIm_14'

    trend = "Neutral"
    status = f"ADX: {round(last.get(adx_col, 0), 2)}"

    # بررسی روند بر اساس EMA
    if last.get(ema_col) is not None:
        if last['Close'] > last[ema_col]:
            trend = "Bullish"
        elif last['Close'] < last[ema_col]:
            trend = "Bearish"

    # بررسی استحکام روند بر اساس ADX
    if last.get(adx_col, 0) > 25:
        if last.get(dipi, 0) > last.get(dimi, 0):
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
                # نمایش ADX و DIs
                "adx": f"ADX: {round(last.get('ADX_14', 0), 2)} | +DI: {round(last.get('DIp_14', 0), 2)} | -DI: {round(last.get('DIm_14', 0), 2)}",
                # نمایش MACD و هیستوگرام (H)
                "macd": f"MACD: {round(last.get('MACD_12_26_9', 0), 4)} | H: {round(last.get('MACDh_12_26_9', 0), 4)}",
                # نمایش استوکاستیک K و D
                "stoch": f"K: {round(last.get('STOCHk_14_3_3', 0), 2)} | D: {round(last.get('STOCHd_14_3_3', 0), 2)}",
                # نمایش باندهای بولینگر (بالا، میانی، پایین)
                "bbands": f"U: {round(last.get('BBU_5_2.0', 0), 2)} | M: {round(last.get('BB_5_2.0', 0), 2)} | L: {round(last.get('BBL_5_2.0', 0), 2)}",
                "atr": round(last.get('ATR_14', 0), 4),
                "regime": f"Regime: {last.get('Regime', 'N/A')}",
                # نمایش کانال دانچین (پایین، بالا)
                "donchian": f"L: {round(last.get('DCL_20', 0), 4)} | R: {round(last.get('DCU_20', 0), 4)}",
                "divergence": div_msg,
                "ai_report": {
                    "message": ml_report.get("message"),
                    "ensemble_score": ml_report.get("ensemble_score"), 
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
        # چاپ کامل جزئیات خطا برای دیباگ کردن
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

