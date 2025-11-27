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

API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "f24a3dec20104e639d1995e42dc4673c")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

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
ML_SCORE_NORMALIZER = 40.0

GLOBAL_RF_IMPORTANCES = {"RSI_14": 0.25, "ADX": 0.2, "EMA_Diff_Fast": 0.15}
GLOBAL_TEST_ACCURACY = "N/A (Offline Training Required)"

# ---------------------------------------------------------
# مدل‌ها و متغیرهای lazy-load
# ---------------------------------------------------------
tf = None
lstm_model = None
rf_model = None
lr_model = None
xgb_model = None
scaler = None
GLOBAL_MODELS_LOADED = False
MODELS_LOADING_ATTEMPTED = False  # جلوگیری از تلاش‌های مکرر

# دیتابیس (اختیاری)
database = None
try:
    import database
except Exception:
    database = None

# ---------------------------------------------------------
# توابع کمکی برای lazy loading مدل‌ها (اجباراً sync و محافظت‌شده)
# ---------------------------------------------------------
def ensure_models_loaded():
    global tf, lstm_model, rf_model, lr_model, xgb_model, scaler, GLOBAL_MODELS_LOADED, MODELS_LOADING_ATTEMPTED

    if GLOBAL_MODELS_LOADED or MODELS_LOADING_ATTEMPTED:
        return

    MODELS_LOADING_ATTEMPTED = True
    try:
        try:
            import tensorflow as _tf
            tf = _tf
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        except Exception:
            tf = None

        models_dir = "models"
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        rf_path = os.path.join(models_dir, "rf_model.pkl")
        lr_path = os.path.join(models_dir, "lr_model.pkl")
        xgb_path = os.path.join(models_dir, "xgb_model.pkl")
        lstm_path = os.path.join(models_dir, "lstm_model.h5")

        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
            except Exception as e:
                print(f"⚠️ Failed to load scaler: {e}")
                scaler = None
        if os.path.exists(rf_path):
            try:
                rf_model = joblib.load(rf_path)
            except Exception as e:
                print(f"⚠️ Failed to load rf_model: {e}")
                rf_model = None
        if os.path.exists(lr_path):
            try:
                lr_model = joblib.load(lr_path)
            except Exception as e:
                print(f"⚠️ Failed to load lr_model: {e}")
                lr_model = None
        if os.path.exists(xgb_path):
            try:
                xgb_model = joblib.load(xgb_path)
            except Exception as e:
                print(f"⚠️ Failed to load xgb_model: {e}")
                xgb_model = None

        if tf is not None and os.path.exists(lstm_path):
            try:
                lstm_model = tf.keras.models.load_model(lstm_path, compile=False)
            except Exception as e:
                print(f"⚠️ Failed to load LSTM model: {e}")
                lstm_model = None

        if scaler is not None or rf_model is not None or lr_model is not None or xgb_model is not None or lstm_model is not None:
            GLOBAL_MODELS_LOADED = True
            print("✅ Models loaded lazily.")
        else:
            print("⚠️ No models found or loading failed. Running in basic mode.")
            GLOBAL_MODELS_LOADED = False

    except Exception as e:
        print(f"❌ Unexpected error during model loading: {e}")
        GLOBAL_MODELS_LOADED = False

# ---------------------------------------------------------
# توابع تبدیل برای JSON
# ---------------------------------------------------------
def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

# ---------------------------------------------------------
# دریافت کندل‌ها (sync, بدون YFinance)
# ---------------------------------------------------------
def get_candles(symbol, interval, size=2000):
    df_db = pd.DataFrame()
    try:
        if database:
            df_db = database.get_all_candles(symbol, interval)
    except Exception:
        df_db = pd.DataFrame()

    req_size = 500 if not df_db.empty else size
    df_new = pd.DataFrame()

    # TwelveData
    try:
        api_symbol = symbol.replace("/", "")
        url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={req_size}"
        response = requests.get(url, timeout=5)
        data = response.json()
        if "values" in data and data["values"]:
            df_new = pd.DataFrame(data["values"])
            cols = ['open', 'high', 'low', 'close', 'volume']
            for c in cols:
                if c in df_new.columns:
                    df_new[c] = pd.to_numeric(df_new[c], errors='coerce')
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            df_new = df_new.dropna().iloc[::-1].reset_index(drop=True)
            try:
                if database:
                    database.save_candles(df_new, symbol, interval)
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ TwelveData Error: {e}")

    df_final = pd.DataFrame()
    if not df_db.empty and not df_new.empty:
        df_final = pd.concat([df_db, df_new])
    elif not df_db.empty:
        df_final = df_db
    elif not df_new.empty:
        df_final = df_new

    if not df_final.empty:
        df_final['datetime'] = pd.to_datetime(df_final['datetime'])
        df_final = df_final.drop_duplicates(subset=['datetime'], keep='last')
        df_final = df_final.sort_values(by='datetime').reset_index(drop=True)
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df_final.columns:
                df_final[c] = pd.to_numeric(df_final[c], errors='coerce')
        return df_final.dropna(subset=['close']).tail(size).reset_index(drop=True)

    return None

# ---------------------------------------------------------
# پردازش داده‌ها و اندیکاتورها
# ---------------------------------------------------------
def process_data(df):
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if len(df) < 60:
            return df

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

        # ویژگی‌های تکمیلی
        df.ta.stoch(k=14, d=3, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.supertrend(length=10, multiplier=3.0, append=True)

        if 'ATRr_14' in df.columns:
            df['ATR_14'] = df['ATRr_14']
        if 'ADX_14' not in df.columns and 'ADX' in df.columns:
            df['ADX_14'] = df['ADX']
        if 'STOCHk_14_3_3' in df.columns:
            df['STOCH_K'] = df['STOCHk_14_3_3']
        else:
            df['STOCH_K'] = 0
        if 'SUPERTd_10_3.0' in df.columns:
            df['SUPERT_D'] = df['SUPERTd_10_3.0']
        else:
            df['SUPERT_D'] = 0
        if 'MFI_14' not in df.columns:
            df['MFI_14'] = 0

        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        df['DCL'] = df.get('DCL_20_20', df['low'])
        df['DCU'] = df.get('DCU_20_20', df['high'])
        df['Returns'] = df['close'].pct_change().fillna(0)
        df['Volatility'] = np.where(df['close'] != 0, (df['high'] - df['low']) / df['close'], 0)
        df['EMA_Diff_Fast'] = np.where(df['close'] != 0, (df.get('EMA_20', df['close']) - df.get('EMA_50', df['close'])) / df['close'], 0)
        df['EMA_Diff_Slow'] = np.where(df['close'] != 0, (df.get('EMA_50', df['close']) - df.get('EMA_100', df['close'])) / df['close'], 0)
        df['Hour'] = df['datetime'].dt.hour
        df['DayOfWeek'] = df['datetime'].dt.dayofweek
        df['HV_20'] = df['Returns'].rolling(20).std().fillna(0)

        return df.reset_index(drop=True)
    except Exception as e:
        traceback.print_exc()
        return df

# ---------------------------------------------------------
# ادامه کد: ML prediction، sentiment، divergence، position size، Flask routes
# ---------------------------------------------------------
# تابع get_ml_prediction و بقیه کد بدون تغییر است، فقط YFinance حذف شد

# ---------------------------------------------------------
# entrypoint Flask
# ---------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
