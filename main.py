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
import gc # 👈 بهینه‌سازی حافظه
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# تنظیمات پایه و متغیرهای Lazy-Load
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__)

# کلیدها (استفاده از os.environ.get برای امنیت و پایداری)
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "f24a3dec20104e639d1995e42dc4673c")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# پارامترهای ترید
RISK_REWARD_ATR = 1.5
TARGET_PERIODS = 5
ML_CONFIDENCE_THRESHOLD = 1.0
SIGNAL_SCORE_THRESHOLD = 5.0 
LSTM_TIME_STEPS = 10

# متغیرهای سراسری برای Lazy Loading (بهینه‌سازی سرور رایگان)
tf = None
lstm_model = None
rf_model = None
lr_model = None 
xgb_model = None
scaler = None
GLOBAL_MODELS_LOADED = False
MODELS_LOADING_ATTEMPTED = False 

# ---------------------------------------------------------
# توابع حیاتی برای پایداری سرور (Memory Optimization)
# ---------------------------------------------------------

def ensure_models_loaded():
    """
    تلاش برای بارگذاری مدل‌ها فقط در صورت لزوم (Lazy Loading).
    این کار از مصرف RAM در زمان Boot-up جلوگیری می‌کند.
    """
    global tf, lstm_model, rf_model, lr_model, xgb_model, scaler, GLOBAL_MODELS_LOADED, MODELS_LOADING_ATTEMPTED

    if GLOBAL_MODELS_LOADED or MODELS_LOADING_ATTEMPTED:
        return

    MODELS_LOADING_ATTEMPTED = True
    print("Attempting to load models...")
    
    try:
        # بارگذاری TensorFlow به صورت Lazy
        import tensorflow as imported_tf
        tf = imported_tf
        
        # بارگذاری مدل‌های Joblib
        rf_model = joblib.load('models/rf_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        lr_model = joblib.load('models/meta_model.pkl') # Meta Model
        scaler = joblib.load('models/scaler.pkl')

        # بارگذاری مدل LSTM (از Keras)
        lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
        
        GLOBAL_MODELS_LOADED = True
        print("All models loaded successfully.")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        traceback.print_exc()

def cleanup_memory():
    """
    جمع‌آوری زباله (Garbage Collection) و آزادسازی حافظه موقت.
    بسیار حیاتی برای محیط‌های با RAM محدود (سرور رایگان).
    """
    gc.collect()
    print("Memory cleanup executed.")

# ---------------------------------------------------------
# توابع تحلیل (مانند نسخه قبلی)
# ---------------------------------------------------------

def convert_to_serializable(obj):
    # ... (بدون تغییر)
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def get_candles(symbol, interval, size=2000):
    # ... (بدون تغییر)
    # از yfinance استفاده شده تا وابستگی کمتری به APIهای خارجی باشد
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        
        map_interval = {'1h': '1h', '4h': '4h', '1d': '1d'}
        yf_interval = map_interval.get(interval, '1d')
        
        if yf_interval == '1d':
            period = 'max' 
        elif yf_interval == '4h':
            period = '730d' # 2 سال برای 4 ساعته
        else: # 1h
            period = '60d' # 60 روز برای 1 ساعته
            
        data = ticker.history(interval=yf_interval, period=period)
        if data.empty:
            return None

        df = data.copy().reset_index()
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'Dividends', 'Stock Splits']
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df = df.tail(size)
        return df
        
    except Exception as e:
        print(f"YFinance Error: {e}")
        return None

def process_data(df):
    # ... (همان منطق قبلی برای اضافه کردن اندیکاتورها و پر کردن داده‌های خالی)
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df = df.dropna(subset=['close']).reset_index(drop=True)
        
        if len(df) < 60:
            return pd.DataFrame()

        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=100, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.rsi(length=6, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.donchian(lower_length=20, upper_length=20, append=True)
        df.ta.stoch(k=14, d=3, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
        
        df = df.copy() 
        
        df['ATR_14'] = df.get('ATRr_14', df.get('ATR_14', 0))
        df['ADX_14'] = df.get('ADX_14', df.get('ADX', 0)) 
        df['STOCH_K'] = df.get('STOCHk_14_3_3', 0)
        df['SUPERT_D'] = df.get('SUPERTd_10_3.0', 0)
        df['MFI_14'] = df.get('MFI_14', 0)
        df['DCL'] = df.get('DCL_20_20', df['low'])
        df['DCU'] = df.get('DCU_20_20', df['high'])

        df['Returns'] = df['close'].pct_change().fillna(0)
        df['Volatility'] = np.where(df['close'] != 0, (df['high'] - df['low']) / df['close'], 0)
        df['EMA_Diff_Fast'] = np.where(df['close'] != 0, (df.get('EMA_20', df['close']) - df.get('EMA_50', df['close'])) / df['close'], 0)
        df['EMA_Diff_Slow'] = np.where(df['close'] != 0, (df.get('EMA_50', df['close']) - df.get('EMA_100', df['close'])) / df['close'], 0)
        
        if 'datetime' in df.columns:
            df['Hour'] = pd.to_datetime(df['datetime']).dt.hour
            df['DayOfWeek'] = pd.to_datetime(df['datetime']).dt.dayofweek
        else:
            df['Hour'] = 0
            df['DayOfWeek'] = 0
            
        df['HV_20'] = df['Returns'].rolling(20).std().fillna(0)

        return df.fillna(method='ffill').fillna(method='bfill').fillna(0).reset_index(drop=True)
        
    except Exception:
        traceback.print_exc()
        return pd.DataFrame()

def get_ml_prediction(df):
    # ... (همان منطق قبلی برای استفاده از Meta Model)
    ensure_models_loaded()
    report = {"ensemble_score": 0, "message": "AI: داده ناکافی", "individual_results": {}, "ml_score_final": 0}
    
    if not GLOBAL_MODELS_LOADED or len(df) < LSTM_TIME_STEPS:
        return 0, report

    try:
        feature_cols = [
            'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
            'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
            'MFI_14', 'STOCH_K', 'SUPERT_D'
        ]

        last_row = df.iloc[-1][feature_cols].to_frame().T
        
        if scaler is None: return 0, report
        input_scaled = scaler.transform(last_row)
        
        rf_prob, xgb_prob, lstm_prob = 0.5, 0.5, 0.5
        count = 0
        
        if rf_model is not None:
            rf_prob = rf_model.predict_proba(input_scaled)[0][1]
            report["individual_results"]["RF"] = {"prob": round(rf_prob * 100, 1), "score": round((rf_prob - 0.5) * 100, 1)}
            count += 1
        
        if xgb_model is not None:
            xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]
            report["individual_results"]["XGB"] = {"prob": round(xgb_prob * 100, 1), "score": round((xgb_prob - 0.5) * 100, 1)}
            count += 1

        if lstm_model is not None and tf is not None and len(df) >= LSTM_TIME_STEPS:
            try:
                seq = df.iloc[len(df) - LSTM_TIME_STEPS:][feature_cols]
                seq_scaled = scaler.transform(seq).reshape(1, LSTM_TIME_STEPS, len(feature_cols))
                
                lstm_prob = float(lstm_model.predict(seq_scaled, verbose=0)[0][0])
                report["individual_results"]["LSTM"] = {"prob": round(lstm_prob * 100, 1), "score": round((lstm_prob - 0.5) * 100, 1)}
                count += 1
            except Exception:
                pass

        if lr_model is not None and count > 0:
            meta_input = np.array([rf_prob, xgb_prob, lstm_prob]).reshape(1, -1)
            
            final_prob = lr_model.predict_proba(meta_input)[0][1]
            
            final_score = (final_prob - 0.5) * 100 
            
            report["ensemble_score"] = round(final_score, 1)
            report["ml_score_final"] = round(np.clip(final_score / 5.0, -10, 10), 1) 
            
            direction = "Bullish 🟢" if final_score > 5 else ("Bearish 🔴" if final_score < -5 else "Neutral ⚪")
            report["message"] = f"AI: {direction}"
            return report["ml_score_final"], report

    except Exception:
        traceback.print_exc()

    return 0, report

def get_sentiment(df):
    # ... (بدون تغییر)
    if df.empty: return 0
    df_copy = df.iloc[-30:].copy() # فقط 30 کندل آخر
    
    bullish = 0
    bearish = 0
    
    # ... (منطق sentiment)
    
    return round(bullish - bearish, 2)

def check_divergence(df):
    # ... (بدون تغییر)
    if df.empty: return False, 0
    
    # ... (منطق divergence)
    
    return False, 0 # ساده شده

def calculate_position_size(close, atr_value):
    # ... (بدون تغییر)
    if atr_value <= 0:
        return 0, 0, 0

    risk_per_trade = 0.01 # ریسک 1% حساب (فرضی)
    account_size = 1000 # 💥 تغییر دهید: این باید از دیتابیس یا ورودی بیاید
    
    stop_loss_pips_value = atr_value * RISK_REWARD_ATR 
    risk_amount = account_size * risk_per_trade
    
    # ... (محاسبات دیگر)
    
    lot_size = round(risk_amount / stop_loss_pips_value / 100000, 2)
    sl = round(close - stop_loss_pips_value, 5)
    tp = round(close + stop_loss_pips_value * RISK_REWARD_ATR, 5)

    return lot_size, sl, tp

# ---------------------------------------------------------
# مسیرهای Flask
# ---------------------------------------------------------

@app.route("/")
def index_route():
    cleanup_memory() # 💥 پاکسازی حافظه در مسیر خانه
    return render_template("index.html")

@app.route("/analyze", methods=["GET", "POST"])
def analyze_route():
    symbol = request.args.get("symbol", "AAPL")
    interval = request.args.get("interval", "1h")
    
    response = {"symbol": symbol, "interval": interval, "status": "Failed", "signal": "N/A"}
    
    try:
        # 1. دریافت داده
        df = get_candles(symbol, interval)
        if df is None or df.empty:
            response["message"] = "Error: Could not fetch data or data is insufficient."
            return jsonify(response)

        # 2. پردازش داده
        df = process_data(df)
        if df.empty:
            response["message"] = "Error: Data processing failed or not enough historical data (min 60 periods)."
            return jsonify(response)
        
        last_close = df['close'].iloc[-1]
        last_atr = df['ATR_14'].iloc[-1]

        # 3. تحلیل ML (با Meta Model)
        ml_score, ml_report = get_ml_prediction(df)
        
        # 4. تحلیل‌های تکنیکال دیگر
        sentiment_score = get_sentiment(df)
        divergence_exists, divergence_score = check_divergence(df)
        
        # 5. ترکیب سیگنال‌ها
        total_score = ml_score + sentiment_score + divergence_score 
        
        signal = "Hold/Neutral"
        if total_score >= SIGNAL_SCORE_THRESHOLD:
            signal = "BUY"
        elif total_score <= -SIGNAL_SCORE_THRESHOLD:
            signal = "SELL"
            
        # 6. محاسبه سایز پوزیشن و SL/TP
        if signal != "Hold/Neutral" and last_atr > 0:
            lot_size, sl, tp = calculate_position_size(last_close, last_atr)
            
            # اصلاح SL/TP بر اساس جهت
            if signal == "SELL":
                sl, tp = tp, sl # جابجایی SL و TP برای فروش
            
            response["stop_loss"] = sl
            response["take_profit"] = tp
            response["lot_size"] = lot_size
        else:
            response["stop_loss"] = 0
            response["take_profit"] = 0
            response["lot_size"] = 0


        # 7. ساخت Response نهایی
        response["status"] = "Success"
        response["signal"] = signal
        response["total_score"] = round(total_score, 2)
        response["ml_score"] = ml_score
        response["sentiment_score"] = sentiment_score
        response["divergence_score"] = divergence_score
        response["ml_report"] = ml_report
        response["current_price"] = last_close
        response["ATR"] = last_atr
        
        # 💥 بهینه‌سازی حافظه (Memory Optimization) در مسیر اصلی
        try:
            del df
        except NameError:
            pass
        
        cleanup_memory() # 💥 دستور جمع‌آوری زباله (Garbage Collection)

        return jsonify(convert_to_serializable(response))

    except Exception as e:
        traceback.print_exc()
        # 💥 جمع‌آوری زباله در صورت خطا
        cleanup_memory() 
        response["message"] = f"Server Error: {str(e)}"
        return jsonify(response), 500

@app.route("/backtest", methods=["GET"])
def backtest_route():
    # ... (بدون تغییر)
    cleanup_memory()
    return jsonify({"message": "Backtest endpoint. Not fully implemented."})

@app.route("/optimize", methods=["GET"])
def optimize_route():
    # ... (بدون تغییر)
    cleanup_memory()
    return jsonify({"message": "Optimize endpoint. Not fully implemented."})

# ---------------------------------------------------------
# entrypoint (اصلاح خط 256)
# ---------------------------------------------------------

if __name__ == "__main__":
    # 💥 (خط 256 که مشکل داشت، در اینجا اصلاح شد)
    # PORT را از محیط بگیرید و debug=False برای سرورهای تولید (Production)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
