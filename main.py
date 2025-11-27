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
# تنظیمات پایه (از کد اصلی شما)
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__)

# استفاده از os.environ.get برای کلیدها (برای امنیت)
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "f24a3dec20104e639d1995e42dc4673c")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

RISK_REWARD_ATR = 1.5
TARGET_PERIODS = 5
ML_CONFIDENCE_THRESHOLD = 1.0
# پیشنهاد: برای افزایش سیگنال، آستانه را کمی کاهش دهید (اگرچه 5.0 قوی‌تر است)
SIGNAL_SCORE_THRESHOLD = 5.0 
LSTM_TIME_STEPS = 10
# ... (بقیه مپ‌ها و تنظیمات ثابت)

# ---------------------------------------------------------
# مدل‌ها و متغیرهای lazy-load (از کد اصلی شما)
# ---------------------------------------------------------
tf = None
lstm_model = None
rf_model = None
lr_model = None # Meta Model (رگرسیون لجستیک)
xgb_model = None
scaler = None
GLOBAL_MODELS_LOADED = False
MODELS_LOADING_ATTEMPTED = False 

# دیتابیس (اختیاری)
# ... (بخش دیتابیس)

# ---------------------------------------------------------
# توابع کمکی برای lazy loading مدل‌ها (بدون تغییر)
# ---------------------------------------------------------
# ... تابع ensure_models_loaded() که شامل import tensorflow و joblib.load است، تغییری نمی‌کند
# ...

# ---------------------------------------------------------
# توابع تبدیل برای JSON (بدون تغییر)
# ---------------------------------------------------------
# ... تابع convert_to_serializable(obj) تغییری نمی‌کند
# ...

# ---------------------------------------------------------
# دریافت کندل‌ها (sync, with yfinance fallback) (بدون تغییر)
# ---------------------------------------------------------
# ... تابع get_candles(symbol, interval, size=2000) تغییری نمی‌کند
# ...

# ---------------------------------------------------------
# پردازش داده‌ها و اندیکاتورها (اصلاح شده برای پایداری)
# ---------------------------------------------------------
def process_data(df):
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        # تنظیم ستون‌ها برای pandas_ta و اطمینان از نوع عددی
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # حذف سطرهای ناقص برای محاسبات
        df = df.dropna(subset=['close']).reset_index(drop=True)
        
        if len(df) < 60:
            return pd.DataFrame() # برگرداندن DataFrame خالی در صورت ناکافی بودن داده

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
        df.ta.stoch(k=14, d=3, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
        
        # 💥 اصلاح: اطمینان از نام ستون‌ها و Fill (برای پایداری)
        df = df.copy() # جلوگیری از SettingWithCopyWarning
        
        df['ATR_14'] = df.get('ATRr_14', df.get('ATR_14', 0))
        df['ADX_14'] = df.get('ADX_14', df.get('ADX', 0)) # ADX نامگذاری‌های مختلفی دارد
        df['STOCH_K'] = df.get('STOCHk_14_3_3', 0)
        df['SUPERT_D'] = df.get('SUPERTd_10_3.0', 0)
        df['MFI_14'] = df.get('MFI_14', 0)
        df['DCL'] = df.get('DCL_20_20', df['low'])
        df['DCU'] = df.get('DCU_20_20', df['high'])

        # ویژگی‌های مشتق شده (بدون تغییر)
        df['Returns'] = df['close'].pct_change().fillna(0)
        df['Volatility'] = np.where(df['close'] != 0, (df['high'] - df['low']) / df['close'], 0)
        df['EMA_Diff_Fast'] = np.where(df['close'] != 0, (df.get('EMA_20', df['close']) - df.get('EMA_50', df['close'])) / df['close'], 0)
        df['EMA_Diff_Slow'] = np.where(df['close'] != 0, (df.get('EMA_50', df['close']) - df.get('EMA_100', df['close'])) / df['close'], 0)
        df['Hour'] = df['datetime'].dt.hour
        df['DayOfWeek'] = df['datetime'].dt.dayofweek
        df['HV_20'] = df['Returns'].rolling(20).std().fillna(0)

        # پر کردن NaNها با روشی پایدار
        return df.fillna(method='ffill').fillna(method='bfill').fillna(0).reset_index(drop=True)
        
    except Exception:
        traceback.print_exc()
        return pd.DataFrame()

# ---------------------------------------------------------
# ML: (اصلاح شده برای استفاده از Meta Model)
# ---------------------------------------------------------
def get_ml_prediction(df):
    """
    wrapper sync for ML prediction. Calls ensure_models_loaded() on demand.
    استفاده از Meta Model (lr_model) برای ترکیب نهایی.
    """
    ensure_models_loaded()
    report = {"ensemble_score": 0, "message": "AI: داده ناکافی", "individual_results": {}, "ml_score_final": 0}
    
    # اگر مدل‌ها لود نشده‌اند یا داده ناکافی است، بلافاصله خارج شوید
    if not GLOBAL_MODELS_LOADED or len(df) < LSTM_TIME_STEPS:
        return 0, report

    try:
        feature_cols = [
            'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
            'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
            'MFI_14', 'STOCH_K', 'SUPERT_D'
        ]

        # اطمینان از اینکه آخرین سطر برای پیش‌بینی استفاده می‌شود
        last_row = df.iloc[-1][feature_cols].to_frame().T
        
        # نرمال‌سازی داده
        if scaler is None: return 0, report
        input_scaled = scaler.transform(last_row)
        
        # 1. پیش‌بینی مدل‌های کلاسیک (ورودی: آخرین سطر نرمال شده)
        rf_prob, xgb_prob, lr_prob_base = 0.5, 0.5, 0.5
        count = 0
        
        # RF
        if rf_model is not None:
            rf_prob = rf_model.predict_proba(input_scaled)[0][1]
            report["individual_results"]["RF"] = {"prob": round(rf_prob * 100, 1), "score": round((rf_prob - 0.5) * 100, 1)}
            count += 1
        
        # XGB
        if xgb_model is not None:
            xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]
            report["individual_results"]["XGB"] = {"prob": round(xgb_prob * 100, 1), "score": round((xgb_prob - 0.5) * 100, 1)}
            count += 1

        # 2. پیش‌بینی LSTM (ورودی: توالی زمانی)
        lstm_prob = 0.5
        if lstm_model is not None and tf is not None and len(df) >= LSTM_TIME_STEPS:
            try:
                # آماده‌سازی توالی برای LSTM
                seq = df.iloc[len(df) - LSTM_TIME_STEPS:][feature_cols]
                seq_scaled = scaler.transform(seq).reshape(1, LSTM_TIME_STEPS, len(feature_cols))
                
                lstm_prob = float(lstm_model.predict(seq_scaled, verbose=0)[0][0])
                report["individual_results"]["LSTM"] = {"prob": round(lstm_prob * 100, 1), "score": round((lstm_prob - 0.5) * 100, 1)}
                count += 1
            except Exception:
                pass

        # 3. ترکیب نهایی با Meta Model (LR Model)
        if lr_model is not None and count > 0:
            # ساخت ورودی برای Meta Model از پراببیلیتی‌های خام
            meta_input = np.array([rf_prob, xgb_prob, lstm_prob]).reshape(1, -1)
            
            # پیش‌بینی نهایی (احتمال کلاس ۱)
            final_prob = lr_model.predict_proba(meta_input)[0][1]
            
            # تبدیل احتمال نهایی به اسکور (از 0.5 به عنوان خط وسط)
            final_score = (final_prob - 0.5) * 100 
            
            report["ensemble_score"] = round(final_score, 1)
            # نرمال‌سازی اسکور به مقیاس 10- تا 10
            report["ml_score_final"] = round(np.clip(final_score / 5.0, -10, 10), 1) 
            
            direction = "Bullish 🟢" if final_score > 5 else ("Bearish 🔴" if final_score < -5 else "Neutral ⚪")
            report["message"] = f"AI: {direction}"
            return report["ml_score_final"], report

    except Exception:
        traceback.print_exc()

    return 0, report

# ---------------------------------------------------------
# توابع کمکی دیگر (sentiment, divergence, position size) (بدون تغییر)
# ---------------------------------------------------------
# ... توابع get_sentiment، check_divergence و calculate_position_size تغییری نمی‌کنند
# ...

# ---------------------------------------------------------
# مسیرهای Flask (sync) (بدون تغییر)
# ---------------------------------------------------------
@app.route("/")
# ... (index_route)

@app.route("/analyze", methods=["GET", "POST"])
def analyze_route():
    # ... (بخش پارس ورودی و دریافت کندل‌ها)

    # 💥 بهینه‌سازی حافظه (Memory Optimization) در مسیر اصلی
    try:
        # ... (تمام منطق تحلیل و سیگنال‌دهی)
        
        # ... (بخش محاسبه sl, tp, lot_size و ساخت response)
        
        # پاکسازی حافظه‌های موقت (برای سرور رایگان)
        try:
            del df
            if 'df_htf' in locals():
                del df_htf
        except Exception:
            pass
        gc.collect() # 💥 دستور جمع‌آوری زباله (Garbage Collection)

        return jsonify(convert_to_serializable(response))

    except Exception as e:
        traceback.print_exc()
        # 💥 جمع‌آوری زباله در صورت خطا
        gc.collect() 
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

@app.route("/backtest", methods=["GET"])
# ... (backtest_route)

@app.route("/optimize", methods=["GET"])
# ... (optimize_route)

# ---------------------------------------------------------
# entrypoint (بدون تغییر)
# ---------------------------------------------------------
if __name__ == "__main__":
    # مهم: debug=False برای اجرا در Gunicorn/Railway
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
