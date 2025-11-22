import os
import json
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import joblib # برای بارگذاری مدل‌های Scikit-learn
import tensorflow as tf # برای بارگذاری مدل LSTM
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# ۱. پیکربندی، بارگذاری مدل‌ها و ابزارهای کمکی (GLOBAL)
# ---------------------------------------------------------

warnings.filterwarnings('ignore')

# کلاس کمکی برای تبدیل خروجی NumPy به JSON استاندارد
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder

# 🔑 API KEYS
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY") 
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY") 

# 📊 پارامترها و تنظیمات
RISK_REWARD_ATR = 1.5           
TARGET_PERIODS = 5              
ML_CONFIDENCE_THRESHOLD = 1.0   
SIGNAL_SCORE_THRESHOLD = 5.0    
LSTM_TIME_STEPS = 10 
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
ML_SCORE_NORMALIZER = 40.0 

# متغیرهای سراسری برای ذخیره خروجی‌های آموزش (که باید آفلاین پر شوند)
GLOBAL_RF_IMPORTANCES = {}
GLOBAL_TEST_ACCURACY = "N/A (Offline Training Required)"

# 🧠 بارگذاری مدل‌ها و Scaler فقط یک بار در زمان راه‌اندازی
try:
    # ⚠️ مسیرها را چک کنید: models/lstm_model.h5
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    rf_model = joblib.load('models/rf_model.pkl')
    lr_model = joblib.load('models/lr_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl') 
    GLOBAL_MODELS_LOADED = True
    print("✅ All ML models and scaler loaded successfully at startup.")
except Exception as e:
    GLOBAL_MODELS_LOADED = False
    print(f"❌ WARNING: Failed to load models. Running in basic mode. Error: {e}")

# ---------------------------------------------------------
# ۲. توابع کمکی (Helper Functions)
# ---------------------------------------------------------

# ⚠️ مهم: برای اینکه کد اجرا شود، باید توابع زیر را در این بخش جایگذاری کنید. 
# این توابع شامل منطق پیچیده شما هستند و در این خلاصه حذف شده‌اند:
# - get_candles(symbol, interval, size=2000)
# - check_target(row, df_full, periods, rr_atr)
# - check_divergence(df)
# - get_market_sentiment(symbol)
# - calculate_smart_sl_tp(entry, signal, atr, support, resistance)
# - calculate_indicators_and_targets(df)
# - create_lstm_dataset (برای آماده‌سازی داده سه‌بعدی)

# *** تابع استنتاج (Prediction) - بازنویسی شده برای پایداری ***
def get_ml_prediction_inference(df_full):
    report = {"ensemble_score": 0, "ml_score_final": 0, "individual_results": {}, "message": "AI: خنثی"}

    if not GLOBAL_MODELS_LOADED:
        report["message"] = "AI: مدل‌ها بارگذاری نشدند. (Global Load Failed)"
        return 0, report

    try:
        # ۱. آماده‌سازی داده (باید دقیقاً مشابه زمان آموزش باشد)
        feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        
        # بررسی کافی بودن داده‌ها
        if len(df_full) < LSTM_TIME_STEPS:
            report["message"] = "AI: دیتای کافی برای پنجره LSTM وجود ندارد."
            return 0, report

        # داده‌های 2D (برای RF, XGB, LR)
        last_data_2d = df_full.iloc[-1].to_frame().T[feature_cols]
        X_scaled_2d = scaler.transform(last_data_2d)
        
        # داده‌های 3D (برای LSTM)
        X_scaled_window = scaler.transform(df_full.iloc[-LSTM_TIME_STEPS:][feature_cols])
        X_scaled_3d = X_scaled_window.reshape(1, LSTM_TIME_STEPS, len(feature_cols))

        ensemble_score_total = 0
        
        # ۲. پیش‌بینی مدل‌های 2D
        for name, model in [('RF', rf_model), ('LR', lr_model), ('XGB', xgb_model)]:
            prob_p = model.predict_proba(X_scaled_2d)[0][1] 
            confidence_score = (prob_p - 0.5) * 100 
            ensemble_score_total += confidence_score
            report["individual_results"][name] = round(confidence_score, 1)
            
        # ۳. پیش‌بینی مدل 3D (LSTM)
        prob_p_lstm = lstm_model.predict(X_scaled_3d, verbose=0)[0][0]
        confidence_score_lstm = (prob_p_lstm - 0.5) * 100
        ensemble_score_total += confidence_score_lstm
        report["individual_results"]["LSTM"] = round(confidence_score_lstm, 1)

        # محاسبه امتیاز نهایی Ensemble
        ml_score = ensemble_score_total / (4 * ML_SCORE_NORMALIZER) 
        
        report["ensemble_score"] = float(round(ensemble_score_total, 1))
        report["ml_score_final"] = float(round(ml_score, 2))
        
        # پیام نهایی بر اساس امتیاز
        confidence_percent = round(ml_score * 40 * 100 / 400 + 50, 1) # تبدیل امتیاز به درصد اطمینان
        if abs(ml_score) < ML_CONFIDENCE_THRESHOLD:
            report["message"] = f"Ensemble: {confidence_percent}% ⚪ Neutral (Low Confidence)"
        else:
            signal = "Bullish 🟢" if ml_score > 0 else "Bearish 🔴"
            report["message"] = f"Ensemble: {confidence_percent}% {signal}"
        
        return ml_score, report

    except Exception as e:
        # در صورت بروز خطا در استنتاج، به جای کرش سرور، پیام خطا را در خروجی AI قرار می‌دهد.
        report["message"] = f"AI Inference Error: Check Data/Scaler Compatibility ({str(e)[:50]}...)"
        print(f"FATAL AI INFERENCE ERROR: {e}")
        return 0, report


# ---------------------------------------------------------
# ۳. مسیرهای Flask (ROUTES)
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """مسیر ریشه: بارگذاری فرانت‌اند HTML."""
    return render_template("index.html")

@app.route("/analyze", methods=["GET"])
def analyze():
    # ⚠️ تمام منطق پیچیده قبلی خود را در اینجا جایگذاری کنید
    try:
        symbol = request.args.get("symbol", "EUR/USD")
        interval = request.args.get("interval", "1h")
        # ... (بقیه پارامترها و منطق) ...

        df_raw = get_candles(symbol, interval, size=2000)
        if df_raw is None or df_raw.empty: return jsonify({"error": "API Error: Could not fetch market data."}), 500
        
        df = calculate_indicators_and_targets(df_raw.copy()) 
        if df.empty: return jsonify({"error": "Not enough processed data for analysis."}), 500
        
        # فراخوانی تابع استنتاج جدید
        ml_score, ml_report = get_ml_prediction_inference(df.copy())
        
        # ... (ادامه منطق محاسبه امتیازدهی دستی، سیگنال نهایی و SL/TP) ...
        # ... (این منطق باید از کد پیشرفته شما کپی شده باشد) ...
        
        # خروجی نهایی
        return jsonify({
            "symbol": symbol,
            "interval": interval,
            "price": price, # فرض بر این است که price محاسبه شده است
            "signal": final_signal, # فرض بر این است که final_signal محاسبه شده است
            "score": round(score, 1), # فرض بر این است که score محاسبه شده است
            "setup": {"sl": sl, "tp": tp, "rr_ratio": 2.0, "risk_unit_atr": round(atr * 1.5, 5)}, # فرض بر این است که sl, tp, atr محاسبه شده‌اند
            "indicators": {
                "trend": "صعودی ↗" if trend == "uptrend" else "نزولی ↘", 
                "rsi": round(rsi, 2), # فرض بر این است که rsi محاسبه شده است
                "macd": macd_status, # فرض بر این است که macd_status محاسبه شده است
                "ai_report": {
                    "ensemble_score": ml_report["ensemble_score"],
                    "ml_score_final": ml_report["ml_score_final"],
                    "individual_results": ml_report["individual_results"],
                    "message": ml_report["message"],
                    "accuracy": GLOBAL_TEST_ACCURACY, # استفاده از متغیر سراسری
                    "importances": GLOBAL_RF_IMPORTANCES, # استفاده از متغیر سراسری
                }, 
            }
        })

    except Exception as e:
        return jsonify({"error": f"Internal Error during Analysis: {str(e)}", "status": 500}), 500

# ---------------------------------------------------------
# ۴. غیرفعال‌سازی مسیرهای سنگین برای پایداری (حل مشکل 500 کنسول)
# ---------------------------------------------------------

@app.route("/backtest", methods=["GET"])
def backtest_route():
    """مسیر بک‌تست: غیرفعال شده برای پایداری سرور."""
    return jsonify({
        "status": "⚠️ Error: Backtest is Disabled on Live Server.", 
        "reason": "Training and Backtesting are resource-intensive tasks and must be run offline (locally) to maintain server stability.",
        "solution": "Run your backtesting script locally or upgrade to a high-memory/GPU-enabled server."
    }), 501 # 501: Not Implemented

@app.route("/optimize", methods=["GET"])
def optimize_route():
    """مسیر بهینه‌سازی: غیرفعال شده برای پایداری سرور."""
    return jsonify({
        "status": "⚠️ Error: Optimization is Disabled on Live Server.",
        "reason": "Optimization requires training and backtesting hundreds of times, which consumes too many resources and will crash the server.",
        "solution": "Run your optimization script locally or upgrade to a high-memory/GPU-enabled server."
    }), 501 # 501: Not Implemented

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
