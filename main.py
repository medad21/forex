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

# 📊 پارامترهای تریدینگ (باید با بهترین پارامترهای بهینه‌سازی شده شما پر شوند)
RISK_REWARD_ATR = 1.5           
TARGET_PERIODS = 5              
ML_CONFIDENCE_THRESHOLD = 1.0   
SIGNAL_SCORE_THRESHOLD = 5.0    
LSTM_TIME_STEPS = 10 

# 🧠 بارگذاری مدل‌ها و Scaler فقط یک بار در زمان راه‌اندازی
try:
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
    # اگر مدل‌ها پیدا نشوند، برنامه کرش نمی‌کند اما تحلیل AI غیرفعال می‌شود.

# ---------------------------------------------------------
# ۲. توابع کمکی (Helper Functions)
# ---------------------------------------------------------
# توابع پیچیده شما (calculate_indicators_and_targets، check_target، get_candles، check_divergence، calculate_smart_sl_tp) 
# که منطق آن‌ها پیچیده است، در اینجا حفظ می‌شوند.

# توجه: به دلیل طولانی بودن کد توابع، از درج مجدد آن‌ها در اینجا خودداری می‌کنم.
# فرض بر این است که منطق کامل توابع زیر را در فایل main.py حفظ کرده‌اید:
# - get_candles(symbol, interval, size=2000)
# - check_target(row, df_full, periods, rr_atr)
# - check_divergence(df)
# - get_market_sentiment(symbol)
# - calculate_smart_sl_tp(entry, signal, atr, support, resistance)
# - calculate_indicators_and_targets(df)

# *** تغییر: تابع get_ml_prediction فقط برای استنتاج (Inference) ***
def get_ml_prediction_inference(df_full):
    report = {"ensemble_score": 0, "ml_score_final": 0, "individual_results": {}, "message": "AI: خنثی"}

    if not GLOBAL_MODELS_LOADED:
        report["message"] = "AI: مدل‌ها بارگذاری نشدند."
        return 0, report

    try:
        # ۱. آماده‌سازی داده (همانند بخش آموزش)
        feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        
        # آخرین داده‌ها را برای پیش‌بینی انتخاب کنید
        last_data = df_full.iloc[-1].to_frame().T
        last_data = last_data[['close']].copy() # برای جلوگیری از خطای Key
        
        # محاسبه اندیکاتورهای مورد نیاز برای پیش‌بینی
        # (باید مطمئن شوید که تمام ستون‌های feature_cols در df_full وجود دارند)
        
        # مقیاس‌بندی
        X_scaled_2d = scaler.transform(last_data[feature_cols])
        
        # آماده‌سازی داده سه‌بعدی برای LSTM (فرض بر این است که شما یک پنجره ۱۰ کندلی نیاز دارید)
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

        ml_score = ensemble_score_total / (4 * 40) # 4 مدل * 40 (ML_SCORE_NORMALIZER)
        
        report["ensemble_score"] = float(round(ensemble_score_total, 1))
        report["ml_score_final"] = float(round(ml_score, 2))
        report["message"] = f"AI: Score {ml_score:.2f}"
        
        return ml_score, report

    except Exception as e:
        report["message"] = f"AI Inference Error: {str(e)[:100]}..."
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
    try:
        # ... (منطق کامل تابع analyze از کد پیشرفته قبلی) ...
        
        # *** تغییر اصلی در اینجا: فراخوانی تابع استنتاج جدید ***
        ml_score, ml_report = get_ml_prediction_inference(df.copy())
        
        # ... (ادامه منطق محاسبه امتیاز نهایی، سیگنال و SL/TP) ...
        
        # ⚠️ اطمینان حاصل کنید که پارامترهای symbol, interval, size و use_htf در اینجا درست خوانده شوند.
        # ... (ادامه کد) ...

    except Exception as e:
        # اگر خطایی در تابع analyze رخ دهد، به‌جای کرش 500، پیام خطا را برمی‌گرداند.
        return jsonify({"error": f"Internal Error during Analysis: {str(e)}", "status": 500}), 500

# مسیرهای /backtest و /optimize به دلیل حذف آموزش در این نسخه، باید حذف یا غیرفعال شوند
# یا به گونه‌ای بازنویسی شوند که به جای آموزش، از مدل‌های از قبل ذخیره شده استفاده کنند.
# در حال حاضر، این مسیرها را جهت جلوگیری از کرش، غیرفعال یا حذف کنید.

# ---------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
