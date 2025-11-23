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

# 💡 اصلاح مسیر: اگر فایل index.html در پوشه 'templates' قرار دارد، 
# باید template_folder را حذف کنیم یا به 'templates' تغییر دهیم.
# ما template_folder را حذف می‌کنیم تا Flask از حالت پیش‌فرض (پوشه templates) استفاده کند.
app = Flask(__name__) # ⚠️ تغییر: template_folder='.' حذف شد.

# کلیدهای API (در صورت نیاز از Environment Variables بخوانید)
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "df521019db9f44899bfb172fdce6b454")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# پارامترهای استراتژی
RISK_REWARD_ATR = 1.5
SIGNAL_SCORE_THRESHOLD = 5.0
LSTM_TIME_STEPS = 10 
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
ML_SCORE_NORMALIZER = 40.0 

# متغیرهای سراسری برای گزارش‌دهی
GLOBAL_RF_IMPORTANCES = {} 
GLOBAL_TEST_ACCURACY = "N/A (Offline Training Required)"

# ---------------------------------------------------------
# ۲. لود مدل‌ها و ابزارهای مورد نیاز
# ---------------------------------------------------------

rf_model = None
lr_model = None
xgb_model = None
scaler = None

def load_models():
    """لود مدل‌های آموزش دیده و StandardScaler."""
    global rf_model, lr_model, xgb_model, lstm_model, scaler, GLOBAL_RF_IMPORTANCES, GLOBAL_TEST_ACCURACY

    try:
        # لود مدل‌های pkl
        rf_model = joblib.load('models/rf_model.pkl')
        lr_model = joblib.load('models/lr_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        scaler = joblib.load('models/scaler.pkl')

        # لود مدل h5/keras (فقط اگر TensorFlow لود شده باشد)
        if tf:
            # مطمئن شوید که مسیر ذخیره مدل LSTM شما 'models/lstm_model.h5' باشد.
            # اگر در train.py با پسوند .keras ذخیره کرده‌اید، پسوند را اصلاح کنید.
            lstm_model = tf.keras.models.load_model('models/lstm_model.h5') 
            print("✅ LSTM Model Loaded.")

        # استخراج و لود اطلاعات گزارش نهایی از یک فایل مجزا (اختیاری)
        # برای سادگی، فعلاً از مقادیر پیش‌فرض استفاده می‌کنیم.
        
        # اگر در مدل RF ویژگی‌های مهم استخراج شده باشند
        if hasattr(rf_model, 'feature_importances_'):
            feature_names = scaler.feature_names_in_
            importances = dict(zip(feature_names, rf_model.feature_importances_))
            # تبدیل به دیکشنری ساده شده برای نمایش
            sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
            GLOBAL_RF_IMPORTANCES = {k: round(float(v), 2) for k, v in sorted_importances[:3]}
        
        print("✅ Core AI Models Loaded Successfully.")

    except Exception as e:
        # اگر مدل‌ها پیدا نشوند یا مشکلی در لود پیش آید.
        # این حالت در اجرای اول بعد از Train کردن رخ می‌دهد
        print(f"❌ Error loading models: {e}. Using dummy models.")
        # اگر مدل‌ها لود نشدند، از مدل‌های تصادفی استفاده کنید تا برنامه کرش نکند.
        pass # اجازه می‌دهیم مدل‌ها None بمانند و بعداً در تابع predict مدیریت می‌شوند.

# ---------------------------------------------------------
# ۳. توابع اصلی تحلیل
# ---------------------------------------------------------

# ... (تمام توابع fetch_data, calculate_indicators, predict_ensemble_signal, 
# calculate_smart_sl_tp, convert_to_serializable) ...

# ⚠️ برای اینکه این فایل قابل اجرا باشد، باید توابع اصلی (که احتمالاً در main (31).py شما هستند)
# را اضافه کنیم. چون فایل main (31).py قبلی در اختیار من است، فرض می‌کنم توابع در آنجا هستند
# و فقط بخش فلاسک را اصلاح می‌کنم.

# ---------------------------------------------------------
# ۴. مسیردهی Flask
# ---------------------------------------------------------

@app.before_request
def initialize():
    """اجرای یک‌بار برای لود مدل‌ها و دیتابیس در هنگام شروع سرور."""
    if not any([rf_model, lr_model, xgb_model]):
        load_models()
        database.init_db()

@app.route("/", methods=["GET"])
def index():
    """
    ⚠️ اصلاح مسیر رندرینگ. 
    اگر فایل index.html در پوشه templates باشد، فقط نام آن را برمی‌گرداند.
    """
    return render_template('index.html')

@app.route("/analyze", methods=["POST"])
def analyze_market():
    # ... (محتوای تابع analyze_market از main (31).py) ...
    # برای جلوگیری از طولانی شدن کد، محتوای اصلی analyze_market را در اینجا حذف می‌کنم
    # اما شما باید آن را در فایل خود نگه دارید.

    # ❗❗ نکته مهم: اگر مدل‌ها لود نشده‌اند، از اجرای تابع صرف نظر کنید
    if not all([rf_model, lr_model, xgb_model, scaler]):
        return jsonify({
            "error": "AI Models not loaded. Please ensure training was successful.", 
            "status": 503
        }), 503

    # ... (ادامه کد تحلیل بازار) ...
    
    # ⚠️ به دلیل اینکه توابع fetch_data, calculate_indicators و ... در این فایل کامل نیستند،
    # من محتوای analyze_market را به طور کامل از نسخه قبلی شما کپی نمی‌کنم.
    # اما اگر خطا ادامه داشت، مشکل در اینجا و توابع داخلی است.
    
    return jsonify({"status": "OK", "message": "Analysis function is ready to run."})


if __name__ == "__main__":
    # در محیط محلی (Local)
    load_models()
    database.init_db()
    # در Railway، Gunicorn از app استفاده می‌کند و این بخش اجرا نمی‌شود.
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
