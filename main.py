import os
import json
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# ۱. رفع مشکل JSON برای خروجی‌های NumPy
# ---------------------------------------------------------
# چون NumPy از نوع داده‌های استاندارد پایتون نیست، این کلاس آن را به JSON قابل خواندن تبدیل می‌کند.
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# ---------------------------------------------------------
# ۲. پیکربندی اولیه و بارگذاری مدل‌ها
# ---------------------------------------------------------
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.json_encoder = NumpyEncoder

# خواندن کلیدهای API از متغیرهای محیطی Railway
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

# بارگذاری مدل‌های هوش مصنوعی
try:
    # مدل‌ها باید در مسیر 'models/' در ریشه پروژه شما قرار داشته باشند
    # برای TensorFlow (Keras) از tf.keras.models.load_model استفاده کنید
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5') 
    rf_model = joblib.load('models/rf_model.pkl')
    lr_model = joblib.load('models/lr_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl') # Scaler برای پیش‌پردازش داده
    print("✅ All models and scaler loaded successfully.")
except Exception as e:
    # در صورت شکست، برای دیباگ، خطا را نمایش دهید
    print(f"❌ ERROR: Failed to load a model or scaler. Ensure 'models/' directory and files are correct. Error: {e}")
    # اگر مدل‌ها بارگذاری نشوند، برنامه با منطق ساده‌تر (بدون ML) ادامه پیدا می‌کند

# ---------------------------------------------------------
# ۳. توابع کمکی (Helper Functions)
# ---------------------------------------------------------
# توجه: این توابع، اسکلت منطق شما هستند و باید منطق کامل را در داخل آن‌ها پیاده کنید.

def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """دریافت داده‌های تاریخی از TwelveData یا AlphaVantage"""
    
    # در اینجا باید منطق فراخوانی API و تبدیل به DataFrame را پیاده کنید.
    # مثال ساده:
    if not TWELVEDATA_API_KEY:
        print("TWELVEDATA_API_KEY not set.")
        return pd.DataFrame()
        
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={TWELVEDATA_API_KEY}&outputsize=200&format=JSON"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if 'values' in data:
            df = pd.DataFrame(data['values'])
            df = df.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            df = df.astype({'Open': float, 'High': float, 'Low': float, 'Close': float, 'Volume': float})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            return df
    return pd.DataFrame()


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """محاسبه اندیکاتورهای فنی و آماده‌سازی داده‌ها برای مدل‌ها"""
    
    # محاسبه اندیکاتورهای مورد نیاز
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.bbands(append=True)
    df.ta.atr(append=True)
    
    # فیلتر کردن ستون‌های مورد نیاز برای پیش‌بینی
    features = df[['Close', 'RSI_14', 'MACD_12_26_9', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0', 'ATR_14']].iloc[-50:] # برای مثال ۵۰ داده آخر
    
    indicators = {
        "rsi": round(df['RSI_14'].iloc[-1], 2),
        "atr": round(df['ATR_14'].iloc[-1], 5),
        "macd": round(df['MACDH_12_26_9'].iloc[-1], 5),
        "bb_pos": "Breakout" if df['Close'].iloc[-1] > df['BBU_5_2.0'].iloc[-1] else "Inside"
    }

    # پیش‌پردازش برای مدل‌های ML/DL
    scaled_features = scaler.transform(features)
    df_features = pd.DataFrame(scaled_features, columns=features.columns)
    
    return df_features, indicators

def get_market_sentiment(symbol: str) -> dict:
    """دریافت سنتیمنت کلی بازار (اخبار، احساسات)"""
    # این تابع می‌تواند داده‌های اخبار را از یک API دیگر دریافت کند.
    # در اینجا یک خروجی ساده برای تست قرار می‌دهیم.
    return {
        "fa": "در حال حاضر، بازار تحت تأثیر نرخ بهره و گزارش‌های اشتغال، گرایش خنثی تا کمی صعودی دارد.",
        "en": "The market is currently under the influence of interest rates and employment reports, showing a neutral to slightly bullish bias."
    }

def predict_signals(df_features: pd.DataFrame) -> dict:
    """پیش‌بینی سیگنال از مدل‌های مختلف (LSTM, RF, LR, XGBoost)"""
    
    # فرض بر این است که مدل‌ها برای پیش‌بینی Close/Next_Close آموزش دیده‌اند.
    # برای LSTM نیاز به تغییر شکل داده (Reshape) است.
    
    # اجرای پیش‌بینی‌ها
    # مثال: پیش‌بینی باینری (۰: فروش/خنثی، ۱: خرید)
    
    lstm_pred = lstm_model.predict(df_features.values.reshape(1, -1, df_features.shape[1]))[0][0]
    rf_pred = rf_model.predict(df_features.iloc[-1].values.reshape(1, -1))[0]
    xgb_pred = xgb_model.predict(df_features.iloc[-1].values.reshape(1, -1))[0]
    lr_pred = lr_model.predict(df_features.iloc[-1].values.reshape(1, -1))[0]
    
    return {
        "LSTM": "buy" if lstm_pred > 0.5 else "sell",
        "RandomForest": "buy" if rf_pred == 1 else "sell",
        "XGBoost": "buy" if xgb_pred == 1 else "sell",
        "LogisticRegression": "buy" if lr_pred == 1 else "sell",
    }

def get_final_signal(signal_results: dict, df: pd.DataFrame) -> tuple[str, dict]:
    """اجماع و سیگنال نهایی (Ensemble) به همراه جزئیات Stop Loss و Take Profit"""
    
    buy_votes = sum(1 for v in signal_results.values() if v == 'buy')
    sell_votes = sum(1 for v in signal_results.values() if v == 'sell')
    
    final_signal = "neutral"
    if buy_votes >= 3:
        final_signal = "buy"
    elif sell_votes >= 3:
        final_signal = "sell"
    
    # محاسبه ساده SL/TP بر اساس ATR
    atr = df['ATR_14'].iloc[-1]
    last_price = df['Close'].iloc[-1]
    
    sl_value = round(last_price - (atr * 1.5) if final_signal == 'buy' else last_price + (atr * 1.5), 5)
    tp_value = round(last_price + (atr * 3) if final_signal == 'buy' else last_price - (atr * 3), 5)

    setup_details = {
        "entry": round(last_price, 5),
        "sl": sl_value,
        "tp": tp_value
    }
    return final_signal, setup_details

# ---------------------------------------------------------
# ۴. مسیرهای Flask (ROUTES)
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """👈 مسیر ریشه: بارگذاری فرانت‌اند HTML."""
    # این مسیر index.html را از پوشه 'templates' بارگذاری می‌کند.
    return render_template("index.html")

@app.route("/analyze", methods=["GET"])
def analyze():
    """مسیر اصلی: تحلیل لحظه‌ای بازار و تولید سیگنال."""
    symbol = request.args.get("symbol", default="EUR/USD", type=str)
    interval = request.args.get("interval", default="1h", type=str)
    
    df = fetch_data(symbol, interval)
    if df.empty or len(df) < 50:
        return jsonify({"error": "Failed to fetch data or not enough data points (min 50)."}), 500

    df_features, indicators = prepare_features(df)
    sentiment_data = get_market_sentiment(symbol)
    
    # فرض بر این است که مدل‌ها با موفقیت بارگذاری شده‌اند
    signal_results = predict_signals(df_features)
    final_signal, setup_details = get_final_signal(signal_results, df) 
    
    response = {
        "symbol": symbol,
        "interval": interval,
        "signal": final_signal,
        "setup": setup_details,
        "indicators": indicators,
        "models": signal_results,
        "sentiment": sentiment_data,
        "latest_price": df['Close'].iloc[-1]
    }
    return jsonify(response)

@app.route("/backtest", methods=["GET"])
def backtest_route():
    """مسیر بک‌تست: اجرای استراتژی روی داده‌های تاریخی."""
    # ... (منطق بک‌تست باید در اینجا پیاده‌سازی شود) ...
    return jsonify({"status": "Backtest started successfully.", "results": "Placeholder for results."})

@app.route("/optimize", methods=["GET"])
def optimize_route():
    """مسیر بهینه‌سازی: تنظیم پارامترها برای بهترین عملکرد."""
    # ... (منطق بهینه‌سازی باید در اینجا پیاده‌سازی شود) ...
    return jsonify({"status": "Optimization in progress.", "best_params": "Placeholder for best parameters."})

if __name__ == "__main__":
    # در محیط Railway، Gunicorn از این بخش استفاده می‌کند. 
    # این بخش بیشتر برای اجرای لوکال است.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
