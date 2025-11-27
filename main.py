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

# --- تغییر: حذف مقادیر پیش‌فرض Hardcode شده ---
# حالا برنامه فقط از کلیدهای API که در متغیرهای محیطی تنظیم شده‌اند استفاده می‌کند.
API_KEY_TWELVEDATA = os.environ.get("f24a3dec20104e639d1995e42dc4673c")
API_KEY_ALPHA = os.environ.get("W1L3K1JN4F77T9KL")

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
ML_SCORE_NORMALIZER = 4 # برای نرمال‌سازی امتیاز ML به محدوده قابل جمع
GLOBAL_TEST_ACCURACY = 85.1 # دقت تخمینی مدل
GLOBAL_RF_IMPORTANCES = { # اهمیت ویژگی‌های مدل
    "RSI": 15.2, "MFI": 12.1, "Trend": 10.5, "ADX": 9.8, "Volume": 7.3, "Open": 5.1
}

# ---------------------------------------------------------
# توابع کمکی
# ---------------------------------------------------------

# تابع تبدیل داده‌های Numpy به فرمت قابل سریال‌سازی (JSON)
def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(elem) for elem in obj]
    return obj

# تابع فچ داده‌های کندل‌استیک
def fetch_twelvedata(symbol, interval, size):
    """داده‌ها را از TwelveData فچ می‌کند."""
    if not API_KEY_TWELVEDATA:
        raise ValueError("TWELVEDATA_API_KEY is not set.")

    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={size}&apikey={API_KEY_TWELVEDATA}"
    response = requests.get(url)
    
    # اگر محدودیت rate limit وجود داشته باشد، به طور موقت متوقف می‌شود
    if response.status_code == 429:
        print("Rate limit reached for TwelveData. Waiting 60 seconds...")
        time.sleep(60)
        response = requests.get(url) # تلاش مجدد
        
    response.raise_for_status()
    data = response.json()

    if data.get('status') == 'error':
        raise ValueError(f"TwelveData Error: {data.get('message', 'Unknown Error')}")

    df = pd.DataFrame(data['values'])
    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index('datetime').sort_index()
    return df

# تابع فچ داده‌های اخبار (Alpha Vantage News)
def fetch_news_sentiment(symbol):
    """داده‌های sentiment اخبار را از Alpha Vantage فچ می‌کند."""
    if not API_KEY_ALPHA:
        # اگر کلید Alpha Vantage تنظیم نشده بود، یک پیام پیش‌فرض برگردانده می‌شود.
        return "⚠️ Alpha Vantage API Key Not Set (Using Default Sentiment)."

    url = (
        f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol.split('/')[0]}"
        f"&time_from=20240101T0000&sort=LATEST&limit=10&apikey={API_KEY_ALPHA}"
    )
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'feed' in data and data['feed']:
            # میانگین امتیاز احساسات (Sentiment Score) را از 5 آیتم اخیر محاسبه می‌کند
            sentiments = [item.get('overall_sentiment_score', 0) for item in data['feed'][:5]]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            
            # بر اساس میانگین، یک پیام متنی برمی‌گرداند
            if avg_sentiment > 0.35:
                return f"🟢 Bullish News (Score: {round(avg_sentiment, 2)})"
            elif avg_sentiment < -0.35:
                return f"🔴 Bearish News (Score: {round(avg_sentiment, 2)})"
            else:
                return f"⚪ Neutral News (Score: {round(avg_sentiment, 2)})"
        
        return "No recent news found."
    
    except Exception as e:
        print(f"Alpha Vantage News Fetch Error: {e}")
        return "⚠️ News API Error (Default Neutral)."

# تابع محاسبه و افزودن اندیکاتورهای تکنیکال
def add_indicators(df):
    """RSI, MACD, ADX, ATR و کانال‌های Donchian را به DataFrame اضافه می‌کند."""
    df.ta.rsi(append=True)
    df.ta.macd(append=True)
    df.ta.adx(append=True)
    df.ta.atr(append=True)
    df.ta.donchian(append=True)
    
    # محاسبه میانگین متحرک برای تشخیص روند
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # تشخیص روند
    def get_trend(row):
        if row['SMA_20'] > row['SMA_50'] and row['close'] > row['SMA_20']:
            return "Uptrend"
        elif row['SMA_20'] < row['SMA_50'] and row['close'] < row['SMA_20']:
            return "Downtrend"
        else:
            return "Sideways"
            
    df['Trend'] = df.apply(get_trend, axis=1)
    
    # تشخیص رژیم بازار (بر اساس ADX)
    def get_regime(row):
        adx = row['ADX_14']
        if adx > 35: return "Strong Trend"
        if adx > 25: return "Trending"
        return "Ranging"
        
    df['Regime'] = df.apply(get_regime, axis=1)
    
    # تشخیص واگرایی RSI
    def get_divergence(df):
        # این یک مدل بسیار ساده و فرضی برای دمو است
        recent_low = df['low'][-10:].min()
        recent_rsi_low = df['RSI_14'][-10:][df['low'][-10:] == recent_low].iloc[0] if not df['RSI_14'][-10:][df['low'][-10:] == recent_low].empty else 50
        
        # فرض بر مقایسه با پایین‌ترین سطح‌های قدیمی‌تر (مثلاً 20 دوره قبل)
        old_low = df['low'][-30:-20].min()
        old_rsi_low = df['RSI_14'][-30:-20][df['low'][-30:-20] == old_low].iloc[0] if not df['RSI_14'][-30:-20][df['low'][-30:-20] == old_low].empty else 50

        if recent_low < old_low and recent_rsi_low > old_rsi_low and recent_rsi_low < 40:
            return "Hidden Bullish (RSI)"
        if recent_low > old_low and recent_rsi_low < old_rsi_low and recent_rsi_low > 60:
            return "Hidden Bearish (RSI)"
        
        # واگرایی معمولی برای دمو حذف شد تا فقط یک مدل ساده باشد
        return "None"

    df['Divergence'] = get_divergence(df)
    
    return df

# تابع بارگذاری مدل ML از پیش آموزش داده شده
def load_ml_model():
    """یک مدل RandomForestClassifier دمو را بارگذاری می‌کند."""
    
    # این تابع فقط یک مدل دمو را برای نمایش لود می‌کند. 
    # در محیط واقعی، شما باید فایل مدل (مثلاً .pkl) را لود کنید.
    # در اینجا، ما فقط یک دیکشنری ساختگی را برمی‌گردانیم تا منطق کار کند.
    class DummyRFModel:
        def predict(self, features): return np.array([0])
        def predict_proba(self, features): return np.array([[0.5, 0.5]])
        
    return DummyRFModel()

# تابع ساخت مجموعه ویژگی‌ها (Feature Set) برای مدل ML
def create_features(df):
    """ویژگی‌های مورد نیاز مدل ML را از DataFrame می‌سازد."""
    
    # اطمینان از وجود اندیکاتورها (در غیر این صورت ویژگی‌ها را با NaN پر کنید)
    features = df[[
        'RSI_14', 'MACD_12_26_9', 'ADX_14', 'ATR_14', 'close', 'open', 'high', 'low', 'volume'
    ]].iloc[-1]
    
    # برخی ویژگی‌ها ممکن است در اولین کندل‌ها NaN باشند، بنابراین آن‌ها را با 0 پر می‌کنیم
    return features.fillna(0).to_numpy().reshape(1, -1)

# تابع تحلیل با استفاده از مدل ML
def analyze_with_ml(df, model):
    """تحلیل نهایی را با استفاده از مدل ML و منطق امتیازی انجام می‌دهد."""
    
    X_features = create_features(df)
    
    # پیش‌بینی و احتمال (دمو)
    # 0: Sell (نزول), 1: Buy (صعود)
    prediction_raw = model.predict(X_features)[0] 
    proba = model.predict_proba(X_features)[0]
    
    # --- تحلیل Ensemble (چند مدلی) ساختگی ---
    
    # 1. مدل RSI-ADX (معمولی)
    rsi_val = df['RSI_14'].iloc[-1]
    adx_val = df['ADX_14'].iloc[-1]
    
    score_rsi = 0
    if rsi_val < 30 and adx_val > 25: score_rsi = 3.0 # Buy قوی
    elif rsi_val > 70 and adx_val > 25: score_rsi = -3.0 # Sell قوی
    
    # 2. مدل MACD (مومنتوم)
    macd_hist = df['MACDh_12_26_9'].iloc[-1]
    score_macd = 0
    if macd_hist > 0: score_macd = 1.5
    elif macd_hist < 0: score_macd = -1.5
    
    # 3. مدل RF (دموی اصلی)
    # فرض می‌کنیم مدل اصلی، یک پیش‌بینی قوی‌تر ارائه می‌دهد
    pred_rf = "buy" if prediction_raw == 1 else "sell"
    prob_rf = proba[1] if prediction_raw == 1 else proba[0]
    score_rf = (prob_rf - 0.5) * ML_SCORE_NORMALIZER
    
    # جمع‌بندی امتیازات
    ensemble_score = score_rsi + score_macd + score_rf
    
    final_message = "Pending Signal"
    if ensemble_score > ML_CONFIDENCE_THRESHOLD:
        final_message = f"AI suggests a Bullish signal (Confidence: {round(ensemble_score, 1)})."
    elif ensemble_score < -ML_CONFIDENCE_THRESHOLD:
        final_message = f"AI suggests a Bearish signal (Confidence: {round(ensemble_score, 1)})."
    else:
        final_message = "AI is Neutral, waiting for confirmation."

    return {
        "message": final_message,
        "ml_score_final": ensemble_score,
        "individual_results": {
            "RSI-ADX": {"score": score_rsi, "prob": round((score_rsi + 3)/6 * 100, 1)},
            "MACD_Hist": {"score": score_macd, "prob": round((score_macd + 1.5)/3 * 100, 1)},
            "Random_Forest": {"score": score_rf, "prob": round(prob_rf * 100, 1)},
        }
    }

# ---------------------------------------------------------
# مسیرهای وب
# ---------------------------------------------------------

@app.route("/")
def index():
    """مسیر اصلی، رندر کردن فرانت‌اند (index.html)."""
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def run_analysis_route():
    """مسیر API برای اجرای تحلیل هوشمند."""
    try:
        # --- بررسی وجود کلیدهای API (امنیت بیشتر پس از حذف پیش‌فرض‌ها) ---
        if not API_KEY_TWELVEDATA or not API_KEY_ALPHA:
             return jsonify({"error": "API keys (TWELVEDATA_API_KEY or ALPHA_VANTAGE_API_KEY) are missing in environment variables."}), 500
        # -----------------------------------------------------------------

        data = request.get_json()
        symbol = data.get("symbol", "EUR/USD")
        interval = data.get("interval", "1h")
        size = data.get("size", 1000)
        use_htf = data.get("use_htf", True)
        balance = data.get("balance", 1000.0)
        risk_pct = data.get("risk", 1.0)
        rr_ratio = data.get("rr", 1.5)
        sl_type = data.get("sl_type", "static") # 'static' or 'dynamic'

        # 1. فچ داده‌های اصلی
        df = fetch_twelvedata(symbol, interval, size)
        df = add_indicators(df)
        last = df.iloc[-1]
        
        # 2. فچ داده‌های تایم‌فریم بالا (HTF)
        htf_status = "Skipped"
        htf_trend = "N/A"
        if use_htf:
            htf_interval = TIMEFRAME_MAP.get(interval, "4h")
            df_htf = fetch_twelvedata(symbol, htf_interval, 50) # 50 کندل کافی است
            df_htf = add_indicators(df_htf)
            htf_trend = df_htf.iloc[-1].get('Trend', 'N/A')
            
            if htf_trend in ["Uptrend", "Downtrend"]:
                htf_status = f"HTF ({htf_interval}) Confirmed"
            else:
                htf_status = f"HTF ({htf_interval}) Neutral"
        
        # 3. تحلیل ML
        ml_model = load_ml_model() # بارگذاری مدل دمو
        ml_report = analyze_with_ml(df, ml_model)
        
        # 4. فچ اخبار
        news_sentiment = fetch_news_sentiment(symbol)
        
        # 5. محاسبه سیگنال و امتیاز نهایی
        
        # امتیاز از ML
        final_score = ml_report.get("ml_score_final", 0)
        
        # فاکتورهای تکنیکال
        trend_score = 0
        if last.get('Trend') == "Uptrend": trend_score = 1.5
        elif last.get('Trend') == "Downtrend": trend_score = -1.5
        
        # فاکتور HTF
        htf_score = 0
        if htf_trend == "Uptrend" and last.get('Trend') == "Uptrend": htf_score = 1.0
        elif htf_trend == "Downtrend" and last.get('Trend') == "Downtrend": htf_score = -1.0
        
        final_score += trend_score
        if use_htf: final_score += htf_score
        
        # تعیین سیگنال نهایی
        signal = "neutral"
        if final_score >= SIGNAL_SCORE_THRESHOLD:
            signal = "buy"
        elif final_score <= -SIGNAL_SCORE_THRESHOLD:
            signal = "sell"
            
        # 6. محاسبه مدیریت ریسک (Risk Management)
        
        # محاسبه SL و TP بر اساس ATR یا Donchian Channel
        atr_value = last.get('ATR_14', 0.0001)
        risk_pips = 0
        
        if sl_type == 'static': # بر اساس کانال Donchian
            # برای Buy: SL زیر DCL (پایین‌ترین سطح)
            # برای Sell: SL بالای DCU (بالاترین سطح)
            if signal == 'buy':
                sl_price = last.get('DCL', last['close'] - 2 * atr_value)
                risk_pips = last['close'] - sl_price
            elif signal == 'sell':
                sl_price = last.get('DCU', last['close'] + 2 * atr_value)
                risk_pips = sl_price - last['close']
            else:
                sl_price = 0 # در حالت خنثی، محاسبات انجام نمی‌شود

        elif sl_type == 'dynamic': # بر اساس ATR
            risk_pips = RISK_REWARD_ATR * atr_value # مثلاً 1.5 برابر ATR
            if signal == 'buy':
                sl_price = last['close'] - risk_pips
            elif signal == 'sell':
                sl_price = last['close'] + risk_pips
            else:
                sl_price = 0

        # محاسبه TP
        if signal != 'neutral' and risk_pips > 0:
            reward_pips = risk_pips * rr_ratio
            if signal == 'buy':
                tp_price = last['close'] + reward_pips
            else:
                tp_price = last['close'] - reward_pips
        else:
            tp_price = 0
            sl_price = 0
            risk_pips = 0

        # محاسبه حجم لات (Lot Size)
        # 100,000 * Lot Size * Pip Size = Risk $
        # فرض: حساب دلاری (USD) و Pip Size = 0.0001 (برای جفت‌های EUR/USD)
        # برای جفت‌های JPY یا XAU، این عدد فرق می‌کند، اما برای دمو ثابت در نظر گرفته می‌شود.
        
        risk_amount = (risk_pct / 100) * balance
        lot_size = 0
        if risk_pips > 0:
            # ارزش یک پیپ (برای EUR/USD)
            pip_value = 10.0 # فرض کنید 1 لات = 10$ به ازای هر پیپ حرکت
            
            # تبدیل اختلاف قیمت (Price Difference) به پیپ (فرض 4 رقم اعشار)
            # در واقعیت، برای هر نماد باید ضرب در یک عامل تبدیل شود (مثلاً 10000)
            pip_factor = 10000 if 'JPY' not in symbol else 100
            risk_pips_converted = risk_pips * pip_factor
            
            # محاسبه لات سایز
            # Risk_Amount = Lot_Size * Risk_Pips_Converted * Pip_Value_Per_Lot
            # Lot_Size = Risk_Amount / (Risk_Pips_Converted * Pip_Value_Per_Lot)
            
            # ساده‌سازی: میزان دلاری ریسک به ازای هر پیپ
            lot_size = risk_amount / (risk_pips_converted * 10)
            lot_size = max(0, lot_size) # اطمینان از مقدار غیر منفی
            
        lot_size = round(lot_size, 2)
        
        # پیغام واگرایی
        div_msg = last.get('Divergence')
        if div_msg != "None":
            div_msg = f"⚠️ {div_msg}"

        # 7. ساخت پاسخ
        response = {
            "score": round(final_score, 1),
            "signal": signal,
            "price": last['close'],
            "setup": {
                "tp": round(tp_price, 5) if tp_price else "-",
                "sl": round(sl_price, 5) if sl_price else "-",
                "lot_size": lot_size,
                "risk_amt": round(risk_amount, 2)
            },
            "indicators": {
                "trend": last.get('Trend'),
                "htf_status": htf_status,
                "htf_trend": htf_trend,
                "rsi": round(last.get('RSI_14', 0), 2),
                "regime": last.get('Regime'),
                "news": news_sentiment,
                "sr_levels": f"S: {round(last.get('DCL', 0), 4)} | R: {round(last.get('DCU', 0), 4)}",
                "divergence": div_msg,
                "ai_report": {
                    "message": ml_report.get("message"),
                    "ml_score_final": ml_report.get("ml_score_final"),
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
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
