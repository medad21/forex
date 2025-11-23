import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import joblib
import warnings
import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# ✅ ایمپورت ماژول دیتابیس جدید
import database

# ---------------------------------------------------------
# ۱. پیکربندی
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__, template_folder='.')

# کلیدهای API (در صورت نیاز از Environment Variables بخوانید)
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "df521019db9f44899bfb172fdce6b454")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# پارامترهای استراتژی
RISK_REWARD_ATR = 1.5
TARGET_PERIODS = 5
ML_CONFIDENCE_THRESHOLD = 1.0
SIGNAL_SCORE_THRESHOLD = 5.0
LSTM_TIME_STEPS = 10
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
ML_SCORE_NORMALIZER = 40.0

# متغیرهای سراسری مدل
GLOBAL_MODELS_LOADED = False
rf_model = None
lr_model = None
xgb_model = None
lstm_model = None
scaler = None

# اهمیت ویژگی‌ها (می‌تواند بعداً از مدل خوانده شود)
GLOBAL_RF_IMPORTANCES = {"RSI_14": 0.25, "ADX": 0.2, "EMA_Diff_Fast": 0.15}

# ---------------------------------------------------------
# ۲. بارگذاری مدل‌ها (یک بار در شروع برنامه)
# ---------------------------------------------------------
try:
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        lr_model = joblib.load('models/lr_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
        GLOBAL_MODELS_LOADED = True
        print("✅ AI Models Loaded Successfully.")
    else:
        print("⚠️ Warning: Model files not found in 'models/' directory. AI features disabled.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ---------------------------------------------------------
# ۳. توابع کمکی و منطق ترید
# ---------------------------------------------------------

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

def get_candles(symbol, interval, size=2000):
    """
    دریافت کندل‌ها با اولویت دیتابیس Postgres
    """
    # ۱. خواندن از دیتابیس
    df_db = database.get_all_candles(symbol, interval)
    
    # ۲. تعیین زمان شروع برای دانلود دیتای جدید
    output_size = size
    if not df_db.empty:
        # اگر دیتا داریم، فقط جدیدها را بگیریم (به اضافه کمی بافر)
        output_size = 500 # تعداد کمتری از API می‌گیریم چون تاریخچه را داریم
    
    # ۳. درخواست به API
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={output_size}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if "values" in data:
            df_new = pd.DataFrame(data["values"])
            # تمیزکاری دیتا
            cols = ['open', 'high', 'low', 'close', 'volume']
            for c in cols:
                df_new[c] = pd.to_numeric(df_new[c], errors='coerce')
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            df_new = df_new.dropna()
            # معکوس کردن (چون API جدیدترین را اول می‌دهد)
            df_new = df_new.iloc[::-1].reset_index(drop=True)
            
            # ۴. ذخیره در دیتابیس (فقط جدیدها)
            database.save_candles(df_new, symbol, interval)
            
            # ۵. ترکیب دیتابیس و دیتای جدید
            if not df_db.empty:
                df_final = pd.concat([df_db, df_new]).drop_duplicates(subset=['datetime'], keep='last')
                df_final = df_final.sort_values(by='datetime').reset_index(drop=True)
            else:
                df_final = df_new
                
            return df_final.tail(size).reset_index(drop=True)
            
    except Exception as e:
        print(f"⚠️ API Fetch Error: {e}")
    
    # اگر API خطا داد، هرچه در دیتابیس داریم را برگردان
    return df_db.tail(size).reset_index(drop=True) if not df_db.empty else None

def calculate_smart_sl_tp(entry, signal, atr, support, resistance):
    if not atr or atr == 0: return None, None
    rr = 2.0
    sl, tp = None, None
    
    if signal == "buy":
        sl_base = entry - (atr * 1.5)
        # اگر حمایت نزدیک است، استاپ را کمی زیر حمایت بگذار
        if support != 0 and (entry - support) < (atr * 2.5):
            sl_base = min(sl_base, support - (atr * 0.1))
        tp = entry + ((entry - sl_base) * rr)
        sl = sl_base
    elif signal == "sell":
        sl_base = entry + (atr * 1.5)
        # اگر مقاومت نزدیک است، استاپ را کمی بالای مقاومت بگذار
        if resistance != 0 and (resistance - entry) < (atr * 2.5):
            sl_base = max(sl_base, resistance + (atr * 0.1))
        tp = entry - ((sl_base - entry) * rr)
        sl = sl_base
        
    return (round(sl, 5) if sl else None), (round(tp, 5) if tp else None)

def get_market_sentiment(symbol):
    # کدهای ساده شده برای دریافت سنتیمنت (برای جلوگیری از کندی، تایم‌اوت کوتاه دارد)
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol: av_symbol = "CRYPTO:BTC"
        elif "XAU" in symbol: av_symbol = "FOREX:XAUUSD"
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=2)
        data = r.json()
        if "feed" in data and data["feed"]:
            score = float(data["feed"][0].get("overall_sentiment_score", 0))
            label = data["feed"][0].get("overall_sentiment_label", "Neutral")
            text = f"{label} ({score})"
            return score * 5, text # امتیاز اسکیل شده
    except:
        pass
    return 0, "Neutral (No News)"

def check_divergence(df):
    # بررسی واگرایی ساده RSI
    if len(df) < 20: return 0, "---"
    price = df['close'].values
    rsi = df['RSI_14'].values
    
    # نگاه به 10 کندل آخر
    idx_p_high = np.argmax(price[-10:]) + (len(price)-10)
    idx_r_high = np.argmax(rsi[-10:]) + (len(rsi)-10)
    
    # واگرایی مخفی (ساده شده)
    curr_p = price[-1]
    curr_r = rsi[-1]
    
    msg = "No Divergence"
    score = 0
    
    # اگر قیمت سقف جدید زد ولی RSI نزد (Bearish)
    if price[idx_p_high] > price[idx_p_high-5] and rsi[idx_r_high] < rsi[idx_r_high-5]: 
        # منطق دقیق نیاز به پیوت دارد، این یک تقریب سریع است
        pass 
        
    return score, msg

def process_data(df):
    # محاسبه اندیکاتورها با pandas_ta
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.donchian(lower_length=20, upper_length=20, append=True)
    
    # نام‌گذاری استاندارد ستون‌ها برای دسترسی راحت‌تر
    df['RSI_14'] = df.get('RSI_14', 50)
    df['RSI_6'] = df.get('RSI_6', 50)
    df['ADX'] = df.get('ADX_14', 0)
    df['ATR'] = df.get('ATRr_14', 0)
    df['EMA_20'] = df.get('EMA_20', df['close'])
    df['EMA_50'] = df.get('EMA_50', df['close'])
    df['EMA_100'] = df.get('EMA_100', df['close'])
    
    # سطوح Donchian به عنوان حمایت/مقاومت
    df['DCL'] = df.get('DCL_20_20', df['low'])
    df['DCU'] = df.get('DCU_20_20', df['high'])

    # ویژگی‌های مدل ML (باید دقیقاً مثل زمان آموزش باشد)
    df['Returns'] = df['close'].pct_change()
    df['Volatility'] = (df['high'] - df['low']) / df['close'] # درصدی برای تعمیم‌پذیری
    df['EMA_Diff_Fast'] = (df['EMA_20'] - df['EMA_50']) / df['close']
    df['EMA_Diff_Slow'] = (df['EMA_50'] - df['EMA_100']) / df['close']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()
    
    return df.dropna().reset_index(drop=True)

def get_ml_prediction(df):
    report = {"ensemble_score": 0, "message": "AI: غیرفعال", "individual_results": {}}
    
    if not GLOBAL_MODELS_LOADED or len(df) < LSTM_TIME_STEPS + 5:
        return 0, report

    try:
        # ویژگی‌های مورد نیاز مدل
        feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        
        # آماده‌سازی دیتای ورودی
        last_row = df.iloc[-1][feature_cols].to_frame().T
        input_scaled = scaler.transform(last_row)
        
        # دیتای LSTM (سری زمانی)
        last_seq = df.iloc[-LSTM_TIME_STEPS:][feature_cols]
        input_seq_scaled = scaler.transform(last_seq).reshape(1, LSTM_TIME_STEPS, len(feature_cols))

        score_sum = 0
        
        # 1. Random Forest
        prob_rf = rf_model.predict_proba(input_scaled)[0][1]
        score_rf = (prob_rf - 0.5) * 100
        score_sum += score_rf
        report["individual_results"]["RF"] = {"prob": round(prob_rf*100, 1), "score": round(score_rf, 1)}

        # 2. Logistic Regression
        prob_lr = lr_model.predict_proba(input_scaled)[0][1]
        score_lr = (prob_lr - 0.5) * 100
        score_sum += score_lr
        report["individual_results"]["LR"] = {"prob": round(prob_lr*100, 1), "score": round(score_lr, 1)}

        # 3. XGBoost
        prob_xgb = xgb_model.predict_proba(input_scaled)[0][1]
        score_xgb = (prob_xgb - 0.5) * 100
        score_sum += score_xgb
        report["individual_results"]["XGB"] = {"prob": round(prob_xgb*100, 1), "score": round(score_xgb, 1)}
        
        # 4. LSTM
        prob_lstm = float(lstm_model.predict(input_seq_scaled, verbose=0)[0][0])
        score_lstm = (prob_lstm - 0.5) * 100
        score_sum += score_lstm
        report["individual_results"]["LSTM"] = {"prob": round(prob_lstm*100, 1), "score": round(score_lstm, 1)}

        # تجمیع
        final_ml_score = score_sum / 4.0 # میانگین امتیازها
        report["ensemble_score"] = round(score_sum, 1)
        
        direction = "Bullish 🟢" if final_ml_score > 0 else "Bearish 🔴"
        if abs(final_ml_score) < 5: direction = "Neutral ⚪"
        
        report["message"] = f"AI: {direction} ({int(abs(final_ml_score))}%)"
        
        # نرمال‌سازی برای سیستم امتیازدهی اصلی (بین -10 تا +10)
        normalized_score = np.clip(final_ml_score / 5.0, -10, 10)
        return normalized_score, report

    except Exception as e:
        print(f"❌ AI Inference Error: {e}")
        report["message"] = "AI Error"
        return 0, report

# ---------------------------------------------------------
# ۴. مسیرهای وب (Routes)
# ---------------------------------------------------------

@app.route("/")
def index():
    # اگر فایل index.html دارید آن را رندر کنید، وگرنه پیام ساده
    if os.path.exists("index (16).html"):
        return render_template("index (16).html") # نام فایل را دقیقا چک کنید
    # برای اطمینان از نام فایل آپلودی شما:
    return render_template("index.html") if os.path.exists("index.html") else "<h1>Server is Running</h1>"

@app.route("/analyze")
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    use_htf = request.args.get("use_htf") == "true"
    
    # ۱. دریافت دیتا
    df = get_candles(symbol, interval)
    if df is None or len(df) < 50:
        return jsonify({"error": "Not enough data"}), 500
        
    # ۲. پردازش تکنیکال
    df = process_data(df)
    last = df.iloc[-1]
    
    # ۳. پیش‌بینی هوش مصنوعی
    ml_score_norm, ml_report = get_ml_prediction(df)
    
    # ۴. سیستم امتیازدهی ترکیبی
    score = ml_score_norm # پایه امتیاز از AI می‌آید
    
    # قوانین تکنیکال کلاسیک
    trend = "uptrend" if last['EMA_20'] > last['EMA_50'] else "downtrend"
    rsi = last['RSI_14']
    adx = last['ADX']
    
    if trend == "uptrend": score += 1
    else: score -= 1
    
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    
    if adx > 25: score *= 1.2 # تقویت سیگنال در بازار رونددار
    
    # سنتیمنت اخبار
    news_score, news_text = get_market_sentiment(symbol)
    score += news_score
    
    # تاییدیه تایم بالا (HTF)
    htf_text = "غیرفعال"
    if use_htf:
        htf_int = TIMEFRAME_MAP.get(interval)
        if htf_int:
            df_htf = get_candles(symbol, htf_int, size=100)
            if df_htf is not None:
                df_htf.ta.ema(length=50, append=True)
                htf_last = df_htf.iloc[-1]
                htf_trend = "uptrend" if htf_last['close'] > htf_last['EMA_50'] else "downtrend"
                if htf_trend == trend: score += 2
                else: score -= 2
                htf_text = f"{htf_trend} ({htf_int})"

    # ۵. سیگنال نهایی
    signal = "neutral"
    if score >= SIGNAL_SCORE_THRESHOLD: signal = "buy"
    elif score <= -SIGNAL_SCORE_THRESHOLD: signal = "sell"
    
    # ۶. مدیریت ریسک
    sl, tp = calculate_smart_sl_tp(last['close'], signal, last['ATR'], last['DCL'], last['DCU'])
    
    # ۷. خروجی JSON
    response = {
        "symbol": symbol,
        "price": last['close'],
        "score": round(score, 1),
        "signal": signal,
        "setup": {"sl": sl, "tp": tp},
        "indicators": {
            "rsi": round(rsi, 1),
            "trend": trend,
            "macd": "Bullish" if last['MACD_12_26_9'] > last['MACDs_12_26_9'] else "Bearish",
            "adx": round(adx, 1),
            "regime": "Trending" if adx > 25 else "Ranging",
            "news": news_text,
            "htf_status": htf_text,
            "sr_levels": f"S: {round(last['DCL'],4)} | R: {round(last['DCU'],4)}",
            "divergence": "---", # محاسبه دقیق زمان‌بر است
            "ai_report": {
                "message": ml_report["message"],
                "ensemble_score": ml_report["ensemble_score"],
                "individual_results": ml_report["individual_results"],
                "accuracy": "N/A (Live)",
                "importances": GLOBAL_RF_IMPORTANCES
            }
        }
    }
    
    return jsonify(convert_to_serializable(response))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
