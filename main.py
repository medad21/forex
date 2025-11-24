import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import warnings
import yfinance as yf
import traceback
import gc  # ✅ مدیریت حافظه (بسیار مهم برای سرورهای رایگان)
from flask import Flask, request, jsonify, render_template
from urllib.parse import urlparse

# ✅ 1. ایمپورت ایمن TensorFlow (جلوگیری از کرش در صورت عدم نصب)
tf = None
lstm_model = None
try:
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("✅ TensorFlow imported successfully.")
except ImportError:
    print("⚠️ TensorFlow not installed. LSTM will be disabled.")
except Exception as e:
    print(f"⚠️ TensorFlow import failed. Error: {e}")

# ✅ 2. ایمپورت ماژول دیتابیس (جلوگیری از توقف برنامه در صورت نبود فایل)
try:
    import database
except ImportError:
    print("⚠️ database.py not found. Saving disabled.")

# ---------------------------------------------------------
# تنظیمات برنامه
# ---------------------------------------------------------
warnings.filterwarnings('ignore')
app = Flask(__name__) 

API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "f24a3dec20104e639d1995e42dc4673c")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")

# تنظیمات پیش‌فرض مدیریت ریسک
DEFAULT_RISK_REWARD = 1.5
SIGNAL_SCORE_THRESHOLD = 5.0
LSTM_TIME_STEPS = 10

# نقشه تایم‌فریم‌ها
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

# متغیرهای سراسری مدل‌ها
GLOBAL_MODELS_LOADED = False
rf_model, lr_model, xgb_model, scaler = None, None, None, None

# اهمیت ویژگی‌ها (برای نمایش در فرانت‌اند)
GLOBAL_RF_IMPORTANCES = {
    "RSI_14": 0.15, 
    "ADX": 0.10, 
    "SUPERT_D": 0.20, 
    "STOCH_K": 0.10,
    "MFI_14": 0.10
} 

# ---------------------------------------------------------
# بارگذاری مدل‌های هوش مصنوعی
# ---------------------------------------------------------
try:
    if os.path.exists('models/scaler.pkl'):
        scaler = joblib.load('models/scaler.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        lr_model = joblib.load('models/lr_model.pkl')
        xgb_model = joblib.load('models/xgb_model.pkl')
        
        if tf is not None and os.path.exists('models/lstm_model.h5'):
            try:
                lstm_model = tf.keras.models.load_model('models/lstm_model.h5', compile=False)
                print("✅ LSTM Model Loaded.")
            except: 
                lstm_model = None
                print("⚠️ LSTM Model found but failed to load.")
        
        GLOBAL_MODELS_LOADED = True
        print("✅ All AI Models Loaded Successfully.")
    else:
        print("⚠️ Warning: Models directory or files not found.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# ---------------------------------------------------------
# توابع دریافت داده (Get Candles)
# ---------------------------------------------------------

def get_candles(symbol, interval, size=2000):
    """ دریافت کندل با سیستم فال‌بک (Database -> TwelveData -> YFinance) """
    
    # 1. تلاش اول: دیتابیس
    df_db = pd.DataFrame()
    try:
        df_db = database.get_all_candles(symbol, interval)
    except: pass

    req_size = 500 if not df_db.empty else size
    df_new = pd.DataFrame()
    
    # 2. تلاش دوم: TwelveData API
    try:
        api_symbol = symbol.replace("/", "")
        url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={req_size}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if "values" in data:
            df_new = pd.DataFrame(data["values"])
            cols = ['open', 'high', 'low', 'close', 'volume']
            for c in cols: df_new[c] = pd.to_numeric(df_new[c], errors='coerce')
            df_new['datetime'] = pd.to_datetime(df_new['datetime'])
            df_new = df_new.dropna().iloc[::-1].reset_index(drop=True)
            
            try: database.save_candles(df_new, symbol, interval)
            except: pass
            
    except Exception as e:
        print(f"⚠️ TwelveData Error: {e}")

    # 3. تلاش سوم: YFinance Fallback
    if df_new.empty:
        try:
            print(f"🔄 Trying YFinance fallback for {symbol}...")
            # نگاشت نمادها
            if "BTC" in symbol: yf_symbol = "BTC-USD"
            elif "ETH" in symbol: yf_symbol = "ETH-USD"
            elif "XAU" in symbol: yf_symbol = "GC=F"
            elif "EUR" in symbol: yf_symbol = "EURUSD=X"
            elif "GBP" in symbol: yf_symbol = "GBPUSD=X"
            elif "JPY" in symbol: yf_symbol = "JPY=X"
            else: yf_symbol = symbol.replace("/", "") + "=X"
            
            # نگاشت اینتروال
            yf_int = "1h"
            if interval == "5min": yf_int = "5m"
            elif interval == "15min": yf_int = "15m"
            elif interval == "30min": yf_int = "30m"
            elif interval == "1h": yf_int = "1h"
            elif interval == "4h": yf_int = "1h" # 4h در نسخه رایگان نیست، 1h می‌گیریم
            elif interval == "1day": yf_int = "1d"
            elif interval == "1week": yf_int = "1wk"
            elif interval == "1month": yf_int = "1mo"
            
            period = "1mo"
            if interval in ["5min", "15min", "30min"]: period = "5d"
            elif interval in ["1week", "1month"]: period = "2y"

            df_yf = yf.download(yf_symbol, period=period, interval=yf_int, progress=False)
            
            if not df_yf.empty:
                df_yf = df_yf.reset_index()
                # رفع مشکل MultiIndex
                if isinstance(df_yf.columns, pd.MultiIndex):
                    df_yf.columns = df_yf.columns.get_level_values(0)
                
                rename_map = {'Date': 'datetime', 'Datetime': 'datetime', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}
                df_yf.rename(columns=rename_map, inplace=True)
                
                if 'datetime' in df_yf.columns:
                    df_yf['datetime'] = pd.to_datetime(df_yf['datetime']).dt.tz_localize(None)

                req_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                df_new = df_yf[[c for c in req_cols if c in df_yf.columns]].dropna()
                
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    if c in df_new.columns: df_new[c] = pd.to_numeric(df_new[c], errors='coerce')

                try: database.save_candles(df_new, symbol, interval)
                except: pass
                
        except Exception as e:
            print(f"❌ YFinance Error: {e}")

    # ادغام داده‌ها
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
        
        df_final = df_final.dropna(subset=['close'])
        return df_final.tail(size).reset_index(drop=True)

    return None

# ---------------------------------------------------------
# پردازش داده‌ها (Calculations)
# ---------------------------------------------------------

def process_data(df):
    """ محاسبه اندیکاتورها (شامل ۳ اندیکاتور جدید برای رفع مشکل AI) """
    if df is None or df.empty: return pd.DataFrame()
    
    try:
        # 1. تبدیل اولیه
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        if len(df) < 60: return df # حداقل داده مورد نیاز

        # 2. اندیکاتورهای پایه
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=100, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.rsi(length=6, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.macd(append=True)
        df.ta.donchian(lower_length=20, upper_length=20, append=True)
        
        # 3. ✅ ویژگی‌های جدید (هماهنگ با train.py)
        df.ta.stoch(k=14, d=3, append=True)        # نوسانگر استوکستیک
        df.ta.mfi(length=14, append=True)          # جریان نقدینگی
        df.ta.supertrend(length=10, multiplier=3.0, append=True) # سوپرترند
        
        # 4. یکسان‌سازی نام ستون‌ها
        if 'ATRr_14' in df.columns: df['ATR_14'] = df['ATRr_14']
        if 'ADX_14' not in df.columns and 'ADX' in df.columns: df['ADX_14'] = df['ADX']
        
        # نگاشت نام‌های جدید
        if 'STOCHk_14_3_3' in df.columns: df['STOCH_K'] = df['STOCHk_14_3_3']
        else: df['STOCH_K'] = 0
            
        if 'SUPERTd_10_3.0' in df.columns: df['SUPERT_D'] = df['SUPERTd_10_3.0']
        else: df['SUPERT_D'] = 0

        if 'MFI_14' not in df.columns: df['MFI_14'] = 0

        # پر کردن مقادیر خالی
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 5. محاسبات تکمیلی
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
        print(f"⚠️ Error in process_data: {e}")
        traceback.print_exc()
        return df

# ---------------------------------------------------------
# پیش‌بینی هوش مصنوعی (Ensemble AI)
# ---------------------------------------------------------

def get_ml_prediction(df):
    report = {"ensemble_score": 0, "message": "AI: غیرفعال (داده کم)", "individual_results": {}, "ml_score_final": 0}
    
    if not GLOBAL_MODELS_LOADED or len(df) < 5: 
        return 0, report

    try:
        # ✅ لیست ویژگی‌ها باید دقیقاً ۱۳ مورد باشد (هماهنگ با train.py جدید)
        feature_cols = [
            'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
            'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
            'MFI_14', 'STOCH_K', 'SUPERT_D'
        ]
        
        # اطمینان از وجود ستون‌ها
        for col in feature_cols:
            if col not in df.columns: df[col] = 0
            
        last_row = df.iloc[-1][feature_cols].to_frame().T
        input_scaled = scaler.transform(last_row)
        
        score_sum = 0
        count = 0
        
        # پیش‌بینی مدل‌های کلاسیک
        for name, model in [('RF', rf_model), ('LR', lr_model), ('XGB', xgb_model)]:
             if model:
                 try:
                     p = model.predict_proba(input_scaled)[0][1]
                     s = (p - 0.5) * 100 # تبدیل احتمال به امتیاز (-50 تا +50)
                     score_sum += s
                     count += 1
                     report["individual_results"][name] = {"prob": round(p*100, 1), "score": round(s, 1)}
                 except: pass
        
        # پیش‌بینی LSTM
        if lstm_model and len(df) >= LSTM_TIME_STEPS:
            try:
                required_start_index = len(df) - LSTM_TIME_STEPS
                if required_start_index >= 0:
                    seq = df.iloc[required_start_index:][feature_cols]
                    seq_scaled = scaler.transform(seq).reshape(1, LSTM_TIME_STEPS, len(feature_cols))
                    p = float(lstm_model.predict(seq_scaled, verbose=0)[0][0])
                    s = (p - 0.5) * 100
                    score_sum += s
                    count += 1
                    report["individual_results"]["LSTM"] = {"prob": round(p*100, 1), "score": round(s, 1)}
            except Exception as e:
                 print(f"❌ LSTM Prediction Error: {e}")

        # جمع‌بندی نهایی
        if count > 0:
            final_score = score_sum / count
            report["ensemble_score"] = round(score_sum, 1)
            # نرمال‌سازی امتیاز بین -10 تا +10
            report["ml_score_final"] = round(np.clip(final_score / 5.0, -10, 10), 1)
            
            direction = "Bullish 🟢" if final_score > 5 else ("Bearish 🔴" if final_score < -5 else "Neutral ⚪")
            report["message"] = f"AI: {direction}"
            
            return report["ml_score_final"], report
            
    except Exception as e:
        print(f"❌ AI Prediction Error: {e}")
        traceback.print_exc()
        
    return 0, report

# ---------------------------------------------------------
# توابع کمکی تحلیل (Sentiment, Divergence, SL/TP)
# ---------------------------------------------------------

def get_sentiment(symbol):
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol: av_symbol = "CRYPTO:BTC"
        elif "XAU" in symbol: av_symbol = "FOREX:XAUUSD"
        
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=3)
        data = r.json()
        if "feed" in data and data["feed"]:
            item = data["feed"][0]
            score = float(item.get("overall_sentiment_score", 0))
            label = item.get("overall_sentiment_label", "Neutral")
            return score * 2, f"{label} ({score})"
    except: pass
    return 0, "No News / API Limit"

def check_divergence(df):
    if len(df) < 20: return 0, "---" 
    try:
        price = df['close'].values; rsi = df['RSI_14'].values
        # بررسی واگرایی در 15 کندل اخیر
        prev_max_idx = np.argmax(price[-20:-5]) + (len(price)-20)
        if price[-1] > price[prev_max_idx] and rsi[-1] < rsi[prev_max_idx]: 
            return -3, "Bearish Div 📉"
            
        prev_min_idx = np.argmin(price[-20:-5]) + (len(price)-20)
        if price[-1] < price[prev_min_idx] and rsi[-1] > rsi[prev_min_idx]: 
            return 3, "Bullish Div 📈"
            
        return 0, "No Divergence"
    except: return 0, "---"

def calculate_position_size(balance, risk_pct, sl_pips, symbol):
    """ ✅ تابع جدید: محاسبه حجم ورود """
    try:
        if sl_pips <= 0: return 0
        risk_amount = balance * (risk_pct / 100)
        
        # ارزش هر پیپ (تقریبی)
        pip_value_per_lot = 10 # برای جفت‌های اصلی USD
        if "JPY" in symbol: pip_value_per_lot = 1000 / 110 # تخمینی برای ین
        if "BTC" in symbol: return round(risk_amount / sl_pips, 4) # برای بیتکوین (حجم بر اساس قیمت)
        if "XAU" in symbol: pip_value_per_lot = 100 # طلا
        
        lot_size = risk_amount / (sl_pips * pip_value_per_lot)
        return round(lot_size, 2)
    except: return 0

# ---------------------------------------------------------
# مسیرهای وب (Routes)
# ---------------------------------------------------------

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, int)): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"]) 
def analyze():
    try:
        # 1. دریافت داده‌ها از کلاینت
        data = request.get_json()
        if not data: return jsonify({"error": "No JSON received"}), 400

        symbol = data.get("symbol", "EUR/USD")
        interval = data.get("interval", "1h")
        use_htf = str(data.get("use_htf")).lower() == 'true'
        size = int(data.get("size", 1000))
        
        # پارامترهای جدید مدیریت سرمایه و استراتژی
        balance = float(data.get("balance", 1000))     # موجودی حساب
        risk_pct = float(data.get("risk", 1.0))        # درصد ریسک
        dynamic_sl = data.get("dynamic_sl", False)     # استاپ داینامیک
        
        # 2. دریافت و پردازش کندل‌ها
        df = get_candles(symbol, interval, size)
        
        if df is None or len(df) < 60: 
            return jsonify({"error": "Not enough data (Min 60 candles). Check symbol/API."}), 500
            
        df = process_data(df)
        if df.empty: return jsonify({"error": "Data processing failed."}), 500

        last = df.iloc[-1]
        
        # 3. تحلیل هوش مصنوعی
        ml_score, ml_report = get_ml_prediction(df)
        score = ml_score
        
        # 4. سیستم امتیازدهی کلاسیک + ویژگی‌های جدید
        trend = "Uptrend" if last.get('EMA_20', 0) > last.get('EMA_50', 0) else "Downtrend"
        rsi = last.get('RSI_14', 50)
        adx = last.get('ADX_14', 0)
        supert_d = last.get('SUPERT_D', 0)
        
        # منطق امتیازدهی
        if trend == "Uptrend": score += 1
        else: score -= 1
        
        if rsi < 30: score += 2
        elif rsi > 70: score -= 2
        
        if adx > 25: score *= 1.2
        
        # تاثیر سوپرترند (ویژگی جدید)
        if supert_d == 1: score += 1.5 
        elif supert_d == -1: score -= 1.5
        
        # اخبار و واگرایی
        news_score, news_msg = get_sentiment(symbol); score += news_score
        div_score, div_msg = check_divergence(df); score += div_score
        
        # تحلیل تایم‌فریم بالا (HTF)
        htf_status = "Inactive"; htf_trend = "N/A"
        if use_htf and interval in TIMEFRAME_MAP:
            htf_int = TIMEFRAME_MAP[interval]
            df_htf = get_candles(symbol, htf_int, 200)
            if df_htf is not None:
                df_htf = process_data(df_htf)
                if not df_htf.empty:
                    htf_last = df_htf.iloc[-1]
                    htf_trend = "Bullish" if htf_last.get('EMA_20', 0) > htf_last.get('EMA_50', 0) else "Bearish"
                    htf_status = f"Active: {htf_trend} ({htf_int})"
                    
                    if (htf_trend == "Bullish" and trend == "Uptrend") or \
                       (htf_trend == "Bearish" and trend == "Downtrend"): 
                        score += 2
                    else: 
                        score -= 2

        # 5. تولید سیگنال نهایی
        signal = "neutral"
        if score >= SIGNAL_SCORE_THRESHOLD: signal = "buy"
        elif score <= -SIGNAL_SCORE_THRESHOLD: signal = "sell"
        
        # 6. محاسبات خروج (SL/TP) - ✅ بخش جدید داینامیک/استاتیک
        price = last['close']
        atr = last.get('ATR_14', 0)
        sl, tp = 0, 0
        
        if signal != 'neutral' and atr > 0:
            if signal == 'buy':
                # اگر داینامیک باشد از سوپرترند یا EMA استفاده می‌کنیم، وگرنه کف کانال (DCL)
                if dynamic_sl:
                    # استفاده از خط روند سوپرترند اگر زیر قیمت باشد، وگرنه EMA50
                    support_line = last.get('SUPERT_10_3.0') if last.get('SUPERT_D') == 1 else last.get('EMA_50')
                    sl = support_line if support_line < price else price - (2 * atr)
                else:
                    sl = max(last.get('DCL', 0), price - (2 * atr)) # استاتیک (کف دونچیان)
                
                # تی پی بر اساس ریسک به ریوارد
                dist = price - sl
                tp = price + (dist * DEFAULT_RISK_REWARD)
                
            elif signal == 'sell':
                if dynamic_sl:
                    resistance_line = last.get('SUPERT_10_3.0') if last.get('SUPERT_D') == -1 else last.get('EMA_50')
                    sl = resistance_line if resistance_line > price else price + (2 * atr)
                else:
                    sl = min(last.get('DCU', 0), price + (2 * atr)) # استاتیک (سقف دونچیان)
                
                dist = sl - price
                tp = price - (dist * DEFAULT_RISK_REWARD)

        sl = round(sl, 5)
        tp = round(tp, 5)

        # 7. محاسبه حجم (Lot Size) - ✅ ویژگی درخواستی جدید
        lot_size = 0
        if sl > 0:
            dist_pips = abs(price - sl)
            if "JPY" not in symbol: dist_pips *= 10000 
            else: dist_pips *= 100
            
            lot_size = calculate_position_size(balance, risk_pct, dist_pips, symbol)

        # 8. آماده‌سازی پاسخ
        response = {
            "symbol": symbol, "price": price, "signal": signal,
            "score": round(score, 1),
            "setup": {
                "sl": sl, "tp": tp, 
                "lot_size": lot_size, # ✅ ارسال حجم به فرانت
                "risk_amt": round(balance * (risk_pct/100), 2)
            },
            "indicators": {
                "rsi": round(rsi, 1), "trend": trend,
                "macd": "Bullish" if last.get('MACD_12_26_9', 0) > last.get('MACDs_12_26_9', 0) else "Bearish",
                "adx": round(adx, 1), 
                "regime": "Trending" if adx > 25 else "Ranging",
                "news": news_msg, "htf_status": htf_status, "htf_trend": htf_trend,
                "sr_levels": f"S: {round(last.get('DCL', 0), 4)} | R: {round(last.get('DCU', 0), 4)}",
                "divergence": div_msg,
                "ai_report": {
                    "message": ml_report["message"],
                    "ml_score_final": ml_report["ml_score_final"],
                    "individual_results": ml_report["individual_results"],
                    "accuracy": "N/A", "importances": GLOBAL_RF_IMPORTANCES
                }
            }
        }
        
        # ✅ 9. آزادسازی حافظه (Anti-Crash برای Railway)
        del df
        if 'df_htf' in locals(): del df_htf
        gc.collect()
        
        return jsonify(convert_to_serializable(response))

    except Exception as e:
        print("❌ CRITICAL SERVER ERROR:")
        traceback.print_exc()
        return jsonify({"error": f"Server Logic Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
