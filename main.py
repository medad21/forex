import os
import json
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import joblib 
from sklearn.preprocessing import MinMaxScaler 
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# ۱. پیکربندی و ابزارهای کمکی
# ---------------------------------------------------------

warnings.filterwarnings('ignore')

app = Flask(__name__)

# 🔑 API KEYS (Your Keys Restored)
# In a real environment, replace these with secure environment variables
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "df521019db9f44899bfb172fdce6b454")
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")        

# ⚠️ ۱. ایمپورت ایمن TensorFlow و تنظیمات محیطی (برای رفع مشکل ۴۹۹)
tf = None
lstm_model = None
try:
    # جلوگیری از نمایش پیام‌های اضافی TensorFlow در کنسول
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    print("✅ TensorFlow imported successfully. LSTM model can be loaded.")
except ImportError:
    print("⚠️ TensorFlow not installed. LSTM functionality will be disabled.")
except Exception as e:
    print(f"⚠️ TensorFlow import failed. Error: {e}")

# ⚠️ ۲. ایمپورت ماژول دیتابیس (برای مدیریت اتصال دیتابیس)
database = None
try:
    import database 
    print("✅ Database module imported.")
except ImportError:
    print("⚠️ database.py not found. Saving/Database functionality will be disabled.")
except Exception as e:
    print(f"⚠️ Database import failed: {e}")

# 📊 پارامترها
RISK_REWARD_ATR = 1.5           # ATR Multiplier for SL/TP base
TARGET_PERIODS = 5              # Lookahead periods for target calculation
ML_CONFIDENCE_THRESHOLD = 1.0   # Minimum score for ML signal to be non-neutral
SIGNAL_SCORE_THRESHOLD = 5.0    # Minimum total score for final signal (Buy/Sell)
LSTM_TIME_STEPS = 10 
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
ML_SCORE_NORMALIZER = 40.0 

# متغیرهای سراسری
GLOBAL_RF_IMPORTANCES = {"RSI_14": 0.25, "ADX": 0.2, "EMA_Diff_Fast": 0.15} 
GLOBAL_TEST_ACCURACY = "N/A (Offline Training Required)"

# 🧠 بارگذاری مدل‌ها
try:
    # مطمئن شوید که TensorFlow برای بارگذاری مدل LSTM با موفقیت بارگذاری شده باشد
    if tf is None:
        raise ImportError("TensorFlow is not available. Skipping LSTM model load.")
        
    # NOTE: The models folder/files must exist for these to load successfully
    # In a production environment, ensure 'models/' is accessible.
    lstm_model = tf.keras.models.load_model('models/lstm_model.h5')
    rf_model = joblib.load('models/rf_model.pkl')
    lr_model = joblib.load('models/lr_model.pkl')
    xgb_model = joblib.load('models/xgb_model.pkl')
    scaler = joblib.load('models/scaler.pkl') 
    GLOBAL_MODELS_LOADED = True
    print("✅ All ML models and scaler loaded successfully at startup.")
except ImportError as e:
    GLOBAL_MODELS_LOADED = False
    print(f"❌ WARNING: ML models skipped (TensorFlow/Library issue). Running in basic mode. Error: {e}")
except Exception as e:
    GLOBAL_MODELS_LOADED = False
    print(f"❌ WARNING: Failed to load models. Running in basic mode. Error: {e}")

# ---------------------------------------------------------
# ۲. توابع کمکی
# ---------------------------------------------------------

def convert_to_serializable(obj):
    # Converts NumPy types to standard Python types for JSON serialization
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

# ⚠️ تابع get_candles با پشتیبانی از دیتابیس (ترکیب ML و Persistence)
async def get_candles(symbol, interval, size=2000):
    
    df_db = pd.DataFrame() # Start with an empty DataFrame
    # 1. Try to read existing data from the database
    if database:
        df_db = await database.get_all_candles(symbol, interval)
    
    # 2. Determine the timestamp for new data fetch
    start_timestamp = None
    output_size = size # Default API size
    if not df_db.empty:
        # Get the time of the latest candle available in the DB
        # Add 1 second to the max time to prevent fetching the last candle again
        last_time = df_db['datetime'].max() + pd.Timedelta(seconds=1) 
        start_timestamp = int(last_time.timestamp()) 
        output_size = 5000 # If DB exists, fetch a large buffer to ensure all gaps are filled
    else:
        # If DB is empty, get a good initial set from the API
        output_size = 5000 

    # 3. Construct the API URL
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={output_size}"
    
    if start_timestamp:
        # Add start_date parameter to only fetch data newer than what's in the DB
        start_date_iso = pd.to_datetime(start_timestamp, unit='s').strftime('%Y-%m-%d %H:%M:%S')
        url += f"&start_date={start_date_iso.replace(' ', '%20')}" # URL encode the space

    try:
        response = requests.get(url, timeout=15)
        data = response.json()
        
        if "values" not in data or not data["values"]: 
            # If no new data or an API error, return the DB data (if available)
            if not df_db.empty:
                print("API returned no new values. Using only DB data.")
                return df_db.tail(size).reset_index(drop=True) 
            print(f"API Error Response: {data}")
            return None
        
        # 4. Process the new data from API
        df_new = pd.DataFrame(data["values"])
        for c in ['open', 'high', 'low', 'close', 'volume']: 
            df_new[c] = pd.to_numeric(df_new[c], errors='coerce')
        df_new = df_new.dropna()
        # TwelveData returns newest first; reverse it
        df_new = df_new.iloc[::-1].reset_index(drop=True) 
        df_new['datetime'] = pd.to_datetime(df_new['datetime'])
        
        # 5. Save the newly fetched data to the database (only if database module is loaded)
        if database:
            await database.save_candles(df_new, symbol, interval)
        
        # 6. Combine Database and new API data
        df_combined = pd.concat([df_db, df_new], ignore_index=True)
        # Drop duplicates based on the 'datetime' column
        df_combined.drop_duplicates(subset=['datetime'], keep='last', inplace=True)
        
        # 7. Return the final required size (e.g., last 2000 candles) for analysis
        return df_combined.sort_values(by='datetime').tail(size).reset_index(drop=True)

    except Exception as e: 
        print(f"Data fetch/DB save error: {e}")
        # On fatal error, still try to return the DB data
        if not df_db.empty:
            return df_db.tail(size).reset_index(drop=True)
        return None

def check_target(row, df_full, periods, rr_atr):
    # Calculates if the price hits TP (1) or SL/Breakeven (2) or neither (-1)
    idx = row.name
    current_close = row['close']
    atr = row['ATR_Value']
    if idx + periods >= len(df_full) or atr == 0: return -1
    future_data = df_full.loc[idx+1 : idx+periods]
    if future_data.empty: return -1
    
    tp_buy = current_close + (atr * rr_atr)
    sl_buy = current_close - (atr * rr_atr)
    tp_sell = current_close - (atr * rr_atr)
    sl_sell = current_close + (atr * rr_atr)

    for i in range(len(future_data)):
        buy_win = (future_data['high'].iloc[i] >= tp_buy)
        buy_loss = (future_data['low'].iloc[i] <= sl_buy)
        sell_win = (future_data['low'].iloc[i] <= tp_sell)
        sell_loss = (future_data['high'].iloc[i] >= sl_sell)
        
        if buy_win: return 1 # Buy successful
        if buy_loss: return 2 # Buy failure/Stop Loss
        if sell_win: return 0 # Sell successful (equivalent to class 0)
        if sell_loss: return 2 # Sell failure/Stop Loss
            
    return -1 # No target hit

def check_divergence(df):
    # Simple check for RSI divergence over the last 15 candles
    if 'RSI_14' not in df.columns: df.ta.rsi(length=14, append=True)
    subset = df.iloc[-15:].reset_index(drop=True)
    price, rsi = subset['close'], subset['RSI_14']
    price_high_idx = price.idxmax()
    price_low_idx = price.idxmin()
    curr_price, curr_rsi = price.iloc[-1], rsi.iloc[-1]
    score, msg = 0, "بدون واگرایی"
    # Bearish Divergence (Price Higher, RSI Lower)
    if price_high_idx < 14 and curr_price > price[price_high_idx] and curr_rsi < rsi[price_high_idx]: 
        msg, score = "Bearish Div 📉 (کاهش)", -3
    # Bullish Divergence (Price Lower, RSI Higher)
    elif price_low_idx < 14 and curr_price < price[price_low_idx] and curr_rsi > rsi[price_low_idx]: 
        msg, score = "Bullish Div 📈 (افزایش)", 3
    return score, msg

def get_market_sentiment(symbol):
    # Fetches market news sentiment from Alpha Vantage
    sentiment_score = 0
    sentiment_text = "اخبار خنثی (بدون رویداد مهم)"
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol or "ETH" in symbol: av_symbol = "CRYPTO:" + symbol.split('/')[0] # Use CRYPTO prefix for crypto
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=3)
        data = r.json()
        if "feed" in data and len(data["feed"]) > 0:
            label = data["feed"][0].get("overall_sentiment_label", "Neutral")
            score = float(data["feed"][0].get("overall_sentiment_score", 0))
            if "Bullish" in label: sentiment_text = "🟢 اخبار مثبت (Bullish)"
            elif "Bearish" in label: sentiment_text = "🔴 اخبار منفی (Bearish)"
            sentiment_score = score * 5 # Scale sentiment score for use in the final score
            return sentiment_score, sentiment_text
    except: pass
    return sentiment_score, sentiment_text

def calculate_smart_sl_tp(entry, signal, atr, support, resistance):
    # Calculates Stop Loss (SL) and Take Profit (TP) based on ATR and S/R levels
    if atr is None or np.isnan(atr) or atr == 0: return None, None
    rr = 2.0 
    if signal == "buy":
        sl_base = entry - (atr * 1.5)
        # Adjust SL if support is very close
        if support != 0 and (entry - support) < (atr * 2.0): sl_base = min(sl_base, support)
        tp = entry + ((entry - sl_base) * rr)
        sl = sl_base
    elif signal == "sell":
        sl_base = entry + (atr * 1.5)
        # Adjust SL if resistance is very close
        if resistance != 0 and (resistance - entry) < (atr * 2.0): sl_base = max(sl_base, resistance)
        tp = entry - ((sl_base - entry) * rr)
        sl = sl_base
    else:
        return None, None
    return round(float(sl), 5) if sl is not None else None, round(float(tp), 5) if tp is not None else None

def calculate_indicators_and_targets(df):
    # Calculates all necessary technical indicators using pandas_ta
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)
    df.ta.donchian(lower_length=20, upper_length=20, append=True)
    
    # Standardize indicator column names
    df['ADX'] = df.get(next((c for c in df.columns if c.startswith('ADX')), ''), 0)
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()
    df['ATR_Value'] = df.get(next((c for c in df.columns if c.startswith('ATRr')), ''), 0)
    df['RSI_14'] = df.get(next((c for c in df.columns if c.startswith('RSI_14')), ''), 0)
    df['RSI_6'] = df.ta.rsi(length=6) 
    df['EMA_20'] = df.get(next((c for c in df.columns if c.startswith('EMA_20')), ''), 0)
    df['EMA_50'] = df.get(next((c for c in df.columns if c.startswith('EMA_50')), ''), 0)
    df['EMA_100'] = df.get(next((c for c in df.columns if c.startswith('EMA_100')), ''), 0)
    df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
    df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_100']
    df['DCL'] = df.get(next((c for c in df.columns if c.startswith('DCL')), ''), 0) # Donchian Channel Lower
    df['DCU'] = df.get(next((c for c in df.columns if c.startswith('DCU')), ''), 0) # Donchian Channel Upper

    # Target column for ML training (0: Sell, 1: Buy, 2: SL, -1: No Target)
    df['Target'] = df.apply(check_target, axis=1, args=(df, TARGET_PERIODS, RISK_REWARD_ATR)) 
    return df.dropna().reset_index(drop=True)

# ✅ تابع اصلاح شده برای رفع مشکل Undefined
def get_ml_prediction_inference(df_full):
    # Generates predictions using the loaded ensemble models (RF, LR, XGB, LSTM)
    report = {"ensemble_score": 0, "ml_score_final": 0, "individual_results": {}, "message": "AI: خنثی"}

    if not GLOBAL_MODELS_LOADED:
        report["message"] = "AI: مدل‌ها بارگذاری نشدند."
        return 0, report

    # اگر TensorFlow با موفقیت بارگذاری نشده، مدل LSTM کار نمی‌کند.
    if tf is None and 'LSTM' in report["individual_results"]:
        del report["individual_results"]["LSTM"] 
        
    try:
        feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        
        if len(df_full) < LSTM_TIME_STEPS:
            report["message"] = "AI: دیتای کافی نیست."
            return 0, report

        # Data preparation
        missing_cols = [col for col in feature_cols if col not in df_full.columns]
        if missing_cols:
             raise ValueError(f"Missing features in DataFrame: {missing_cols}")

        last_data_2d = df_full.iloc[-1].to_frame().T[feature_cols]
        X_scaled_2d = scaler.transform(last_data_2d)
        
        ensemble_score_total = 0
        model_count = 0
        
        # Predict 2D models
        for name, model in [('RF', rf_model), ('LR', lr_model), ('XGB', xgb_model)]:
            # Predict probability for the "Buy" class (Target=1)
            prob_p = model.predict_proba(X_scaled_2d)[0][1] 
            # Convert probability (0 to 1) to confidence score (-50 to +50)
            confidence_score = (prob_p - 0.5) * 100 
            ensemble_score_total += confidence_score
            model_count += 1
            report["individual_results"][name] = {
                "score": round(confidence_score, 1),
                "prob": round(prob_p * 100, 1)
            }
            
        # Predict LSTM model (3D) - only if TensorFlow and model loaded
        if tf is not None and lstm_model is not None:
            X_scaled_window = scaler.transform(df_full.iloc[-LSTM_TIME_STEPS:][feature_cols])
            X_scaled_3d = X_scaled_window.reshape(1, LSTM_TIME_STEPS, len(feature_cols))
            prob_p_lstm = lstm_model.predict(X_scaled_3d, verbose=0)[0][0]
            confidence_score_lstm = (prob_p_lstm - 0.5) * 100
            ensemble_score_total += confidence_score_lstm
            model_count += 1
            report["individual_results"]["LSTM"] = {
                "score": round(confidence_score_lstm, 1),
                "prob": round(prob_p_lstm * 100, 1)
            }
        
        if model_count == 0:
            report["message"] = "AI: هیچ مدلی برای پیش‌بینی بارگذاری نشد."
            return 0, report

        ml_score = ensemble_score_total / (model_count * ML_SCORE_NORMALIZER) # Normalize to a smaller range (e.g. -2.5 to 2.5)
        report["ensemble_score"] = float(round(ensemble_score_total, 1))
        report["ml_score_final"] = float(round(ml_score, 2))
        
        # Calculate the displayed confidence percentage
        confidence_percent = round((ensemble_score_total / (model_count * 100) * 50) + 50, 1) 
        if abs(ml_score) < ML_CONFIDENCE_THRESHOLD:
            report["message"] = f"Ensemble: {confidence_percent}% ⚪ Neutral"
        else:
            signal = "Bullish 🟢" if ml_score > 0 else "Bearish 🔴"
            report["message"] = f"Ensemble: {confidence_percent}% {signal}"
        
        return ml_score, report

    except Exception as e:
        report["message"] = f"AI Error: {str(e)[:50]}"
        print(f"FATAL AI ERROR: {e}")
        return 0, report

# ---------------------------------------------------------
# ۳. مسیرهای Flask
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    # Show the initial status or simple UI
    tf_status = "✅ Loaded" if tf else "❌ Disabled"
    db_status = "✅ Loaded" if database else "❌ Disabled"
    return f"""
    <h1>Crypto Analysis Service Running (Advanced)!</h1>
    <p>ML models loaded: {'Yes' if GLOBAL_MODELS_LOADED else 'No (Running in Basic Mode)'}</p>
    <p>TensorFlow Status: {tf_status}</p>
    <p>Database Status: {db_status}</p>
    <p>Check /analyze?symbol=EUR/USD&interval=1h</p>
    """

@app.route("/analyze", methods=["GET"])
async def analyze():
    try:
        symbol = request.args.get("symbol", "EUR/USD")
        interval = request.args.get("interval", "1h")
        use_htf = request.args.get("use_htf") == "true"
        
        # 1. Fetch data (uses database persistence)
        df_raw = await get_candles(symbol, interval, size=2000)
        if df_raw is None or df_raw.empty: return jsonify({"error": "API Error: Could not fetch market data."}), 500
        
        # 2. Calculate indicators and targets
        df = calculate_indicators_and_targets(df_raw.copy()) 
        if df.empty or len(df) < 50: return jsonify({"error": "Not enough data (min 50)."}), 500
        
        # 3. ML Prediction
        ml_score, ml_report = get_ml_prediction_inference(df.copy())
        
        last = df.iloc[-1]
        price = float(last['close'])
        
        # 4. Extract indicators
        rsi = float(last['RSI_14'])
        atr = float(last['ATR_Value'])
        ema20 = float(last['EMA_20'])
        ema50 = float(last['EMA_50'])
        trend = "uptrend" if ema20 > ema50 else "downtrend"
        # Safely extract MACD line and signal
        macd_line_col = next((c for c in df.columns if c.startswith('MACD_')), None)
        macd_sig_col = next((c for c in df.columns if c.startswith('MACDs_')), None)
        macd_line = float(last.get(macd_line_col, 0))
        macd_sig = float(last.get(macd_sig_col, 0))
        macd_status = "Bullish 🟢" if macd_line > macd_sig else "Bearish 🔴"
        
        adx_val = float(last['ADX'])
        regime = "Ranging (رنج)"
        if adx_val > 25: regime = "Trending"
        if adx_val > 50: regime = "Strong Trend"
        
        support = float(last['DCL'])
        resistance = float(last['DCU'])
        
        div_score, div_msg = check_divergence(df)
        news_score, news_text = get_market_sentiment(symbol)
        
        # 5. Higher Timeframe Confirmation
        htf_trend, htf_status, htf_score = "neutral", "غیرفعال", 0
        if use_htf:
            htf_int = TIMEFRAME_MAP.get(interval)
            if htf_int:
                df_h_raw = await get_candles(symbol, htf_int, size=100)
                if df_h_raw is not None and not df_h_raw.empty:
                    df_h_raw.ta.ema(length=20, append=True)
                    df_h_raw.ta.ema(length=50, append=True)
                    l_h = df_h_raw.iloc[-1]
                    e20_h_col = next((c for c in df_h_raw.columns if c.startswith('EMA_20')), None)
                    e50_h_col = next((c for c in df_h_raw.columns if c.startswith('EMA_50')), None)
                    e20_h = float(l_h.get(e20_h_col, 0))
                    e50_h = float(l_h.get(e50_h_col, 0))
                    htf_trend = "uptrend" if e20_h > e50_h else "downtrend"
                    htf_status = f"فعال ({htf_int})"
                    if trend == htf_trend: htf_score = 2
                    else: htf_score = -1

        # 6. Signal Scoring Logic
        score = 0
        current_ml_score = ml_score
        if abs(ml_score) < ML_CONFIDENCE_THRESHOLD:
            current_ml_score = 0

        score += current_ml_score 

        if adx_val > 25: # Trending Market Logic
            score += 3 if trend == "uptrend" else -3
            score += 1 if macd_line > macd_sig else -1
        else: # Ranging Market Logic
            score += 1 if trend == "uptrend" else -1
            if rsi < 30: score += 3 # Oversold Buy Signal
            elif rsi > 70: score -= 3 # Overbought Sell Signal
            
        # Proximity to Donchian S/R
        dist_to_res = resistance - price
        dist_to_sup = price - support
        if atr > 0:
            if dist_to_res < (atr * 0.5): score -= 2
            if dist_to_sup < (atr * 0.5): score += 2

        score += div_score # Divergence score
        score += news_score # News sentiment score
        score += htf_score # Higher Timeframe score

        final_signal = "neutral"
        if score >= SIGNAL_SCORE_THRESHOLD: final_signal = "buy"
        elif score <= -SIGNAL_SCORE_THRESHOLD: final_signal = "sell"

        # 7. Calculate SL/TP
        sl, tp = calculate_smart_sl_tp(price, final_signal, atr, support, resistance)
        
        # 8. Prepare Response
        response_data = {
            "symbol": symbol,
            "interval": interval,
            "price": price,
            "signal": final_signal,
            "score": round(score, 1),
            "setup": {"sl": sl, "tp": tp, "rr_ratio": 2.0, "risk_unit_atr": round(atr * 1.5, 5)},
            "indicators": {
                "trend": "صعودی ↗" if trend == "uptrend" else "نزولی ↘", 
                "rsi": round(rsi, 2),
                "atr": round(atr, 5),
                "macd": macd_status,
                "news": news_text, 
                "htf_status": htf_status,
                "htf_trend": htf_trend,
                "regime": f"{regime} (ADX: {int(adx_val)})",
                "sr_levels": f"S: {round(support, 5)} | R: {round(resistance, 5)}",
                "divergence": div_msg,
                "ai_report": {
                    "ensemble_score": ml_report["ensemble_score"],
                    "ml_score_final": ml_report["ml_score_final"],
                    "individual_results": ml_report["individual_results"],
                    "message": ml_report["message"],
                    "accuracy": GLOBAL_TEST_ACCURACY,
                    "importances": GLOBAL_RF_IMPORTANCES,
                }, 
            }
        }
        return jsonify(convert_to_serializable(response_data))

    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"Internal Error: Check data size or feature calculation. Error: {str(e)}"
        return jsonify({"error": error_msg, "status": 500}), 500

@app.route("/backtest", methods=["GET"])
def backtest_route():
    return jsonify({"status": "⚠️ Backtest Disabled on Server"}), 501 

@app.route("/optimize", methods=["GET"])
def optimize_route():
    return jsonify({"status": "⚠️ Optimization Disabled on Server"}), 501 

if __name__ == "__main__":
    # Flask development server setup (use a proper WSGI server in production)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
