import os
import json
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time
import joblib 
import tensorflow as tf 
from flask import Flask, request, jsonify, render_template

# ---------------------------------------------------------
# ۱. پیکربندی و ابزارهای کمکی
# ---------------------------------------------------------

warnings.filterwarnings('ignore')

app = Flask(__name__)

# 🔑 API KEYS
API_KEY_TWELVEDATA = "df521019db9f44899bfb172fdce6b454" 
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"              

# 📊 پارامترها
RISK_REWARD_ATR = 1.5           
TARGET_PERIODS = 5              
ML_CONFIDENCE_THRESHOLD = 1.0   
SIGNAL_SCORE_THRESHOLD = 5.0    
LSTM_TIME_STEPS = 10 
TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
ML_SCORE_NORMALIZER = 40.0 

# متغیرهای سراسری
GLOBAL_RF_IMPORTANCES = {"RSI_14": 0.25, "ADX": 0.2, "EMA_Diff_Fast": 0.15} 
GLOBAL_TEST_ACCURACY = "N/A (Offline Training Required)"

# 🧠 بارگذاری مدل‌ها
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

# ---------------------------------------------------------
# ۲. توابع کمکی
# ---------------------------------------------------------

def convert_to_serializable(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, dict): return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [convert_to_serializable(i) for i in obj]
    return obj

def get_candles(symbol, interval, size=2000):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={size}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "values" not in data: 
            print(f"API Error Response: {data}")
            return None
        df = pd.DataFrame(data["values"])
        for c in ['open', 'high', 'low', 'close']: df[c] = pd.to_numeric(df[c])
        df = df.iloc[::-1].reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e: 
        print(f"Data fetch error: {e}")
        return None

def check_target(row, df_full, periods, rr_atr):
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
        
        if buy_win: return 1 
        if buy_loss: return 2 
        if sell_win: return 0 
        if sell_loss: return 2 
            
    return -1

def check_divergence(df):
    if 'RSI_14' not in df.columns: df.ta.rsi(length=14, append=True)
    subset = df.iloc[-15:].reset_index(drop=True)
    price, rsi = subset['close'], subset['RSI_14']
    price_high_idx = price.idxmax()
    price_low_idx = price.idxmin()
    curr_price, curr_rsi = price.iloc[-1], rsi.iloc[-1]
    score, msg = 0, "بدون واگرایی"
    if price_high_idx < 14 and curr_price > price[price_high_idx] and curr_rsi < rsi[price_high_idx]: 
        msg, score = "Bearish Div 📉 (کاهش)", -3
    elif price_low_idx < 14 and curr_price < price[price_low_idx] and curr_rsi > rsi[price_low_idx]: 
        msg, score = "Bullish Div 📈 (افزایش)", 3
    return score, msg

def get_market_sentiment(symbol):
    sentiment_score = 0
    sentiment_text = "اخبار خنثی (بدون رویداد مهم)"
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol: av_symbol = "CRYPTO:BTC"
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=3)
        data = r.json()
        if "feed" in data and len(data["feed"]) > 0:
            label = data["feed"][0].get("overall_sentiment_label", "Neutral")
            score = float(data["feed"][0].get("overall_sentiment_score", 0))
            if "Bullish" in label: sentiment_text = "🟢 اخبار مثبت (Bullish)"
            elif "Bearish" in label: sentiment_text = "🔴 اخبار منفی (Bearish)"
            sentiment_score = score * 5
            return sentiment_score, sentiment_text
    except: pass
    return sentiment_score, sentiment_text

def calculate_smart_sl_tp(entry, signal, atr, support, resistance):
    if atr is None or np.isnan(atr) or atr == 0: return None, None
    rr = 2.0 
    if signal == "buy":
        sl_base = entry - (atr * 1.5)
        if support != 0 and (entry - support) < (atr * 2.0): sl_base = min(sl_base, support)
        tp = entry + ((entry - sl_base) * rr)
        sl = sl_base
    elif signal == "sell":
        sl_base = entry + (atr * 1.5)
        if resistance != 0 and (resistance - entry) < (atr * 2.0): sl_base = max(sl_base, resistance)
        tp = entry - ((sl_base - entry) * rr)
        sl = sl_base
    else:
        return None, None
    return round(float(sl), 5) if sl is not None else None, round(float(tp), 5) if tp is not None else None

def calculate_indicators_and_targets(df):
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)
    df.ta.donchian(lower_length=20, upper_length=20, append=True)
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
    df['DCL'] = df.get(next((c for c in df.columns if c.startswith('DCL')), ''), 0)
    df['DCU'] = df.get(next((c for c in df.columns if c.startswith('DCU')), ''), 0)
    df['Target'] = df.apply(check_target, axis=1, args=(df, TARGET_PERIODS, RISK_REWARD_ATR)) 
    return df.dropna().reset_index(drop=True)

# ✅ تابع اصلاح شده برای رفع مشکل Undefined
def get_ml_prediction_inference(df_full):
    report = {"ensemble_score": 0, "ml_score_final": 0, "individual_results": {}, "message": "AI: خنثی"}

    if not GLOBAL_MODELS_LOADED:
        report["message"] = "AI: مدل‌ها بارگذاری نشدند."
        return 0, report

    try:
        feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        
        if len(df_full) < LSTM_TIME_STEPS:
            report["message"] = "AI: دیتای کافی نیست."
            return 0, report

        last_data_2d = df_full.iloc[-1].to_frame().T[feature_cols]
        X_scaled_2d = scaler.transform(last_data_2d)
        
        X_scaled_window = scaler.transform(df_full.iloc[-LSTM_TIME_STEPS:][feature_cols])
        X_scaled_3d = X_scaled_window.reshape(1, LSTM_TIME_STEPS, len(feature_cols))

        ensemble_score_total = 0
        
        # پیش‌بینی مدل‌های 2D
        for name, model in [('RF', rf_model), ('LR', lr_model), ('XGB', xgb_model)]:
            prob_p = model.predict_proba(X_scaled_2d)[0][1] 
            confidence_score = (prob_p - 0.5) * 100 
            ensemble_score_total += confidence_score
            # 👇 اصلاح اینجا: ارسال Score و Prob به صورت دیکشنری برای فرانت‌اند
            report["individual_results"][name] = {
                "score": round(confidence_score, 1),
                "prob": round(prob_p * 100, 1)
            }
            
        # پیش‌بینی مدل LSTM
        prob_p_lstm = lstm_model.predict(X_scaled_3d, verbose=0)[0][0]
        confidence_score_lstm = (prob_p_lstm - 0.5) * 100
        ensemble_score_total += confidence_score_lstm
        # 👇 اصلاح اینجا
        report["individual_results"]["LSTM"] = {
            "score": round(confidence_score_lstm, 1),
            "prob": round(prob_p_lstm * 100, 1)
        }

        ml_score = ensemble_score_total / (4 * ML_SCORE_NORMALIZER) 
        report["ensemble_score"] = float(round(ensemble_score_total, 1))
        report["ml_score_final"] = float(round(ml_score, 2))
        
        confidence_percent = round((ensemble_score_total / 400 * 50) + 50, 1) 
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
    return render_template("index.html")

@app.route("/analyze", methods=["GET"])
def analyze():
    try:
        symbol = request.args.get("symbol", "EUR/USD")
        interval = request.args.get("interval", "1h")
        use_htf = request.args.get("use_htf") == "true"
        
        df_raw = get_candles(symbol, interval, size=2000)
        if df_raw is None or df_raw.empty: return jsonify({"error": "API Error: Could not fetch market data."}), 500
        
        df = calculate_indicators_and_targets(df_raw.copy()) 
        if df.empty or len(df) < 50: return jsonify({"error": "Not enough data (min 50)."}), 500
        
        ml_score, ml_report = get_ml_prediction_inference(df.copy())
        
        last = df.iloc[-1]
        price = float(last['close'])
        
        rsi = float(last['RSI_14'])
        atr = float(last['ATR_Value'])
        ema20 = float(last['EMA_20'])
        ema50 = float(last['EMA_50'])
        trend = "uptrend" if ema20 > ema50 else "downtrend"
        macd_line = float(last.get(next((c for c in df.columns if c.startswith('MACD_')), ''), 0))
        macd_sig = float(last.get(next((c for c in df.columns if c.startswith('MACDs_')), ''), 0))
        macd_status = "Bullish 🟢" if macd_line > macd_sig else "Bearish 🔴"
        
        adx_val = float(last['ADX'])
        regime = "Ranging (رنج)"
        if adx_val > 25: regime = "Trending"
        if adx_val > 50: regime = "Strong Trend"
        
        support = float(last['DCL'])
        resistance = float(last['DCU'])
        
        div_score, div_msg = check_divergence(df)
        news_score, news_text = get_market_sentiment(symbol)
        
        htf_trend, htf_status, htf_score = "neutral", "غیرفعال", 0
        if use_htf:
            htf_int = TIMEFRAME_MAP.get(interval)
            if htf_int:
                df_h_raw = get_candles(symbol, htf_int, size=100)
                if df_h_raw is not None and not df_h_raw.empty:
                    df_h_raw.ta.ema(length=20, append=True)
                    df_h_raw.ta.ema(length=50, append=True)
                    l_h = df_h_raw.iloc[-1]
                    e20_h = float(l_h.get(next((c for c in df_h_raw.columns if c.startswith('EMA_20')), ''), 0))
                    e50_h = float(l_h.get(next((c for c in df_h_raw.columns if c.startswith('EMA_50')), ''), 0))
                    htf_trend = "uptrend" if e20_h > e50_h else "downtrend"
                    htf_status = f"فعال ({htf_int})"
                    if trend == htf_trend: htf_score = 2
                    else: htf_score = -1

        score = 0
        current_ml_score = ml_score
        if abs(ml_score) < ML_CONFIDENCE_THRESHOLD:
            current_ml_score = 0

        score += current_ml_score 

        if adx_val > 25: 
            score += 3 if trend == "uptrend" else -3
            score += 1 if macd_line > macd_sig else -1
        else: 
            score += 1 if trend == "uptrend" else -1
            if rsi < 30: score += 3
            elif rsi > 70: score -= 3
            
        dist_to_res = resistance - price
        dist_to_sup = price - support
        if atr > 0:
            if dist_to_res < (atr * 0.5): score -= 2
            if dist_to_sup < (atr * 0.5): score += 2

        score += div_score 
        score += news_score 
        score += htf_score 

        final_signal = "neutral"
        if score >= SIGNAL_SCORE_THRESHOLD: final_signal = "buy"
        elif score <= -SIGNAL_SCORE_THRESHOLD: final_signal = "sell"

        sl, tp = calculate_smart_sl_tp(price, final_signal, atr, support, resistance)
        
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
        return jsonify({"error": f"Internal Error during Analysis: {str(e)}", "status": 500}), 500

@app.route("/backtest", methods=["GET"])
def backtest_route():
    return jsonify({"status": "⚠️ Backtest Disabled on Server"}), 501 

@app.route("/optimize", methods=["GET"])
def optimize_route():
    return jsonify({"status": "⚠️ Optimization Disabled on Server"}), 501 

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
