from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import tensorflow as tf
import pandas_ta as ta
from datetime import datetime, timedelta
import os
import warnings
import traceback
import gc

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ----------------------------
# تنظیمات و بارگذاری مدل‌ها
# ----------------------------
MODELS_PATH = "models"
SYMBOL_MAP_YF = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",
    "BTCUSD": "BTC-USD"
}
FEATURE_COLS = [
    'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
    'MFI_14', 'STOCH_K', 'SUPERT_D'
]
TIME_STEPS = 10
GLOBAL_TEST_ACCURACY = 51.38

# بارگذاری مدل‌ها
try:
    SCALER = joblib.load(os.path.join(MODELS_PATH, "scaler.pkl"))
    RF_MODEL = joblib.load(os.path.join(MODELS_PATH, "rf_model.pkl"))
    XGB_MODEL = joblib.load(os.path.join(MODELS_PATH, "xgb_model.pkl"))
    META_MODEL = joblib.load(os.path.join(MODELS_PATH, "meta_model.pkl"))
    try:
        LSTM_MODEL = tf.keras.models.load_model(os.path.join(MODELS_PATH, "lstm_model.h5"))
        USE_LSTM = True
        print("✅ LSTM Model Loaded.")
    except:
        LSTM_MODEL = None
        USE_LSTM = False
        print("⚠️ LSTM not found. Only RF+XGB will be used.")
    print("✅ All Base Models Loaded.")
except Exception as e:
    print(f"❌ Failed to load models: {e}")
    SCALER = RF_MODEL = XGB_MODEL = META_MODEL = None
    USE_LSTM = False

# ----------------------------
# توابع کمکی
# ----------------------------
def create_sequences(X, steps=TIME_STEPS):
    if len(X) >= steps:
        return np.array([X[-steps:]])
    return np.array([])

def get_last_data_point(df, col_name, default=0.0):
    if col_name in df.columns and not df[col_name].empty:
        val = df[col_name].iloc[-1]
        return val if pd.notna(val) else default
    return default

# ----------------------------
# دانلود داده زنده
# ----------------------------
def fetch_live_data(symbol_key, interval='1h', lookback_days=90):
    yf_sym = SYMBOL_MAP_YF.get(symbol_key)
    if not yf_sym:
        return pd.DataFrame()
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    try:
        df = yf.download(yf_sym, start=start_date, interval=interval, progress=False, auto_adjust=True)
        if df.empty or len(df) < 50:
            return pd.DataFrame()
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={'date': 'datetime', 'adj close': 'close'}, inplace=True)
        if 'volume' not in df.columns: df['volume'] = 0.0
        df = df.sort_values('datetime').reset_index(drop=True)
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"❌ YahooFinance error: {e}")
        return pd.DataFrame()

# ----------------------------
# محاسبه اندیکاتورها
# ----------------------------
def calculate_indicators(df):
    if df.empty: return pd.DataFrame()
    df = df.copy()
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.mfi(length=14, append=True)
    try: df.ta.stoch(k=14, d=3, append=True)
    except: pass
    try: df.ta.supertrend(length=10, multiplier=3.0, append=True)
    except: pass
    stoch_k_col = next((c for c in df.columns if 'STOCHk' in c), None)
    supertd_col = next((c for c in df.columns if 'SUPERTd' in c), None)
    df['STOCH_K'] = df[stoch_k_col] if stoch_k_col else 50.0
    df['SUPERT_D'] = df[supertd_col] if supertd_col else 1.0
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()
    ema20 = df.get('EMA_20', df['close'])
    ema50 = df.get('EMA_50', df['close'])
    ema100 = df.get('EMA_100', df['close'])
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100
    df = df.fillna(0.0)
    final_cols = FEATURE_COLS + ['close', 'datetime', 'ATR_14']
    return df.tail(TIME_STEPS + 50).filter(items=final_cols, axis=1)

# ----------------------------
# پیش‌بینی Ensemble
# ----------------------------
def run_ensemble_prediction(df):
    if df.empty or len(df) < TIME_STEPS:
        return 0.5, None, None, {}
    X = df[FEATURE_COLS].values
    current_price = df['close'].iloc[-1]
    current_time = df['datetime'].iloc[-1]
    X_scaled = SCALER.transform(X)
    X_last = X_scaled[-1].reshape(1, -1)
    rf_prob = RF_MODEL.predict_proba(X_last)[0,1]
    xgb_prob = XGB_MODEL.predict_proba(X_last)[0,1]
    meta_inputs = np.array([rf_prob, xgb_prob]).reshape(1,-1)
    individual_results = {
        "RF": {"score": round(rf_prob*2-1,2), "prob": round(rf_prob*100,2)},
        "XGB": {"score": round(xgb_prob*2-1,2), "prob": round(xgb_prob*100,2)}
    }
    if USE_LSTM and LSTM_MODEL:
        X_lstm = create_sequences(X_scaled, TIME_STEPS)
        if len(X_lstm) > 0:
            lstm_prob = LSTM_MODEL.predict(X_lstm, verbose=0)[-1,0]
            meta_inputs = np.column_stack([meta_inputs, np.array([[lstm_prob]])])
            individual_results["LSTM"] = {"score": round(lstm_prob*2-1,2), "prob": round(lstm_prob*100,2)}
    final_prob = META_MODEL.predict_proba(meta_inputs)[0,1]
    final_score = round((final_prob - 0.5) * 200,2)
    return final_score, current_time, current_price, individual_results

# ----------------------------
# API Endpoint
# ----------------------------
@app.route("/api/v1/data", methods=["GET"])
def get_data():
    try:
        symbol = request.args.get("symbol", "EURUSD").upper()
        timeframe = request.args.get("timeframe", "1h")
        if not SCALER or not META_MODEL:
            return jsonify({"error": "Models not loaded. Run train.py first."}), 500
        df_live = fetch_live_data(symbol, interval=timeframe, lookback_days=90)
        if df_live.empty:
            return jsonify({"error": f"Failed to fetch live data for {symbol}."}), 500
        df_processed = calculate_indicators(df_live)
        if df_processed.empty:
            return jsonify({"error": f"Insufficient data after indicator calculation for {symbol}."}), 500
        final_score, current_time, current_price, ind_results = run_ensemble_prediction(df_processed)
        last_atr = get_last_data_point(df_processed, 'ATR_14', 0.0005)
        last_rsi = get_last_data_point(df_processed, 'RSI_14', 50.0)
        signal_message = "NO SIGNAL"
        if final_score >= 10:
            signal_message = "BUY (Long)"
            trend_msg = "Bullish"
        elif final_score <= -10:
            signal_message = "SELL (Short)"
            trend_msg = "Bearish"
        else:
            trend_msg = "Neutral"
        min_atr = max(last_atr, 0.0005)
        tp_price = current_price + (min_atr*1.5) if final_score>0 else current_price - (min_atr*1.5)
        sl_price = current_price - min_atr if final_score>0 else current_price + min_atr
        response = {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": {
                "symbol": symbol,
                "price": round(current_price,5),
                "signal": signal_message,
                "entry": round(current_price,5),
                "tp": round(tp_price,5) if signal_message!="NO SIGNAL" else None,
                "sl": round(sl_price,5) if signal_message!="NO SIGNAL" else None,
                "lot_size": 0,
                "risk_amt": 0
            },
            "indicators": {
                "trend": trend_msg,
                "rsi": round(last_rsi,2),
                "htf_status": "Enabled",
                "htf_trend": trend_msg,
                "ai_report": {
                    "message": f"Ensemble Score: {final_score:.2f}% ({signal_message})",
                    "ensemble_score": final_score,
                    "individual_results": ind_results,
                    "accuracy": GLOBAL_TEST_ACCURACY,
                    "importances": {}
                }
            }
        }
        gc.collect()
        return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

# ----------------------------
# صفحه HTML
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
