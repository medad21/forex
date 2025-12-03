from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import requests
import os

app = Flask(__name__)

# --- CONFIG ---
MODEL_DIR = "models"
API_KEY = os.environ.get("TWELVEDATA_API_KEY", "f24a3dec20104e639d1995e42dc4673c")
TIME_STEPS = 10

# Mapping برای درخواست کلاینت به فرمت API
SYMBOL_MAP = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",
    "BTCUSD": "BTC/USD"
}

# --- Load Models (Global) ---
try:
    SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    RF = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    XGB = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    META = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))
    LSTM = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
    print("✅ Models Loaded Successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")

# --- Helper Functions ---
def fetch_twelvedata(symbol, interval):
    """گرفتن داده زنده از TwelveData"""
    api_symbol = SYMBOL_MAP.get(symbol, symbol)
    url = f"https://api.twelvedata.com/time_series?symbol={api_symbol}&interval={interval}&apikey={API_KEY}&outputsize=150"
    
    try:
        resp = requests.get(url, timeout=5).json()
        if 'values' not in resp: return pd.DataFrame()
        
        df = pd.DataFrame(resp['values'])
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols: df[c] = pd.to_numeric(df[c])
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Reverse to Chronological order (Old -> New)
        return df.iloc[::-1].reset_index(drop=True)
    except:
        return pd.DataFrame()

def process_data(df):
    if len(df) < 30: return None
    df = df.copy()
    
    # Indicators matches train.py
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
    
    df = df.fillna(0)
    
    stoch_k_col = next((c for c in df.columns if 'STOCHk' in c), None)
    supertd_col = next((c for c in df.columns if 'SUPERTd' in c), None)
    df['STOCH_K'] = df[stoch_k_col] if stoch_k_col else 50.0
    df['SUPERT_D'] = df[supertd_col] if supertd_col else 1.0

    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std().fillna(0)

    ema20 = df.get('EMA_20', df['close'])
    ema50 = df.get('EMA_50', df['close'])
    ema100 = df.get('EMA_100', df['close'])
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100
    
    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]
    
    return df, feature_cols

# --- API Route ---
@app.route('/predict', methods=['GET'])
def predict():
    sym = request.args.get('symbol', 'EURUSD')
    interval = request.args.get('interval', '1h')
    
    df = fetch_twelvedata(sym, interval)
    if df.empty:
        return jsonify({"error": "Failed to fetch data from TwelveData"}), 500
        
    processed_data = process_data(df)
    if not processed_data:
        return jsonify({"error": "Not enough data for indicators"}), 400
        
    df_proc, feats = processed_data
    
    # Prepare Input
    X = df_proc[feats].values
    X_scaled = SCALER.transform(X)
    
    # RF & XGB (Last row)
    last_row = X_scaled[-1].reshape(1, -1)
    rf_prob = RF.predict_proba(last_row)[:, 1][0]
    xgb_prob = XGB.predict_proba(last_row)[:, 1][0]
    
    meta_in = [rf_prob, xgb_prob]
    
    # LSTM
    if LSTM:
        X_seq = np.array([X_scaled[-TIME_STEPS:]])
        lstm_prob = LSTM.predict(X_seq, verbose=0)[0][0]
        meta_in.append(lstm_prob)
        
    # Final Meta Prediction
    final_prob = META.predict_proba(np.array([meta_in]))[:, 1][0]
    
    atr = df_proc['ATR_14'].iloc[-1]
    price = df_proc['close'].iloc[-1]
    
    signal = "NEUTRAL"
    if final_prob > 0.65: signal = "STRONG BUY"
    elif final_prob > 0.55: signal = "BUY"
    elif final_prob < 0.35: signal = "STRONG SELL"
    elif final_prob < 0.45: signal = "SELL"
    
    return jsonify({
        "symbol": sym,
        "price": price,
        "probability": round(final_prob, 4),
        "signal": signal,
        "sl": round(price - (1.5 * atr) if "BUY" in signal else price + (1.5 * atr), 5),
        "tp": round(price + (2.0 * atr) if "BUY" in signal else price - (2.0 * atr), 5),
        "details": {
            "rf": round(rf_prob, 2),
            "xgb": round(xgb_prob, 2),
            "lstm": round(lstm_prob, 2) if LSTM else 0
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
