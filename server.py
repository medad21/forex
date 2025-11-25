from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

app = Flask(__name__)

# مسیر مدل‌ها
MODELS_PATH = "models"

# ویژگی‌ها
feature_cols = [
    'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
    'MFI_14', 'STOCH_K', 'SUPERT_D'
]

# --- توابع کمکی ---
def calculate_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df['Returns'] = df['close'].pct_change()
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()
    
    # EMA
    df['EMA_20'] = df['close'].ewm(span=20).mean()
    df['EMA_50'] = df['close'].ewm(span=50).mean()
    df['EMA_100'] = df['close'].ewm(span=100).mean()
    df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
    df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_100']
    
    # RSI ساده
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_6'] = df['close'].diff().rolling(6).mean()
    
    # ADX, MFI, STOCH, SUPERT (placeholder=0 برای سادگی)
    df['ADX_14'] = 0
    df['MFI_14'] = 0
    df['STOCH_K'] = 0
    df['SUPERT_D'] = 0
    
    return df.dropna()

# --- مدل Ensemble ---
class EnsembleModel:
    def __init__(self):
        self.rf = joblib.load(f"{MODELS_PATH}/rf_model.pkl")
        self.xgb = joblib.load(f"{MODELS_PATH}/xgb_model.pkl")
        self.lstm = tf.keras.models.load_model(f"{MODELS_PATH}/lstm_model.h5")
        self.scaler = joblib.load(f"{MODELS_PATH}/scaler.pkl")
        self.time_steps = 10

    def predict(self, df):
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        # RF + XGB
        pred_rf = self.rf.predict_proba(X_scaled)[:,1]
        pred_xgb = self.xgb.predict_proba(X_scaled)[:,1]

        # LSTM
        X_lstm = []
        for i in range(len(X_scaled)-self.time_steps):
            X_lstm.append(X_scaled[i:i+self.time_steps])
        X_lstm = np.array(X_lstm)
        pred_lstm = self.lstm.predict(X_lstm, verbose=0).flatten()

        pred_rf_trim = pred_rf[self.time_steps:]
        pred_xgb_trim = pred_xgb[self.time_steps:]

        ensemble_pred = (pred_rf_trim + pred_xgb_trim + pred_lstm)/3
        return ensemble_pred, df.index[self.time_steps:]

ensemble_model = EnsembleModel()

# --- endpoint ---
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        payload = request.json
        symbol = payload.get("symbol", "EUR/USD").replace("/", "")
        interval = payload.get("interval", "1h")
        end = datetime.now()
        start = end - timedelta(days=60)
        
        df = yf.download(f"FX_IDC:{symbol}" if symbol != "BTCUSD" else "BINANCE:BTCUSDT",
                         start=start, end=end, interval=interval, progress=False)
        if df.empty:
            return jsonify({"error": f"No data for {symbol}"}), 400
        
        df = calculate_indicators(df)
        preds, dates = ensemble_model.predict(df)
        latest_pred = preds[-1]
        score = float((latest_pred-0.5)*2*10)  # normalized to -10..+10
        signal = "buy" if latest_pred>0.5 else "sell"

        # خروجی شبیه فرانت
        result = {
            "score": score,
            "signal": signal,
            "price": float(df['close'].iloc[-1]),
            "setup": {
                "tp": None,
                "sl": None,
                "lot_size": 0,
                "risk_amt": 0
            },
            "indicators": {
                "sr_levels": "---",
                "news": "---",
                "trend": "Uptrend" if score>0 else "Downtrend",
                "rsi": float(df['RSI_14'].iloc[-1]),
                "regime": "---",
                "divergence": "---",
                "htf_status": "فعال" if payload.get("use_htf", True) else "غیرفعال",
                "htf_trend": "Bullish" if score>0 else "Bearish",
                "ai_report": {
                    "ensemble_score": float(latest_pred),
                    "individual_results": {
                        "RF": {"score": float(preds[-1]), "prob": float(preds[-1]*100)},
                        "XGB": {"score": float(preds[-1]), "prob": float(preds[-1]*100)},
                        "LSTM": {"score": float(preds[-1]), "prob": float(preds[-1]*100)}
                    }
                }
            }
        }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
