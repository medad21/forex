import os
import requests
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import RobustScaler

# ⚠️ کلید خود را اینجا وارد کنید یا در Environment Variable تنظیم کنید
API_KEY = os.environ.get("TWELVEDATA_API_KEY", "f24a3dec20104e639d1995e42dc4673c")
MODEL_DIR = "models"
TIME_STEPS = 10

# نگاشت نمادها برای TwelveData
SYMBOLS_MAP = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY", 
    "XAUUSD": "XAU/USD",
    "BTCUSD": "BTC/USD"
}

class TradingAI:
    def __init__(self):
        self.scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        self.rf = joblib.load(f"{MODEL_DIR}/rf_model.pkl")
        self.xgb = joblib.load(f"{MODEL_DIR}/xgb_model.pkl")
        self.meta = joblib.load(f"{MODEL_DIR}/meta_model.pkl")
        self.lstm = tf.keras.models.load_model(f"{MODEL_DIR}/lstm_model.h5")

    def get_data(self, symbol, interval="1h"):
        """دریافت داده از TwelveData و معکوس کردن برای محاسبات"""
        td_sym = SYMBOLS_MAP.get(symbol, symbol)
        url = f"https://api.twelvedata.com/time_series?symbol={td_sym}&interval={interval}&apikey={API_KEY}&outputsize=100"
        
        try:
            r = requests.get(url).json()
            if 'values' not in r: return None
            df = pd.DataFrame(r['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            cols = ['open', 'high', 'low', 'close', 'volume']
            for c in cols: df[c] = pd.to_numeric(df[c])
            
            # معکوس کردن: تبدیل از (جدید->قدیم) به (قدیم->جدید)
            return df.iloc[::-1].reset_index(drop=True)
        except Exception as e:
            print(e)
            return None

    def prepare_features(self, df):
        """دقیقاً مشابه train.py"""
        if len(df) < 30: return None, None
        df = df.copy()

        # 1. Stationary Features
        df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))
        
        df['RSI_Norm'] = df.ta.rsi(length=14) / 100.0
        df['MFI_Norm'] = df.ta.mfi(length=14) / 100.0
        
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df['Dist_EMA20'] = (df['close'] - df['EMA_20']) / df['EMA_20']
        df['Dist_EMA50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
        
        df.ta.atr(length=14, append=True) # برای محاسبه حد سود/ضرر
        
        roll_std = df['Log_Ret'].rolling(window=20).std()
        roll_mean = df['Log_Ret'].rolling(window=20).mean()
        df['Vol_ZScore'] = (df['Log_Ret'] - roll_mean) / (roll_std + 1e-8)
        
        df = df.dropna()
        if df.empty: return None, None

        feat_cols = ['Log_Ret', 'Dist_EMA20', 'Dist_EMA50', 'RSI_Norm', 'MFI_Norm', 'Vol_ZScore']
        context_cols = ['RSI_Norm', 'Vol_ZScore']
        
        return df, (feat_cols, context_cols)

    def predict(self, symbol):
        df, cols = self.prepare_features(self.get_data(symbol))
        if df is None: return {"error": "No Data"}
        
        feats, ctxs = cols
        
        # آخرین داده برای پیش‌بینی
        X_raw = df[feats].values
        X_scaled = self.scaler.transform(X_raw)
        
        # 1. Base Predictions
        last_row = X_scaled[-1].reshape(1, -1)
        rf_p = self.rf.predict_proba(last_row)[:, 1][0]
        xgb_p = self.xgb.predict_proba(last_row)[:, 1][0]
        
        # 2. LSTM Prediction
        lstm_seq = np.array([X_scaled[-TIME_STEPS:]])
        lstm_p = self.lstm.predict(lstm_seq, verbose=0)[0][0]
        
        # 3. Context Extraction
        # RSI و Volatility آخرین کندل
        last_ctx = df[ctxs].iloc[-1].values 
        
        # 4. Meta Prediction [RF, XGB, LSTM, RSI, Vol]
        meta_in = np.column_stack([
            [rf_p], [xgb_p], [lstm_p], [last_ctx]
        ])
        
        final_prob = self.meta.predict_proba(meta_in)[:, 1][0]
        
        # مدیریت سرمایه (ATR Based)
        atr = df['ATRr_14'].iloc[-1]
        price = df['close'].iloc[-1]
        
        signal = "NEUTRAL"
        if final_prob > 0.60: signal = "BUY"
        elif final_prob < 0.40: signal = "SELL" # اگر تارگت فروش هم آموزش داده بودید
        
        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": round(final_prob * 100, 1),
            "price": price,
            "sl": round(price - (1.5 * atr), 4),
            "tp": round(price + (2.0 * atr), 4)
        }

if __name__ == "__main__":
    bot = TradingAI()
    print(bot.predict("EURUSD"))
