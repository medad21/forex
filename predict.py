import os
import joblib
import requests
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf

# تنظیمات
TIME_STEPS = 10
MODEL_DIR = "models"
API_KEY = os.environ.get("TWELVEDATA_API_KEY", "f24a3dec20104e639d1995e42dc4673c")

# لیست نمادها باید با فرمت TwelveData باشد (اسلش دارد)
SYMBOLS_MAPPING = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",
    "BTCUSD": "BTC/USD"
}

# --- دریافت داده از Twelve Data ---
def get_live_data(symbol, interval="1h", outputsize=200):
    """دریافت داده مستقیم از Twelve Data برای همخوانی با داده‌های آموزشی"""
    td_symbol = SYMBOLS_MAPPING.get(symbol, symbol.replace("=X", ""))
    
    url = f"https://api.twelvedata.com/time_series?symbol={td_symbol}&interval={interval}&apikey={API_KEY}&outputsize={outputsize}"
    try:
        resp = requests.get(url, timeout=10).json()
        if 'values' not in resp:
            print(f"⚠️ API Error for {symbol}: {resp.get('message', 'Unknown error')}")
            return pd.DataFrame()

        df = pd.DataFrame(resp['values'])
        # تبدیل ستون‌ها به عددی
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            df[c] = pd.to_numeric(df[c])
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # نکته حیاتی: TwelveData داده‌ها را از جدید به قدیم می‌دهد.
        # ما برای اندیکاتورها و LSTM نیاز داریم از قدیم به جدید باشد.
        df = df.iloc[::-1].reset_index(drop=True)
        
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return pd.DataFrame()

# --- محاسبات (باید دقیقاً کپی logic فایل train باشد) ---
def calculate_features(df):
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
    
    # فقط آخرین ردیف را برنمی‌گردانیم، کل دیتافریم را برای LSTM نیاز داریم
    return df, feature_cols

def create_lstm_sequence(data_scaled, steps):
    return np.array([data_scaled[-steps:]])

# --- کلاس Ensemble ---
class EnsemblePredictor:
    def __init__(self):
        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
        self.xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
        self.meta = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))
        try:
            self.lstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
        except:
            self.lstm = None

    def predict(self, symbol):
        df_raw = get_live_data(symbol)
        if len(df_raw) < 50:
            print(f"⚠️ Not enough data for {symbol}")
            return None

        df_proc, feats = calculate_features(df_raw)
        
        # گرفتن داده‌های لازم برای مدل
        X = df_proc[feats].values
        X_scaled = self.scaler.transform(X)

        # آخرین نمونه برای RF/XGB
        last_sample = X_scaled[-1].reshape(1, -1)
        
        rf_p = self.rf.predict_proba(last_sample)[:, 1][0]
        xgb_p = self.xgb.predict_proba(last_sample)[:, 1][0]
        
        meta_input = [rf_p, xgb_p]

        if self.lstm:
            # توالی زمانی برای LSTM
            lstm_seq = create_lstm_sequence(X_scaled, TIME_STEPS)
            lstm_p = self.lstm.predict(lstm_seq, verbose=0)[0][0]
            meta_input.append(lstm_p)

        meta_input = np.array([meta_input])
        final_prob = self.meta.predict_proba(meta_input)[:, 1][0]
        
        return {
            "symbol": symbol,
            "price": df_raw['close'].iloc[-1],
            "prob": round(final_prob, 4),
            "signal": "BUY" if final_prob > 0.6 else ("SELL" if final_prob < 0.4 else "NEUTRAL")
        }

if __name__ == "__main__":
    predictor = EnsemblePredictor()
    print("\n🔍 Live Prediction from TwelveData:")
    
    for sym in SYMBOLS_MAPPING.keys():
        res = predictor.predict(sym)
        if res:
            print(f"🔹 {res['symbol']}: {res['signal']} (Score: {res['prob']}) | Price: {res['price']}")
