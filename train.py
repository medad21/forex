# real_time_predict.py
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf

MODEL_DIR = "models"
CSV_FILE = "data.csv"  # CSV آپدیت شده با داده‌های جدید
TIME_STEPS = 10

# -------------------------
# بارگذاری مدل‌ها و scaler
# -------------------------
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
meta_model = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))

# -------------------------
# محاسبه فیچرها
# -------------------------
def calculate_features(df):
    df = df.copy()
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.supertrend(length=10, multiplier=3.0, append=True)

    df['RSI_14'] = df.get('RSI_14', df.get('ta_rsi_14',0))
    df['RSI_6']  = df.get('RSI_6', df.get('ta_rsi_6',0))
    df['ADX_14'] = df.get('ADX_14', df.get('ta_adx_14',0))
    df['STOCH_K'] = df.get('STOCHk_14_3_3',0)
    df['SUPERT_D'] = df.get('SUPERTd_10_3.0',0)
    df['MFI_14'] = df.get('MFI_14',0)
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()

    ema20 = df.get('EMA_20', df['close'])
    ema50 = df.get('EMA_50', df['close'])
    ema100 = df.get('EMA_100', df['close'])
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    df = df.dropna().reset_index(drop=True)
    return df

# -------------------------
# ایجاد sequence برای LSTM
# -------------------------
def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

# -------------------------
# اجرای پیش‌بینی
# -------------------------
if __name__ == "__main__":
    df = pd.read_csv(CSV_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = calculate_features(df)

    feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
                    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
                    'MFI_14', 'STOCH_K', 'SUPERT_D']

    X = df[feature_cols].values
    X_scaled = scaler.transform(X)

    # RF & XGB
    rf_probs = rf_model.predict_proba(X_scaled)[:,1]
    xgb_probs = xgb_model.predict_proba(X_scaled)[:,1]

    # LSTM
    if len(X_scaled) > TIME_STEPS:
        X_lstm = create_sequences(X_scaled, TIME_STEPS)
        lstm_probs = lstm_model.predict(X_lstm).reshape(-1)
        rf_aligned = rf_probs[TIME_STEPS:][:len(lstm_probs)]
        xgb_aligned = xgb_probs[TIME_STEPS:][:len(lstm_probs)]
        X_meta = np.column_stack([rf_aligned, xgb_aligned, lstm_probs])
        meta_probs = meta_model.predict_proba(X_meta)[:,1]

        # آخرین پیش‌بینی
        print("📊 آخرین پیش‌بینی Meta-Model:")
        print(f"احتمال رشد قیمت: {meta_probs[-1]:.4f}")
        print("سیگنال پیشنهادی:", "BUY" if meta_probs[-1]>0.5 else "SELL")
    else:
        print("⚠️ داده کافی برای LSTM / Meta prediction موجود نیست. فقط RF/XGB قابل پیش‌بینی است.")
        print(f"RF احتمال رشد: {rf_probs[-1]:.4f}")
        print(f"XGB احتمال رشد: {xgb_probs[-1]:.4f}")
