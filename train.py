# train.py
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta
import requests

# ----------------------------
# تنظیمات مسیر و فایل‌ها
# ----------------------------
DATA_DIR = "data/raw"
DATA_FILE = os.path.join(DATA_DIR, "data.csv")
MODEL_DIR = "models"
TIME_STEPS = 10
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# API کلیدها
# ----------------------------
ALPHA_VANTAGE_KEY = "W1L3K1JN4F77T9KL"  # اگر دارید
DUKASCOPY_KEY = ""  # اگر لازم بود

# ----------------------------
# دانلود داده‌ها
# ----------------------------
def download_alpha_vantage(symbol):
    """داده 1H از Alpha Vantage"""
    try:
        from alpha_vantage.foreignexchange import ForeignExchange
        fx = ForeignExchange(key=ALPHA_VANTAGE_KEY, output_format='pandas')
        df, _ = fx.get_currency_exchange_intraday(
            from_symbol=symbol[:3],
            to_symbol=symbol[3:], interval="60min"
        )
        df.reset_index(inplace=True)
        df.rename(columns={
            "date": "datetime",
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close"
        }, inplace=True)
        df["volume"] = 0
        return df
    except Exception as e:
        print(f"❌ AlphaVantage error {symbol}: {e}")
        return pd.DataFrame()

def download_dukascopy(symbol, start=None, end=None):
    """داده تاریخی Dukascopy"""
    try:
        import dukascopy
        jc = dukascopy.JCDownloader()
        df = jc.download(symbol, start=start, end=end)
        df.reset_index(inplace=True)
        df.rename(columns={
            "datetime": "datetime",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        }, inplace=True)
        return df
    except Exception as e:
        print(f"❌ Dukascopy error {symbol}: {e}")
        return pd.DataFrame()

# ----------------------------
# مدیریت داده: merge داده‌های جدید با قدیمی
# ----------------------------
def update_data(symbol):
    if os.path.exists(DATA_FILE):
        old_df = pd.read_csv(DATA_FILE)
        old_df['datetime'] = pd.to_datetime(old_df['datetime'])
    else:
        old_df = pd.DataFrame()

    # دانلود جدید
    df_new_av = download_alpha_vantage(symbol)
    df_new_du = download_dukascopy(symbol)
    
    df_new = pd.concat([df_new_av, df_new_du], ignore_index=True)
    df_new.drop_duplicates(subset=["datetime"], inplace=True)
    df_new.sort_values("datetime", inplace=True)
    
    if not old_df.empty:
        df_final = pd.concat([old_df, df_new], ignore_index=True)
        df_final.drop_duplicates(subset=["datetime"], inplace=True)
    else:
        df_final = df_new

    df_final.to_csv(DATA_FILE, index=False)
    print(f"✅ Updated data saved to {DATA_FILE}")
    return df_final

# ----------------------------
# محاسبه فیچرها
# ----------------------------
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

# ----------------------------
# ایجاد sequence برای LSTM
# ----------------------------
def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

# ----------------------------
# آموزش مدل‌ها
# ----------------------------
def train_models(df):
    feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
                    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
                    'MFI_14', 'STOCH_K', 'SUPERT_D']

    X = df[feature_cols].values
    y = (df['Returns'] > 0).astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # RF
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf_model.fit(X_scaled, y)
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))

    # XGB
    xgb_model = XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric="logloss")
    xgb_model.fit(X_scaled, y)
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    # LR
    lr_model = LogisticRegression()
    lr_model.fit(X_scaled, y)
    joblib.dump(lr_model, os.path.join(MODEL_DIR, "lr_model.pkl"))

    # LSTM
    X_lstm = create_sequences(X_scaled)
    y_lstm = y[TIME_STEPS:]
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.Input((TIME_STEPS, X_scaled.shape[1])),
        tf.keras.layers.LSTM(64, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam')
    lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=32, verbose=1)
    lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

    # Meta-Model
    rf_probs = rf_model.predict_proba(X_scaled)[:,1]
    xgb_probs = xgb_model.predict_proba(X_scaled)[:,1]
    lstm_probs = np.zeros_like(rf_probs)
    if len(X_scaled) > TIME_STEPS:
        X_lstm_seq = create_sequences(X_scaled)
        lstm_probs[TIME_STEPS:] = lstm_model.predict(X_lstm_seq).reshape(-1)
    X_meta = np.column_stack([rf_probs, xgb_probs, lstm_probs])
    meta_model = LogisticRegression()
    meta_model.fit(X_meta[TIME_STEPS:], y[TIME_STEPS:])
    joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.pkl"))

    print("✅ All models trained and saved.")

# ----------------------------
# پیش‌بینی لحظه‌ای
# ----------------------------
def predict_latest():
    if not os.path.exists(DATA_FILE):
        print("❌ No data available.")
        return

    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = calculate_features(df)

    feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
                    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
                    'MFI_14', 'STOCH_K', 'SUPERT_D']

    X = df[feature_cols].values
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    X_scaled = scaler.transform(X)

    rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    lstm_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
    meta_model = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))

    rf_probs = rf_model.predict_proba(X_scaled)[:,1]
    xgb_probs = xgb_model.predict_proba(X_scaled)[:,1]

    if len(X_scaled) > TIME_STEPS:
        X_lstm_seq = create_sequences(X_scaled)
        lstm_probs = lstm_model.predict(X_lstm_seq).reshape(-1)
        rf_aligned = rf_probs[TIME_STEPS:][:len(lstm_probs)]
        xgb_aligned = xgb_probs[TIME_STEPS:][:len(lstm_probs)]
        X_meta = np.column_stack([rf_aligned, xgb_aligned, lstm_probs])
        meta_probs = meta_model.predict_proba(X_meta)[:,1]

        print("📊 آخرین پیش‌بینی Meta-Model:")
        print(f"احتمال رشد قیمت: {meta_probs[-1]:.4f}")
        print("سیگنال پیشنهادی:", "BUY" if meta_probs[-1]>0.5 else "SELL")
    else:
        print("⚠️ داده کافی برای LSTM/Meta prediction موجود نیست.")
        print(f"RF احتمال رشد: {rf_probs[-1]:.4f}")
        print(f"XGB احتمال رشد: {xgb_probs[-1]:.4f}")

# ----------------------------
# دکمه آپدیت خودکار
# ----------------------------
def update_and_train(symbol):
    update_data(symbol)
    df = pd.read_csv(DATA_FILE)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = calculate_features(df)
    train_models(df)
    predict_latest()

# ----------------------------
# اجرای اصلی
# ----------------------------
if __name__ == "__main__":
    SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "GC", "BTC"]
    for sym in SYMBOLS:
        update_and_train(sym)
