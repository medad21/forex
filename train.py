# train.py
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -------------------------
# تنظیمات
# -------------------------
CSV_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
TIME_STEPS = 10
META_HOLDOUT_FRAC = 0.2
FEATURE_COLS = [
    'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
    'MFI_14', 'STOCH_K', 'SUPERT_D'
]

# -------------------------
# محاسبه اندیکاتورها و Target
# -------------------------
def calculate_indicators_and_target(df):
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

    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.dropna().reset_index(drop=True)

# -------------------------
# ایجاد sequences برای LSTM
# -------------------------
def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

# -------------------------
# اجرای اصلی
# -------------------------
if __name__ == "__main__":
    all_dfs = []
    for file in os.listdir(CSV_DIR):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(CSV_DIR, file), parse_dates=['datetime'])
            df = calculate_indicators_and_target(df)
            all_dfs.append(df)

    if not all_dfs:
        raise SystemExit("❌ No data found in CSVs.")

    df_all = pd.concat(all_dfs, ignore_index=True)

    X = df_all[FEATURE_COLS].values
    y = df_all['Target'].values

    # تقسیم train/holdout برای meta model
    X_train_full, X_meta, y_train_full, y_meta = train_test_split(
        X, y, test_size=META_HOLDOUT_FRAC, random_state=42, shuffle=True, stratify=y
    )

    # StandardScaler
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_meta_scaled = scaler.transform(X_meta)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_full_scaled, y_train_full)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

    # XGBoost
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric='logloss', use_label_encoder=False)
    xgb.fit(X_train_full_scaled, y_train_full)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    # پیش‌بینی روی meta holdout
    rf_probs = rf.predict_proba(X_meta_scaled)[:,1]
    xgb_probs = xgb.predict_proba(X_meta_scaled)[:,1]

    # LSTM
    if len(X_train_full_scaled) <= TIME_STEPS:
        raise SystemExit(f"Not enough rows for TIME_STEPS={TIME_STEPS}")

    X_lstm_train = create_sequences(X_train_full_scaled, TIME_STEPS)
    y_lstm_train = y_train_full[TIME_STEPS:]

    lstm_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TIME_STEPS,X.shape[1])),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=64, verbose=1)
    lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

    # LSTM برای meta
    X_lstm_meta = create_sequences(X_meta_scaled, TIME_STEPS)
    y_lstm_meta = y_meta[TIME_STEPS:]
    lstm_probs = lstm_model.predict(X_lstm_meta).reshape(-1)

    # ساخت meta features
    rf_meta_aligned = rf_probs[TIME_STEPS:][:len(lstm_probs)]
    xgb_meta_aligned = xgb_probs[TIME_STEPS:][:len(lstm_probs)]
    y_meta_aligned = y_meta[TIME_STEPS:][:len(lstm_probs)]

    X_meta_for_meta = np.column_stack([rf_meta_aligned, xgb_meta_aligned, lstm_probs])
    y_meta_for_meta = y_meta_aligned

    meta_model = LogisticRegression()
    meta_model.fit(X_meta_for_meta, y_meta_for_meta)
    joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.pkl"))

    print("✅ Training complete. All models saved in ./models/")
