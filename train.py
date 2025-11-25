# train_and_update.py
import os
import pandas as pd
import numpy as np
import datetime
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from alpha_vantage.foreignexchange import ForeignExchange

# ----------------------------
# Config
# ----------------------------
SYMBOLS = ["EURUSD","GBPUSD","USDJPY","GC","BTC"]
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CSV_FILE = os.path.join(DATA_DIR, "merged_forex_data.csv")
MODEL_DIR = "models"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TIME_STEPS = 10
META_HOLDOUT_FRAC = 0.2
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"  # AlphaVantage key

# ----------------------------
# Helper: download AlphaVantage 1h data
# ----------------------------
def download_alpha_vantage(symbol, api_key=API_KEY_ALPHA, save_path=RAW_DIR):
    os.makedirs(save_path, exist_ok=True)
    outfile = os.path.join(save_path, f"{symbol}_av.csv")

    if not api_key:
        print(f"⚠️ No AlphaVantage API key for {symbol}. Skipping.")
        return None

    try:
        fx = ForeignExchange(key=api_key, output_format='pandas')
        df, _ = fx.get_currency_exchange_intraday(
            from_symbol=symbol[:3],
            to_symbol=symbol[3:],
            interval="60min"
        )

        df.reset_index(inplace=True)
        df.rename(columns={
            "date":"datetime",
            "1. open":"open",
            "2. high":"high",
            "3. low":"low",
            "4. close":"close"
        }, inplace=True)
        df["volume"] = 0
        df.to_csv(outfile, index=False)
        print(f"✅ AlphaVantage saved: {outfile}")
        return df
    except Exception as e:
        print(f"❌ AlphaVantage error {symbol}: {e}")
        return None

# ----------------------------
# Merge existing CSV and new data
# ----------------------------
def update_csv():
    existing = pd.read_csv(CSV_FILE, parse_dates=['datetime']) if os.path.exists(CSV_FILE) else pd.DataFrame()
    combined = existing.copy()
    
    for sym in SYMBOLS:
        df_new = download_alpha_vantage(sym)
        if df_new is not None and not df_new.empty:
            combined = pd.concat([combined, df_new])
    
    if not combined.empty:
        combined.drop_duplicates(['datetime'], inplace=True)
        combined.sort_values('datetime', inplace=True)
        combined.to_csv(CSV_FILE, index=False)
        print(f"✅ CSV updated: {CSV_FILE}")
    return combined

# ----------------------------
# Feature calculation
# ----------------------------
def calculate_features(df):
    df = df.copy()
    df['Returns'] = df['close'].pct_change()
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['datetime']).dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()
    df['EMA_20'] = df['close'].ewm(span=20).mean()
    df['EMA_50'] = df['close'].ewm(span=50).mean()
    df['EMA_100'] = df['close'].ewm(span=100).mean()
    df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
    df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_100']
    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.dropna().reset_index(drop=True)

# ----------------------------
# Create LSTM sequences
# ----------------------------
def create_sequences(X, steps=TIME_STEPS):
    return np.array([X[i:i+steps] for i in range(len(X)-steps)])

# ----------------------------
# Train models
# ----------------------------
def train_models(df):
    feature_cols = ['Returns','Volatility','Hour','DayOfWeek','HV_20','EMA_Diff_Fast','EMA_Diff_Slow']
    X = df[feature_cols].values
    y = df['Target'].values

    from sklearn.model_selection import train_test_split
    X_train, X_meta, y_train, y_meta = train_test_split(
        X, y, test_size=META_HOLDOUT_FRAC, random_state=42, shuffle=True, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_meta_scaled = scaler.transform(X_meta)
    joblib.dump(scaler, os.path.join(MODEL_DIR,"scaler.pkl"))

    # RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR,"rf_model.pkl"))

    # XGB
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric='logloss', use_label_encoder=False)
    xgb.fit(X_train_scaled, y_train)
    joblib.dump(xgb, os.path.join(MODEL_DIR,"xgb_model.pkl"))

    # LR
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, os.path.join(MODEL_DIR,"lr_model.pkl"))

    # LSTM
    if len(X_train_scaled) > TIME_STEPS:
        X_lstm_train = create_sequences(X_train_scaled)
        y_lstm_train = y_train[TIME_STEPS:]
        lstm = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TIME_STEPS,X.shape[1])),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=64, verbose=1)
        lstm.save(os.path.join(MODEL_DIR,"lstm_model.h5"))

        # Meta-model
        rf_probs = rf.predict_proba(X_meta_scaled)[:,1]
        xgb_probs = xgb.predict_proba(X_meta_scaled)[:,1]
        X_lstm_meta = create_sequences(X_meta_scaled)
        lstm_probs = lstm.predict(X_lstm_meta).reshape(-1)
        rf_aligned = rf_probs[TIME_STEPS:][:len(lstm_probs)]
        xgb_aligned = xgb_probs[TIME_STEPS:][:len(lstm_probs)]
        X_meta_model = np.column_stack([rf_aligned, xgb_aligned, lstm_probs])
        y_meta_aligned = y_meta[TIME_STEPS:][:len(lstm_probs)]
        meta_model = LogisticRegression()
        meta_model.fit(X_meta_model, y_meta_aligned)
        joblib.dump(meta_model, os.path.join(MODEL_DIR,"meta_model.pkl"))

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    df = update_csv()
    if not df.empty:
        df_feat = calculate_features(df)
        train_models(df_feat)
        print("✅ All models trained and saved")
