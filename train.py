# train.py
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -------------------------
# تنظیمات
# -------------------------
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "BTC-USD"]
INTERVAL = "1h"

# پارامترها
TOTAL_DAYS = 650          # مجموع روزهایی که می‌خواهیم دانلود کنیم (کمتر از 730 برای جلوگیری از ارور YF)
BATCH_DAYS = 200          # اندازه هر بچ دانلود
TIME_STEPS = 10           # برای LSTM؛ در صورت کمبود دیتا این را کاهش بده
META_HOLDOUT_FRAC = 0.20  # نسبت نگهدارنده برای meta-model

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# دانلود ایمن داده (batch)
# -------------------------
def download_in_batches(symbol, total_days=TOTAL_DAYS, batch_days=BATCH_DAYS, interval=INTERVAL):
    end = datetime.datetime.now()
    parts = []
    while total_days > 0:
        start = end - datetime.timedelta(days=batch_days)
        print(f"📥 Downloading {symbol} → {start.date()} .. {end.date()}")
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        if not df.empty:
            parts.append(df)
        else:
            print("⚠️ empty batch (Yahoo grumpy) — skipping this interval")
        end = start
        total_days -= batch_days
    if not parts:
        return pd.DataFrame()
    df_full = pd.concat(parts).sort_index().drop_duplicates()
    return df_full

# -------------------------
# محاسبهٔ اندیکاتورها / فیچرها
# -------------------------
def calculate_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df['Returns'] = df['close'].pct_change()

    # اندیکاتورهای پایه
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

    # ستون‌های تمیز شده
    df['RSI_14'] = df.get("RSI_14", df.get("ta_rsi_14", 0))
    df['RSI_6']  = df.get("RSI_6", df.get("ta_rsi_6", 0))
    df['ADX_14'] = df.get("ADX_14", df.get("ta_adx_14", 0))
    df['STOCH_K'] = df.get("STOCHk_14_3_3", 0)
    df['SUPERT_D'] = df.get("SUPERTd_10_3.0", 0)
    df['MFI_14'] = df.get("MFI_14", 0)

    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()

    ema20 = df.get("EMA_20", df.get("ta_ema_20", df['close']))
    ema50 = df.get("EMA_50", df.get("ta_ema_50", df['close']))
    ema100 = df.get("EMA_100", df.get("ta_ema_100", df['close']))
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    return df.dropna().reset_index(drop=True)

# -------------------------
# ساخت تارگت (مثال: تحقق TP براساس ATR)
# -------------------------
def create_target(df, future_period=5, atr_multiplier=1.5):
    closes = df['close'].values
    highs = df['high'].values
    atrs = df.get('ATRr_14', df.get('ATR_14', pd.Series([0]*len(df)))).values

    targets = []
    for i in range(len(closes) - future_period):
        current_close = closes[i]
        atr = max(atrs[i], 0.001)
        tp = current_close + (atr * atr_multiplier)
        future_highs = highs[i+1 : i+future_period+1]
        targets.append(1 if np.max(future_highs) >= tp else 0)
    df = df.iloc[:len(targets)].copy()
    df['Target'] = targets
    return df

# -------------------------
# تابع کمکی برای ساخت sequence برای LSTM
# -------------------------
def create_sequences(array_2d, steps):
    Xs = []
    for i in range(len(array_2d) - steps):
        Xs.append(array_2d[i: i + steps])
    return np.array(Xs)

# -------------------------
# اجرای اصلی
# -------------------------
if __name__ == "__main__":
    print("⏳ Downloading symbols...")
    all_dfs = []
    for symbol in SYMBOLS:
        df_sym = download_in_batches(symbol)
        if df_sym.empty:
            print(f"❌ {symbol} produced no data — skipping")
            continue
        df_sym = calculate_indicators(df_sym)
        df_sym = create_target(df_sym)
        all_dfs.append(df_sym)

    if not all_dfs:
        raise SystemExit("No data downloaded for any symbol — aborting")

    df = pd.concat(all_dfs, ignore_index=True).dropna().reset_index(drop=True)
    print(f"✅ Combined dataframe shape: {df.shape}")

    # فیچرها (می‌توانی این لیست را گسترش دهی)
    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]

    X_all = df[feature_cols].values
    y_all = df['Target'].values

    # تقسیم به train_full و meta_holdout (برای آموزش meta-model)
    X_train_full, X_meta, y_train_full, y_meta = train_test_split(
        X_all, y_all, test_size=META_HOLDOUT_FRAC, random_state=42, shuffle=True, stratify=y_all
    )

    # مقیاس‌بندی
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_meta_scaled = scaler.transform(X_meta)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # === مدل RF و XGB روی train_full ===
    print("⚙️ Training RandomForest and XGBoost...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_full_scaled, y_train_full)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric='logloss', use_label_encoder=False)
    xgb.fit(X_train_full_scaled, y_train_full)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    # پیش‌بینی-prob روی نگهدارنده
    rf_meta_probs = rf.predict_proba(X_meta_scaled)[:,1]
    xgb_meta_probs = xgb.predict_proba(X_meta_scaled)[:,1]

    # === آموزش LSTM ===
    print("🧠 Preparing and training LSTM...")
    if len(X_train_full_scaled) <= TIME_STEPS:
        raise SystemExit(f"Not enough training rows for TIME_STEPS={TIME_STEPS}. Reduce TIME_STEPS or collect more data.")

    X_lstm_train = create_sequences(X_train_full_scaled, TIME_STEPS)
    y_lstm_train = y_train_full[TIME_STEPS:]

    lstm_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TIME_STEPS, X_all.shape[1])),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=64, verbose=1)
    lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

    # آماده‌سازی نگهدارنده برای LSTM (برای meta)
    if len(X_meta_scaled) <= TIME_STEPS:
        raise SystemExit("Meta holdout too small to create LSTM sequences — increase holdout size or reduce TIME_STEPS.")

    X_lstm_meta = create_sequences(X_meta_scaled, TIME_STEPS)
    y_lstm_meta = y_meta[TIME_STEPS:]
    lstm_meta_probs = lstm_model.predict(X_lstm_meta).reshape(-1)

    # همترازی RF/XGB با offset مربوط به TIME_STEPS
    rf_meta_aligned = rf_meta_probs[TIME_STEPS:][:len(lstm_meta_probs)]
    xgb_meta_aligned = xgb_meta_probs[TIME_STEPS:][:len(lstm_meta_probs)]
    y_meta_aligned = y_meta[TIME_STEPS:][:len(lstm_meta_probs)]

    # ساخت فیچرهای متا و آموزش LogisticRegression
    X_meta_for_meta = np.column_stack([rf_meta_aligned, xgb_meta_aligned, lstm_meta_probs])
    y_meta_for_meta = y_meta_aligned

    print("🧩 Training Meta-Model (LogisticRegression)...")
    meta_model = LogisticRegression()
    meta_model.fit(X_meta_for_meta, y_meta_for_meta)
    joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.pkl"))

    print("\n✅ Training complete. Models saved in ./models:")
    print(os.listdir(MODEL_DIR))
