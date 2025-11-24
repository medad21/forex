import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import datetime

SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "BTC-USD"]
INTERVAL = "1h"

# ======================================================================
# دانلود داده به صورت Batch (مطمئن و بدون خطای Yahoo)
# ======================================================================
def download_in_batches(symbol, total_days=650, batch_days=200, interval="1h"):
    end = datetime.datetime.now()
    all_parts = []

    while total_days > 0:
        start = end - datetime.timedelta(days=batch_days)
        print(f"📥 Batch: {symbol}  →  {start.date()} تا {end.date()}")

        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

        if not df.empty:
            all_parts.append(df)
        else:
            print("⚠️ Batch خالی بود، ادامه می‌دهیم...")

        end = start
        total_days -= batch_days

    if not all_parts:
        return pd.DataFrame()

    df_full = pd.concat(all_parts).sort_index().drop_duplicates()
    return df_full


# ======================================================================
# اندیکاتورها
# ======================================================================
def calculate_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                             'Close': 'close', 'Volume': 'volume'})

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

    df['RSI_14'] = df.get("RSI_14", df.get("ta_rsi_14", 0))
    df['RSI_6'] = df.get("RSI_6", df.get("ta_rsi_6", 0))
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


# ======================================================================
# تارگت
# ======================================================================
def create_target(df):
    future_period = 5
    atr_multiplier = 1.5

    targets = []
    closes = df['close'].values
    highs = df['high'].values
    atrs = df.get('ATRr_14', df.get('ATR_14', 0.001)).values

    for i in range(len(closes) - future_period):
        current_close = closes[i]
        atr = max(atrs[i], 0.001)

        take_profit = current_close + (atr * atr_multiplier)
        future_highs = highs[i + 1:i + future_period + 1]

        targets.append(1 if np.max(future_highs) >= take_profit else 0)

    df = df.iloc[:len(targets)]
    df['Target'] = targets
    return df


# ======================================================================
# اجرای اصلی
# ======================================================================
if __name__ == "__main__":

    all_data = []

    for symbol in SYMBOLS:
        print(f"\n⏳ Downloading {symbol} ...")
        # 🛑 استفاده از تابع جدید Batch Download
        df_symbol = download_in_batches(symbol, total_days=650, batch_days=200, interval="1h")

        if df_symbol.empty:
            print(f"❌ {symbol} هیچ داده‌ای نداد!")
            continue

        df_symbol = calculate_indicators(df_symbol)
        df_symbol = create_target(df_symbol)
        all_data.append(df_symbol)

    if not all_data:
        print("❌ هیچ دیتایی دانلود نشد!")
        exit()

    print("⚙️ Combining data...")
    df = pd.concat(all_data, ignore_index=True).dropna().reset_index(drop=True)

    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]

    X = df[feature_cols].values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y
    )

    print("⚖️ Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if not os.path.exists("models"):
        os.makedirs("models")

    print("🧠 Training Models...")

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, 'models/rf_model.pkl')

    lr = LogisticRegression(C=1.0, random_state=42)
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, 'models/lr_model.pkl')

    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    joblib.dump(xgb, 'models/xgb_model.pkl')

    # LSTM
    time_steps = 10

    def create_lstm_data(data, steps):
        X = []
        for i in range(len(data) - steps):
            X.append(data[i:i + steps])
        return np.array(X)

    X_lstm = create_lstm_data(scaler.transform(X), time_steps)
    y_lstm = y[time_steps:]
    split = int(len(X_lstm) * 0.8)

    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, len(feature_cols))),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_lstm[:split], y_lstm[:split], epochs=3, batch_size=32, verbose=0)
    lstm.save('models/lstm_model.h5')

    joblib.dump(scaler, 'models/scaler.pkl')

    print("\n✅ Done! All models trained successfully.")
