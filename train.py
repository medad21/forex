import os
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from tensorflow import keras
import requests
from utils.dukascopy import download_duka
from utils.alpha_vantage import download_av


# -----------------------------
# پیکربندی مسیرها
# -----------------------------
RAW_DIR = "data/raw"
MERGED_DIR = "data/merged"
MODELS_DIR = "models"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

symbols = ["EURUSD", "GBPUSD", "USDJPY", "GC", "BTC"]


# -----------------------------
# ادغام دیتا از منابع مختلف
# -----------------------------
def merge_data(symbol):
    files = []
    duka_path = f"{RAW_DIR}/{symbol}_duka.csv"
    av_path = f"{RAW_DIR}/{symbol}_av.csv"

    if os.path.exists(duka_path):
        files.append(pd.read_csv(duka_path))
    if os.path.exists(av_path):
        files.append(pd.read_csv(av_path))

    if not files:
        return None

    df = pd.concat(files).drop_duplicates(subset="time").sort_values("time")
    df.to_csv(f"{MERGED_DIR}/{symbol}.csv", index=False)

    return df


# -----------------------------
# ویژگی‌سازی (۱۳ ویژگی)
# -----------------------------
def add_features(df):
    df["returns"] = df["close"].pct_change()
    df["ma7"] = df["close"].rolling(7).mean()
    df["ma21"] = df["close"].rolling(21).mean()
    df["vol"] = df["close"].rolling(7).std()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"] = compute_macd(df["close"])
    df["high_low_spread"] = df["high"] - df["low"]
    df["open_close_spread"] = df["close"] - df["open"]
    df["target"] = df["close"].shift(-1)

    df = df.dropna()
    return df


# -----------------------------
# اندیکاتورهای مورد نیاز
# -----------------------------
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def compute_macd(series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal


# -----------------------------
# آموزش مدل‌ها
# -----------------------------
def train_models(df, symbol):
    X = df.drop(columns=["target"])
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pd.to_pickle(scaler, f"{MODELS_DIR}/{symbol}_scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    # RandomForest
    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(X_train, y_train)
    pd.to_pickle(rf, f"{MODELS_DIR}/{symbol}_rf.pkl")

    # XGBoost
    model_xgb = xgb.XGBRegressor(n_estimators=300)
    model_xgb.fit(X_train, y_train)
    model_xgb.save_model(f"{MODELS_DIR}/{symbol}_xgb.json")

    # LSTM
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    lstm = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(1, X.shape[1])),
        keras.layers.LSTM(32),
        keras.layers.Dense(1)
    ])
    lstm.compile(optimizer="adam", loss="mse")
    lstm.fit(X_train_lstm, y_train, epochs=12, batch_size=32, verbose=1)
    lstm.save(f"{MODELS_DIR}/{symbol}_lstm.h5")


# -----------------------------
# اجرای کامل: آپدیت → ادغام → فیچر → آموزش
# -----------------------------
def update_and_train(include_training=True):
    print("🚀 شروع آپدیت دیتا...")

    for sym in symbols:
        print(f"\n=== {sym} ===")

        download_duka(sym)
        download_av(sym)

        df = merge_data(sym)
        if df is None or len(df) < 50:
            print(f"❌ دیتای کافی برای {sym} یافت نشد.")
            continue

        df = add_features(df)

        if include_training:
            print(f"🎯 آموزش مدل‌ها برای {sym}...")
            train_models(df, sym)

    print("\n🎉 عملیات تکمیل شد!")


if __name__ == "__main__":
    update_and_train(include_training=True)
