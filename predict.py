import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import yfinance as yf
import datetime

# لیست نمادها
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "BTC-USD"]
INTERVAL = "1h"

# فیچرهای اصلی مدل
feature_cols = [
    'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
    'MFI_14', 'STOCH_K', 'SUPERT_D'
]

# sanity-check
def sanity_check_features(df):
    missing_cols = [col for col in feature_cols if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in feature_cols and col != 'Target']

    if missing_cols:
        raise ValueError(f"❌ خطا: ویژگی‌های زیر در دیتا وجود ندارند: {missing_cols}")
    if extra_cols:
        print(f"⚠️ هشدار: ویژگی‌های اضافی وجود دارند و نادیده گرفته می‌شوند: {extra_cols}")
    print("✅ Sanity check passed: همه ویژگی‌ها موجود هستند و درست‌اند.")

# محاسبه اندیکاتورها
def calculate_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df['Returns'] = df['close'].pct_change()

    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stoch(k=14,d=3,append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.supertrend(length=10, multiplier=3.0, append=True)

    df['RSI_14'] = df.get("RSI_14", df.get("ta_rsi_14",0))
    df['RSI_6'] = df.get("RSI_6", df.get("ta_rsi_6",0))
    df['ADX_14'] = df.get("ADX_14", df.get("ta_adx_14",0))
    df['STOCH_K'] = df.get("STOCHk_14_3_3",0)
    df['SUPERT_D'] = df.get("SUPERTd_10_3.0",0)
    df['MFI_14'] = df.get("MFI_14",0)

    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()

    ema20 = df.get("EMA_20", df.get("ta_ema_20", df['close']))
    ema50 = df.get("EMA_50", df.get("ta_ema_50", df['close']))
    ema100 = df.get("EMA_100", df.get("ta_ema_100", df['close']))
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    return df.dropna().reset_index()

# کلاس Ensemble
class EnsembleModel:
    def __init__(self, models_path='models', time_steps=10):
        self.rf = joblib.load(os.path.join(models_path, 'rf_model.pkl'))
        self.xgb = joblib.load(os.path.join(models_path, 'xgb_model.pkl'))
        self.lstm = tf.keras.models.load_model(os.path.join(models_path, 'lstm_model.h5'))
        self.scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
        self.time_steps = time_steps

    def predict(self, df):
        sanity_check_features(df)
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        pred_rf = self.rf.predict_proba(X_scaled)[:,1]
        pred_xgb = self.xgb.predict_proba(X_scaled)[:,1]

        X_lstm = []
        for i in range(len(X_scaled)-self.time_steps):
            X_lstm.append(X_scaled[i:i+self.time_steps])
        X_lstm = np.array(X_lstm)
        pred_lstm = self.lstm.predict(X_lstm, verbose=0).flatten()

        pred_rf_trim = pred_rf[self.time_steps:]
        pred_xgb_trim = pred_xgb[self.time_steps:]

        ensemble_pred = (pred_rf_trim + pred_xgb_trim + pred_lstm)/3
        return ensemble_pred, df['Datetime'].values[self.time_steps:]

# --- اجرای پیش‌بینی روی تمام نمادها ---
if __name__ == "__main__":
    model = EnsembleModel()
    all_results = []

    for symbol in SYMBOLS:
        print(f"⏳ Downloading {symbol} …")
        try:
            df = yf.download(symbol, period="60d", interval=INTERVAL, progress=False)
            if df.empty:
                print(f"⚠️ داده‌ای برای {symbol} موجود نیست، رد شد.")
                continue
            df = calculate_indicators(df)
            preds, dates = model.predict(df)
            signals = ["BUY" if p>0.5 else "SELL" for p in preds]

            for d, p, s in zip(dates, preds, signals):
                all_results.append({"Datetime": d, "Symbol": symbol, "Ensemble_Pred": p, "Signal": s})

        except Exception as e:
            print(f"❌ خطا در پردازش {symbol}: {e}")

    df_out = pd.DataFrame(all_results)
    df_out.to_csv("ensemble_predictions.csv", index=False)
    print("✅ پیش‌بینی‌ها ذخیره شد در ensemble_predictions.csv")
