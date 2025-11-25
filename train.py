# predict.py
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf

MODEL_DIR = "models"
TIME_STEPS = 10  # باید همان مقدار TIME_STEPS در train.py باشد
feature_cols = [
    'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow',
    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
    'MFI_14', 'STOCH_K', 'SUPERT_D'
]

# بارگذاری مدل‌ها
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
rf = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
xgb = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
meta = joblib.load(os.path.join(MODEL_DIR, "meta_model.pkl"))
lstm = tf.keras.models.load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))

def calculate_indicators_local(df):
    # همان تابع محاسبهٔ اندیکاتور (مختصر شده) — df باید با ایندکس Datetime باشد
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    ema20 = df.get("EMA_20", df.get("ta_ema_20", df['close']))
    ema50 = df.get("EMA_50", df.get("ta_ema_50", df['close']))
    df['EMA_Diff_Fast'] = ema20 - ema50
    return df.dropna()

def ensemble_predict_from_df(df_recent):
    """
    df_recent: dataframe شامل آخرین ردیف‌ها با ایندکس زمانی — 
               باید حداقل TIME_STEPS ردیف داشته باشد و ستون‌های OHLCV
    """
    df_feat = calculate_indicators_local(df_recent)
    df_feat = df_feat.reset_index(drop=True)
    X = df_feat[feature_cols].values

    if len(X) < TIME_STEPS:
        raise ValueError("Not enough rows for TIME_STEPS")

    # استفاده از آخرین ردیف برای RF/XGB
    last_row = X[-1].reshape(1, -1)
    last_row_scaled = scaler.transform(last_row)
    rf_p = rf.predict_proba(last_row_scaled)[:,1][0]
    xgb_p = xgb.predict_proba(last_row_scaled)[:,1][0]

    # برای LSTM باید sequence بسازیم از آخرین TIME_STEPS ردیف‌ها
    seq = X[-TIME_STEPS:]
    seq_scaled = scaler.transform(seq)
    seq_input = seq_scaled.reshape(1, seq_scaled.shape[0], seq_scaled.shape[1])
    lstm_p = lstm.predict(seq_input).reshape(-1)[0]

    meta_X = np.array([[rf_p, xgb_p, lstm_p]])
    final_prob = meta.predict_proba(meta_X)[:,1][0]
    final_class = 1 if final_prob > 0.5 else 0
    return final_class, final_prob, dict(rf=rf_p, xgb=xgb_p, lstm=lstm_p)

# مثال استفاده:
if __name__ == "__main__":
    # ساخت یک دیتافریم نمونه: اینجا باید دیتای واقعی  time-series وارد کنی
    # df_recent باید شامل حداقل TIME_STEPS ردیف OHLCV باشد
    print("نمونه اجرا: predict.py — لطفا df_recent واقعی وارد کن")
