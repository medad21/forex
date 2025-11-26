# train.py
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import datetime
import time

# -------------------------
# بررسی نصب بودن کتابخانه Twelve Data
# -------------------------
try:
    from twelvedata import TDClient
    print("✅ TDClient imported successfully.")
except ImportError:
    raise SystemExit("❌ Library 'twelvedata' not found. Please run: pip install twelvedata")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -------------------------
# تنظیمات برنامه
# -------------------------
SYMBOL_MAP = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "XAUUSD": "XAU/USD",
    "BTCUSD": "BTC/USD"
}
SYMBOLS = list(SYMBOL_MAP.keys()) 
INTERVAL = "1h"
TOTAL_DAYS = 650
TIME_STEPS = 10
META_HOLDOUT_FRAC = 0.2
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# 🔑👇 کلید API خود را دقیقا در خط زیر داخل "" قرار دهید 👇🔑
# ==========================================
TD_API_KEY = "f24a3dec20104e639d1995e42dc4673c" 
# مثال: TD_API_KEY = "a1b2c3d4e5f6..."


# اتصال به کلاینت
td = None
if TD_API_KEY and "API_KEY_KHOD" not in TD_API_KEY:
    try:
        td = TDClient(apikey=TD_API_KEY)
    except Exception as e:
        print(f"⚠️ Error initializing TDClient: {e}")
else:
    print("⚠️ هشدار: کلید API وارد نشده است! لطفا خط 46 فایل را ویرایش کنید.")

# -------------------------
# تابع دانلود داده (مقاوم در برابر خطای Volume)
# -------------------------
def download_td(symbol_key, interval='1h', days=TOTAL_DAYS):
    """دانلود داده از Twelve Data با مدیریت خطای ستون حجم"""
    td_symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
    
    if td is None:
        print(f"❌ TD API Key is invalid or not set in the file.")
        return pd.DataFrame()

    print(f"⏳ (TD) Downloading {td_symbol}...")
    
    # محدودیت 5000 کندل برای اکانت‌های رایگان
    output_size = min(days * 24, 5000)
    
    try:
        # دریافت داده به صورت JSON (انعطاف‌پذیرتر از as_pandas)
        ts = td.time_series(
            symbol=td_symbol,
            interval=interval,
            outputsize=output_size,
            timezone="Exchange"
        ).as_json()
        
        if not ts or len(ts) < 50:
             print(f"⚠️ Insufficient data for {td_symbol}.")
             return pd.DataFrame()

        df = pd.DataFrame(ts)
        
        # استانداردسازی نام ستون‌ها به حروف کوچک
        df = df.rename(columns={c: c.lower() for c in df.columns})

        # 🛠️ فیکس مهم: اگر volume نبود یا ارور داد، با 0 پر کن
        if 'volume' not in df.columns:
            df['volume'] = 0.0
        
        # تبدیل ستون datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            return pd.DataFrame()

        # تبدیل داده‌های عددی و پر کردن مقادیر خالی با 0
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0 

        # مرتب‌سازی زمانی
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df[['datetime','open','high','low','close','volume']]
        
    except Exception as e:
        print(f"⚠️ Download failed for {td_symbol}. Error: {e}")
        return pd.DataFrame()

# -------------------------
# محاسبه اندیکاتورها
# -------------------------
def calculate_indicators_and_target(df):
    if len(df) < 50: return pd.DataFrame()
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
    
    try:
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
    except:
        pass 

    df['RSI_14'] = df.get('RSI_14', df.get('ta_rsi_14', 50))
    df['RSI_6']  = df.get('RSI_6', df.get('ta_rsi_6', 50))
    df['ADX_14'] = df.get('ADX_14', df.get('ta_adx_14', 0))
    df['STOCH_K'] = df.get('STOCHk_14_3_3', 50)
    df['SUPERT_D'] = df.get('SUPERTd_10_3.0', 0)
    df['MFI_14'] = df.get('MFI_14', 50)
    
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()

    ema20 = df.get('EMA_20', df['close'])
    ema50 = df.get('EMA_50', df['close'])
    
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - df.get('EMA_100', df['close'])

    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.dropna().reset_index(drop=True)

# -------------------------
# آماده‌سازی داده برای LSTM
# -------------------------
def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

# -------------------------
# بدنه اصلی برنامه
# -------------------------
if __name__ == "__main__":
    all_dfs = []
    print("🚀 Starting Training Pipeline...")

    if not TD_API_KEY or "API_KEY_KHOD" in TD_API_KEY:
        print("❌ ERROR: لطفا کلید API خود را در خط 46 فایل جایگزین کنید!")
        raise SystemExit(1)
    
    for sym in SYMBOLS:
        df = download_td(sym)
        
        if df.empty:
            print(f"❌ Skipping {sym} (No Data).")
            continue
            
        print(f"✅ {sym}: Downloaded {len(df)} rows.")
        df = calculate_indicators_and_target(df)
        
        if not df.empty:
            all_dfs.append(df)
        
        time.sleep(1.5) 

    if not all_dfs:
        print("❌ CRITICAL: No data collected from any symbol. Exiting.")
        raise SystemExit(1)

    df_all = pd.concat(all_dfs, ignore_index=True).dropna().reset_index(drop=True)
    print(f"📊 Total Training Data: {len(df_all)} rows")

    feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20','MFI_14','STOCH_K','SUPERT_D']
    
    valid_features = [c for c in feature_cols if c in df_all.columns]
    X = df_all[valid_features].values
    y = df_all['Target'].values

    # تقسیم داده‌ها
    X_train_full, X_meta, y_train_full, y_meta = train_test_split(X, y, test_size=META_HOLDOUT_FRAC, random_state=42, shuffle=True, stratify=y)

    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_meta_scaled = scaler.transform(X_meta)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("🌲 Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_full_scaled, y_train_full)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

    print("🚀 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
    xgb.fit(X_train_full_scaled, y_train_full)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    rf_probs = rf.predict_proba(X_meta_scaled)[:,1]
    xgb_probs = xgb.predict_proba(X_meta_scaled)[:,1]

    print("🧠 Training LSTM...")
    if len(X_train_full_scaled) > TIME_STEPS + 50:
        X_lstm_train = create_sequences(X_train_full_scaled, TIME_STEPS)
        y_lstm_train = y_train_full[TIME_STEPS:]

        lstm_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TIME_STEPS, len(valid_features))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=64, verbose=0)
        lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

        X_lstm_meta = create_sequences(X_meta_scaled, TIME_STEPS)
        lstm_probs = lstm_model.predict(X_lstm_meta, verbose=0).reshape(-1)

        min_len = min(len(rf_probs[TIME_STEPS:]), len(lstm_probs))
        
        rf_meta_aligned = rf_probs[TIME_STEPS:][:min_len]
        xgb_meta_aligned = xgb_probs[TIME_STEPS:][:min_len]
        lstm_probs = lstm_probs[:min_len]
        y_meta_aligned = y_meta[TIME_STEPS:][:min_len]

        X_meta_for_meta = np.column_stack([rf_meta_aligned, xgb_meta_aligned, lstm_probs])
        
        print(f"🔗 Training Meta Model with {len(X_meta_for_meta)} samples...")
        meta_model = LogisticRegression()
        meta_model.fit(X_meta_for_meta, y_meta_aligned)
        joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.pkl"))
        
        print("✅✅ Training pipeline completed successfully!")
    else:
        print("⚠️ Data insufficient for LSTM sequences.")
