# train.py
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import datetime
import requests
import time
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -------------------------
# تنظیمات نمادها (Mapping)
# -------------------------
# کلیدها: نام‌های ساده برای فایل‌ها و مدل‌ها
# مقادیر: نماد دقیق در Yahoo Finance
SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",  # استاندارد فارکس (پوند به دلار)
    "USDJPY": "JPY=X",     # یا USDJPY=X (گاهی یاهو عوض می‌کند، JPY=X معمولا نرخ برابری است)
    "GC":     "GC=F",      # طلا (Futures)
    "BTC":    "BTC-USD"    # بیت‌کوین
}

SYMBOLS = list(SYMBOL_MAP.keys()) # لیست نام‌های ساده
INTERVAL = "1h"
TOTAL_DAYS = 650
TIME_STEPS = 10
META_HOLDOUT_FRAC = 0.2
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔑 کلید API خود را اینجا وارد کنید
TD_API_KEY = os.getenv('TD_API_KEY', 'f24a3dec20104e639d1995e42dc4673c') 

# -------------------------
# دانلود داده از Twelve Data (روش اصلی - دقیق‌تر)
# -------------------------
def download_td(symbol_key, interval='1h', days=TOTAL_DAYS):
    if not TD_API_KEY or "YOUR_TWELVE" in TD_API_KEY:
        print(f"⚠️ TD API Key not set. Skipping Twelve Data for {symbol_key}")
        return pd.DataFrame()

    # تبدیل به فرمت Twelve Data
    if symbol_key == "BTC": td_symbol = "BTC/USD"
    elif symbol_key == "GC": td_symbol = "XAU/USD" # در TD طلا XAU است
    elif symbol_key == "EURUSD": td_symbol = "EUR/USD"
    elif symbol_key == "GBPUSD": td_symbol = "GBP/USD"
    elif symbol_key == "USDJPY": td_symbol = "USD/JPY"
    else: td_symbol = symbol_key

    print(f"⏳ (TD) Downloading {td_symbol}...")
    output_size = min(days * 24, 5000) # رعایت لیمیت
    url = f'https://api.twelvedata.com/time_series?symbol={td_symbol}&interval={interval}&outputsize={output_size}&apikey={TD_API_KEY}&format=CSV'
    
    try:
        df = pd.read_csv(url)
        if 'code' in df.columns and df['code'].iloc[0] == 429:
             print(f"⚠️ Twelve Data Rate Limit.")
             return pd.DataFrame()
        if df.empty or 'datetime' not in df.columns:
            return pd.DataFrame()
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df = df.sort_values('datetime').reset_index(drop=True)
        return df[['datetime','open','high','low','close','volume']]
    except Exception as e:
        print(f"⚠️ TD Error for {symbol_key}: {e}")
        return pd.DataFrame()

# -------------------------
# دانلود داده از Yahoo Finance (با نمادهای اصلاح شده شما)
# -------------------------
def download_yf(symbol_key, interval='1h', days=TOTAL_DAYS):
    # دریافت نماد صحیح یاهو از مپینگ
    yf_ticker = SYMBOL_MAP.get(symbol_key, f"{symbol_key}=X")
    
    print(f"🔎 (YF) Trying Yahoo Finance for: {yf_ticker}")
    
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    
    try:
        # دانلود بدون مولتی ایندکس برای جلوگیری از پیچیدگی
        df = yf.download(yf_ticker, start=start, end=end, interval=interval, progress=False, multi_level_index=False)
        
        if df.empty:
            print(f"❌ Yahoo returned empty data for {yf_ticker}")
            return pd.DataFrame()
            
        df = df.reset_index()
        # نرمال‌سازی نام ستون‌ها (کوچک کردن حروف)
        df.columns = [c.lower() for c in df.columns]
        
        # پیدا کردن ستون تاریخ
        date_col = None
        for c in df.columns:
            if 'date' in c or 'time' in c:
                date_col = c
                break
        
        if date_col:
            df = df.rename(columns={date_col: 'datetime'})
            return df[['datetime','open','high','low','close','volume']]
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"❌ YF Exception for {yf_ticker}: {e}")
        return pd.DataFrame()

# -------------------------
# محاسبات اندیکاتور و آموزش
# -------------------------
def calculate_indicators_and_target(df):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    # تبدیل به float جهت اطمینان
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()

    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.mfi(length=14, append=True)

    # نام‌گذاری‌های استاندارد
    df['RSI_14'] = df.get('RSI_14', df.get('ta_rsi_14', 50))
    df['ADX_14'] = df.get('ADX_14', df.get('ta_adx_14', 0))
    df['STOCH_K'] = df.get('STOCHk_14_3_3', 50)
    df['MFI_14'] = df.get('MFI_14', 50)
    
    # Supertrend (با مدیریت خطا)
    try:
        st = df.ta.supertrend(length=10, multiplier=3.0)
        df['SUPERT_D'] = st['SUPERTd_10_3.0']
    except:
        df['SUPERT_D'] = 0

    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()

    ema20 = df.get('EMA_20', df['close'])
    ema50 = df.get('EMA_50', df['close'])
    df['EMA_Diff'] = ema20 - ema50

    # Target: 1 if Close in 5 hours > Current Close
    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.dropna().reset_index(drop=True)

def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

if __name__ == "__main__":
    all_dfs = []
    print("🚀 Starting Pipeline...")

    for sym in SYMBOLS:
        # 1. تلاش با Twelve Data
        df = download_td(sym)
        # 2. اگر نشد، تلاش با Yahoo Finance (با نماد صحیح)
        if df.empty:
            df = download_yf(sym)
        
        if df.empty:
            print(f"❌ Skipping {sym}")
            continue
            
        print(f"✅ Data OK for {sym}: {len(df)} rows")
        df = calculate_indicators_and_target(df)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)

    if not all_dfs:
        raise SystemExit("❌ No data available from ANY source.")

    df_all = pd.concat(all_dfs, ignore_index=True).dropna().reset_index(drop=True)
    
    # فیچرها
    feature_cols = ['RSI_14', 'ADX_14', 'EMA_Diff', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20', 'MFI_14', 'STOCH_K', 'SUPERT_D']
    
    # بررسی وجود ستون‌ها
    valid_features = [c for c in feature_cols if c in df_all.columns]
    X = df_all[valid_features].values
    y = df_all['Target'].values

    # تقسیم داده
    X_train, X_meta, y_train, y_meta = train_test_split(X, y, test_size=META_HOLDOUT_FRAC, shuffle=True, stratify=y)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_meta_s = scaler.transform(X_meta)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # 1. RandomForest
    print("🌲 RF...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

    # 2. XGBoost
    print("🚀 XGB...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train_s, y_train)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    # Meta Data Prep
    rf_p = rf.predict_proba(X_meta_s)[:,1]
    xgb_p = xgb.predict_proba(X_meta_s)[:,1]

    # 3. LSTM
    print("🧠 LSTM...")
    if len(X_train_s) > TIME_STEPS + 50:
        X_lstm_train = create_sequences(X_train_s, TIME_STEPS)
        y_lstm_train = y_train[TIME_STEPS:]
        
        lstm = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TIME_STEPS, len(valid_features))),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm.fit(X_lstm_train, y_lstm_train, epochs=3, batch_size=32, verbose=0)
        lstm.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
        
        # Meta inputs
        X_lstm_meta = create_sequences(X_meta_s, TIME_STEPS)
        lstm_p = lstm.predict(X_lstm_meta, verbose=0).reshape(-1)
        
        # Alignment
        min_len = min(len(rf_p[TIME_STEPS:]), len(lstm_p))
        meta_X = np.column_stack([rf_p[TIME_STEPS:][:min_len], xgb_p[TIME_STEPS:][:min_len], lstm_p[:min_len]])
        meta_y = y_meta[TIME_STEPS:][:min_len]
        
        # 4. Meta Model
        lr = LogisticRegression()
        lr.fit(meta_X, meta_y)
        joblib.dump(lr, os.path.join(MODEL_DIR, "meta_model.pkl"))
        
    print("✅ Training Finished.")
