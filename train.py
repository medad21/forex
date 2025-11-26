# train.py (نسخه نهایی و پایدار)
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import datetime
import requests
import time
import yfinance as yf # بازگشت به yfinance استاندارد
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -------------------------
# تنظیمات نمادها (Mapping)
# -------------------------
SYMBOL_MAP = {
    # کلید: نام ساده | مقادیر: نماد دقیق در Twelve Data (TD) و Yahoo Finance (YF)
    "EURUSD": {"TD": "EUR/USD", "YF": "EURUSD=X"},
    "GBPUSD": {"TD": "GBP/USD", "YF": "GBPUSD=X"},
    "USDJPY": {"TD": "USD/JPY", "YF": "JPY=X"},
    "XAUUSD": {"TD": "XAU/USD", "YF": "GC=F"},
    "BTCUSD": {"TD": "BTC/USD", "YF": "BTC-USD"}
}

SYMBOLS = list(SYMBOL_MAP.keys()) 
INTERVAL = "1h" # اگر این را به '1d' تغییر دهید، دانلود با YF کمی بهتر کار می‌کند
TOTAL_DAYS = 650
TIME_STEPS = 10
META_HOLDOUT_FRAC = 0.2
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔑 تنظیم TD_API_KEY از متغیر محیطی
TD_API_KEY = os.getenv('TD_API_KEY', 'YOUR_TWELVE_DATA_KEY') 

# -------------------------
# دانلود داده از Twelve Data (روش اصلی)
# -------------------------
def download_td(symbol_key, interval='1h', days=TOTAL_DAYS):
    """اولویت اول: Twelve Data"""
    td_symbol = SYMBOL_MAP[symbol_key]['TD']
    
    if not TD_API_KEY or "YOUR_TWELVE" in TD_API_KEY:
        # 🛑 این لاگ به شما می‌گوید که چرا Twelve Data کار نکرده است
        print(f"⚠️ TD API Key is missing or default. Skipping Twelve Data for {symbol_key}.")
        return pd.DataFrame()

    print(f"⏳ (TD) Downloading {td_symbol}...")
    output_size = min(days * 24, 5000) # رعایت لیمیت رایگان
    url = f'https://api.twelvedata.com/time_series?symbol={td_symbol}&interval={interval}&outputsize={output_size}&apikey={TD_API_KEY}&format=CSV'
    
    try:
        df = pd.read_csv(url)
        # بررسی خطاهای API یا Empty Response
        if 'code' in df.columns and df['code'].iloc[0] >= 400:
             print(f"⚠️ Twelve Data API Error for {symbol_key}. Code: {df['code'].iloc[0]}")
             return pd.DataFrame()
        if df.empty or 'datetime' not in df.columns:
            return pd.DataFrame()
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df = df.sort_values('datetime').reset_index(drop=True)
        return df[['datetime','open','high','low','close','volume']]
    except Exception as e:
        print(f"⚠️ TD Connection Error for {symbol_key}: {e}")
        return pd.DataFrame()

# -------------------------
# دانلود داده از Yahoo Finance (روش جایگزین - احتمالا بلاک شده)
# -------------------------
def download_yf(symbol_key, interval='1h', days=TOTAL_DAYS):
    """اولویت دوم: yfinance (احتمال شکست زیاد است)"""
    yf_ticker = SYMBOL_MAP[symbol_key]['YF']
    print(f"🔎 (YF) Trying {yf_ticker}...")
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    
    try:
        df = yf.download(yf_ticker, start=start, end=end, interval=interval, progress=False, multi_level_index=False)
        
        if df.empty: return pd.DataFrame()
            
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c or 'time' in c), None)
        
        if date_col:
            df = df.rename(columns={date_col: 'datetime'})
            return df[['datetime','open','high','low','close','volume']]
        return pd.DataFrame()
    except Exception as e:
        # 🛑 این بخش خطای ImpersonateError را رد می‌کند
        print(f"❌ YF Download Failed for {symbol_key} (Known Blocking Issue).")
        return pd.DataFrame()

# -------------------------
# محاسبات اندیکاتور و آموزش
# -------------------------
def calculate_indicators_and_target(df):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()

    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.mfi(length=14, append=True)

    df['RSI_14'] = df.get('RSI_14', df.get('ta_rsi_14', 50))
    df['ADX_14'] = df.get('ADX_14', df.get('ta_adx_14', 0))
    df['STOCH_K'] = df.get('STOCHk_14_3_3', 50)
    df['MFI_14'] = df.get('MFI_14', 50)
    
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
        df = download_td(sym) # 1. Twelve Data (اولویت اصلی)
        if df.empty:
            df = download_yf(sym) # 2. YFinance (احتمال شکست)
        
        if df.empty:
            print(f"❌ Skipping {sym}")
            continue
            
        print(f"✅ Data OK for {sym}: {len(df)} rows")
        df = calculate_indicators_and_target(df)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)

    if not all_dfs:
        print("❌ CRITICAL: No data available from ANY source. Please check your TD_API_KEY.")
        raise SystemExit(1)
        
    df_all = pd.concat(all_dfs, ignore_index=True).dropna().reset_index(drop=True)
    
    feature_cols = ['RSI_14', 'ADX_14', 'EMA_Diff', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20', 'MFI_14', 'STOCH_K', 'SUPERT_D']
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
        lstm_p = lstm.predict(X_lstm_meta, verbose=0).flatten()
        
        # Alignment
        min_len = min(len(rf_p[TIME_STEPS:]), len(lstm_p))
        meta_X = np.column_stack([rf_p[TIME_STEPS:][:min_len], xgb_p[TIME_STEPS:][:min_len], lstm_p[:min_len]])
        meta_y = y_meta[TIME_STEPS:][:min_len]
        
        # 4. Meta Model
        lr = LogisticRegression()
        lr.fit(meta_X, meta_y)
        joblib.dump(lr, os.path.join(MODEL_DIR, "meta_model.pkl"))
        
    print("✅ Training Finished.")
