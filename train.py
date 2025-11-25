# train.py (نسخه نهایی و اصلاح شده برای تضمین دانلود و رفع خطای Yfinance)
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import datetime
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import yfinance as yf

# -------------------------
# تنظیمات
# -------------------------
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "GC", "BTC"] 
INTERVAL = "1h"
TOTAL_DAYS = 700 
TIME_STEPS = 10
META_HOLDOUT_FRAC = 0.2
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ✅ اصلاح مهم: کلید API مستقیم برای تضمین اتصال (حل مشکل TD API Key not found)
TD_API_KEY = "f24a3dec20104e639d1995e42dc4673c"

# -------------------------
# دانلود داده از Twelve Data (اولویت اول)
# -------------------------
def download_td(symbol, interval='1h', days=TOTAL_DAYS):
    if not TD_API_KEY:
        print(f"⚠️ TD API Key not set.")
        return pd.DataFrame()
    
    # اصلاح نمادها برای Twelve Data
    td_symbol = symbol
    if symbol == "GC": td_symbol = "XAU/USD"
    elif symbol == "BTC": td_symbol = "BTC/USD"
    elif symbol == "EURUSD": td_symbol = "EUR/USD"
    elif symbol == "GBPUSD": td_symbol = "GBP/USD"
    elif symbol == "USDJPY": td_symbol = "USD/JPY"
    
    # Twelve Data حداکثر 5000 کندل می‌دهد. برای 700 روز 1 ساعته، حدود 16800 کندل نیاز است.
    output_size = 5000 

    url = f'https://api.twelvedata.com/time_series?symbol={td_symbol}&interval={interval}&outputsize={output_size}&apikey={TD_API_KEY}&format=CSV'
    
    try:
        print(f"🌍 Requesting TwelveData for {td_symbol}...")
        df = pd.read_csv(url)
        
        if df.empty or 'datetime' not in df.columns:
            if 'code' in df.columns:
                print(f"⚠️ Twelve Data Error: {df.iloc[0].get('message', 'Unknown error')}")
            return pd.DataFrame()
            
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df = df.sort_values('datetime').reset_index(drop=True)
        return df[['datetime','open','high','low','close','volume']]
    except Exception as e:
        print(f"⚠️ Twelve Data download failed: {e}")
        return pd.DataFrame()

# -------------------------
# دانلود داده از Yahoo Finance (اولویت دوم)
# -------------------------
def download_yf(symbol, interval='1h', days=TOTAL_DAYS):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    
    # ✅ اصلاح نمادها برای Yahoo (حل مشکل symbol not found و YFTzMissingError)
    ticker_map = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",   # نماد استاندارد برای USDJPY
        "GC": "GC=F",        # طلا
        "BTC": "BTC-USD"     # بیت کوین
    }
    
    ticker = ticker_map.get(symbol, f"{symbol}=X")
    print(f"🔎 Trying Yahoo Finance: {ticker}")
    
    try:
        # ✅ اصلاح مهم: استفاده از session برای حل خطای Impersonate
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, session=session)
        
        if df.empty:
            return pd.DataFrame()
            
        df = df.reset_index()
        # پاکسازی نام ستون‌ها
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [c.lower() for c in df.columns]
        
        rename_map = {'date': 'datetime', 'adj close': 'close'}
        df = df.rename(columns=rename_map)
        
        req_cols = ['datetime','open','high','low','close','volume']
        valid_cols = [c for c in req_cols if c in df.columns]
        
        if len(valid_cols) < 5:
            return pd.DataFrame()

        return df[valid_cols]
    except Exception as e:
        print(f"❌ YF Error for {ticker}: {e}")
        return pd.DataFrame()

# -------------------------
# محاسبات و آموزش (بدون تغییر)
# -------------------------
def calculate_indicators_and_target(df):
    if len(df) < 50: return pd.DataFrame()
    
    df = df.copy()
    cols = ['open', 'high', 'low', 'close', 'volume']
    for c in cols: 
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()

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
    
    # نگاشت نام ستون‌ها
    df['RSI_14'] = df.get('RSI_14', df.get('ta_rsi_14', 50))
    df['RSI_6']  = df.get('RSI_6', df.get('ta_rsi_6', 50))
    df['ADX_14'] = df.get('ADX_14', df.get('ta_adx_14', 0))
    df['STOCH_K'] = df.get('STOCHk_14_3_3', 0)
    df['MFI_14'] = df.get('MFI_14', 0)
    
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

def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

if __name__ == "__main__":
    print("🚀 Starting Training...")
    all_dfs = []
    
    for sym in SYMBOLS:
        print(f"\nProcessing {sym}...")
        df = download_td(sym) # اولویت ۱: Twelve Data
        if df.empty:
            df = download_yf(sym) # اولویت ۲: Yahoo Finance
            
        if df.empty:
            print(f"❌ No data for {sym}")
            continue
        
        print(f"✅ Downloaded {len(df)} candles for {sym}")
        df_processed = calculate_indicators_and_target(df)
        if not df_processed.empty:
            all_dfs.append(df_processed)

    if not all_dfs:
        print("\n❌ CRITICAL: No data available. Check your internet connection or API Key.")
        exit()

    df_all = pd.concat(all_dfs, ignore_index=True).dropna().reset_index(drop=True)
    print(f"\n📊 Total Samples: {len(df_all)}")
    
    feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20','MFI_14','STOCH_K']
    
    for c in feature_cols:
        if c not in df_all.columns: df_all[c] = 0

    X = df_all[feature_cols].values
    y = df_all['Target'].values

    X_train_full, X_meta, y_train_full, y_meta = train_test_split(X, y, test_size=META_HOLDOUT_FRAC, random_state=42, shuffle=True, stratify=y)
    
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_meta_scaled = scaler.transform(X_meta)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    print("🌲 Training RF...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_full_scaled, y_train_full)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

    print("🚀 Training XGB...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05)
    xgb.fit(X_train_full_scaled, y_train_full)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    print("🧠 Training LSTM...")
    if len(X_train_full_scaled) > TIME_STEPS:
        X_lstm_train = create_sequences(X_train_full_scaled, TIME_STEPS)
        y_lstm_train = y_train_full[TIME_STEPS:]

        lstm_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TIME_STEPS, len(feature_cols))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=64, verbose=1)
        lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
        
        print("🤖 Training Meta Model...")
        rf_probs = rf.predict_proba(X_meta_scaled)[:,1]
        xgb_probs = xgb.predict_proba(X_meta_scaled)[:,1]
        
        X_lstm_meta = create_sequences(X_meta_scaled, TIME_STEPS)
        lstm_probs = lstm_model.predict(X_lstm_meta).reshape(-1)
        
        min_len = min(len(rf_probs), len(xgb_probs), len(lstm_probs))
        # همترازی داده‌های متا با خروجی LSTM (که TIME_STEPS ردیف را از دست می‌دهد)
        X_meta_final = np.column_stack([rf_probs[-min_len:], xgb_probs[-min_len:], lstm_probs[-min_len:]])
        y_meta_aligned = y_meta[-min_len:]
        
        meta_model = LogisticRegression()
        meta_model.fit(X_meta_final, y_meta_aligned)
        joblib.dump(meta_model, os.path.join(MODEL_DIR, "lr_model.pkl"))

    print("\n✅ Training Finished! Models saved.")
