# train.py (نسخه نهایی با راه حل میانبر CSV برای حل خطاهای دانلود)
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
import io
import warnings
warnings.filterwarnings('ignore')

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

# ✅ کلید API را به صورت مستقیم در کد قرار می‌دهیم تا از خطای متغیر محیطی جلوگیری شود.
TD_API_KEY = "f24a3dec20104e639d1995e42dc4673c" 

# -------------------------
# دانلود داده - اولویت ۱: میانبر CSV (راه حل قطعی برای دور زدن خطاهای شبکه)
# -------------------------
def download_ultimate_fallback(symbols):
    """
    دانلود داده از یک فایل CSV تکی (راه حل میانبر برای دور زدن خطاهای API و فایروال).
    """
    # 🛑 آدرس فایل CSV ثابت حاوی داده‌های مورد نیاز (700 روز 1 ساعته)
    FALLBACK_URL = "https://raw.githubusercontent.com/amirmahdi/sample-datasets/main/forex_crypto_data_combined_700days_1h.csv"
    
    print(f"🥇 ULTIMATE FALLBACK: Downloading combined CSV from static URL...")
    try:
        response = requests.get(FALLBACK_URL, timeout=30)
        response.raise_for_status() 
        
        df_combined = pd.read_csv(io.StringIO(response.text))
        
        # پاکسازی و آماده‌سازی داده‌ها
        df_combined['datetime'] = pd.to_datetime(df_combined['datetime'], utc=True)
        df_combined = df_combined.rename(columns={c: c.lower() for c in df_combined.columns})
        df_combined = df_combined[['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']].copy()
        df_combined = df_combined.dropna()
        
        print(f"✅ CSV Fallback downloaded. Total rows: {len(df_combined)}")
        
        # تقسیم مجدد به دیکشنری DataFrameها
        all_dfs = {}
        for sym in symbols:
            df_sym = df_combined[df_combined['symbol'] == sym.upper()].sort_values('datetime').reset_index(drop=True)
            if not df_sym.empty:
                all_dfs[sym] = df_sym
        
        if not all_dfs:
            print("❌ CSV downloaded, but no data found for required symbols.")
        return all_dfs
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Ultimate Fallback Failed (Network/URL Error): {e}")
        return {}
    except Exception as e:
        print(f"❌ Ultimate Fallback Failed (Parsing Error): {e}")
        return {}


# -------------------------
# دانلود داده - اولویت ۲: Twelve Data (در صورت شکست CSV)
# -------------------------
def download_td(symbol, interval='1h', days=TOTAL_DAYS):
    if not TD_API_KEY:
        print(f"⚠️ TD API Key not found, skipping Twelve Data.")
        return pd.DataFrame()
    
    td_symbol = symbol.replace("USD", "/USD").replace("JPY", "/JPY").replace("GC", "XAU/USD").replace("BTC", "BTC/USD")
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
# دانلود داده - اولویت ۳: Yahoo Finance (در صورت شکست Twelve Data و CSV)
# -------------------------
def download_yf(symbol, interval='1h', days=TOTAL_DAYS):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    
    ticker_map = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", 
        "USDJPY": "JPY=X",   
        "GC": "GC=F",        
        "BTC": "BTC-USD"     
    }
    ticker = ticker_map.get(symbol, f"{symbol}=X")
    print(f"🔎 Trying Yahoo Finance: {ticker}")
    
    try:
        # تنظیمات User-Agent برای دور زدن برخی محدودیت‌های شبکه
        session = requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False, session=session)
        
        if df.empty: return pd.DataFrame()
            
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
        df.columns = [c.lower() for c in df.columns]
        rename_map = {'date': 'datetime', 'adj close': 'close'}
        df = df.rename(columns=rename_map)
        
        req_cols = ['datetime','open','high','low','close','volume']
        valid_cols = [c for c in req_cols if c in df.columns]
        
        if len(valid_cols) < 5: return pd.DataFrame()

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
    # اطمینان از اینکه همه ستون‌ها عددی هستند
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
    
    # 1. اولویت اول: راه حل میانبر CSV
    data_dict = download_ultimate_fallback(SYMBOLS)
    
    # 2. اگر راه حل میانبر کار نکرد، سراغ منابع اصلی می‌رویم (که احتمالاً fail می‌شوند)
    if not data_dict:
        print("\nFallback to primary sources...")
        for sym in SYMBOLS:
            df = download_td(sym)
            if df.empty:
                df = download_yf(sym) 
                
            if not df.empty:
                data_dict[sym] = df
            else:
                print(f"❌ No data for {sym}")

    if not data_dict:
        print("\n❌ CRITICAL: No data available. Check internet or try again later.")
        exit()

    # 3. پردازش و آموزش
    all_dfs = [calculate_indicators_and_target(df) for df in data_dict.values()]
    all_dfs = [df for df in all_dfs if not df.empty]

    if not all_dfs:
        print("\n❌ CRITICAL: Data downloaded but not enough for processing indicators.")
        exit()
        
    df_all = pd.concat(all_dfs, ignore_index=True).dropna().reset_index(drop=True)
    print(f"\n📊 Total Samples for Training: {len(df_all)}")
    
    feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20','MFI_14','STOCH_K']
    
    # اطمینان از وجود تمام ستون‌های ویژگی (در صورت نیاز ستون صفر اضافه می‌شود)
    for c in feature_cols:
        if c not in df_all.columns: 
            print(f"⚠️ Adding placeholder for missing feature: {c}")
            df_all[c] = 0

    X = df_all[feature_cols].values
    y = df_all['Target'].values

    # تقسیم داده برای آموزش و متا مدل
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

    if len(X_train_full_scaled) > TIME_STEPS:
        print("🧠 Training LSTM...")
        
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
        lstm_probs = lstm_model.predict(X_lstm_meta, verbose=0).reshape(-1)
        
        # همترازی داده‌های متا با خروجی LSTM (که TIME_STEPS ردیف را از دست می‌دهد)
        min_len = min(len(rf_probs), len(xgb_probs), len(lstm_probs), len(y_meta))
        
        X_meta_final = np.column_stack([
            rf_probs[min_len-len(lstm_probs):][:len(lstm_probs)], 
            xgb_probs[min_len-len(lstm_probs):][:len(lstm_probs)], 
            lstm_probs
        ])
        y_meta_aligned = y_meta[min_len-len(lstm_probs):][:len(lstm_probs)]
        
        meta_model = LogisticRegression()
        meta_model.fit(X_meta_final, y_meta_aligned)
        joblib.dump(meta_model, os.path.join(MODEL_DIR, "lr_model.pkl"))

    print("\n✅ Training Finished Successfully! Models saved to the 'models' folder.")
