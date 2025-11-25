# train.py (Fixed & Optimized)
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

# ✅ اصلاح مهم: کلید را مستقیم قرار دادیم (os.getenv حذف شد)
TD_API_KEY = "f24a3dec20104e639d1995e42dc4673c"

# -------------------------
# دانلود داده از Twelve Data
# -------------------------
def download_td(symbol, interval='1h', days=TOTAL_DAYS):
    """تلاش برای دانلود داده از Twelve Data"""
    if not TD_API_KEY:
        print(f"⚠️ TD API Key missing.")
        return pd.DataFrame()
    
    # نگاشت نمادها برای Twelve Data
    td_symbol = symbol
    if symbol == "GC": td_symbol = "XAU/USD" # طلا
    if symbol == "BTC": td_symbol = "BTC/USD"
    
    # محاسبه حجم دیتا (محدودیت 5000 تایی برای پلن رایگان)
    output_size = days * 24 
    if output_size > 5000: output_size = 5000

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
# دانلود داده از Yahoo Finance (Fallback)
# -------------------------
def download_yf(symbol, interval='1h', days=TOTAL_DAYS):
    """تلاش برای دانلود داده از Yahoo Finance"""
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=days)
    
    # ✅ اصلاح نمادها برای جلوگیری از ارور
    ticker_map = {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
        "GC": "GC=F",
        "BTC": "BTC-USD"
    }
    
    ticker = ticker_map.get(symbol, f"{symbol}=X")
        
    print(f"🔎 Trying Yahoo Finance: {ticker}")
    try:
        # دانلود بدون افکت‌های اضافی برای جلوگیری از ارور Impersonate
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        
        if df.empty:
            return pd.DataFrame()
            
        df = df.reset_index()
        # استانداردسازی نام ستون‌ها (حذف MultiIndex اگر وجود داشت)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = [c.lower() for c in df.columns]
        
        # هندل کردن نام ستون تاریخ
        if 'date' in df.columns: df = df.rename(columns={'date': 'datetime'})
        
        # انتخاب ستون‌های مورد نیاز
        req_cols = ['datetime','open','high','low','close','volume']
        valid_cols = [c for c in req_cols if c in df.columns]
        
        if len(valid_cols) < 5:
            print(f"⚠️ Missing columns in YF data for {symbol}")
            return pd.DataFrame()

        return df[valid_cols]
    except Exception as e:
        print(f"❌ YF Error for {symbol}: {e}")
        return pd.DataFrame()

# -------------------------
# محاسبهٔ اندیکاتورها
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
    
    # نگاشت نام ستون‌ها (برای سازگاری با ورژن‌های مختلف pandas-ta)
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
    ema100 = df.get('EMA_100', df['close'])
    
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    # Target: پیش‌بینی جهت قیمت در 5 کندل آینده
    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.dropna().reset_index(drop=True)

# -------------------------
# توابع کمکی
# -------------------------
def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

# -------------------------
# اجرای اصلی
# -------------------------
if __name__ == "__main__":
    print("🚀 Starting Training Process...")
    all_dfs = []
    
    for sym in SYMBOLS:
        print(f"\nProcessing {sym}...")
        # اولویت ۱: دانلود از سرور اصلی (TwelveData) با کلید صحیح
        df = download_td(sym)
        
        # اولویت ۲: دانلود از یاهو (اگر سرور اصلی خطا داد)
        if df.empty:
            df = download_yf(sym)
            
        if df.empty:
            print(f"❌ No data found for {sym} from any source.")
            continue
        
        print(f"✅ Downloaded {len(df)} candles for {sym}")
        
        df_processed = calculate_indicators_and_target(df)
        if not df_processed.empty:
            all_dfs.append(df_processed)
            print(f"   -> Processed samples: {len(df_processed)}")

    if not all_dfs:
        print("\n❌ CRITICAL: No data available to train models.")
        exit()

    df_all = pd.concat(all_dfs, ignore_index=True).dropna().reset_index(drop=True)
    print(f"\n📊 Total Training Samples: {len(df_all)}")
    
    feature_cols = ['RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20','MFI_14','STOCH_K']
    
    # پر کردن مقادیر خالی احتمالی
    for c in feature_cols:
        if c not in df_all.columns: df_all[c] = 0

    X = df_all[feature_cols].values
    y = df_all['Target'].values

    # تقسیم داده‌ها
    X_train_full, X_meta, y_train_full, y_meta = train_test_split(X, y, test_size=META_HOLDOUT_FRAC, random_state=42, shuffle=True, stratify=y)
    
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_meta_scaled = scaler.transform(X_meta)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # 1. RandomForest
    print("🌲 Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_full_scaled, y_train_full)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

    # 2. XGBoost
    print("🚀 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric='logloss')
    xgb.fit(X_train_full_scaled, y_train_full)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    # 3. LSTM
    print("🧠 Training LSTM...")
    if len(X_train_full_scaled) > TIME_STEPS:
        X_lstm_train = create_sequences(X_train_full_scaled, TIME_STEPS)
        y_lstm_train = y_train_full[TIME_STEPS:]

        lstm_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TIME_STEPS, len(feature_cols))),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=64, verbose=1)
        lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
        
        # Meta Model Prep (Logistic Regression)
        print("🤖 Training Meta Model...")
        rf_probs = rf.predict_proba(X_meta_scaled)[:,1]
        xgb_probs = xgb.predict_proba(X_meta_scaled)[:,1]
        
        X_lstm_meta = create_sequences(X_meta_scaled, TIME_STEPS)
        lstm_probs = lstm_model.predict(X_lstm_meta).reshape(-1)
        
        # همتراز کردن طول آرایه‌ها
        min_len = min(len(rf_probs), len(xgb_probs), len(lstm_probs))
        X_meta_final = np.column_stack([rf_probs[-min_len:], xgb_probs[-min_len:], lstm_probs[-min_len:]])
        y_meta_aligned = y_meta[-min_len:]
        
        meta_model = LogisticRegression()
        meta_model.fit(X_meta_final, y_meta_aligned)
        joblib.dump(meta_model, os.path.join(MODEL_DIR, "lr_model.pkl"))
        
    else:
        print("⚠️ Not enough data for LSTM.")

    print("\n✅✅ Training Finished Successfully! Models saved.")
