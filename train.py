import os
import time
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from twelvedata import TDClient
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
import math

# غیرفعال کردن هشدارهای غیرمهم
warnings.filterwarnings("ignore")

# -------------------------
# بررسی نصب بودن کتابخانه Twelve Data
# -------------------------
try:
    from twelvedata import TDClient
    print("✅ TDClient imported successfully.")
except ImportError:
    raise SystemExit("❌ Library 'twelvedata' not found. Please run: pip install twelvedata")

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
# 🛑 مقدار 650 روز مناسب است، چون گارد 5000 کندل جلوی خطا را می‌گیرد
TOTAL_DAYS = 650 
TIME_STEPS = 10
# 🛑 ما از تقسیم‌بندی زمانی (Walk-Forward) استفاده خواهیم کرد، نه Shuffle
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
USE_LSTM = True # از LSTM استفاده شود

# ==========================================
# 🔑👇 کلید API شما 👇🔑
# ==========================================
TD_API_KEY = "c15e9b87795a49aebc5b246e156b68bb" # خط ۴۲

# اتصال به کلاینت
td = None
if TD_API_KEY and "API_KEY" not in TD_API_KEY:
    try:
        td = TDClient(apikey=TD_API_KEY)
    except Exception as e:
        print(f"⚠️ Error initializing TDClient: {e}")
else:
    print("⚠️ هشدار: کلید API معتبر نیست.")

# -------------------------
# تابع دانلود داده (اصلاح شده و ضد باگ)
# -------------------------
def download_td(symbol_key, interval='1h', days=TOTAL_DAYS):
    td_symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
    
    if td is None:
        print(f"❌ TD API Key is invalid or TD client failed to initialize.")
        return pd.DataFrame()

    print(f"⏳ (TD) Downloading {td_symbol}...")
    
    # تنظیم حداکثر خروجی به 4500 برای جلوگیری از خطای مرز 5000 در پلن رایگان
    output_size = min(days * 24, 4500) 
    
    try:
        # دریافت داده خام
        ts_data = td.time_series(
            symbol=td_symbol,
            interval=interval,
            outputsize=output_size,
            timezone="Exchange"
        ).as_json()
        
        # 1️⃣ بررسی اینکه آیا داده دریافت شده یا پیام خطا است
        if isinstance(ts_data, dict) and ('code' in ts_data or 'status' in ts_data):
            # اگر خروجی دیکشنری باشد و دارای code باشد، یعنی ارور سمت سرور است
            msg = ts_data.get('message', 'Unknown API Error')
            print(f"⚠️ API Error for {td_symbol}: {msg}")
            return pd.DataFrame()

        # 2️⃣ بررسی خالی بودن لیست و نوع آن
        if not ts_data or not isinstance(ts_data, list) or len(ts_data) < 50:
             print(f"⚠️ Insufficient data for {td_symbol} (Len: {len(ts_data) if isinstance(ts_data, list) else 0}).")
             return pd.DataFrame()

        # تبدیل به DataFrame
        df = pd.DataFrame(ts_data)
        
        # کوچک کردن نام ستون‌ها (Open -> open)
        df = df.rename(columns={c: c.lower() for c in df.columns})

        # 3️⃣ بررسی و ساخت ستون‌های حیاتی
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ Critical column '{col}' missing for {td_symbol}.")
                return pd.DataFrame()
            # تبدیل به عدد
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # هندل کردن Volume (چون فارکس گاهی volume ندارد)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
        else:
            df['volume'] = 0.0
        
        # هندل کردن Datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            print(f"❌ Date column missing for {td_symbol}.")
            return pd.DataFrame()

        # مرتب‌سازی زمانی
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df[['datetime','open','high','low','close','volume']]
        
    except Exception as e:
        # چاپ کامل خطا برای دیباگ بهتر
        print(f"⚠️ Exception in download for {td_symbol}: {e}")
        return pd.DataFrame()

# -------------------------
# تابع محاسبه اندیکاتورها 
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
    df.ta.mfi(length=14, append=True)
    
    try: df.ta.stoch(k=14, d=3, append=True)
    except: pass
    
    try: df.ta.supertrend(length=10, multiplier=3.0, append=True)
    except: pass

    stoch_k_col = next((c for c in df.columns if 'STOCHk' in c), None)
    supertd_col = next((c for c in df.columns if 'SUPERTd' in c), None)

    df['STOCH_K'] = df[stoch_k_col] if stoch_k_col else 50.0
    df['SUPERT_D'] = df[supertd_col] if supertd_col else 1.0

    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()

    ema20 = df.get('EMA_20', df['close'])
    ema50 = df.get('EMA_50', df['close'])
    ema100 = df.get('EMA_100', df['close'])

    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    # Target: 5 کندل بعد
    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)

    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0 
        else:
            df[col] = df[col].fillna(0.0) 
            
    df_cleaned = df.iloc[:-5].copy()
    
    # اطمینان از وجود ستون 'close'
    final_cols = feature_cols + ['Target', 'close']
    df_cleaned = df_cleaned.filter(items=final_cols, axis=1)
    
    return df_cleaned

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

    for sym in SYMBOLS:
        df = download_td(sym)
        
        if df.empty:
            print(f"❌ Skipping {sym} (No Data).")
            continue
            
        print(f"✅ {sym}: Downloaded {len(df)} rows.")
        df = calculate_indicators_and_target(df)
        
        if df.empty:
            print(f"❌ Skipping {sym} (Data vanished or too short).")
            continue
            
        print(f"✅ {sym}: Data prepared. Rows: {len(df)}")
        all_dfs.append(df)
        time.sleep(2.0) # تأخیر برای جلوگیری از محدودیت نرخ API (Rate Limit)

    if not all_dfs:
        print("❌ CRITICAL: No usable data collected. Exiting.")
        raise SystemExit(1)

    df_all = pd.concat(all_dfs, ignore_index=True).reset_index(drop=True)
    print(f"📊 Total Training Data: {len(df_all)} rows")

    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]
    
    X = df_all[feature_cols].values
    y = df_all['Target'].values

    # 🛑 اصلاح Data Leakage: تقسیم‌بندی زمانی (Chronological Split)
    # 70% Train, 15% Validation (برای متا مدل), 15% Test (ارزیابی نهایی)
    total_len = len(X)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    
    X_train_full, y_train_full = X[:train_size], y[:train_size]
    X_meta, y_meta = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:] # داده‌های آینده مطلق

    # نرمال‌سازی
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_meta_scaled = scaler.transform(X_meta)
    X_test_scaled = scaler.transform(X_test) # نرمال‌سازی داده تست
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # 1. آموزش RandomForest
    print("🌲 Training RandomForest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_full_scaled, y_train_full)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

    # 2. آموزش XGBoost
    print("🚀 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_train_full_scaled, y_train_full)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    # پیش‌بینی روی داده متا (Validation Set)
    rf_probs = rf.predict_proba(X_meta_scaled)[:,1]
    xgb_probs = xgb.predict_proba(X_meta_scaled)[:,1]
    meta_inputs = np.column_stack([rf_probs, xgb_probs])
    
    y_meta_aligned = y_meta
    
    # 3. آموزش LSTM (اختیاری)
    if USE_LSTM and len(X_train_full_scaled) > TIME_STEPS + 50:
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
        lstm_model.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=64, verbose=0)
        lstm_model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

        # پیش‌بینی LSTM روی متا
        X_lstm_meta = create_sequences(X_meta_scaled, TIME_STEPS)
        lstm_probs = lstm_model.predict(X_lstm_meta, verbose=0).reshape(-1)

        # هم‌تراز کردن طول آرایه‌ها (چون LSTM چند داده اول را می‌خورد)
        # RF/XGB به اندازه val_size داده دارند. LSTM به اندازه val_size - TIME_STEPS
        min_len = min(len(meta_inputs), len(lstm_probs))
        
        # تنها داده‌هایی که هر سه مدل برایشان پیش‌بینی دارند استفاده می‌شوند
        meta_inputs = meta_inputs[-min_len:] 
        lstm_probs = lstm_probs[-min_len:]
        y_meta_aligned = y_meta[-min_len:]

        # آموزش Meta Model با LSTM
        X_meta_for_meta = np.column_stack([meta_inputs, lstm_probs])
        
    else:
        # آموزش Meta Model بدون LSTM
        X_meta_for_meta = meta_inputs
        
    print(f"🔗 Training Meta Model with {len(X_meta_for_meta)} samples...")
    meta_model = LogisticRegression()
    meta_model.fit(X_meta_for_meta, y_meta_aligned)
    joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.pkl"))
        
    # ==========================
    # ⚖️ تست نهایی (روی Test Set)
    # ==========================
    
    # پیش‌بینی مدل‌های پایه روی داده‌های تست
    rf_test = rf.predict_proba(X_test_scaled)[:, 1]
    xgb_test = xgb.predict_proba(X_test_scaled)[:, 1]
    final_input = np.column_stack([rf_test, xgb_test])
    
    # اگر LSTM آموزش دیده بود (یعنی متا مدل سه ورودی دارد)
    if USE_LSTM and len(X_test_scaled) > TIME_STEPS and len(X_meta_for_meta[0]) == 3: 
        # پیش‌بینی LSTM روی داده‌های تست
        X_lstm_test = create_sequences(X_test_scaled, TIME_STEPS)
        lstm_test = lstm_model.predict(X_lstm_test, verbose=0).flatten()
        
        # هم‌تراز کردن طول‌ها برای تست
        min_len_test = min(len(final_input), len(lstm_test))
        final_input = np.column_stack([final_input[-min_len_test:], lstm_test[-min_len_test:]])
        y_test = y_test[-min_len_test:] # آپدیت y_test
    
    # پیش‌بینی نهایی توسط متا مدل
    meta_probs = meta_model.predict_proba(final_input)[:, 1]
    y_pred_final = (meta_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred_final)

    print("\n" + "="*40)
    print("⚖️  FINAL TEST RESULTS (UNSEEN DATA)")
    print("="*40)
    print(f"🎯 META MODEL ACCURACY: {acc*100:.2f}%")
    print("-" * 30)
    
    if acc > 0.53:
        print("✅ GREAT! Model has a real statistical edge.")
    elif acc > 0.50:
        print("⚠️ OKAY. Model is slightly better than random.")
    else:
        print("❌ BAD. Model is confusing signals (Needs more data/features).")
        
    print(f"✅ Models saved in '{MODEL_DIR}/'")
