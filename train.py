import os
import time
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# غیرفعال کردن هشدارهای غیرمهم
warnings.filterwarnings("ignore")

# -------------------------
# تنظیمات برنامه
# -------------------------
# نگاشت نام نماد به نام فایل CSV آپلود شده
CSV_FILES = {
    "EURUSD": "EURUSD_data.csv",
    "GBPUSD": "GBPUSD_data.csv",
    "USDJPY": "USDJPY_data.csv",
    "XAUUSD": "XAUUSD_data.csv",
    "BTCUSD": "BTCUSD_data.csv"
}

SYMBOLS = list(CSV_FILES.keys()) 
TIME_STEPS = 10
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
USE_LSTM = True 

# -------------------------
# تابع خواندن فایل CSV (جایگزین دانلود)
# -------------------------
def load_local_data(symbol_key):
    file_name = CSV_FILES.get(symbol_key)
    
    if not os.path.exists(file_name):
        print(f"❌ File not found: {file_name}")
        return pd.DataFrame()

    print(f"📂 Loading local file: {file_name}...")

    try:
        # خواندن فایل با جداکننده ; (فرمت Twelve Data)
        df = pd.read_csv(file_name, sep=';')
        
        # تبدیل نام ستون‌ها به حروف کوچک
        df.columns = [c.lower() for c in df.columns]
        
        # اطمینان از فرمت datetime
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            print(f"⚠️ 'datetime' column missing in {file_name}")
            return pd.DataFrame()

        # مرتب‌سازی زمانی (از قدیم به جدید) - خیلی مهم برای آموزش
        df = df.sort_values('datetime').reset_index(drop=True)
        
        # اطمینان از وجود ستون‌های عددی
        req_cols = ['open', 'high', 'low', 'close', 'volume']
        for c in req_cols:
            if c not in df.columns:
                df[c] = 0.0
            else:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"❌ Error reading {file_name}: {e}")
        return pd.DataFrame()

# -------------------------
# تابع محاسبه اندیکاتورها (بدون تغییر)
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
    print("🚀 Starting Offline Training Pipeline (Using CSV Files)...")

    for sym in SYMBOLS:
        # استفاده از تابع خواندن فایل محلی
        df = load_local_data(sym)
        
        if df.empty:
            print(f"❌ Skipping {sym} (Empty or not found).")
            continue
            
        print(f"✅ {sym}: Loaded {len(df)} rows.")
        df = calculate_indicators_and_target(df)
        
        if df.empty:
            print(f"❌ Skipping {sym} (Not enough data after processing).")
            continue
            
        print(f"✅ {sym}: Indicators calculated. Rows: {len(df)}")
        all_dfs.append(df)

    if not all_dfs:
        print("❌ CRITICAL: No usable data found. Please check CSV files.")
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

    # تقسیم‌بندی زمانی (Chronological Split)
    total_len = len(X)
    train_size = int(total_len * 0.70)
    val_size = int(total_len * 0.15)
    
    X_train_full, y_train_full = X[:train_size], y[:train_size]
    X_meta, y_meta = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # نرمال‌سازی
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_meta_scaled = scaler.transform(X_meta)
    X_test_scaled = scaler.transform(X_test)
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

    # پیش‌بینی روی داده متا
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

        # هم‌تراز کردن طول آرایه‌ها
        min_len = min(len(meta_inputs), len(lstm_probs))
        
        meta_inputs = meta_inputs[-min_len:] 
        lstm_probs = lstm_probs[-min_len:]
        y_meta_aligned = y_meta[-min_len:]

        X_meta_for_meta = np.column_stack([meta_inputs, lstm_probs])
        
    else:
        X_meta_for_meta = meta_inputs
        
    print(f"🔗 Training Meta Model with {len(X_meta_for_meta)} samples...")
    meta_model = LogisticRegression()
    meta_model.fit(X_meta_for_meta, y_meta_aligned)
    joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.pkl"))
        
    # ==========================
    # ⚖️ تست نهایی (روی Test Set)
    # ==========================
    
    rf_test = rf.predict_proba(X_test_scaled)[:, 1]
    xgb_test = xgb.predict_proba(X_test_scaled)[:, 1]
    final_input = np.column_stack([rf_test, xgb_test])
    
    if USE_LSTM and len(X_test_scaled) > TIME_STEPS and len(X_meta_for_meta[0]) == 3: 
        X_lstm_test = create_sequences(X_test_scaled, TIME_STEPS)
        lstm_test = lstm_model.predict(X_lstm_test, verbose=0).flatten()
        
        min_len_test = min(len(final_input), len(lstm_test))
        final_input = np.column_stack([final_input[-min_len_test:], lstm_test[-min_len_test:]])
        y_test = y_test[-min_len_test:]
    
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
