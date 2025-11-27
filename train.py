import os
import time
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import warnings

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

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
# 🔑👇 کلید API خود را دقیقاً در خط زیر قرار دهید 👇🔑
# ==========================================
TD_API_KEY = "f24a3dec20104e639d1995e42dc4673c" 
# اگر کلید بالا کار نکرد، کلید جدید خود را جایگزین کنید

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
# تابع دانلود داده
# -------------------------
def download_td(symbol_key, interval='1h', days=TOTAL_DAYS):
    td_symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
    
    if td is None:
        print(f"❌ TD API Key is invalid.")
        return pd.DataFrame()

    print(f"⏳ (TD) Downloading {td_symbol}...")
    output_size = min(days * 24, 5000)
    
    try:
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
        df = df.rename(columns={c: c.lower() for c in df.columns})

        if 'volume' not in df.columns:
            df['volume'] = 0.0
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        else:
            return pd.DataFrame()

        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0 

        df = df.sort_values('datetime').reset_index(drop=True)
        return df[['datetime','open','high','low','close','volume']]
        
    except Exception as e:
        print(f"⚠️ Download failed for {td_symbol}. Error: {e}")
        return pd.DataFrame()

# -------------------------
# تابع محاسبه اندیکاتورها (نسخه اصلاح شده و مقاوم)
# -------------------------
def calculate_indicators_and_target(df):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()

    # 1. محاسبات اصلی
    df['Returns'] = df['close'].pct_change()
    
    # محاسبه اندیکاتورها با مدیریت خطا
    # ما از try-except کلی استفاده نمی کنیم تا بفهمیم کدام بخش مشکل دارد
    # اما برای جلوگیری از کرش کردن روی نسخه نامپای، fillna را بلافاصله اعمال میکنیم
    
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.mfi(length=14, append=True)
    
    # اندیکاتورهای پیچیده تر
    try:
        df.ta.stoch(k=14, d=3, append=True)
    except: pass
    
    try:
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
    except: pass

    # 2. پیدا کردن نام ستون‌های متغیر (مثل STOCHk_14_3_3)
    stoch_k_col = next((c for c in df.columns if 'STOCHk' in c), None)
    supertd_col = next((c for c in df.columns if 'SUPERTd' in c), None)

    df['STOCH_K'] = df[stoch_k_col] if stoch_k_col else 50.0
    df['SUPERT_D'] = df[supertd_col] if supertd_col else 1.0

    # 3. سایر محاسبات
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()

    # EMA Cross
    # اول بررسی میکنیم ستون‌ها وجود داشته باشند، اگر نبودند با قیمت close پر میشوند (خنثی)
    ema20 = df['EMA_20'] if 'EMA_20' in df.columns else df['close']
    ema50 = df['EMA_50'] if 'EMA_50' in df.columns else df['close']
    ema100 = df['EMA_100'] if 'EMA_100' in df.columns else df['close']

    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    # 4. ساخت تار겟 (Target)
    # اگر قیمت 5 ساعت بعد بالاتر بود = 1، در غیر این صورت = 0
    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)

    # 5. لیست نهایی فیچرها (Features)
    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]

    # 6. پاکسازی نهایی (بسیار مهم: به جای dropna کلی، فقط فیچرها را پر میکنیم)
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0 # اگر ستونی ساخته نشد، صفر بگذار
        else:
            df[col] = df[col].fillna(0.0) # جاهای خالی را صفر کن
            
    # حذف 5 ردیف آخر که Target ندارند (چون شیفت دادیم)
    df_cleaned = df.iloc[:-5].copy()
    
    # فقط ستون‌های مورد نیاز را نگه میداریم
    final_cols = feature_cols + ['Target', 'close']
    df_cleaned = df_cleaned[final_cols]

    print(f"DEBUG: Rows before cleanup: {len(df)}. Rows after cleanup: {len(df_cleaned)}")
    
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
            print(f"❌ Skipping {sym} (Data vanished).")
            continue
            
        print(f"✅ {sym}: Data prepared. Rows: {len(df)}")
        all_dfs.append(df)
        time.sleep(1.0) 

    if not all_dfs:
        print("❌ CRITICAL: No usable data collected. Exiting.")
        raise SystemExit(1)

    # ادغام تمام دیتاها
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"📊 Total Training Data: {len(df_all)} rows")

    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]
    
    # استخراج X و y
    X = df_all[feature_cols].values
    y = df_all['Target'].values

    # تقسیم داده‌ها
    X_train_full, X_meta, y_train_full, y_meta = train_test_split(
        X, y, test_size=META_HOLDOUT_FRAC, random_state=42, shuffle=True, stratify=y
    )

    # نرمال‌سازی
    scaler = StandardScaler()
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_meta_scaled = scaler.transform(X_meta)
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

    # 3. آموزش LSTM
    print("🧠 Training LSTM...")
    if len(X_train_full_scaled) > TIME_STEPS + 50:
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
        min_len = min(len(rf_probs[TIME_STEPS:]), len(lstm_probs))
        
        rf_meta_aligned = rf_probs[TIME_STEPS:][:min_len]
        xgb_meta_aligned = xgb_probs[TIME_STEPS:][:min_len]
        lstm_probs = lstm_probs[:min_len]
        y_meta_aligned = y_meta[TIME_STEPS:][:min_len]

        # 4. آموزش Meta Model
        X_meta_for_meta = np.column_stack([rf_meta_aligned, xgb_meta_aligned, lstm_probs])
        
        print(f"🔗 Training Meta Model with {len(X_meta_for_meta)} samples...")
        meta_model = LogisticRegression()
        meta_model.fit(X_meta_for_meta, y_meta_aligned)
        joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.pkl"))
        
        print("✅✅ Training pipeline completed successfully!")
    else:
        print("⚠️ Data insufficient for LSTM sequences. Meta model skipped.")
