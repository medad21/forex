import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# -------------------------
# تنظیمات برنامه
# -------------------------
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
# 1. تابع خواندن فایل CSV
# -------------------------
def load_local_data(symbol_key):
    file_name = CSV_FILES.get(symbol_key)
    if not os.path.exists(file_name):
        print(f"❌ File not found: {file_name}")
        return pd.DataFrame()

    print(f"📂 Loading local file: {file_name}...")
    try:
        # فرض بر این است که جداکننده ; است (فرمت خروجی متاتریدر یا برخی ابزارها)
        # اگر فایل شما کاما است، sep=',' بگذارید
        df = pd.read_csv(file_name, sep=';') 
        df.columns = [c.lower() for c in df.columns]
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # مرتب‌سازی: قدیمی‌ترین به جدیدترین (بسیار مهم برای TimeSeries)
        df = df.sort_values('datetime').reset_index(drop=True)
        
        req_cols = ['open', 'high', 'low', 'close', 'volume']
        for c in req_cols:
            if c not in df.columns: df[c] = 0.0
            else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"❌ Error reading {file_name}: {e}")
        return pd.DataFrame()

# -------------------------
# 2. تابع محاسبه اندیکاتورها و هدف‌گذاری هوشمند (اصلاح شده)
# -------------------------
def calculate_indicators_and_target(df):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()

    # محاسبه تغییرات
    df['Returns'] = df['close'].pct_change()
    
    # اندیکاتورها
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True) # برای تارگت نیاز داریم
    df.ta.adx(length=14, append=True)
    df.ta.mfi(length=14, append=True)
    
    try: df.ta.stoch(k=14, d=3, append=True)
    except: pass
    try: df.ta.supertrend(length=10, multiplier=3.0, append=True)
    except: pass

    # پرکردن مقادیر خالی اندیکاتورها
    df = df.fillna(method='bfill').fillna(0)

    # نام‌گذاری استاندارد ستون‌ها
    stoch_k_col = next((c for c in df.columns if 'STOCHk' in c), None)
    supertd_col = next((c for c in df.columns if 'SUPERTd' in c), None)
    atr_col = next((c for c in df.columns if 'ATRr_14' in c or 'ATR_14' in c), 'ATR_14')

    df['STOCH_K'] = df[stoch_k_col] if stoch_k_col else 50.0
    df['SUPERT_D'] = df[supertd_col] if supertd_col else 1.0
    df['ATR_14'] = df[atr_col]

    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()

    # EMA Cross diffs
    ema20 = df.get('EMA_20', df['close'])
    ema50 = df.get('EMA_50', df['close'])
    ema100 = df.get('EMA_100', df['close'])
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    # ======================================================
    # 🔥 اصلاحیه مهم: تعریف هدف (Target) با فیلتر نویز (ATR)
    # ======================================================
    # شرط: قیمت در 5 کندل آینده باید حداقل (0.5 * ATR) رشد کند.
    # این یعنی فقط حرکات "معنادار" را 1 در نظر می‌گیریم، نه نوسانات کوچک.
    
    future_close = df['close'].shift(-5)
    threshold = df['ATR_14'] * 0.5  # ضریب سخت‌گیری (می‌توانید به 0.3 یا 0.8 تغییر دهید)
    
    # 1 = خرید (Long)
    # 0 = عدم خرید (می‌تواند نزولی یا رنج باشد)
    df['Target'] = (future_close > (df['close'] + threshold)).astype(int)

    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]

    # تضمین وجود ستون‌ها
    for col in feature_cols:
        if col not in df.columns: df[col] = 0.0 
            
    # حذف 5 ردیف آخر که Target ندارند (NaN هستند چون Shift دادیم)
    df_cleaned = df.iloc[:-5].copy()
    
    final_cols = feature_cols + ['Target']
    df_cleaned = df_cleaned.filter(items=final_cols, axis=1)
    
    return df_cleaned

# -------------------------
# تابع کمکی LSTM
# -------------------------
def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    if len(X) <= steps: return np.array([])
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

# -------------------------
# بدنه اصلی (Training)
# -------------------------
if __name__ == "__main__":
    all_dfs = []
    print("🚀 Starting ROBUST Training Pipeline...")

    for sym in SYMBOLS:
        df = load_local_data(sym)
        if df.empty: continue
        
        df_proc = calculate_indicators_and_target(df)
        if not df_proc.empty:
            all_dfs.append(df_proc)
            print(f"✅ {sym}: Processed {len(df_proc)} samples.")

    if not all_dfs:
        print("❌ No data found.")
        raise SystemExit(1)

    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"📊 Total Dataset: {len(df_all)} rows")

    feature_cols = [c for c in df_all.columns if c != 'Target']
    X = df_all[feature_cols].values
    y = df_all['Target'].values

    # تقسیم داده (Train/Val/Test) - بدون به هم ریختن ترتیب زمانی
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    X_meta = X[train_size:train_size+val_size]
    y_meta = y[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    # نرمال‌سازی
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_meta_scaled = scaler.transform(X_meta)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    # 1. Random Forest
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))

    # 2. XGBoost
    print("🚀 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=6, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_train_scaled, y_train)
    joblib.dump(xgb, os.path.join(MODEL_DIR, "xgb_model.pkl"))

    # آماده‌سازی متا
    rf_meta_prob = rf.predict_proba(X_meta_scaled)[:, 1]
    xgb_meta_prob = xgb.predict_proba(X_meta_scaled)[:, 1]
    meta_inputs = np.column_stack([rf_meta_prob, xgb_meta_prob])
    
    y_meta_aligned = y_meta

    # 3. LSTM (Optional)
    if USE_LSTM:
        print("🧠 Training LSTM...")
        X_lstm_train = create_sequences(X_train_scaled, TIME_STEPS)
        y_lstm_train = y_train[TIME_STEPS:] # LSTM تارگت‌های اولیه را از دست می‌دهد

        model_lstm = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TIME_STEPS, len(feature_cols))),
            tf.keras.layers.LSTM(50, return_sequences=True, dropout=0.2),
            tf.keras.layers.LSTM(30, dropout=0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model_lstm.fit(X_lstm_train, y_lstm_train, epochs=8, batch_size=64, verbose=0)
        model_lstm.save(os.path.join(MODEL_DIR, "lstm_model.h5"))

        # پیش‌بینی روی متا
        X_lstm_meta = create_sequences(X_meta_scaled, TIME_STEPS)
        if len(X_lstm_meta) > 0:
            lstm_probs = model_lstm.predict(X_lstm_meta, verbose=0).flatten()
            
            # همتراز کردن طول‌ها (چون LSTM چند داده اول را می‌خورد)
            min_len = min(len(meta_inputs), len(lstm_probs))
            meta_inputs = np.column_stack([meta_inputs[-min_len:], lstm_probs[-min_len:]])
            y_meta_aligned = y_meta[-min_len:]

    # 4. Meta Model (Logistic Regression)
    print("⚖️ Training Meta Model...")
    meta_model = LogisticRegression()
    meta_model.fit(meta_inputs, y_meta_aligned)
    joblib.dump(meta_model, os.path.join(MODEL_DIR, "meta_model.pkl"))

    # تست نهایی
    print("\n✅ Training Complete. Models Saved.")
