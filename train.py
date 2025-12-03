import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import RobustScaler # مقاوم در برابر نویز
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# --- تنظیمات ---
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

# ---------------------------------------------------------
# 1. مهندسی ویژگی‌ها (Stationary Features)
# ---------------------------------------------------------
def calculate_features(df):
    if len(df) < 50: return pd.DataFrame(), []
    df = df.copy()

    # --- الف: تبدیل قیمت به فرمت ایستا (Stationary) ---
    # استفاده از Log Returns به جای قیمت خام
    df['Log_Ret'] = np.log(df['close'] / df['close'].shift(1))

    # --- ب: اندیکاتورها ---
    # RSI و MFI بین 0 تا 100 هستند -> تقسیم بر 100 می‌کنیم تا بین 0 تا 1 شوند
    df['RSI_Norm'] = df.ta.rsi(length=14) / 100.0
    df['MFI_Norm'] = df.ta.mfi(length=14) / 100.0
    
    # فاصله نسبی قیمت از میانگین‌ها (نه اختلاف پولی)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df['Dist_EMA20'] = (df['close'] - df['EMA_20']) / df['EMA_20']
    df['Dist_EMA50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
    
    # ATR برای محاسبه حد ضرر و تارگت (نیاز به نرمال‌سازی ندارد چون فیچر نیست، تارگت ساز است)
    df.ta.atr(length=14, append=True)

    # --- ج: نرمال‌سازی نوسان (Volatility Z-Score) ---
    # این ویژگی به مدل می‌فهماند الان بازار آرام است یا وحشی
    roll_std = df['Log_Ret'].rolling(window=20).std()
    roll_mean = df['Log_Ret'].rolling(window=20).mean()
    df['Vol_ZScore'] = (df['Log_Ret'] - roll_mean) / (roll_std + 1e-8)

    # --- د: تعریف تارگت هوشمند (ATR Based) ---
    # شرط: قیمت 5 کندل بعد > قیمت فعلی + (0.5 * ATR)
    # یعنی حرکت باید به اندازه کافی قوی باشد تا اسپرد را رد کند
    future_close = df['close'].shift(-5)
    threshold = df['ATRr_14'] * 0.5
    df['Target'] = (future_close > (df['close'] + threshold)).astype(int)

    # لیست فیچرهای نهایی برای آموزش (بدون قیمت خام!)
    feature_cols = [
        'Log_Ret', 'Dist_EMA20', 'Dist_EMA50', 
        'RSI_Norm', 'MFI_Norm', 'Vol_ZScore'
    ]
    
    # فیچرهای کمکی برای مدل متا (Context Features)
    context_cols = ['RSI_Norm', 'Vol_ZScore']

    # تمیزکاری: حذف NaN (بسیار مهم: با صفر پر نکنید!)
    df_clean = df.dropna().copy()
    
    # حذف 5 سطر آخر که تارگت ندارند
    df_clean = df_clean.iloc[:-5]

    return df_clean, feature_cols, context_cols

# ---------------------------------------------------------
# 2. آماده‌سازی داده‌ها
# ---------------------------------------------------------
def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    if len(X) <= steps: return np.array([])
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

def load_data():
    all_data = []
    feat_cols = []
    ctx_cols = []
    
    for sym in SYMBOLS:
        fname = CSV_FILES.get(sym)
        if not os.path.exists(fname): continue
        
        try:
            df = pd.read_csv(fname, sep=';') # جداکننده را چک کنید
            df.columns = [c.lower() for c in df.columns]
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime').reset_index(drop=True)
            
            df_proc, f_cols, c_cols = calculate_features(df)
            if not df_proc.empty:
                all_data.append(df_proc)
                feat_cols = f_cols
                ctx_cols = c_cols
                print(f"✅ {sym}: {len(df_proc)} samples ready.")
        except Exception as e:
            print(f"❌ Error {sym}: {e}")

    if not all_data: return None, None, None, None
    
    full_df = pd.concat(all_data, ignore_index=True)
    return full_df, feat_cols, ctx_cols

# ---------------------------------------------------------
# 3. اجرای اصلی آموزش
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🚀 Starting Professional Training...")
    
    df, features, contexts = load_data()
    if df is None: raise SystemExit("No data!")

    X = df[features].values
    y = df['Target'].values
    X_context = df[contexts].values # داده‌های زمینه برای مدل متا

    # تقسیم زمانی (Train 70%, Validation 15%, Test 15%)
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train = X[:train_end]
    y_train = y[:train_end]
    
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    C_val = X_context[train_end:val_end] # کانتکست برای اعتبارسنجی
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    C_test = X_context[val_end:]

    # --- Scaling (Prevent Leakage) ---
    # فیت کردن فقط روی Train
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    # --- Training Base Models ---
    print("🌲 RF...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")

    print("🚀 XGB...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.03, max_depth=5, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_train_s, y_train)
    joblib.dump(xgb, f"{MODEL_DIR}/xgb_model.pkl")

    print("🧠 LSTM...")
    X_train_seq = create_sequences(X_train_s)
    y_train_seq = y_train[TIME_STEPS:] # LSTM تارگت‌های اول را از دست می‌دهد
    
    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TIME_STEPS, len(features))),
        tf.keras.layers.LSTM(32, return_sequences=False, dropout=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_seq, y_train_seq, epochs=5, batch_size=64, verbose=0)
    lstm.save(f"{MODEL_DIR}/lstm_model.h5")

    # --- Training Meta Model (Context Aware) ---
    print("🔗 Building Meta Inputs...")
    
    # 1. پیش‌بینی‌های مدل‌های پایه روی داده‌های Validation
    rf_pred = rf.predict_proba(X_val_s)[:, 1]
    xgb_pred = xgb.predict_proba(X_val_s)[:, 1]
    
    X_val_seq = create_sequences(X_val_s)
    lstm_pred = lstm.predict(X_val_seq, verbose=0).flatten()
    
    # هم‌تراز کردن طول‌ها (چون LSTM چند داده اول را می‌خورد)
    min_len = min(len(rf_pred), len(lstm_pred))
    
    # برش داده‌ها به اندازه min_len (از آخر)
    rf_p = rf_pred[-min_len:]
    xgb_p = xgb_pred[-min_len:]
    lstm_p = lstm_pred[-min_len:]
    context_p = C_val[-min_len:] # RSI و Volatility
    y_meta_target = y_val[-min_len:]

    # ورودی متا: [RF, XGB, LSTM, RSI, Volatility]
    # اینجاست که مدل هوشمند می‌شود!
    meta_input = np.column_stack([rf_p, xgb_p, lstm_p, context_p])
    
    print("⚖️  Training Meta Logic...")
    meta_model = LogisticRegression()
    meta_model.fit(meta_input, y_meta_target)
    joblib.dump(meta_model, f"{MODEL_DIR}/meta_model.pkl")

    print("✅ Done! Models Saved.")
