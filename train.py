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
from sklearn.metrics import accuracy_score, classification_report
import warnings
import math

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ تنظیمات (Config)
# ==========================================
USE_LSTM = True   # اگر سرور ضعیف است False کنید
TD_API_KEY = "f24a3dec20104e639d1995e42dc4673c" # کلید API
SYMBOL_MAP = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY", "XAUUSD": "XAU/USD", "BTCUSD": "BTC/USD"}
SYMBOLS = list(SYMBOL_MAP.keys())
TOTAL_DAYS = 700
TIME_STEPS = 10
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

td = None
try:
    if TD_API_KEY: td = TDClient(apikey=TD_API_KEY)
except: pass

# ==========================================
# 📥 توابع دانلود و مهندسی ویژگی (Feature Engineering)
# ==========================================
def download_td(symbol_key, interval='1h', days=TOTAL_DAYS):
    if not td: return pd.DataFrame()
    td_symbol = SYMBOL_MAP.get(symbol_key, symbol_key)
    print(f"⏳ Downloading {td_symbol}...")
    try:
        ts = td.time_series(symbol=td_symbol, interval=interval, outputsize=days*24, timezone="Exchange").as_json()
        if not ts or len(ts) < 100: return pd.DataFrame()
        df = pd.DataFrame(ts).rename(columns=str.lower)
        df['datetime'] = pd.to_datetime(df['datetime'])
        for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
        return df.sort_values('datetime').reset_index(drop=True)
    except Exception as e:
        print(f"❌ Error {td_symbol}: {e}")
        return pd.DataFrame()

def process_data(df):
    if len(df) < 100: return pd.DataFrame()
    df = df.copy()
    
    # 1. اندیکاتورهای پایه
    df['Returns'] = df['close'].pct_change()
    df['Log_Returns'] = np.log(df['close'] / df['close'].shift(1)) # دقیق‌تر از درصد ساده
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.atr(length=14, append=True)
    
    # 2. فیچرهای پیشرفته (Lag Features) - "حافظه بازار"
    # وضعیت RSI و بازدهی در ۱ و ۲ ساعت قبل
    df['RSI_Lag1'] = df['RSI_14'].shift(1)
    df['Returns_Lag1'] = df['Returns'].shift(1)
    df['Returns_Lag2'] = df['Returns'].shift(2)
    
    # 3. فیچرهای زمانی (Cyclical Features)
    # تبدیل ساعت خطی (0-23) به دایره‌ای (Sin/Cos) تا مدل بفهمد ساعت 23 به 0 نزدیک است
    df['Hour_Sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    
    # 4. فیچرهای محاسباتی
    df['Volatility'] = (df['high'] - df['low']) / df['close']
    df['EMA_Diff'] = (df.get('EMA_20', df['close']) - df.get('EMA_50', df['close']))
    
    # 5. تارگت: آیا ۳ ساعت بعد قیمت بالاتر است؟ (1=بله، 0=خیر)
    df['Target'] = (df['close'].shift(-3) > df['close']).astype(int)
    
    # پاکسازی NaN ها (به خاطر Lag و Shift ایجاد می‌شوند)
    df = df.dropna().reset_index(drop=True)
    return df

# ==========================================
# 🧠 اجرای آموزش
# ==========================================
if __name__ == "__main__":
    print("🚀 Starting Advanced Training (With Lag & Time Features)...")
    
    # جمع‌آوری دیتا
    all_data = []
    for sym in SYMBOLS:
        raw = download_td(sym)
        clean = process_data(raw)
        if not clean.empty: all_data.append(clean)
        time.sleep(1)
    
    if not all_data: exit("❌ No Data available.")
    df_full = pd.concat(all_data).sort_values('datetime').reset_index(drop=True)
    
    # لیست نهایی ویژگی‌ها برای آموزش
    features = [
        'RSI_14', 'RSI_Lag1',         # وضعیت مومنتوم الان و قبل
        'ADX_14', 
        'EMA_Diff', 
        'Returns', 'Returns_Lag1',    # وضعیت قیمت الان و قبل
        'Volatility', 
        'ATRr_14',
        'Hour_Sin', 'Hour_Cos'        # زمان بازار
    ]
    
    # اطمینان از وجود ستون‌ها
    for f in features: 
        if f not in df_full.columns: df_full[f] = 0
            
    X = df_full[features].values
    y = df_full['Target'].values
    
    print(f"📊 Total Samples: {len(X)}")

    # تقسیم‌بندی زمانی (Time Series Split) - بدون بُر زدن!
    train_size = int(len(X) * 0.70) # 70% آموزش
    val_size = int(len(X) * 0.15)   # 15% اعتبارسنجی
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:] # 15% تست نهایی
    
    # نرمال‌سازی
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    
    # آموزش RF
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, n_jobs=-1, random_state=42)
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, f"{MODEL_DIR}/rf_model.pkl")
    
    # آموزش XGBoost
    print("🚀 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=6, eval_metric='logloss', n_jobs=-1)
    xgb.fit(X_train_s, y_train)
    joblib.dump(xgb, f"{MODEL_DIR}/xgb_model.pkl")

    # آماده‌سازی متا مدل
    rf_val = rf.predict_proba(X_val_s)[:, 1]
    xgb_val = xgb.predict_proba(X_val_s)[:, 1]
    meta_input = np.column_stack([rf_val, xgb_val])
    
    # LSTM (اختیاری)
    if USE_LSTM and len(X_train_s) > TIME_STEPS + 50:
        print("🧠 Training LSTM...")
        def create_seq(data, steps=TIME_STEPS):
            return np.array([data[i-steps:i] for i in range(steps, len(data))])
            
        X_lstm_train = create_seq(X_train_s)
        y_lstm_train = y_train[TIME_STEPS:]
        
        lstm = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(TIME_STEPS, len(features))),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer='adam', loss='binary_crossentropy')
        lstm.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=32, verbose=0)
        lstm.save(f"{MODEL_DIR}/lstm_model.h5")
        
        # پیش‌بینی روی Val
        X_lstm_val = create_seq(X_val_s)
        lstm_pred_val = lstm.predict(X_lstm_val, verbose=0).flatten()
        
        # تراز کردن طول‌ها
        min_len = min(len(meta_input), len(lstm_pred_val))
        meta_input = np.column_stack([meta_input[-min_len:], lstm_pred_val[-min_len:]])
        y_val = y_val[-min_len:]

    # آموزش Meta Model
    print("🔗 Training Meta Model...")
    meta = LogisticRegression()
    meta.fit(meta_input, y_val)
    joblib.dump(meta, f"{MODEL_DIR}/meta_model.pkl")

    # ==========================
    # ⚖️ تست نهایی (The Moment of Truth)
    # ==========================
    print("\n" + "="*40)
    print("⚖️  FINAL TEST RESULTS (UNSEEN DATA)")
    print("="*40)
    
    # تولید ورودی برای تست نهایی
    rf_test = rf.predict_proba(X_test_s)[:, 1]
    xgb_test = xgb.predict_proba(X_test_s)[:, 1]
    
    final_input = np.column_stack([rf_test, xgb_test])
    
    if USE_LSTM and len(X_test_s) > TIME_STEPS:
        X_lstm_test = create_seq(X_test_s)
        lstm_test = lstm.predict(X_lstm_test, verbose=0).flatten()
        min_len_test = min(len(final_input), len(lstm_test))
        final_input = np.column_stack([final_input[-min_len_test:], lstm_test[-min_len_test:]])
        y_test = y_test[-min_len_test:]

    # پیش‌بینی نهایی
    meta_probs = meta.predict_proba(final_input)[:, 1]
    y_pred_final = (meta_probs > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred_final)
    
    print(f"🎯 META MODEL ACCURACY: {acc*100:.2f}%")
    print("-" * 30)
    
    if acc > 0.53:
        print("✅ GREAT! Model has a real statistical edge.")
    elif acc > 0.50:
        print("⚠️ OKAY. Model is slightly better than random.")
    else:
        print("❌ BAD. Model is confusing signals (Needs more data/features).")
        
    print(f"✅ Models saved in '{MODEL_DIR}/'")
