import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# تنظیمات دریافت دیتا
SYMBOL = "EURUSD=X"
PERIOD = "2y"   # دریافت دیتای 2 سال گذشته
INTERVAL = "1h" # تایم فریم 1 ساعته

# --- توابع کمکی ---
def calculate_indicators(df):
    # اصلاح نام ستون‌ها
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    # محاسبه اندیکاتورها
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    
    # ساخت ویژگی‌ها
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()
    
    # نام‌گذاری دقیق برای هماهنگی با main.py
    df['RSI_14'] = df.get(f"RSI_14", df['ta_rsi_14'] if 'ta_rsi_14' in df else 0)
    df['RSI_6'] = df.get(f"RSI_6", df['ta_rsi_6'] if 'ta_rsi_6' in df else 0)
    df['ADX'] = df.get(f"ADX_14", df['ta_adx_14'] if 'ta_adx_14' in df else 0)
    
    ema20 = df.get(f"EMA_20", df['ta_ema_20'] if 'ta_ema_20' in df else df['close'])
    ema50 = df.get(f"EMA_50", df['ta_ema_50'] if 'ta_ema_50' in df else df['close'])
    ema100 = df.get(f"EMA_100", df['ta_ema_100'] if 'ta_ema_100' in df else df['close'])
    
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    return df.dropna()

def create_target(df):
    # هدف: رشد قیمت به اندازه 1.5 برابر ATR در 5 کندل آینده
    future_period = 5
    atr_multiplier = 1.5
    targets = []
    closes = df['close'].values
    highs = df['high'].values
    atrs = df['ATRr_14'].values
    
    for i in range(len(closes) - future_period):
        current_close = closes[i]
        atr = atrs[i]
        take_profit = current_close + (atr * atr_multiplier)
        future_highs = highs[i+1 : i+future_period+1]
        
        if np.max(future_highs) >= take_profit:
            targets.append(1) 
        else:
            targets.append(0) 
            
    df = df.iloc[:len(targets)]
    df['Target'] = targets
    return df

# --- شروع عملیات ---
if __name__ == "__main__":
    print(f"⏳ Downloading real data for {SYMBOL}...")
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False)
    
    if df.empty:
        print("❌ Error: Could not download data.")
        exit()
        
    print("⚙️ Processing data...")
    df = calculate_indicators(df)
    df = create_target(df)
    print(f"📊 Data ready: {len(df)} candles")

    feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
    X = df[feature_cols].values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("⚖️ Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if not os.path.exists('models'):
        os.makedirs('models')

    print("🧠 Training Models (Wait a few seconds)...")
    
    # 1. RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, 'models/rf_model.pkl')

    # 2. LR
    lr = LogisticRegression(C=1.0, random_state=42)
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, 'models/lr_model.pkl')

    # 3. XGB
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    joblib.dump(xgb, 'models/xgb_model.pkl')

    # 4. LSTM
    time_steps = 10
    def create_lstm_data(data, steps):
        X = []
        for i in range(len(data) - steps):
            X.append(data[i:(i + steps)])
        return np.array(X)

    X_lstm = create_lstm_data(scaler.transform(X), time_steps)
    y_lstm = y[time_steps:]
    split = int(len(X_lstm) * 0.8)
    
    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, len(feature_cols))),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_lstm[:split], y_lstm[:split], epochs=3, batch_size=32, verbose=0)
    lstm.save('models/lstm_model.h5')

    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✅ Done! All models are updated with REAL data.")
