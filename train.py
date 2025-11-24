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

# تنظیمات
SYMBOL = "EURUSD=X"
PERIOD = "2y"
INTERVAL = "1h"

def calculate_indicators(df):
    # اصلاح نام ستون‌ها
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    # 1. محاسبه اندیکاتورهای پایه
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    
    # 2. اندیکاتورهای جدید (برای هماهنگی با main.py)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.mfi(length=14, append=True)
    df.ta.supertrend(length=10, multiplier=3.0, append=True)

    # 3. نام‌گذاری و استانداردسازی ستون‌ها
    # نگاشت نام‌های pandas_ta به نام‌های مورد نظر ما
    if 'STOCHk_14_3_3' in df.columns: df['STOCH_K'] = df['STOCHk_14_3_3']
    if 'SUPERTd_10_3.0' in df.columns: df['SUPERT_D'] = df['SUPERTd_10_3.0']
    if 'ADX_14' not in df.columns and 'ADX' in df.columns: df['ADX_14'] = df['ADX']
    if 'ATRr_14' in df.columns: df['ATR_14'] = df['ATRr_14']

    # ساخت ویژگی‌های ترکیبی
    df['Volatility'] = (df['high'] - df['low']) / df['close']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()
    
    df['EMA_Diff_Fast'] = (df.get('EMA_20', df['close']) - df.get('EMA_50', df['close'])) / df['close']
    df['EMA_Diff_Slow'] = (df.get('EMA_50', df['close']) - df.get('EMA_100', df['close'])) / df['close']

    return df.dropna()

def create_target(df):
    # هدف: اگر قیمت در 5 کندل آینده به اندازه 1.5 برابر ATR رشد کرد = 1 (خرید)
    future_period = 5
    atr_multiplier = 1.5
    
    targets = []
    closes = df['close'].values
    highs = df['high'].values
    atrs = df['ATR_14'].values
    
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

if __name__ == "__main__":
    print(f"⏳ Downloading real data for {SYMBOL}...")
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False)
    
    if df.empty:
        print("❌ Error: Could not download data.")
        exit()
        
    print("⚙️ Calculating indicators...")
    df = calculate_indicators(df)
    df = create_target(df)
    
    print(f"📊 Training Data Size: {len(df)} candles")

    # لیست ویژگی‌ها (دقیقاً مطابق با main.py جدید)
    feature_cols = [
        'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
        'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
        'MFI_14', 'STOCH_K', 'SUPERT_D'
    ]
    
    # بررسی وجود ستون‌ها
    for col in feature_cols:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            exit()

    X = df[feature_cols].values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("⚖️ Training Scaler (with 13 features)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if not os.path.exists('models'):
        os.makedirs('models')

    # 1. RF
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, 'models/rf_model.pkl')

    # 2. LR
    print("📈 Training Logistic Regression...")
    lr = LogisticRegression(C=1.0, random_state=42)
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, 'models/lr_model.pkl')

    # 3. XGB
    print("🚀 Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    joblib.dump(xgb, 'models/xgb_model.pkl')

    # 4. LSTM
    print("🧠 Training LSTM...")
    time_steps = 10
    def create_lstm_data(data, steps):
        X = []
        for i in range(len(data) - steps):
            X.append(data[i:(i + steps)])
        return np.array(X)

    X_lstm = create_lstm_data(scaler.transform(X), time_steps)
    y_lstm = y[time_steps:]
    
    split = int(len(X_lstm) * 0.8)
    X_train_lstm, y_train_lstm = X_lstm[:split], y_lstm[:split]

    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(time_steps, len(feature_cols))),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=1)
    lstm.save('models/lstm_model.h5')

    # ذخیره Scaler جدید
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✅ Done! New models (13 features) saved in 'models/'.")
```

۲. **اجرای آموزش:**
در ترمینال Codespaces دستور زیر را بزنید تا مدل‌های جدید ساخته شوند:
```bash
python train.py
```
(مطمئن شوید پیام `✅ Done! New models...` را در انتها می‌بینید).

۳. **ارسال مدل‌های جدید به گیت‌هاب:**
حالا باید فایل‌های مدل جدید (`.pkl` و `.h5`) را آپلود کنید. دستورات زیر را به ترتیب بزنید:

```bash
git add models/
git commit -m "Retrain models with 13 features (Added MFI, Stoch, SuperTrend)"
git push origin main
