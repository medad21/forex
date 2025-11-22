import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf  # کتابخانه جدید برای دانلود دیتا
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# تنظیمات
SYMBOL = "EURUSD=X"
PERIOD = "2y"  # دو سال دیتا
INTERVAL = "1h" # تایم فریم یک ساعته

def calculate_indicators(df):
    # پاکسازی نام ستون‌ها (مخصوص yfinance جدید)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
    
    # محاسبه اندیکاتورها دقیقاً مشابه main.py
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.rsi(length=6, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    
    # ویژگی‌های مهندسی شده
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()
    
    # نام‌گذاری دقیق ستون‌ها برای جلوگیری از خطا
    # توجه: نام‌ها باید دقیقاً با main.py یکی باشند
    df['RSI_14'] = df.get(f"RSI_14", df['ta_rsi_14'] if 'ta_rsi_14' in df else 0)
    df['RSI_6'] = df.get(f"RSI_6", df['ta_rsi_6'] if 'ta_rsi_6' in df else 0)
    df['ADX'] = df.get(f"ADX_14", df['ta_adx_14'] if 'ta_adx_14' in df else 0)
    
    # EMA Diffs
    ema20 = df.get(f"EMA_20", df['ta_ema_20'] if 'ta_ema_20' in df else df['close'])
    ema50 = df.get(f"EMA_50", df['ta_ema_50'] if 'ta_ema_50' in df else df['close'])
    ema100 = df.get(f"EMA_100", df['ta_ema_100'] if 'ta_ema_100' in df else df['close'])
    
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100

    return df.dropna()

def create_target(df):
    # هدف: اگر قیمت در 5 کندل آینده به اندازه 1.5 برابر ATR رشد کرد = 1 (خرید)
    # اگر افت کرد = 0 (فروش/خنثی)
    future_period = 5
    atr_multiplier = 1.5
    
    targets = []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['ATRr_14'].values
    
    for i in range(len(closes) - future_period):
        current_close = closes[i]
        atr = atrs[i]
        take_profit = current_close + (atr * atr_multiplier)
        stop_loss = current_close - (atr * atr_multiplier)
        
        # بررسی آینده
        future_highs = highs[i+1 : i+future_period+1]
        future_lows = lows[i+1 : i+future_period+1]
        
        if np.max(future_highs) >= take_profit:
            targets.append(1) # سیگنال خرید موفق
        else:
            targets.append(0) # عدم موفقیت خرید (یا نزول)
            
    # همگام‌سازی طول دیتافریم با تارگت‌ها
    df = df.iloc[:len(targets)]
    df['Target'] = targets
    return df

# --- اجرای اصلی ---
if __name__ == "__main__":
    print(f"⏳ در حال دانلود داده‌های واقعی {SYMBOL}...")
    df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False)
    
    if df.empty:
        print("❌ خطا در دانلود دیتا. وی‌پی‌ان خود را چک کنید.")
        exit()
        
    print("⚙️ محاسبه اندیکاتورها...")
    df = calculate_indicators(df)
    df = create_target(df)
    
    print(f"📊 تعداد داده‌های آموزشی: {len(df)} کندل")

    # فیچرهایی که مدل می‌بیند (دقیقاً مثل main.py)
    feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
    
    X = df[feature_cols].values
    y = df['Target'].values

    # تقسیم داده
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # استانداردسازی
    print("⚖️ آموزش Scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ساخت پوشه models
    if not os.path.exists('models'):
        os.makedirs('models')

    # 1. آموزش RF
    print("🌲 آموزش Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    joblib.dump(rf, 'models/rf_model.pkl')

    # 2. آموزش LR
    print("📈 آموزش Logistic Regression...")
    lr = LogisticRegression(C=1.0, random_state=42)
    lr.fit(X_train_scaled, y_train)
    joblib.dump(lr, 'models/lr_model.pkl')

    # 3. آموزش XGB
    print("🚀 آموزش XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, eval_metric='logloss')
    xgb.fit(X_train_scaled, y_train)
    joblib.dump(xgb, 'models/xgb_model.pkl')

    # 4. آموزش LSTM
    print("🧠 آموزش LSTM...")
    time_steps = 10
    def create_lstm_data(data, steps):
        X = []
        for i in range(len(data) - steps):
            X.append(data[i:(i + steps)])
        return np.array(X)

    # برای LSTM باید دوباره دیتا را فرمت کنیم
    X_lstm = create_lstm_data(scaler.transform(X), time_steps)
    y_lstm = y[time_steps:]
    
    # تقسیم مجدد مخصوص LSTM
    split = int(len(X_lstm) * 0.8)
    X_train_lstm, y_train_lstm = X_lstm[:split], y_lstm[:split]

    lstm = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(time_steps, len(feature_cols))),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, verbose=1)
    lstm.save('models/lstm_model.h5')

    # ذخیره Scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✅ تمام! مدل‌های هوشمند واقعی در پوشه models ذخیره شدند.")
    print("حالا این فایل‌ها را به پروژه Railway خود آپلود کنید.")
