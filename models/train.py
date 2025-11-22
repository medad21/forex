
# train.py (برای اجرای فقط یک بار و ذخیره مدل‌ها)
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# ۱. تعریف توابع لازم (فقط برای تولید داده)
# ------------------------------------------------------------------

# این تابع باید منطق کامل calculate_indicators_and_targets شما باشد.
# ما از یک نسخه ساده شده برای تولید داده استفاده می‌کنیم.
def calculate_indicators_and_targets(df):
    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    
    # اطمینان از وجود تمام ستون‌های ویژگی مورد نیاز
    df['ADX'] = df.ta.adx(length=14).iloc[:, 0] # فرض بر اینکه ADX درست محاسبه شود
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = pd.to_datetime(df['datetime']).dt.hour
    df['DayOfWeek'] = pd.to_datetime(df['datetime']).dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()
    
    df['RSI_14'] = df.get(next((c for c in df.columns if c.startswith('RSI_14')), ''), 0)
    df['RSI_6'] = df.ta.rsi(length=6) 
    df['EMA_Diff_Fast'] = df.get(next((c for c in df.columns if c.startswith('EMA_20')), ''), 0) - df.get(next((c for c in df.columns if c.startswith('EMA_50')), ''), 0)
    df['EMA_Diff_Slow'] = df.get(next((c for c in df.columns if c.startswith('EMA_50')), ''), 0) - df.ta.ema(length=100)
    
    # ایجاد یک Target نمونه (این باید Target واقعی شما باشد)
    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)
    
    return df.dropna().reset_index(drop=True)

# ------------------------------------------------------------------
# ۲. تولید داده نمونه و آموزش
# ------------------------------------------------------------------

print("1. در حال تولید داده‌های نمونه...")
# تولید داده‌های تصادفی برای شبیه‌سازی
data = np.random.rand(2000, 4) * 100
df_sample = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
df_sample['datetime'] = pd.to_datetime(pd.date_range('2024-01-01', periods=2000, freq='h'))

# اجرای تابع پردازش داده
df_processed = calculate_indicators_and_targets(df_sample)

# ستون‌های ویژگی
feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
X = df_processed[feature_cols].values
y = df_processed['Target'].values

# تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 🔑 آموزش StandardScaler و Transform
print("2. در حال آموزش Scaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------------
# ۳. آموزش و ذخیره مدل‌های 2D (RF, LR, XGB)
# ------------------------------------------------------------------

# RF
print("3. آموزش Random Forest و ذخیره...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
joblib.dump(rf_model, 'models/rf_model.pkl')

# LR
print("4. آموزش Logistic Regression و ذخیره...")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
joblib.dump(lr_model, 'models/lr_model.pkl')

# XGB
print("5. آموزش XGBoost و ذخیره...")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
joblib.dump(xgb_model, 'models/xgb_model.pkl')


# ------------------------------------------------------------------
# ۴. آموزش و ذخیره مدل 3D (LSTM)
# ------------------------------------------------------------------

# آماده‌سازی داده برای LSTM (3D: Samples, TimeSteps, Features)
def create_lstm_dataset(X, time_steps=10):
    Xs = []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
    return np.array(Xs)

TIME_STEPS = 10 
X_train_lstm = create_lstm_dataset(X_train_scaled, TIME_STEPS)
y_train_lstm = y_train[TIME_STEPS:] # تطابق دادن برچسب‌ها

print(f"6. در حال آموزش LSTM (TimeSteps={TIME_STEPS}) و ذخیره...")
lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=1, batch_size=32, verbose=0) 
lstm_model.save('models/lstm_model.h5')

# ------------------------------------------------------------------
# ۵. ذخیره Scaler نهایی و پایان
# ------------------------------------------------------------------

# 💡 ذخیره Scaler آموزش دیده (بسیار مهم!)
joblib.dump(scaler, 'models/scaler.pkl')

print("✅ تمام مدل‌ها و Scaler با موفقیت در پوشه models/ ذخیره شدند.")
