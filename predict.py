import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import yfinance as yf
import datetime

# تنظیمات
TIME_STEPS = 10
MODEL_DIR = "models"
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "GC=F", "BTC-USD"]
INTERVAL = "1h"

# فیچرهای اصلی مدل (باید دقیقاً با train.py منطبق باشد)
feature_cols = [
    'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
    'MFI_14', 'STOCH_K', 'SUPERT_D'
]

# --- توابع کمکی ---

def create_sequences(X, steps=TIME_STEPS):
    """تبدیل داده به توالی‌های زمانی برای LSTM"""
    seqs = []
    for i in range(len(X) - steps + 1): # +1 برای اطمینان از اینکه آخرین نقطه پیش‌بینی شود
        seqs.append(X[i:i + steps])
    return np.array(seqs)

def get_last_data_point(df, col_name, default=0.0):
    """دریافت آخرین مقدار ستون یا مقدار پیش‌فرض اگر ستون موجود نبود."""
    if col_name in df.columns:
        return df[col_name].iloc[-1]
    return default

# --- محاسبه اندیکاتورها (اصلاح شده) ---

def calculate_indicators(df):
    
    # 1. تمیزکاری نام ستون‌های yfinance
    df.columns = df.columns.str.lower()
    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    
    if df.empty: return pd.DataFrame()
    df['Returns'] = df['Close'].pct_change()
    
    # 2. محاسبه اندیکاتورها
    try:
        df.ta.ema(length=20, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=100, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.rsi(length=6, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.adx(length=14, append=True)
        df.ta.stoch(k=14, d=3, append=True)
        df.ta.mfi(length=14, append=True)
        df.ta.supertrend(length=10, multiplier=3.0, append=True)
    except Exception as e:
        print(f"⚠️ Warning during TA calculation: {e}")

    # 3. ایجاد و Clean up ستون‌های فیچر
    
    # ایجاد ستون‌های ثابت
    stoch_k_col = next((c for c in df.columns if 'STOCHk' in c), None)
    supertd_col = next((c for c in df.columns if 'SUPERTd' in c), None)

    df['STOCH_K'] = df[stoch_k_col] if stoch_k_col and stoch_k_col in df.columns else 50.0
    df['SUPERT_D'] = df[supertd_col] if supertd_col and supertd_col in df.columns else 1.0
    
    # محاسبات مشتق شده
    df['Volatility'] = df['High'] - df['Low']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()

    ema20 = df.get("EMA_20", df['Close'])
    ema50 = df.get("EMA_50", df['Close'])
    ema100 = df.get("EMA_100", df['Close'])
    df['EMA_Diff_Fast'] = ema20 - ema50
    df['EMA_Diff_Slow'] = ema50 - ema100
    
    # fillna و تضمین وجود ستون‌ها
    for col in feature_cols:
        if col not in df.columns:
            # اگر اندیکاتور نامش تغییر کرد
            df[col] = df.get(col, 0.0) 
        
        # پر کردن NaNها با صفر (یا با آخرین مقدار موجود اگر منطقی بود)
        df[col] = df[col].fillna(0.0)
        
    return df.reset_index(names=['Datetime'])

# --- کلاس Ensemble (اصلاح شده برای Meta Model) ---

class EnsembleModel:
    def __init__(self, models_path=MODEL_DIR, time_steps=TIME_STEPS):
        self.rf = joblib.load(os.path.join(models_path, 'rf_model.pkl'))
        self.xgb = joblib.load(os.path.join(models_path, 'xgb_model.pkl'))
        self.lstm = tf.keras.models.load_model(os.path.join(models_path, 'lstm_model.h5'))
        self.meta = joblib.load(os.path.join(models_path, 'meta_model.pkl')) # بارگذاری Meta Model
        self.scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
        self.time_steps = time_steps

    def predict(self, df):
        
        # 1. Sanity Check (تضمین وجود ستون‌ها)
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ خطا: ویژگی‌های زیر وجود ندارند: {missing_cols}")

        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        # 2. پیش‌بینی مدل‌های پایه (Base Models)
        pred_rf = self.rf.predict_proba(X_scaled)[:, 1]
        pred_xgb = self.xgb.predict_proba(X_scaled)[:, 1]

        # 3. پیش‌بینی LSTM
        X_lstm = create_sequences(X_scaled, self.time_steps)
        pred_lstm = self.lstm.predict(X_lstm, verbose=0).flatten()

        # 4. هم‌تراز کردن طول آرایه‌ها برای Meta Model
        # چون LSTM و Target ستون‌های اولیه را حذف می‌کنند
        min_len = min(len(pred_rf), len(pred_xgb), len(pred_lstm) + self.time_steps)
        
        # ما فقط به آخرین پیش‌بینی نیاز داریم، پس آخرین داده را می‌گیریم:
        
        # داده‌های متا برای آخرین نقطه
        rf_prob = pred_rf[-1]
        xgb_prob = pred_xgb[-1]
        lstm_prob = pred_lstm[-1]
        
        # ورودی Meta Model
        meta_input = np.array([[rf_prob, xgb_prob, lstm_prob]])

        # 5. پیش‌بینی نهایی توسط Meta Model
        final_prob = self.meta.predict_proba(meta_input)[:, 1][0]
        
        # آخرین زمان و قیمت برای نمایش
        last_datetime = df['Datetime'].iloc[-1]
        last_close = df['Close'].iloc[-1]

        return final_prob, last_datetime, last_close

# --- اجرای پیش‌بینی روی تمام نمادها ---
if __name__ == "__main__":
    
    # ⚠️ هشدار: این خط به یک مدل meta (رگرسیون لجستیک) نیاز دارد که در train.py ذخیره شده باشد.
    if not os.path.exists(os.path.join(MODEL_DIR, 'meta_model.pkl')):
        print("❌ CRITICAL ERROR: Meta Model not found! Please run train.py first.")
        raise SystemExit(1)
        
    model = EnsembleModel()
    all_results = []
    
    print("\n🚀 Starting Ensemble Prediction...")

    for symbol in SYMBOLS:
        print(f"\n⏳ Downloading and predicting for {symbol} ...")
        
        # yfinance فقط داده‌های گذشته را می‌دهد، 7 روز برای اطمینان از 500 کندل کافی است
        try:
            df = yf.download(symbol, period="7d", interval=INTERVAL, progress=False) 
            
            if df.empty or len(df) < TIME_STEPS + 5:
                print(f"⚠️ داده ناکافی برای {symbol}، رد شد.")
                continue
            
            df = calculate_indicators(df)
            
            # پیش‌بینی فقط برای آخرین کندل
            final_prob, current_time, current_price = model.predict(df)
            
            signal = "BUY (Long)" if final_prob > 0.55 else ("SELL (Short)" if final_prob < 0.45 else "NEUTRAL")

            all_results.append({
                "Datetime": current_time, 
                "Symbol": symbol, 
                "Last_Price": round(current_price, 4),
                "Buy_Probability": round(final_prob, 4), 
                "Signal": signal
            })

        except ValueError as ve:
             print(f"❌ Error in features for {symbol}: {ve}")
        except Exception as e:
            print(f"❌ خطای عمومی در پردازش {symbol}: {e}")

    df_out = pd.DataFrame(all_results)
    print("\n" + "="*50)
    print("✅ خلاصه پیش‌بینی‌های نهایی:")
    print(df_out.to_string(index=False))
    print("="*50)
    
    df_out.to_csv("ensemble_predictions.csv", index=False)
    print("\n✅ پیش‌بینی‌ها ذخیره شد در ensemble_predictions.csv")
