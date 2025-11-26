# train.py (نسخه نهایی با استفاده از ماژول رسمی twelvedata)
import os
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
import datetime
import time

# 🛑 ایمپورت کتابخانه رسمی Twelve Data
try:
    from twelvedata import TDClient
    print("✅ TDClient imported.")
except ImportError:
    raise SystemExit("❌ twelvedata library not found. Please run: pip install twelvedata")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# -------------------------
# تنظیمات نمادها (Mapping)
# -------------------------
SYMBOL_MAP = {
    # TD Symbol تنها چیزی است که لازم داریم
    "EURUSD": {"TD": "EUR/USD"},
    "GBPUSD": {"TD": "GBP/USD"},
    "USDJPY": {"TD": "USD/JPY"},
    "XAUUSD": {"TD": "XAU/USD"},
    "BTCUSD": {"TD": "BTC/USD"}
}

SYMBOLS = list(SYMBOL_MAP.keys()) 
INTERVAL = "1h" 
TOTAL_DAYS = 650
TIME_STEPS = 10
META_HOLDOUT_FRAC = 0.2
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 🔑 تنظیم TD_API_KEY از متغیر محیطی
TD_API_KEY = os.getenv('TD_API_KEY', 'f24a3dec20104e639d1995e42dc4673c') 

# -------------------------
# اتصال به Twelve Data
# -------------------------
td = None
if TD_API_KEY and "YOUR_TWELVE" not in TD_API_KEY:
    try:
        td = TDClient(apikey=TD_API_KEY)
    except Exception as e:
        print(f"❌ Could not initialize TDClient: {e}")

# -------------------------
# دانلود داده از Twelve Data (تنها منبع)
# -------------------------
def download_td(symbol_key, interval='1h', days=TOTAL_DAYS):
    """تنها منبع: استفاده از TDClient"""
    td_symbol = SYMBOL_MAP[symbol_key]['TD']
    
    if td is None:
        print(f"❌ TD API Key is missing or invalid. Skipping Twelve Data for {symbol_key}.")
        return pd.DataFrame()

    print(f"⏳ (TD) Downloading {td_symbol} via TDClient...")
    
    # 5000: حداکثر تعداد شمع‌ها در طرح رایگان
    output_size = min(days * 24, 5000)
    
    try:
        ts = td.time_series(
            symbol=td_symbol,
            interval=interval,
            outputsize=output_size,
            timezone="Exchange" # یا UTC
        ).as_json() # دریافت داده در فرمت استاندارد
        
        if not ts or len(ts) < 50: # حداقل 50 کندل نیاز است
             print(f"⚠️ Twelve Data returned empty or insufficient data for {td_symbol}.")
             return pd.DataFrame()

        df = pd.DataFrame(ts)
        # تمیزکاری داده‌ها
        df = df.rename(columns={'datetime': 'datetime', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        # حجم در فارکس معمولا صفر یا خیلی کم است، باید آن را به عدد تبدیل کنیم
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.sort_values('datetime').reset_index(drop=True).dropna()

        return df[['datetime','open','high','low','close','volume']]
        
    except Exception as e:
        print(f"⚠️ TDClient failed for {td_symbol}: {e}")
        return pd.DataFrame()

# -------------------------
# محاسبات اندیکاتور و آموزش (بدون تغییر)
# -------------------------
def calculate_indicators_and_target(df):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna()

    df['Returns'] = df['close'].pct_change()
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.stoch(k=14, d=3, append=True)
    df.ta.mfi(length=14, append=True)

    df['RSI_14'] = df.get('RSI_14', df.get('ta_rsi_14', 50))
    df['ADX_14'] = df.get('ADX_14', df.get('ta_adx_14', 0))
    df['STOCH_K'] = df.get('STOCHk_14_3_3', 50)
    df['MFI_14'] = df.get('MFI_14', 50)
    
    try:
        st = df.ta.supertrend(length=10, multiplier=3.0)
        df['SUPERT_D'] = st['SUPERTd_10_3.0']
    except:
        df['SUPERT_D'] = 0

    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()

    ema20 = df.get('EMA_20', df['close'])
    ema50 = df.get('EMA_50', df['close'])
    df['EMA_Diff'] = ema20 - ema50

    df['Target'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.dropna().reset_index(drop=True)

def create_sequences(X, steps=TIME_STEPS):
    seqs = []
    for i in range(len(X)-steps):
        seqs.append(X[i:i+steps])
    return np.array(seqs)

if __name__ == "__main__":
    all_dfs = []
    print("🚀 Starting Pipeline...")

    for sym in SYMBOLS:
        df = download_td(sym) # ⬅️ فقط همین منبع!
        
        if df.empty:
            print(f"❌ Skipping {sym} - Twelve Data failed. Please check TD_API_KEY.")
            continue
            
        print(f"✅ Data OK for {sym}: {len(df)} rows")
        df = calculate_indicators_and_target(df)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)

    if not all_dfs:
        print("❌ CRITICAL: No data available from ANY source. Please check your TD_API_KEY.")
        raise SystemExit(1)
        
    df_all = pd.concat(all_dfs, ignore_index=True).dropna().reset_index(drop=True)
    
    feature_cols = ['RSI_14', 'ADX_14', 'EMA_Diff', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20', 'MFI_14', 'STOCH_K', 'SUPERT_D']
    valid_features = [c for c in feature_cols if c in df_all.columns]
    X = df_all[valid_features].values
    y = df_all['Target'].values

    # تقسیم داده و آموزش ... (بدون تغییر)
    # ...
    
    print("✅ Training Finished.")
