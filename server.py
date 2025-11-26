from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
# import yfinance as yf  # ❌ حذف شد
import joblib
import tensorflow as tf
from datetime import datetime, timedelta
import os # ✅ اضافه شد
import warnings
# ایمپورت TDClient را به بلوک تنظیمات منتقل می‌کنیم تا در صورت نبود پکیج، سرور با خطا مواجه نشود
warnings.filterwarnings('ignore')

app = Flask(__name__)

# مسیر مدل‌ها
MODELS_PATH = "models"

# ---------------------------------------------------------
# تنظیمات API Twelve Data
# ---------------------------------------------------------
# می‌توانید کلید API را از متغیر محیطی بگیرید یا از یک کلید فال‌بک استفاده کنید.
API_KEY_TWELVEDATA = "f24a3dec20104e639d1995e42dc4673c"

# ---------------------------------------------------------
# تنظیم TDClient و توابع دانلود
# ---------------------------------------------------------
td = None
if API_KEY_TWELVEDATA and "YOUR_FALLBACK_KEY" not in API_KEY_TWELVEDATA: 
    try:
        from twelvedata import TDClient
        # TDClient را فقط یک بار در هنگام شروع سرور مقداردهی می‌کنیم
        td = TDClient(apikey=API_KEY_TWELVEDATA)
        print("✅ TDClient initialized successfully.")
    except ImportError:
        print("⚠️ twelvedata not installed. Please add 'twelvedata' to requirements.txt")
    except Exception as e:
        print(f"⚠️ Error initializing TDClient on server: {e}")
        td = None

def download_data_for_prediction(symbol, interval, output_size=150):
    """دانلود داده‌های لازم برای پیش‌بینی از Twelve Data."""
    
    if td is None:
        print("❌ TDClient not initialized. Cannot fetch real-time data.")
        return pd.DataFrame()

    try:
        # Twelve Data نمادها را بدون اسلش می‌پذیرد
        symbol_td = symbol.replace("/", "")
        
        print(f"⏳ Fetching {symbol_td}/{interval} from Twelve Data...")
        
        # output_size 150 برای پوشش EMA_100 کافی است
        ts = td.time_series(
            symbol=symbol_td,
            interval=interval,
            outputsize=output_size, 
            timezone="Exchange"
        ).as_json()
        
        if not ts or len(ts) < 100:
             print(f"⚠️ Insufficient data from Twelve Data for {symbol_td}.")
             return pd.DataFrame()
        
        df = pd.DataFrame(ts)
        # Twelve Data از ستون 'datetime' استفاده می‌کند
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
             # ستون‌های Twelve Data به طور پیش‌فرض رشته هستند
             df[col] = pd.to_numeric(df.get(col, 0.0), errors='coerce').fillna(0)
        
        # تابع calculate_indicators شما انتظار ستون‌های کوچک را دارد
        df = df.rename(columns={'open':'open','high':'high','low':'low','close':'close','volume':'volume'})
        
        print(f"✅ Twelve Data: Downloaded {len(df)} rows for {symbol_td}.")
        return df[['open','high','low','close','volume']].dropna()
        
    except Exception as e:
        print(f"❌ Twelve Data Download Error for {symbol_td}: {e}")
        return pd.DataFrame()

# ویژگی‌ها
feature_cols = [
    'RSI_14', 'RSI_6', 'ADX_14', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 
    'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20',
    'MFI_14', 'STOCH_K', 'SUPERT_D'
]

# --- توابع کمکی ---
def calculate_indicators(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # نیازی به rename نیست اگر داده از تابع download_data_for_prediction بیاید
    # df = df.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'}) 
    df['Returns'] = df['close'].pct_change()
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    df['HV_20'] = df['Returns'].rolling(20).std()
    
    # EMA
    df['EMA_20'] = df['close'].ewm(span=20).mean()
    df['EMA_50'] = df['close'].ewm(span=50).mean()
    df['EMA_100'] = df['close'].ewm(span=100).mean()
    df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
    df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_100']
    
    # RSI ساده
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI_14'] = 100 - (100 / (1 + rs))
    df['RSI_6'] = df['close'].diff().rolling(6).mean()
    
    # ADX, MFI, STOCH, SUPERT (placeholder=0 برای سادگی)
    df['ADX_14'] = 0
    df['MFI_14'] = 0
    df['STOCH_K'] = 0
    df['SUPERT_D'] = 0
    
    return df.dropna()

# --- مدل Ensemble ---
class EnsembleModel:
    def __init__(self):
        # بررسی وجود فایل‌ها قبل از بارگذاری در محیط پروداکشن مهم است
        try:
            self.rf = joblib.load(f"{MODELS_PATH}/rf_model.pkl")
            self.xgb = joblib.load(f"{MODELS_PATH}/xgb_model.pkl")
            self.lstm = tf.keras.models.load_model(f"{MODELS_PATH}/lstm_model.h5")
            self.scaler = joblib.load(f"{MODELS_PATH}/scaler.pkl")
        except FileNotFoundError as e:
            print(f"❌ Error loading model files. Ensure {MODELS_PATH}/ is populated. Error: {e}")
            raise
        self.time_steps = 10

    def predict(self, df):
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)

        # RF + XGB
        pred_rf = self.rf.predict_proba(X_scaled)[:,1]
        pred_xgb = self.xgb.predict_proba(X_scaled)[:,1]

        # LSTM
        X_lstm = []
        for i in range(len(X_scaled)-self.time_steps):
            X_lstm.append(X_scaled[i:i+self.time_steps])
        X_lstm = np.array(X_lstm)
        
        # اگر دیتای کافی برای LSTM وجود نداشته باشد (کمتر از 100 کندل)، باید پیش‌بینی را مدیریت کنیم
        if X_lstm.size == 0:
            return np.array([]), df.index 

        pred_lstm = self.lstm.predict(X_lstm, verbose=0).flatten()

        pred_rf_trim = pred_rf[self.time_steps:]
        pred_xgb_trim = pred_xgb[self.time_steps:]

        # اطمینان از هم‌اندازه بودن
        min_len = min(len(pred_rf_trim), len(pred_xgb_trim), len(pred_lstm))
        pred_rf_trim = pred_rf_trim[-min_len:]
        pred_xgb_trim = pred_xgb_trim[-min_len:]
        pred_lstm = pred_lstm[-min_len:]

        ensemble_pred = (pred_rf_trim + pred_xgb_trim + pred_lstm)/3
        return ensemble_pred, df.index[-min_len:]

ensemble_model = EnsembleModel()

# --- endpoint ---
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        payload = request.json
        # Twelve Data از نمادهای استاندارد بدون پیشوند FX_IDC یا BINANCE استفاده می‌کند
        symbol = payload.get("symbol", "EUR/USD").replace("/", "")
        interval = payload.get("interval", "1h")
        # end و start دیگر در تابع دانلود استفاده نمی‌شوند
        
        # ✅ جایگزینی با Twelve Data
        df = download_data_for_prediction(symbol, interval) 
        
        if df.empty:
            return jsonify({"error": f"No data for {symbol} or data fetch failed."}), 500
        
        df = calculate_indicators(df)
        
        # باید چک کنیم که بعد از محاسبه اندیکاتور و dropna، دیتای کافی باقی مانده باشد
        if len(df) < ensemble_model.time_steps + 1:
            return jsonify({"error": f"Insufficient processed data for prediction after indicator calculation ({len(df)} rows)."}), 500

        preds, dates = ensemble_model.predict(df)
        
        if not preds.size:
            return jsonify({"error": "Prediction failed to generate results (probably insufficient data)."}), 500
            
        latest_pred = preds[-1]
        score = float((latest_pred-0.5)*2*10)  # normalized to -10..+10
        signal = "buy" if latest_pred>0.5 else "sell"

        # خروجی شبیه فرانت
        result = {
            "score": score,
            "signal": signal,
            "price": float(df['close'].iloc[-1]),
            "setup": {
                "tp": None,
                "sl": None,
                "lot_size": 0,
                "risk_amt": 0
            },
            "indicators": {
                "sr_levels": "---",
                "news": "---",
                "trend": "Uptrend" if score>0 else "Downtrend",
                "rsi": float(df['RSI_14'].iloc[-1]),
                "regime": "---",
                "divergence": "---",
                "htf_status": "فعال" if payload.get("use_htf", True) else "غیرفعال",
                "htf_trend": "Bullish" if score>0 else "Bearish",
                "ai_report": {
                    "ensemble_score": float(latest_pred),
                    "individual_results": {
                        # این مقادیر باید از خروجی دقیق مدل‌ها گرفته شوند نه اینکه همه را آخرین ensemble_pred قرار دهید.
                        # برای رفع موقت، از آخرین پراب‌های تراز شده استفاده می‌کنیم:
                        "RF": {"score": float(pred_rf_trim[-1]), "prob": float(pred_rf_trim[-1]*100)},
                        "XGB": {"score": float(pred_xgb_trim[-1]), "prob": float(pred_xgb_trim[-1]*100)},
                        "LSTM": {"score": float(pred_lstm[-1]), "prob": float(pred_lstm[-1]*100)}
                    }
                }
            }
        }

        return jsonify(result)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
