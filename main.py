import os
import json
import warnings
import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
import time

# ---------------------------------------------------------
# ۱. ایمپورت کتابخانه‌های سنگین AI و تنظیمات
# ---------------------------------------------------------
# ⚠️ مطمئن شوید که تمام این کتابخانه‌ها در requirements.txt نصب شده‌اند.
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.utils import class_weight 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# ---------------------------------------------------------
# ۲. پیکربندی و متغیرهای محیطی
# ---------------------------------------------------------

warnings.filterwarnings('ignore') # نادیده گرفتن اخطارهای آموزشی (مثلاً از TensorFlow)

# کلاس کمکی برای تبدیل خروجی NumPy به JSON استاندارد
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
app.json_encoder = NumpyEncoder # اعمال Encoder برای جلوگیری از خطای JSON

# 🔑 API KEYS - خواندن ایمن از متغیرهای محیطی Railway
API_KEY_TWELVEDATA = os.environ.get("TWELVEDATA_API_KEY", "df521019db9f44899bfb172fdce6b454") 
API_KEY_ALPHA = os.environ.get("ALPHA_VANTAGE_API_KEY", "W1L3K1JN4F77T9KL")              

# 📊 پارامترهای تریدینگ قابل بهینه‌سازی
RISK_REWARD_ATR = 1.5           
TARGET_PERIODS = 5              
ML_CONFIDENCE_THRESHOLD = 1.0   
SIGNAL_SCORE_THRESHOLD = 5.0    

TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
LSTM_TIME_STEPS = 10 
ML_SCORE_NORMALIZER = 40.0 

# ---------------------------------------------------------
# ۳. توابع هسته سیستم (Core Functions)
# ---------------------------------------------------------

# دریافت دیتا
def get_candles(symbol, interval, size=2000):
    # این تابع از کد پیشرفته قبلی شما است
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={size}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "values" not in data: return None
        df = pd.DataFrame(data["values"])
        for c in ['open', 'high', 'low', 'close']: df[c] = pd.to_numeric(df[c])
        df = df.iloc[::-1].reset_index(drop=True)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e: 
        print(f"Data fetch error: {e}")
        return None

# تابع محاسبه Actionable Target
def check_target(row, df_full, periods, rr_atr):
    # این تابع از کد پیشرفته قبلی شما است
    idx = row.name
    current_close = row['close']
    atr = row['ATR_Value']
    if idx + periods >= len(df_full) or atr == 0: return -1
    future_data = df_full.loc[idx+1 : idx+periods]
    if future_data.empty: return -1
    
    tp_buy = current_close + (atr * rr_atr)
    sl_buy = current_close - (atr * rr_atr)
    tp_sell = current_close - (atr * rr_atr)
    sl_sell = current_close + (atr * rr_atr)

    for i in range(len(future_data)):
        buy_win = (future_data['high'].iloc[i] >= tp_buy)
        buy_loss = (future_data['low'].iloc[i] <= sl_buy)
        sell_win = (future_data['low'].iloc[i] <= tp_sell)
        sell_loss = (future_data['high'].iloc[i] >= sl_sell)
        
        if buy_win and buy_loss:
            if (future_data['high'].iloc[i] - current_close) > (current_close - future_data['low'].iloc[i]): return 1
            return 2 
        if buy_win: return 1 
        if buy_loss: return 2 
        
        if sell_win and sell_loss:
             if (current_close - future_data['low'].iloc[i]) > (future_data['high'].iloc[i] - current_close): return 0
             return 2
        if sell_win: return 0 
        if sell_loss: return 2 
            
    return -1

# تابع آماده‌سازی داده سه‌بعدی LSTM
def create_lstm_dataset(X_scaled_df, y, time_steps):
    # این تابع از کد پیشرفته قبلی شما است
    Xs, ys = [], []
    for i in range(len(X_scaled_df) - time_steps):
        v = X_scaled_df.iloc[i:(i + time_steps)].values
        ys.append(y.iloc[i + time_steps]) 
        Xs.append(v)
    return np.array(Xs), np.array(ys)

# محاسبه واگرایی
def check_divergence(df):
    # این تابع از کد پیشرفته قبلی شما است
    if 'RSI_14' not in df.columns: df.ta.rsi(length=14, append=True)
    subset = df.iloc[-15:].reset_index(drop=True)
    price, rsi = subset['close'], subset['RSI_14']
    
    price_high_idx = price.idxmax()
    price_low_idx = price.idxmin()
    curr_price, curr_rsi = price.iloc[-1], rsi.iloc[-1]
    
    score, msg = 0, "بدون واگرایی"
    
    if price_high_idx < 14 and curr_price > price[price_high_idx] and curr_rsi < rsi[price_high_idx]: 
        msg, score = "Bearish Div 📉 (کاهش)", -3
    elif price_low_idx < 14 and curr_price < price[price_low_idx] and curr_rsi > rsi[price_low_idx]: 
        msg, score = "Bullish Div 📈 (افزایش)", 3
        
    return score, msg

# دریافت سنتیمنت بازار
def get_market_sentiment(symbol):
    # این تابع از کد پیشرفته قبلی شما است
    sentiment_score = 0
    sentiment_text = "اخبار خنثی (بدون رویداد مهم)"
    try:
        av_symbol = "FOREX:" + symbol.replace("/", "")
        if "BTC" in symbol: av_symbol = "CRYPTO:BTC"
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={av_symbol}&apikey={API_KEY_ALPHA}&limit=1"
        r = requests.get(url, timeout=3)
        data = r.json()
        if "feed" in data and len(data["feed"]) > 0:
            label = data["feed"][0].get("overall_sentiment_label", "Neutral")
            score = float(data["feed"][0].get("overall_sentiment_score", 0))
            if "Bullish" in label: 
                sentiment_text = "🟢 اخبار مثبت (Bullish)"
            elif "Bearish" in label: 
                sentiment_text = "🔴 اخبار منفی (Bearish)"
            sentiment_score = score * 5
            return sentiment_score, sentiment_text
    except: pass
    return sentiment_score, sentiment_text

# محاسبه هوشمند SL و TP
def calculate_smart_sl_tp(entry, signal, atr, support, resistance):
    # این تابع از کد پیشرفته قبلی شما است
    if atr is None or np.isnan(atr) or atr == 0: return None, None
    rr = 2.0 
    
    if signal == "buy":
        sl_base = entry - (atr * 1.5)
        if support != 0 and (entry - support) < (atr * 2.0): 
            sl_base = min(sl_base, support)
            
        tp = entry + ((entry - sl_base) * rr)
        sl = sl_base
        
    elif signal == "sell":
        sl_base = entry + (atr * 1.5)
        if resistance != 0 and (resistance - entry) < (atr * 2.0): 
            sl_base = max(sl_base, resistance)
            
        tp = entry - ((sl_base - entry) * rr)
        sl = sl_base
    else:
        return None, None
        
    return round(float(sl), 5) if sl is not None else None, round(float(tp), 5) if tp is not None else None


# محاسبه تمام اندیکاتورها و هدف عملیاتی
def calculate_indicators_and_targets(df):
    # این تابع از کد پیشرفته قبلی شما است
    df['Returns'] = df['close'].pct_change()
    
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=100, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.adx(length=14, append=True)
    df.ta.donchian(lower_length=20, upper_length=20, append=True)
    
    df['ADX'] = df.get(next((c for c in df.columns if c.startswith('ADX')), ''), 0)
    df['Volatility'] = df['high'] - df['low']
    df['Hour'] = df['datetime'].dt.hour
    df['DayOfWeek'] = df['datetime'].dt.dayofweek
    df['HV_20'] = df['Returns'].rolling(window=20).std()
    df['ATR_Value'] = df.get(next((c for c in df.columns if c.startswith('ATRr')), ''), 0)
    df['RSI_14'] = df.get(next((c for c in df.columns if c.startswith('RSI_14')), ''), 0)
    df['RSI_6'] = df.ta.rsi(length=6) 
    df['EMA_20'] = df.get(next((c for c in df.columns if c.startswith('EMA_20')), ''), 0)
    df['EMA_50'] = df.get(next((c for c in df.columns if c.startswith('EMA_50')), ''), 0)
    df['EMA_100'] = df.get(next((c for c in df.columns if c.startswith('EMA_100')), ''), 0)
    df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
    df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_100']
    df['DCL'] = df.get(next((c for c in df.columns if c.startswith('DCL')), ''), 0)
    df['DCU'] = df.get(next((c for c in df.columns if c.startswith('DCU')), ''), 0)
    
    df['Target'] = df.apply(check_target, axis=1, args=(df, TARGET_PERIODS, RISK_REWARD_ATR)) 

    return df.dropna().reset_index(drop=True)

# آموزش مدل‌های AI و پیش‌بینی
def get_ml_prediction(df_full):
    # این تابع از کد پیشرفته قبلی شما است
    report = {
        "accuracy": 0, "importances": {}, "message": "AI: خنثی",
        "ensemble_score": 0, "ml_score_final": 0, "individual_results": {}
    }
    
    historical_ml_scores = pd.Series()
    
    models = {
        'RF': RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42, class_weight="balanced"),
        'XGB': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'), 
        'LR': LogisticRegression(solver='liblinear', random_state=42, class_weight="balanced"),
    }

    try:
        df = df_full[df_full['Target'] != -1].copy() 
        
        feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        df = df.dropna(subset=feature_cols + ['Target'])

        if len(df) < 200: 
            report["message"] = f"AI: دیتای کافی برای آموزش Actionable ({len(df)}/200)"
            return 0, report, historical_ml_scores

        X = df[feature_cols].copy()
        Y = df['Target'].apply(lambda x: 1 if x == 1 or x == 0 else 0).astype(int).copy() 
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        test_size = max(100, int(len(df) * 0.1)) 
        X_train_2d = X_scaled_df.iloc[:-test_size]
        Y_train_2d = Y.iloc[:-test_size]
        X_test_2d = X_scaled_df.iloc[-test_size:]
        Y_test_2d = Y.iloc[-test_size:]
        
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train_2d), y=Y_train_2d)
        class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
        sample_weights_xgb = Y_train_2d.apply(lambda x: class_weights_dict[x]).values
        
        X_lstm, Y_lstm = create_lstm_dataset(X_scaled_df, Y, LSTM_TIME_STEPS)
        
        test_size_3d = min(test_size, len(X_lstm) - 20)
        
        if len(X_lstm) < 50 or test_size_3d <= 0:
            report["message"] = "AI: دیتای کافی برای LSTM و تست وجود ندارد."
            return 0, report, historical_ml_scores
            
        X_train_lstm = X_lstm[:-test_size_3d]
        Y_train_lstm = Y_lstm[:-test_size_3d]
        X_test_lstm = X_lstm[-test_size_3d:]
        Y_test_lstm = Y_lstm[-test_size_3d:]

        last_features = X.iloc[-1].to_frame().T
        last_features_scaled_2d = scaler.transform(last_features) 
        last_window_data = X_scaled_df.iloc[-LSTM_TIME_STEPS:].values
        last_features_scaled_3d = last_window_data.reshape(1, LSTM_TIME_STEPS, len(feature_cols)) 
        
        if len(np.unique(Y_train_2d)) < 2: 
            report["message"] = "AI: کلاس‌های هدف برای آموزش تنوع کافی ندارند."
            return 0, report, historical_ml_scores
        
        ensemble_score_total = 0
        test_predictions_scores = []
        
        # آموزش و پیش‌بینی مدل‌های 2D
        for name in ['RF', 'LR', 'XGB']:
            model = models[name]
            if name == 'XGB': model.fit(X_train_2d, Y_train_2d, sample_weight=sample_weights_xgb)
            else: model.fit(X_train_2d, Y_train_2d)
            
            test_proba = model.predict_proba(X_test_2d)[:, 1]
            test_predictions_scores.append((test_proba - 0.5) * 100) 
            
            prob_p = model.predict_proba(last_features_scaled_2d)[0][1] 
            confidence_score = (prob_p - 0.5) * 100 
            ensemble_score_total += confidence_score
            
            report["individual_results"][name] = round(confidence_score, 1)
            
        # آموزش و پیش‌بینی مدل 3D (LSTM)
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(LSTM_TIME_STEPS, len(feature_cols))))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(1, activation='sigmoid'))
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss=BinaryCrossentropy(), metrics=['accuracy'])
        
        lstm_class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train_lstm), y=Y_train_lstm)
        lstm_class_weights_dict = {0: lstm_class_weights[0], 1: lstm_class_weights[1]}
        
        lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=5, batch_size=32, verbose=0, class_weight=lstm_class_weights_dict)
        
        test_proba_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        test_predictions_scores.append((test_proba_lstm - 0.5) * 100)
        
        prob_p_lstm = lstm_model.predict(last_features_scaled_3d, verbose=0)[0][0]
        confidence_score_lstm = (prob_p_lstm - 0.5) * 100
        ensemble_score_total += confidence_score_lstm
        
        report["individual_results"]["LSTM"] = round(confidence_score_lstm, 1)

        min_test_size = min(len(X_test_2d), len(X_test_lstm))
        ensemble_test_scores_array = np.mean([s[:min_test_size] for s in test_predictions_scores], axis=0)

        if 'RF' in models: report["accuracy"] = round(models['RF'].score(X_test_2d, Y_test_2d) * 100, 2)
        if hasattr(models['RF'], 'feature_importances_'):
             importances = dict(zip(feature_cols, models['RF'].feature_importances_))
             report["importances"] = {k: round(v, 3) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

        lstm_index_start = len(df_full) - len(X_lstm) + LSTM_TIME_STEPS
        
        historical_ml_scores = pd.Series(ensemble_test_scores_array / ML_SCORE_NORMALIZER, 
                                          index=df_full.iloc[lstm_index_start:].index[-min_test_size:])

        ml_score = ensemble_score_total / ML_SCORE_NORMALIZER 
        
        report["ensemble_score"] = float(round(ensemble_score_total, 1))
        report["ml_score_final"] = float(round(ml_score, 2))
        
        return float(ml_score), report, historical_ml_scores
    
    except Exception as e: 
        report["message"] = f"AI Error: {str(e)[:100]}..."
        return 0, report, historical_ml_scores


# اجرای بک‌تست
def run_backtest(df, historical_ml_scores, ml_conf_threshold, score_threshold, risk_reward=RISK_REWARD_ATR):
    # این تابع از کد پیشرفته قبلی شما است
    df_bt = df.loc[historical_ml_scores.index].copy()
    trades = []
    
    for idx in df_bt.index:
        row = df.loc[idx]
        ml_score = historical_ml_scores.loc[idx]
        current_ml_score = ml_score
        
        if abs(ml_score) < ml_conf_threshold:
            current_ml_score = 0
            
        score = current_ml_score
        
        atr = row['ATR_Value']
        ema20 = row['EMA_20']
        ema50 = row['EMA_50']
        trend = "uptrend" if ema20 > ema50 else "downtrend"
        rsi = row['RSI_14']
        macd_line = row.get(next((c for c in df.columns if c.startswith('MACD_')), ''), 0)
        macd_sig = row.get(next((c for c in df.columns if c.startswith('MACDs_')), ''), 0)
        adx_val = row['ADX']
        support = row['DCL']
        resistance = row['DCU']
        
        div_score = 0
        
        if adx_val > 25: 
            score += 3 if trend == "uptrend" else -3
            score += 1 if macd_line > macd_sig else -1
        else: 
            score += 1 if trend == "uptrend" else -1
            if rsi < 30: score += 3
            elif rsi > 70: score -= 3
            
        dist_to_res = resistance - row['close']
        dist_to_sup = row['close'] - support
        if atr > 0:
            if dist_to_res < (atr * 0.5): score -= 2
            if dist_to_sup < (atr * 0.5): score += 2
        
        score += div_score 
        
        final_signal = "neutral"
        if score >= score_threshold: final_signal = "buy"
        elif score <= -score_threshold: final_signal = "sell"
        
        trade_outcome = row['Target']
        pnl = 0
        
        if final_signal == "buy":
            if trade_outcome == 1: pnl = atr * risk_reward 
            else: pnl = atr * -risk_reward 
        elif final_signal == "sell":
            if trade_outcome == 0: pnl = atr * risk_reward 
            else: pnl = atr * -risk_reward 
        
        if final_signal != "neutral":
            trades.append({"pnl": pnl, "signal": final_signal, "score": score})
            
    df_trades = pd.DataFrame(trades)
    
    if df_trades.empty:
        return {"total_trades": 0, "net_pnl": 0, "win_rate": 0, "max_drawdown": 0, "profit_factor": 0}

    total_trades = len(df_trades)
    wins = (df_trades['pnl'] > 0).sum()
    win_rate = (wins / total_trades) * 100
    
    total_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
    total_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
    profit_factor = round(total_profit / total_loss, 2) if total_loss > 0 else 999.0
    
    df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
    df_trades['peak'] = df_trades['cumulative_pnl'].cummax()
    df_trades['drawdown'] = df_trades['peak'] - df_trades['cumulative_pnl']
    max_drawdown = df_trades['drawdown'].max()
    
    return {
        "total_trades": total_trades,
        "net_pnl": round(df_trades['cumulative_pnl'].iloc[-1], 2),
        "win_rate": round(win_rate, 2),
        "max_drawdown": round(max_drawdown, 2),
        "profit_factor": profit_factor
    }

# اجرای بهینه‌سازی
def run_optimization(df, historical_ml_scores):
    # این تابع از کد پیشرفته قبلی شما است
    best_pnl = -99999.0
    best_params = {}
    optimization_results = []
    
    ml_conf_range = np.arange(0.5, 1.75, 0.25)
    score_thresh_range = np.arange(3.0, 8.0, 1.0) 
    
    for ml_conf in ml_conf_range:
        for score_thresh in score_thresh_range:
            
            results = run_backtest(df.copy(), historical_ml_scores, ml_conf, score_thresh, RISK_REWARD_ATR)
            net_pnl = results.get('net_pnl', -99999.0)
            
            run_summary = {
                "ML_Conf": round(ml_conf, 2),
                "Score_Thresh": round(score_thresh, 1),
                "Total_Trades": results.get('total_trades', 0),
                "Net_PnL": net_pnl,
                "Win_Rate": results.get('win_rate', 0),
                "Max_Drawdown": results.get('max_drawdown', 0),
                "Profit_Factor": results.get('profit_factor', 0),
            }
            optimization_results.append(run_summary)
            
            if net_pnl > best_pnl:
                best_pnl = net_pnl
                best_params = {
                    "ML_CONFIDENCE_THRESHOLD": round(ml_conf, 2),
                    "SIGNAL_SCORE_THRESHOLD": round(score_thresh, 1),
                    "Metrics": results
                }

    return best_params, optimization_results

# ---------------------------------------------------------
# ۴. مسیرهای Flask (ROUTES)
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    """👈 مسیر ریشه: بارگذاری فرانت‌اند HTML."""
    # برای حل مشکل خطای 404 اضافه شد.
    return render_template("index.html")

@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    size_str = request.args.get("size", "2000")
    try: size = int(size_str); size = max(500, min(3000, size))
    except: size = 2000

    df_raw = get_candles(symbol, interval, size=size)
    if df_raw is None or df_raw.empty: return jsonify({"error": "API Error: Could not fetch market data."})
    
    # ۱. محاسبه اندیکاتورها و هدف عملیاتی
    df = calculate_indicators_and_targets(df_raw.copy()) 
    
    # ۲. آموزش AI و دریافت امتیاز کندل آخر (اینجا مدل‌ها آموزش داده می‌شوند)
    ml_score, ml_report, _ = get_ml_prediction(df.copy())
    
    if df.empty: return jsonify({"error": "Not enough processed data for analysis."})
    
    last = df.iloc[-1]
    price = float(last['close'])
    
    # داده‌های اندیکاتور
    rsi = float(last['RSI_14'])
    atr = float(last['ATR_Value'])
    ema20 = float(last['EMA_20'])
    ema50 = float(last['EMA_50'])
    trend = "uptrend" if ema20 > ema50 else "downtrend"
    macd_line = float(last.get(next((c for c in df.columns if c.startswith('MACD_')), ''), 0))
    macd_sig = float(last.get(next((c for c in df.columns if c.startswith('MACDs_')), ''), 0))
    macd_status = "Bullish 🟢" if macd_line > macd_sig else "Bearish 🔴"
    
    adx_val = float(last['ADX'])
    regime = "Ranging (رنج)"
    if adx_val > 25: regime = "Trending (رونددار)"
    if adx_val > 50: regime = "Strong Trend (روند قوی)"
    
    support = float(last['DCL'])
    resistance = float(last['DCU'])
    
    # ۴. محاسبه امتیازدهی دستی
    div_score, div_msg = check_divergence(df)
    news_score, news_text = get_market_sentiment(symbol)
    
    use_htf = request.args.get("use_htf") == "true"
    htf_trend, htf_status, htf_score = "neutral", "غیرفعال", 0
    if use_htf:
        htf_int = TIMEFRAME_MAP.get(interval)
        if htf_int:
            df_h_raw = get_candles(symbol, htf_int, size=100)
            if df_h_raw is not None and not df_h_raw.empty:
                df_h_raw.ta.ema(length=20, append=True)
                df_h_raw.ta.ema(length=50, append=True)
                l_h = df_h_raw.iloc[-1]
                e20_h = float(l_h.get(next((c for c in df_h_raw.columns if c.startswith('EMA_20')), ''), 0))
                e50_h = float(l_h.get(next((c for c in df_h_raw.columns if c.startswith('EMA_50')), ''), 0))
                htf_trend = "uptrend" if e20_h > e50_h else "downtrend"
                htf_status = f"فعال ({htf_int})"
                if trend == htf_trend: htf_score = 2
                else: htf_score = -1

    # ۵. محاسبه امتیاز نهایی (AI + دستی)
    score = 0
    current_ml_score = ml_score
    
    # فیلتر اطمینان AI
    if abs(ml_score) < ML_CONFIDENCE_THRESHOLD:
        current_ml_score = 0
        ml_report["ml_score_final"] = 0
        ml_report["message"] = f"Ensemble: {round(ml_report['ensemble_score'] / 400 * 100 + 50, 1)}% ⚪ Neutral (Low Confidence)"

    score += current_ml_score 

    if adx_val > 25: 
        score += 3 if trend == "uptrend" else -3
        score += 1 if macd_line > macd_sig else -1
    else: 
        score += 1 if trend == "uptrend" else -1
        if rsi < 30: score += 3
        elif rsi > 70: score -= 3
        
    dist_to_res = resistance - price
    dist_to_sup = price - support
    if atr > 0:
        if dist_to_res < (atr * 0.5): score -= 2
        if dist_to_sup < (atr * 0.5): score += 2

    score += div_score 
    score += news_score 
    score += htf_score 

    # ۶. سیگنال نهایی
    final_signal = "neutral"
    if score >= SIGNAL_SCORE_THRESHOLD: final_signal = "buy"
    elif score <= -SIGNAL_SCORE_THRESHOLD: final_signal = "sell"

    # ۷. محاسبه SL/TP هوشمند
    sl, tp = calculate_smart_sl_tp(price, final_signal, atr, support, resistance)
    
    return jsonify({
        "symbol": symbol,
        "interval": interval,
        "price": price,
        "signal": final_signal,
        "score": round(score, 1),
        "setup": {"sl": sl, "tp": tp, "rr_ratio": 2.0, "risk_unit_atr": round(atr * 1.5, 5)},
        "indicators": {
            "trend": "صعودی ↗" if trend == "uptrend" else "نزولی ↘", 
            "rsi": round(rsi, 2),
            "atr": round(atr, 5),
            "macd": macd_status,
            "news": news_text, 
            "htf_status": htf_status,
            "htf_trend": htf_trend,
            "regime": f"{regime} (ADX: {int(adx_val)})",
            "sr_levels": f"S: {round(support, 5)} | R: {round(resistance, 5)}",
            "divergence": div_msg,
            "ai_report": ml_report, 
        }
    })

@app.route("/backtest", methods=["GET"])
def backtest_route():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    size_str = request.args.get("size", "3000")
    try: size = int(size_str); size = max(500, min(3000, size))
    except: size = 3000
    
    ml_conf = request.args.get("ml_conf", ML_CONFIDENCE_THRESHOLD, type=float)
    score_thresh = request.args.get("score_thresh", SIGNAL_SCORE_THRESHOLD, type=float)

    df_raw = get_candles(symbol, interval, size=size)
    if df_raw is None or df_raw.empty: return jsonify({"error": "API Error or not enough data"})
    
    df = calculate_indicators_and_targets(df_raw.copy())
    
    _, ml_report, historical_ml_scores = get_ml_prediction(df.copy())
    
    if historical_ml_scores.empty:
         return jsonify({"error": "Backtest Error: AI model did not generate enough historical predictions. Try increasing data size (max 3000).", "ai_training_summary": ml_report})
    
    results = run_backtest(df, historical_ml_scores, ml_conf, score_thresh, RISK_REWARD_ATR)
    
    return jsonify({
        "symbol": symbol,
        "interval": interval,
        "backtest_range": f"Last {len(historical_ml_scores)} candles (AI Test Set)",
        "backtest_parameters": {
            "risk_reward_atr_target": RISK_REWARD_ATR,
            "target_periods": TARGET_PERIODS,
            "ai_confidence_threshold_used": ml_conf,
            "signal_score_threshold_used": score_thresh
        },
        "performance_metrics": results,
        "ai_training_summary": {
            "test_set_accuracy": f"{ml_report.get('accuracy', 0)}%",
            "feature_importances_top_5": {k: v for k, v in list(ml_report.get('importances', {}).items())[:5]}
        }
    })

@app.route("/optimize", methods=["GET"])
def optimize_route():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    size_str = request.args.get("size", "3000") 
    try: size = int(size_str); size = max(500, min(3000, size))
    except: size = 3000

    df_raw = get_candles(symbol, interval, size=size)
    if df_raw is None or df_raw.empty: return jsonify({"error": "API Error or not enough data"})
    
    df = calculate_indicators_and_targets(df_raw.copy())
    
    start_time = time.time()
    _, ml_report, historical_ml_scores = get_ml_prediction(df.copy())
    ml_train_time = round(time.time() - start_time, 2)
    
    if historical_ml_scores.empty:
         return jsonify({"error": "Optimization Error: AI model did not generate historical predictions for the test set.", "ai_training_summary": ml_report})
    
    start_opt_time = time.time()
    best_params, all_results = run_optimization(df, historical_ml_scores)
    opt_time = round(time.time() - start_opt_time, 2)
    
    return jsonify({
        "symbol": symbol,
        "interval": interval,
        "time_taken": f"ML Training: {ml_train_time}s, Optimization: {opt_time}s",
        "optimized_parameters": ["ML_CONFIDENCE_THRESHOLD", "SIGNAL_SCORE_THRESHOLD"],
        "note": "Optimization based on Maximum Net PnL (Profit Factor is secondary metric).",
        "best_result": best_params,
        "top_5_results_by_pnl": sorted(all_results, key=lambda x: x['Net_PnL'], reverse=True)[:5],
        "top_5_results_by_pf": sorted(all_results, key=lambda x: x['Profit_Factor'], reverse=True)[:5],
        "ai_training_summary": {
             "test_set_accuracy": f"{ml_report.get('accuracy', 0)}%",
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
