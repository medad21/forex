from flask import Flask, request, jsonify, render_template
import requests
import pandas as pd
import pandas_ta as ta
import numpy as np
# 🔑 کتابخانه‌های مورد نیاز برای مدل‌های Ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from xgboost import XGBClassifier 
from sklearn.preprocessing import StandardScaler 
from sklearn.utils import class_weight 

# 🔑 کتابخانه‌های مورد نیاز برای مدل LSTM (نیازمند نصب tensorflow/keras)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------
# 🔑 API KEYS - کلیدهای API
# ---------------------------------------------------------
API_KEY_TWELVEDATA = "df521019db9f44899bfb172fdce6b454" 
API_KEY_ALPHA = "W1L3K1JN4F77T9KL"              
API_KEY_FINNHUB = "d4gd4r9r01qm5b352il0d4gd4r9r01qm5b352ilg"                  
# ---------------------------------------------------------

TIMEFRAME_MAP = { "15min": "1h", "1h": "4h", "4h": "1day" }
LSTM_TIME_STEPS = 10 
# آستانه اطمینان: امتیاز AI باید حداقل 1.0 (معادل 10 امتیاز اطمینان جمعی) باشد تا در سیگنال نهایی محاسبه شود.
ML_CONFIDENCE_THRESHOLD = 1.0 

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

# دریافت دیتا با حجم بالا (اندازه انتخابی)
def get_candles(symbol, interval, size=2000):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY_TWELVEDATA}&outputsize={size}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if "values" not in data: return None
        df = pd.DataFrame(data["values"])
        for c in ['open', 'high', 'low', 'close']: df[c] = pd.to_numeric(df[c])
        df = df.iloc[::-1].reset_index(drop=True)
        return df
    except: return None

# --- توابع کمکی (بدون تغییر) ---
def check_market_regime(df):
    if 'ADX_14' not in df.columns: df.ta.adx(length=14, append=True)
    last = df.iloc[-1]
    adx_col = next((c for c in df.columns if c.startswith('ADX')), None)
    adx_val = last.get(adx_col, 0)
    regime = "Ranging (رنج)"
    if adx_val > 25: regime = "Trending (رونددار)"
    if adx_val > 50: regime = "Strong Trend (روند قوی)"
    return regime, adx_val

def get_sr_levels(df):
    df.ta.donchian(lower_length=20, upper_length=20, append=True)
    last = df.iloc[-1]
    sup_col = next((c for c in df.columns if c.startswith('DCL')), None)
    res_col = next((c for c in df.columns if c.startswith('DCU')), None)
    support = last.get(res_col, 0)
    resistance = last.get(sup_col, 0)
    return float(support), float(resistance) 

def check_divergence(df):
    if 'RSI_14' not in df.columns: df.ta.rsi(length=14, append=True)
    subset = df.iloc[-15:].reset_index(drop=True)
    price, rsi = subset['close'], subset['RSI_14']
    price_high_idx, price_low_idx = price.idxmax(), price.idxmin()
    curr_price, curr_rsi = price.iloc[-1], rsi.iloc[-1]
    score, msg = 0, "بدون واگرایی"
    if price_high_idx < 14 and curr_price > price[price_high_idx] and curr_rsi < rsi[price_high_idx]: msg, score = "Bearish Div 📉", -3
    if price_low_idx < 14 and curr_price < price[price_low_idx] and curr_rsi > rsi[price_low_idx]: msg, score = "Bullish Div 📈", 3
    return score, msg

# تابع بررسی Actionable Target (بدون تغییر)
def check_target(row, df_full, periods, rr_atr):
    idx = row.name
    current_close = row['close']
    atr = row['ATR_Value']
    
    if idx + periods >= len(df_full): return -1
    
    future_data = df_full.loc[idx+1 : idx+periods]
    if future_data.empty: return -1
    
    tp_buy = current_close + (atr * rr_atr)
    sl_buy = current_close - (atr * rr_atr)
    tp_sell = current_close - (atr * rr_atr)
    sl_sell = current_close + (atr * rr_atr)

    tp_hit_buy_idx = future_data[future_data['high'] >= tp_buy].index.min()
    sl_hit_buy_idx = future_data[future_data['low'] <= sl_buy].index.min()
    
    tp_hit_sell_idx = future_data[future_data['low'] <= tp_sell].index.min()
    sl_hit_sell_idx = future_data[future_data['high'] >= sl_sell].index.min()

    is_buy_success = (pd.notna(tp_hit_buy_idx) and 
                      (pd.isna(sl_hit_buy_idx) or tp_hit_buy_idx < sl_hit_buy_idx) and
                      (pd.isna(tp_hit_sell_idx) or tp_hit_buy_idx < tp_hit_sell_idx) and 
                      (pd.isna(sl_hit_sell_idx) or tp_hit_buy_idx < sl_hit_sell_idx))

    is_sell_success = (pd.notna(tp_hit_sell_idx) and 
                       (pd.isna(sl_hit_sell_idx) or tp_hit_sell_idx < sl_hit_sell_idx) and
                       (pd.isna(tp_hit_buy_idx) or tp_hit_sell_idx < tp_hit_buy_idx) and 
                       (pd.isna(sl_hit_buy_idx) or tp_hit_sell_idx < sl_hit_buy_idx))

    if is_buy_success:
        return 1
    elif is_sell_success:
        return 0
        
    return -1

# تابع کمکی برای آماده‌سازی داده سه‌بعدی LSTM (بدون تغییر)
def create_lstm_dataset(X_scaled_df, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X_scaled_df) - time_steps):
        v = X_scaled_df.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- سطح ۴: یادگیری ماشین (مدل ترکیبی با Class Weighting و MTF) ---
def get_ml_prediction(df, size):
    report = {
        "accuracy": 0, "importances": {}, "message": "AI: خنثی",
        "ensemble_score": 0, "ml_score_final": 0, "individual_results": {}
    }
    
    models = {
        'RF': RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42, class_weight="balanced"),
        'XGB': XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1), 
        'LR': LogisticRegression(solver='liblinear', random_state=42, class_weight="balanced"),
        'LSTM': None 
    }

    try:
        # Feature Engineering (ویژگی‌های پایه)
        df['Returns'] = df['close'].pct_change()
        df['ADX'] = df.ta.adx(length=14)[df.ta.adx(length=14).columns[0]]
        df['Volatility'] = df['high'] - df['low']
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['Hour'] = df['datetime'].dt.hour
        df['DayOfWeek'] = df['datetime'].dt.dayofweek
        df['HV_20'] = df['Returns'].rolling(window=20).std()
        df['ATR_Value'] = df.ta.atr(length=14) 
        
        # 🔑 ویژگی‌های چندزمانی شبیه‌سازی شده (Simulated MTF)
        # دیدگاه سریع و کند برای RSI
        df['RSI_14'] = df.ta.rsi(length=14)
        df['RSI_6'] = df.ta.rsi(length=6) 
        
        # دیدگاه کوتاه‌مدت و بلندمدت برای روند (EMA)
        df['EMA_20'] = df.ta.ema(length=20)
        df['EMA_50'] = df.ta.ema(length=50)
        df['EMA_100'] = df.ta.ema(length=100)
        df['EMA_Diff_Fast'] = df['EMA_20'] - df['EMA_50']
        df['EMA_Diff_Slow'] = df['EMA_50'] - df['EMA_100']
        
        # --- مهندسی هدف عملیاتی ---
        RISK_REWARD_ATR = 1.5
        TARGET_PERIODS = 5
        df['Target'] = df.apply(check_target, axis=1, args=(df, TARGET_PERIODS, RISK_REWARD_ATR)) 

        # تمیزکاری داده‌ها
        df = df[df['Target'] != -1]
        
        feature_cols = ['RSI_14', 'RSI_6', 'ADX', 'EMA_Diff_Fast', 'EMA_Diff_Slow', 'Returns', 'Volatility', 'Hour', 'DayOfWeek', 'HV_20']
        df = df.dropna(subset=feature_cols + ['Target'])

        if len(df) < 50: 
            report["message"] = f"AI: دیتای کافی برای آموزش Actionable ({len(df)}/{size})"
            return 0, report

        X = df[feature_cols].copy()
        Y = df['Target'].copy().astype(int)
        
        # --- مقیاس‌دهی ویژگی‌ها (Scaling) ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

        # --- آماده‌سازی داده‌ها ---
        test_size_2d = max(100, int(len(df) * 0.1)) 
        X_train_2d = X_scaled_df.iloc[:-test_size_2d]
        Y_train_2d = Y.iloc[:-test_size_2d]
        X_test_2d = X_scaled_df.iloc[-test_size_2d:-1]
        Y_test_2d = Y.iloc[-test_size_2d:-1]
        
        # محاسبه وزن کلاس‌ها 
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(Y_train_2d),
            y=Y_train_2d
        )
        class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
        sample_weights_xgb = Y_train_2d.apply(lambda x: class_weights_dict[x]).values
        
        # داده‌های 3D (برای LSTM)
        X_lstm, Y_lstm = create_lstm_dataset(X_scaled_df, Y, LSTM_TIME_STEPS)
        
        if len(X_lstm) < 50 + LSTM_TIME_STEPS:
             report["message"] = "AI: دیتای 3D کافی نیست"
             return 0, report
            
        test_size_3d = max(100, int(len(X_lstm) * 0.1))
        X_train_lstm = X_lstm[:-test_size_3d]
        Y_train_lstm = Y_lstm[:-test_size_3d]
        X_test_lstm = X_lstm[-test_size_3d:-1]
        Y_test_lstm = Y_lstm[-test_size_3d:-1]

        # داده ورودی نهایی
        last_features = X.iloc[-1].to_frame().T
        last_features_scaled_2d = scaler.transform(last_features) 
        last_window_data = X_scaled_df.iloc[-LSTM_TIME_STEPS:].values
        last_features_scaled_3d = last_window_data.reshape(1, LSTM_TIME_STEPS, len(feature_cols)) 
        
        if len(np.unique(Y_train_2d)) < 2: return 0, report
        
        ensemble_score_total = 0
        test_predictions = {}
        
        # --- آموزش و پیش‌بینی مدل‌های 2D (RF, XGB, LR) ---
        for name in ['RF', 'LR', 'XGB']:
            model = models[name]
            if name == 'XGB':
                model.fit(X_train_2d, Y_train_2d, sample_weight=sample_weights_xgb)
            else:
                model.fit(X_train_2d, Y_train_2d)
            
            test_pred = model.predict(X_test_2d)
            test_predictions[name] = test_pred
            
            prob_p = model.predict_proba(last_features_scaled_2d)[0][1] 
            confidence_score = (prob_p - 0.5) * 100 
            ensemble_score_total += confidence_score
            
            report["individual_results"][name] = {
                'prob': round(float(prob_p * 100), 1),
                'score': round(float(confidence_score), 1),
                'msg': 'Buy' if confidence_score > 0 else 'Sell'
            }
            
        # --- آموزش و پیش‌بینی مدل 3D (LSTM) ---
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(LSTM_TIME_STEPS, len(feature_cols))))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(1, activation='sigmoid'))
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        models['LSTM'] = lstm_model
        
        lstm_model.fit(X_train_lstm, Y_train_lstm, epochs=5, batch_size=32, verbose=0, class_weight=class_weights_dict)
        
        test_pred_lstm_prob = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        test_pred_lstm = (test_pred_lstm_prob > 0.5).astype(int)
        test_predictions['LSTM'] = test_pred_lstm
        
        prob_p_lstm = lstm_model.predict(last_features_scaled_3d, verbose=0)[0][0]
        confidence_score_lstm = (prob_p_lstm - 0.5) * 100
        ensemble_score_total += confidence_score_lstm
        
        report["individual_results"]['LSTM'] = {
            'prob': round(float(prob_p_lstm * 100), 1),
            'score': round(float(confidence_score_lstm), 1),
            'msg': 'Buy' if confidence_score_lstm > 0 else 'Sell'
        }
        
        # --- محاسبه دقت Ensemble ---
        min_test_size = min(len(Y_test_2d), len(Y_test_lstm))
        
        total_predictions = np.zeros(min_test_size)
        total_predictions += test_predictions['RF'][:min_test_size]
        total_predictions += test_predictions['XGB'][:min_test_size]
        total_predictions += test_predictions['LR'][:min_test_size]
        total_predictions += test_predictions['LSTM'][:min_test_size]
        
        majority_pred = (total_predictions > 2).astype(int)
        ensemble_accuracy = (majority_pred == Y_test_lstm[:min_test_size]).mean() 
        report["accuracy"] = float(round(ensemble_accuracy * 100, 2))

        if 'RF' in models:
            importances = dict(zip(feature_cols, models['RF'].feature_importances_))
            report["importances"] = {k: round(float(v), 3) for k, v in sorted(importances.items(), key=lambda item: item[1], reverse=True)}

        ML_SCORE_NORMALIZER = 40.0 
        ml_score = ensemble_score_total / ML_SCORE_NORMALIZER 

        final_prob_average = ensemble_score_total / (len(models) * 100) + 0.5 
        
        final_message = f"Ensemble: {round(final_prob_average * 100, 1)}%"
        if final_prob_average > 0.6: final_message += " 🚀 Strong Buy (Actionable)"
        elif final_prob_average < 0.4: final_message += " 🔻 Strong Sell (Actionable)"
        else: final_message += " ⚪ Neutral"

        report["ensemble_score"] = float(round(ensemble_score_total, 1))
        report["ml_score_final"] = float(round(ml_score, 1))
        report["message"] = final_message

        return float(ml_score), report
    
    except Exception as e: 
        report["message"] = f"AI Error (MTF/Conf. Threshold): {str(e)[:60]}..."
        return 0, report

# --- توابع کمکی (بدون تغییر) ---
def get_market_sentiment(symbol):
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
            if "Bullish" in label: sentiment_text = "🟢 اخبار مثبت (Bullish)"
            elif "Bearish" in label: sentiment_text = "🔴 اخبار منفی (Bearish)"
            sentiment_score = float(data["feed"][0].get("overall_sentiment_score", 0)) * 2
            return sentiment_score, sentiment_text
    except: pass
    return sentiment_score, sentiment_text

def calculate_smart_sl_tp(entry, signal, atr, support, resistance):
    if not atr or np.isnan(atr): return None, None
    sl_mult, rr = 1.5, 2.0
    if signal == "buy":
        sl_base = support if (entry - support) < (atr * 2.0) and support != 0 else (entry - atr * sl_mult)
        tp = entry + ((entry - sl_base) * rr)
    else:
        sl_base = resistance if (resistance - entry) < (atr * 2.0) and resistance != 0 else (entry + atr * sl_mult)
        tp = entry - ((sl_base - entry) * rr)
    return round(float(sl_base), 5) if sl_base is not None else None, round(float(tp), 5) if tp is not None else None

# =========================================================
# MAIN ROUTE (با Confidence Thresholding)
# =========================================================
@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")
    use_htf = request.args.get("use_htf") == "true"
    
    size_str = request.args.get("size", "2000")
    try: size = int(size_str); size = max(100, min(2500, size))
    except: size = 2000

    df = get_candles(symbol, interval, size=size)
    if df is None or df.empty: return jsonify({"error": "API Error"})

    # محاسبه اندیکاتورهای مورد نیاز برای چک‌های دستی و ATR
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.macd(append=True)

    last = df.iloc[-1]
    price = float(last['close'])
    
    rsi = float(last.get(next((c for c in df.columns if c.startswith('RSI')), ''), 50))
    atr = float(last.get(next((c for c in df.columns if c.startswith('ATRr')), ''), 0))
    ema20 = float(last.get(next((c for c in df.columns if c.startswith('EMA_20')), ''), price))
    ema50 = float(last.get(next((c for c in df.columns if c.startswith('EMA_50')), ''), price))
    trend = "uptrend" if ema20 > ema50 else "downtrend"
    macd_line = float(last.get(next((c for c in df.columns if c.startswith('MACD_')), ''), 0))
    macd_sig = float(last.get(next((c for c in df.columns if c.startswith('MACDs_')), ''), 0))
    macd_status = "Bullish 🟢" if macd_line > macd_sig else "Bearish 🔴"
    
    regime, adx_val = check_market_regime(df)
    support, resistance = get_sr_levels(df)
    div_score, div_msg = check_divergence(df)
    ml_score, ml_report = get_ml_prediction(df, size) 
    news_score, news_text = get_market_sentiment(symbol)
    
    # ... (بررسی تایم فریم بالاتر - بدون تغییر)
    htf_trend, htf_status = "neutral", "غیرفعال"
    if use_htf:
        htf_int = TIMEFRAME_MAP.get(interval)
        if htf_int:
            df_h = get_candles(symbol, htf_int, size=100)
            if df_h is not None:
                df_h.ta.ema(length=20, append=True)
                df_h.ta.ema(length=50, append=True)
                l_h = df_h.iloc[-1]
                e20_h = float(l_h.get(next((c for c in df_h.columns if c.startswith('EMA_20')), ''), 0))
                e50_h = float(l_h.get(next((c for c in df_h.columns if c.startswith('EMA_50')), ''), 0))
                htf_trend = "uptrend" if e20_h > e50_h else "downtrend"
                htf_status = f"فعال ({htf_int})"

    score = 0
    
    # 🔑 اعمال Confidence Thresholding
    current_ml_score = ml_score
    if abs(ml_score) < ML_CONFIDENCE_THRESHOLD:
        # اگر امتیاز AI کافی نبود، آن را صفر در نظر می‌گیریم تا تأثیری بر سیگنال نهایی نگذارد.
        current_ml_score = 0
        ml_report["ml_score_final"] = 0
        ml_report["message"] = f"Ensemble: {round(ml_report['ensemble_score'] / 400 * 100 + 50, 1)}% ⚪ Neutral (Low Confidence)"

    score += current_ml_score

    # --- محاسبه امتیاز سیگنال دستی/سنتی (بدون تغییر) ---
    if adx_val > 25: 
        score += 3 if trend == "uptrend" else -3
        score += 1 if macd_line > macd_sig else -1
    else: 
        score += 1 if trend == "uptrend" else -1
        if rsi < 30: score += 3
        elif rsi > 70: score -= 3
        
    dist_to_res = resistance - price
    dist_to_sup = price - support
    if dist_to_res < (atr * 0.5): score -= 2
    if dist_to_sup < (atr * 0.5): score += 2

    score += div_score
    score += news_score
    
    if use_htf and htf_trend != "neutral":
        if trend == htf_trend: score += 2
        else: score -= 1

    final_signal = "neutral"
    if score >= 5: final_signal = "buy"
    elif score <= -5: final_signal = "sell"

    sl, tp = calculate_smart_sl_tp(price, final_signal, atr, support, resistance)

    return jsonify({
        "price": price,
        "signal": final_signal,
        "score": round(score, 1),
        "setup": {"sl": sl, "tp": tp},
        "indicators": {
            "trend": "صعودی ↗" if trend == "uptrend" else "نزولی ↘", 
            "rsi": round(rsi, 2),
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
