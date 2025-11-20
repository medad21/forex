from flask import Flask, request, jsonify, render_template
import requests
import numpy as np
import random 

app = Flask(__name__)

# کلید API شما (برای تست)
API_KEY = "df521019db9f44899bfb172fdce6b454"

@app.route("/")
def index():
    return render_template("index.html")

# --- توابع کمکی جدید ---

def calculate_support_resistance(highs, lows, current_price):
    """
    سطوح حمایت و مقاومت داینامیک را بر اساس 50 کندل اخیر محاسبه می‌کند.
    """
    # بازه زمانی برای پیدا کردن سقف و کف (مثلا 50 ساعت اخیر)
    lookback = 50
    
    # اطمینان از اینکه دیتا کافی داریم
    if len(highs) < lookback: lookback = len(highs)

    recent_high = np.max(highs[-lookback:])
    recent_low = np.min(lows[-lookback:])
    
    # فاصله قیمت فعلی تا سطوح
    dist_to_resistance = recent_high - current_price
    dist_to_support = current_price - recent_low
    
    return recent_high, recent_low, dist_to_resistance, dist_to_support

def get_market_sentiment():
    """
    این تابع وضعیت روانی بازار (اخبار) را شبیه‌سازی می‌کند.
    در پروژه واقعی، اینجا باید به یک API خبری وصل شوید.
    """
    # تولید یک عدد تصادفی بین -1 (اخبار بد) تا +1 (اخبار خوب)
    # با وزن‌دهی به سمت خنثی برای واقع‌گرایی
    sentiment_score = random.gauss(0, 0.4) 
    
    sentiment_text = "خنثی (بدون خبر مهم)"
    if sentiment_score > 0.4:
        sentiment_text = "مثبت (اخبار صعودی) 🐂"
    elif sentiment_score < -0.4:
        sentiment_text = "منفی (اخبار نزولی) 🐻"
        
    return sentiment_score, sentiment_text

# -----------------------

@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol", "EUR/USD")
    interval = request.args.get("interval", "1h")

    # 1. دریافت دیتا از API
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&apikey={API_KEY}&outputsize=100"
    response = requests.get(url)
    data = response.json()

    if "values" not in data:
        return jsonify({"error": "no data found", "details": data})

    try:
        # پردازش دیتا (معکوس کردن آرایه برای ترتیب زمانی صحیح)
        values = data["values"][::-1]
        close = np.array([float(v["close"]) for v in values])
        high = np.array([float(v["high"]) for v in values])
        low = np.array([float(v["low"]) for v in values])
    except Exception as e:
        return jsonify({"error": "data parsing error"})

    # 2. محاسبات اندیکاتورها
    
    # EMA Trend
    def ema(series, period):
        k = 2 / (period + 1)
        ema_arr = np.zeros_like(series)
        ema_arr[0] = series[0]
        for i in range(1, len(series)):
            ema_arr[i] = series[i] * k + ema_arr[i - 1] * (1 - k)
        return ema_arr

    ema20 = ema(close, 20)
    ema50 = ema(close, 50)
    trend = "uptrend" if ema20[-1] > ema50[-1] else "downtrend"

    # RSI Calculation
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rs = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))

    # ATR Calculation
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = np.mean(tr[-14:])

    # 3. محاسبات پیشرفته (جدید)
    
    last_price = close[-1]
    
    # الف) سطوح حمایت و مقاومت
    res_level, sup_level, dist_res, dist_sup = calculate_support_resistance(high, low, last_price)
    
    # ب) سنتیمنت بازار
    news_score, news_text = get_market_sentiment()

    # 4. سیستم امتیازدهی ترکیبی (Logic Engine)
    score = 0

    # امتیاز تکنیکال (روند)
    if trend == "uptrend": score += 2
    else: score -= 2

    # امتیاز مومنتوم (RSI)
    if rsi < 30: score += 3      # اشباع فروش
    elif rsi > 70: score -= 3    # اشباع خرید
    
    # امتیاز پرایس اکشن (حمایت/مقاومت)
    # اگر قیمت خیلی نزدیک به کف (حمایت) است، شانس برگشت به بالا زیاد است
    if dist_sup < (atr * 1.5): score += 2 
    # اگر قیمت خیلی نزدیک به سقف (مقاومت) است، شانس برگشت به پایین زیاد است
    if dist_res < (atr * 1.5): score -= 2 

    # امتیاز خبری
    if news_score > 0.4: score += 2
    elif news_score < -0.4: score -= 2

    # تصمیم نهایی بر اساس جمع امتیازات
    final_signal = "neutral"
    if score >= 4:  # سخت‌گیری بیشتر برای خرید
        final_signal = "buy"
    elif score <= -4: # سخت‌گیری بیشتر برای فروش
        final_signal = "sell"

    # 5. مدیریت پوزیشن (TP/SL) هوشمند
    entry = float(last_price)
    sl = None
    tp = None

    if final_signal == "buy":
        # استاپ لاس دقیقاً زیر حمایت محاسبه شده قرار می‌گیرد (امن‌تر)
        sl = sup_level - (atr * 0.5) 
        risk = entry - sl
        # ریسک به ریوارد 1 به 1.5
        tp = entry + (risk * 1.5) 
        
    elif final_signal == "sell":
        # استاپ لاس دقیقاً بالای مقاومت محاسبه شده
        sl = res_level + (atr * 0.5)
        risk = sl - entry
        tp = entry - (risk * 1.5)

    # خروجی نهایی
    return jsonify({
        "symbol": symbol,
        "price": round(entry, 5),
        "signal": final_signal,
        "score": score,
        "trend": trend,
        "indicators": {
            "rsi": round(float(rsi), 2),
            "atr": round(float(atr), 5),
            "sentiment": news_text
        },
        "levels": {
            "support": round(sup_level, 5),
            "resistance": round(res_level, 5)
        },
        "setup": {
            "sl": round(sl, 5) if sl else None,
            "tp": round(tp, 5) if tp else None
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
