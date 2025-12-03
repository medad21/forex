from flask import Flask, request, jsonify
import predict  

app = Flask(__name__)

# بارگذاری مدل فقط یکبار هنگام شروع سرور
# این کار کل کلاس TradingAI و مدل‌ها را بارگذاری می‌کند
try:
    bot = predict.TradingAI()
    print("✅ Trading AI engine loaded successfully.")
except Exception as e:
    print(f"❌ FATAL: Failed to load models. Did you run train.py? Error: {e}")
    bot = None

@app.route('/analyze', methods=['GET'])
def analyze():
    if bot is None:
        return jsonify({"error": "AI Engine failed to initialize."}), 500
        
    symbol = request.args.get('symbol', 'EURUSD')
    try:
        # فراخوانی متد پیش‌بینی در کلاس TradingAI
        result = bot.predict(symbol)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        # برای اشکال‌زدایی بهتر
        import traceback
        traceback.print_exc() 
        return jsonify({"error": f"Server Error during prediction: {str(e)}"}), 500

@app.route('/')
def home():
    return "AlgoTrading Server (v2.0) is Running!"

if __name__ == '__main__':
    # در محیط محلی
    app.run(host='0.0.0.0', port=5000, debug=False)
