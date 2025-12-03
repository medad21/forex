from flask import Flask, request, jsonify
import os
import predict  # فایل predict.py را در کنار این فایل داشته باشید

app = Flask(__name__)

# بارگذاری مدل فقط یکبار هنگام شروع سرور
bot = predict.TradingAI()

@app.route('/analyze', methods=['GET'])
def analyze():
    symbol = request.args.get('symbol', 'EURUSD')
    try:
        result = bot.predict(symbol)
        if "error" in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "AlgoTrading Server is Running!"

if __name__ == '__main__':
    # پورت 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
