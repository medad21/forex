from flask import Flask, request, render_template, jsonify
import requests
import urllib.parse

app = Flask(__name__)

API_KEY = "YOUR_API_KEY"   # اینجا API Key که از TwelveData گرفتی را قرار بده

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    symbol   = request.form.get("symbol")
    interval = request.form.get("interval")

    # تبدیل EUR/USD به EUR%2FUSD
    encoded_symbol = urllib.parse.quote(symbol, safe='')

    url = f"https://api.twelvedata.com/time_series?symbol={encoded_symbol}&interval={interval}&apikey={API_KEY}"

    response = requests.get(url).json()

    # بررسی خطای API
    if "values" not in response:
        return jsonify({
            "error": "cannot fetch data",
            "api_response": response
        })

    # داده سالم دریافت شد
    last = float(response["values"][0]["close"])
    prev = float(response["values"][1]["close"])

    direction = "📈 صعودی" if last > prev else "📉 نزولی"

    return jsonify({
        "direction": direction,
        "last_price": last,
        "previous_price": prev
    })


if __name__ == "__main__":
    app.run()
