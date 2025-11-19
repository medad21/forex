from flask import Flask, request, render_template, jsonify
import requests
import urllib.parse

app = Flask(__name__)

API_KEY = "YOUR_API_KEY"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    symbol = request.form.get("symbol")
    interval = request.form.get("interval")

    encoded_symbol = urllib.parse.quote(symbol, safe='')

    url = f"https://api.twelvedata.com/time_series?symbol={encoded_symbol}&interval={interval}&apikey={API_KEY}"
    response = requests.get(url).json()

    if "values" not in response:
        return jsonify({"error": "cannot fetch data", "api_response": response})

    last = float(response["values"][0]["close"])
    prev = float(response["values"][1]["close"])

    direction = "📈 صعودی" if last > prev else "📉 نزولی"

    return jsonify({
        "direction": direction,
        "last_price": last,
        "previous_price": prev
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


