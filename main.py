from flask import Flask, render_template, request, jsonify
import requests
import urllib.parse
import os

app = Flask(__name__)

API_KEY = "df521019db9f44899bfb172fdce6b454"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["GET"])
def analyze():
    symbol = request.args.get("symbol")
    interval = request.args.get("interval")

    if not symbol or not interval:
        return jsonify({"error": "missing parameters"})

    encoded_symbol = urllib.parse.quote(symbol, safe='')

    url = (
        f"https://api.twelvedata.com/time_series?"
        f"symbol={encoded_symbol}&interval={interval}&apikey={API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10).json()
    except Exception as e:
        return jsonify({"error": "request failed", "details": str(e)})

    if "values" not in response:
        return jsonify({"error": "cannot fetch data", "api_response": response})

    try:
        last = float(response["values"][0]["close"])
        prev = float(response["values"][1]["close"])
    except:
        return jsonify({"error": "invalid data structure", "api_response": response})

    direction = "📈 صعودی" if last > prev else "📉 نزولی"

    return jsonify({
        "direction": direction,
        "last_price": last,
        "previous_price": prev
    })






