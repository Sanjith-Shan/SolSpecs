"""
SolSpecs — Phone GPS Server
Runs in Termux on Android. Serves GPS coordinates via HTTP.

Install in Termux:
    pkg install python
    pip install flask

Run:
    python gps_server.py

Then on the UNO Q set:
    PHONE_GPS_URL=http://<phone-ip>:5000
"""

import subprocess
import threading
import time

from flask import Flask, jsonify

app = Flask(__name__)

_location = {"latitude": None, "longitude": None, "accuracy": None, "timestamp": None}
_lock = threading.Lock()


def _update_gps_termux():
    """Poll GPS via Termux:API termux-location command."""
    import json
    while True:
        try:
            result = subprocess.run(
                ["termux-location", "-p", "gps", "-r", "once"],
                capture_output=True, text=True, timeout=15,
            )
            data = json.loads(result.stdout)
            with _lock:
                _location["latitude"] = data.get("latitude")
                _location["longitude"] = data.get("longitude")
                _location["accuracy"] = data.get("accuracy")
                _location["timestamp"] = time.time()
        except Exception as e:
            pass  # GPS unavailable — keep last known fix
        time.sleep(5)


@app.route("/location")
def location():
    with _lock:
        loc = dict(_location)
    if loc["latitude"] is None:
        return jsonify({"error": "GPS fix not yet available"}), 503
    return jsonify(loc)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    t = threading.Thread(target=_update_gps_termux, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000)
