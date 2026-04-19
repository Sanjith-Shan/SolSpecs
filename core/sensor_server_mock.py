"""
SolSpecs — Mock Sensor Server
Standalone server for testing the Quest 3 HUD without the UNO Q armband.
Cycles through four scenarios every 60 seconds:
  0–60s:   normal  (green tier)
  60–120s: heating (yellow tier)
  120–180s: critical (red tier)
  180–240s: recovery (green tier, then loops)

Run:  python core/sensor_server_mock.py
"""

import math
import os
import sys
import time

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

app = Flask(__name__)

_START_TIME = time.time()

SCENARIOS = [
    {
        "name": "normal",
        "heart_rate": 78,
        "spo2": 98,
        "skin_temp_c": 36.8,
        "ambient_temp_c": 32.0,
        "ambient_humidity_pct": 55,
        "wbgt": 27.2,
        "heat_tier": "green",
        "hydration": "ok",
        "sweat_level": 120,
        "gsr": 380,
        "accel_x": 0.02,
        "accel_y": -0.03,
        "accel_z": -0.99,
        "fall_detected": False,
        "sun_exposure_min": 5,
        "noise_exposure_min": 10,
    },
    {
        "name": "heating",
        "heart_rate": 105,
        "spo2": 97,
        "skin_temp_c": 37.4,
        "ambient_temp_c": 36.5,
        "ambient_humidity_pct": 68,
        "wbgt": 30.1,
        "heat_tier": "yellow",
        "hydration": "low",
        "sweat_level": 340,
        "gsr": 510,
        "accel_x": 0.08,
        "accel_y": -0.06,
        "accel_z": -0.97,
        "fall_detected": False,
        "sun_exposure_min": 22,
        "noise_exposure_min": 40,
    },
    {
        "name": "critical",
        "heart_rate": 148,
        "spo2": 94,
        "skin_temp_c": 38.7,
        "ambient_temp_c": 42.0,
        "ambient_humidity_pct": 75,
        "wbgt": 34.8,
        "heat_tier": "red",
        "hydration": "critical",
        "sweat_level": 680,
        "gsr": 820,
        "accel_x": 0.15,
        "accel_y": -0.10,
        "accel_z": -0.95,
        "fall_detected": False,
        "sun_exposure_min": 45,
        "noise_exposure_min": 60,
    },
    {
        "name": "recovery",
        "heart_rate": 88,
        "spo2": 97,
        "skin_temp_c": 37.1,
        "ambient_temp_c": 33.0,
        "ambient_humidity_pct": 58,
        "wbgt": 28.0,
        "heat_tier": "green",
        "hydration": "ok",
        "sweat_level": 200,
        "gsr": 420,
        "accel_x": 0.01,
        "accel_y": -0.01,
        "accel_z": -1.00,
        "fall_detected": False,
        "sun_exposure_min": 45,
        "noise_exposure_min": 60,
    },
]

SCENARIO_DURATION = 60  # seconds each


def _current_scenario() -> dict:
    elapsed = (time.time() - _START_TIME) % (SCENARIO_DURATION * len(SCENARIOS))
    idx = int(elapsed // SCENARIO_DURATION)
    base = SCENARIOS[idx]
    phase = (elapsed % SCENARIO_DURATION) / SCENARIO_DURATION  # 0.0 → 1.0 within scenario

    # Add a tiny heartbeat wobble so numbers aren't frozen
    wobble_hr = int(base["heart_rate"] + math.sin(elapsed * 0.8) * 3)
    wobble_spo2 = round(base["spo2"] + math.sin(elapsed * 0.3) * 0.3, 1)

    return {**base, "heart_rate": wobble_hr, "spo2": wobble_spo2}


def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def add_cors(response):
    return _cors(response)


@app.route("/sensors")
def sensors():
    s = _current_scenario()
    thermal_s = int(time.time() - _START_TIME)
    payload = {
        "heart_rate": s["heart_rate"],
        "spo2": s["spo2"],
        "skin_temp_c": s["skin_temp_c"],
        "ambient_temp_c": s["ambient_temp_c"],
        "ambient_humidity_pct": s["ambient_humidity_pct"],
        "wbgt": s["wbgt"],
        "heat_tier": s["heat_tier"],
        "hydration": s["hydration"],
        "sweat_level": s["sweat_level"],
        "gsr": s["gsr"],
        "accel_x": s["accel_x"],
        "accel_y": s["accel_y"],
        "accel_z": s["accel_z"],
        "fall_detected": s["fall_detected"],
        "sun_exposure_min": s["sun_exposure_min"],
        "noise_exposure_min": s["noise_exposure_min"],
        "thermal_exposure_s": thermal_s,
        "timestamp": int(time.time()),
    }
    return jsonify(payload)


@app.route("/status")
def status():
    s = _current_scenario()
    thermal_s = int(time.time() - _START_TIME)
    air_remaining = max(0, 1800 - thermal_s)
    payload = {
        "heat_tier": s["heat_tier"],
        "alerts": _active_alerts(s),
        "thermal_exposure_s": thermal_s,
        "air_remaining_s": air_remaining,
    }
    return jsonify(payload)


@app.route("/fire-config")
def fire_config():
    payload = {
        "wind_direction": 225,
        "wind_speed": 15,
        "grid_size": config.FIRE_GRID_SIZE,
        "tick_rate": config.FIRE_TICK_RATE,
    }
    return jsonify(payload)


_MOCK_FUELS = [
    {"fuel_type": "dead_grass",         "flammability": "EXTREME",  "priority": 1,
     "box_2d": [420, 310, 580, 490], "position": "2 meters ahead",         "action": "Scrape to mineral soil"},
    {"fuel_type": "pine_needle_litter", "flammability": "EXTREME",  "priority": 2,
     "box_2d": [310, 120, 490, 310], "position": "3 meters, slightly left", "action": "Rake and remove"},
    {"fuel_type": "dead_brush",         "flammability": "HIGH",     "priority": 3,
     "box_2d": [210, 510, 450, 760], "position": "4 meters, right",         "action": "Cut and remove"},
    {"fuel_type": "fallen_branches",    "flammability": "HIGH",     "priority": 4,
     "box_2d": [610, 210, 800, 460], "position": "5 meters ahead",          "action": "Cut and remove"},
    {"fuel_type": "chaparral",          "flammability": "MODERATE", "priority": 5,
     "box_2d": [110, 610, 360, 900], "position": "6 meters, far right",     "action": "Cut at ground level"},
]


@app.route("/analyze-fuel", methods=["POST", "OPTIONS"])
def analyze_fuel():
    if request.method == "OPTIONS":
        return "", 204
    return jsonify(_MOCK_FUELS)


@app.route("/hud")
@app.route("/")
def hud():
    hud_dir = os.path.join(os.path.dirname(__file__), "..", "hud")
    return send_from_directory(os.path.abspath(hud_dir), "index.html")


def _active_alerts(s: dict) -> list:
    alerts = []
    tier = s["heat_tier"]
    if tier == "red":
        alerts.append("DANGER — HEAT STRESS CRITICAL — PULL BACK")
    elif tier == "yellow":
        alerts.append("HEAT STRESS WARNING — HYDRATE NOW")
    if s["heart_rate"] > 140:
        alerts.append("HEART RATE CRITICAL")
    if s["spo2"] < 95:
        alerts.append("LOW BLOOD OXYGEN")
    return alerts


if __name__ == "__main__":
    port = config.SENSOR_SERVER_PORT
    print(f"Mock sensor server running on http://0.0.0.0:{port}")
    print(f"Scenario cycle: {' → '.join(s['name'] for s in SCENARIOS)} (every {SCENARIO_DURATION}s)")
    print(f"Test: curl http://localhost:{port}/sensors")
    app.run(host="0.0.0.0", port=port, debug=False)
