"""
SolSpecs — Sensor HTTP Server
Flask app running on the UNO Q. The Quest 3 browser polls this for biometric data.
Reads live state from a StateMachine instance injected at startup via set_state_machine().
"""

import json
import time
import threading
import os

from flask import Flask, jsonify, request, send_from_directory

import config

app = Flask(__name__)

_sm = None          # StateMachine instance
_sm_lock = threading.Lock()


def set_state_machine(sm):
    global _sm
    with _sm_lock:
        _sm = sm


def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _get_sm_snapshot():
    with _sm_lock:
        sm = _sm
    if sm is None:
        return None, None, None
    with sm._sensor_lock:
        mcu = dict(sm._mcu_data)
        glasses = dict(sm._glasses_data)
    tier = sm.current_tier
    return mcu, glasses, tier


@app.after_request
def add_cors(response):
    return _cors(response)


@app.route("/sensors")
def sensors():
    mcu, glasses, tier = _get_sm_snapshot()
    if mcu is None:
        mcu, glasses, tier = {}, {}, "green"

    from core.heat_stress import compute_wbgt_estimate
    temp_c = glasses.get("ambient_temp_c", 28.0) or 28.0
    humidity = glasses.get("ambient_humidity_pct", 60.0) or 60.0
    in_sun = glasses.get("is_direct_sun", False)
    wbgt = compute_wbgt_estimate(temp_c, humidity, in_sun)

    skin_raw = mcu.get("skin_temp_raw", 620)
    # Linear fit: (620, 36.5), (500, 38.5)
    skin_temp = -0.01667 * skin_raw + (36.5 - (-0.01667 * 620))

    sweat_raw = mcu.get("sweat_raw", 0)

    thermal_s = int(time.time() - (_sm._session_start if _sm else time.time()))

    payload = {
        "heart_rate": mcu.get("heart_rate", 72),
        "spo2": mcu.get("spo2", 98),
        "skin_temp_c": round(skin_temp, 1),
        "ambient_temp_c": round(temp_c, 1),
        "ambient_humidity_pct": round(humidity, 1),
        "wbgt": round(wbgt, 1),
        "heat_tier": tier,
        "hydration": "ok",
        "sweat_level": sweat_raw,
        "gsr": mcu.get("gsr_raw", 450),
        "accel_x": mcu.get("accel_x", 0.05),
        "accel_y": mcu.get("accel_y", -0.05),
        "accel_z": mcu.get("accel_z", -0.97),
        "fall_detected": bool(mcu.get("fall_detected", False)),
        "sun_exposure_min": int(_sm.sun_exposure_minutes if _sm else 0),
        "noise_exposure_min": int((_sm.noise_hours_today * 60) if _sm else 0),
        "thermal_exposure_s": thermal_s,
        "timestamp": int(time.time()),
    }
    return jsonify(payload)


@app.route("/status")
def status():
    mcu, glasses, tier = _get_sm_snapshot()
    if mcu is None:
        tier = "green"

    thermal_s = int(time.time() - (_sm._session_start if _sm else time.time()))

    payload = {
        "heat_tier": tier,
        "alerts": [],
        "thermal_exposure_s": thermal_s,
        "air_remaining_s": 1800,  # 30-min SCBA default; decrement logic in Phase 5
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


_FUEL_PROMPT = (
    "You are a wildfire fuel assessment AI assisting a firefighter building a firebreak. "
    "Analyze this ground-level image.\n\n"
    "Identify every visible fuel source that could feed a wildfire. For each, return:\n"
    "- fuel_type: \"dead_grass\", \"pine_needle_litter\", \"dead_brush\", \"fallen_branches\", "
    "\"chaparral\", \"living_brush\", \"small_trees\", or \"large_trees\"\n"
    "- flammability: \"EXTREME\", \"HIGH\", \"MODERATE\", or \"LOW\"\n"
    "- priority: 1-8 (1=clear first: dead grass/needles, 3-4: dead brush/branches, "
    "5-6: chaparral/living brush, 7-8: trees)\n"
    "- box_2d: [ymin, xmin, ymax, xmax] normalized 0-1000\n"
    "- position: natural description (\"3 meters ahead, slightly left\")\n"
    "- action: what to do (\"scrape to mineral soil\", \"cut and remove\", \"fell with chainsaw\")\n\n"
    "Return ONLY a JSON array sorted by priority. No markdown, no explanation."
)


@app.route("/analyze-fuel", methods=["POST", "OPTIONS"])
def analyze_fuel():
    if request.method == "OPTIONS":
        return "", 204

    image_bytes = request.data
    if not image_bytes:
        return jsonify([])

    if not config.GEMINI_API_KEY:
        return jsonify([])

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                _FUEL_PROMPT,
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        data = json.loads(response.text)
        return jsonify(data)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("analyze-fuel error: %s", exc)
        return jsonify([])


@app.route("/hud")
@app.route("/")
def hud():
    hud_dir = os.path.join(os.path.dirname(__file__), "..", "hud")
    return send_from_directory(os.path.abspath(hud_dir), "index.html")


def run(host="0.0.0.0", port=None, debug=False):
    port = port or config.SENSOR_SERVER_PORT
    app.run(host=host, port=port, debug=debug, use_reloader=False)
