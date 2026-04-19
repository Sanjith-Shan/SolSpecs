"""
SolSpecs — Sensor HTTP Server
Flask app that exposes live biometric data, fire config, the HUD, and fuel analysis.

Reads all computed values from a StateMachine via get_current_state().
Fuel analysis uses the AIPipeline's Gemini client when available.
"""

import json
import logging
import os
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

import config

logger = logging.getLogger("SensorServer")

app = Flask(__name__)

_sm = None          # StateMachine instance
_ai = None          # AIPipeline instance (optional)
_sm_lock = threading.Lock()
_ai_lock = threading.Lock()

# ── Gesture / MAYDAY state ────────────────────────────────────────────────────
_emg_events = []            # pending gesture events not yet polled by HUD
_emg_lock = threading.Lock()
_latest_gesture = None      # most recent confirmed gesture label
_gesture_timestamp = None   # epoch float of latest gesture
_mayday_active = False
_mayday_data = None

# ── Remote sensor state (HTTPBridge / UNO Q WiFi) ─────────────────────────────
_remote_sensor_data: dict = {}
_remote_sensor_lock = threading.Lock()
_remote_last_received: float = 0.0   # epoch float; 0 = never received

# ── EMG connection state ──────────────────────────────────────────────────────
_emg_connected: bool = False


def set_emg_connected(connected: bool):
    global _emg_connected
    _emg_connected = connected


def set_state_machine(sm):
    global _sm
    with _sm_lock:
        _sm = sm


def set_ai_pipeline(ai_pipeline):
    global _ai
    with _ai_lock:
        _ai = ai_pipeline


def record_gesture_event(gesture: str):
    """Called by EMGBridge callbacks or POST /emg-event to register a gesture."""
    global _latest_gesture, _gesture_timestamp
    ts = time.time()
    with _emg_lock:
        _latest_gesture = gesture
        _gesture_timestamp = ts
        _emg_events.append({"gesture": gesture, "timestamp": ts})


def record_mayday(data: dict):
    """Called when MAYDAY is triggered via EMG half-clench."""
    global _mayday_active, _mayday_data
    with _emg_lock:
        _mayday_active = True
        _mayday_data = data


def get_remote_sensor_data() -> tuple:
    """
    Returns (data_dict, last_received_epoch) for the HTTPBridge to poll.
    Thread-safe snapshot.
    """
    with _remote_sensor_lock:
        return dict(_remote_sensor_data), _remote_last_received


# ── CORS ─────────────────────────────────────────────────────────────────────

def _cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def add_cors(response):
    return _cors(response)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/sensors")
def sensors():
    with _sm_lock:
        sm = _sm
    with _emg_lock:
        lg = _latest_gesture
        gt = _gesture_timestamp
        ma = _mayday_active
        md = dict(_mayday_data) if _mayday_data else None

    if sm is None:
        state = {
            "heart_rate": 72, "spo2": 98, "skin_temp_c": 36.5,
            "ambient_temp_c": 28.0, "ambient_humidity_pct": 60.0,
            "wbgt": 25.0, "heat_tier": "green", "hydration": "ok",
            "sweat_level": 0,
            "accel_x": 0.0, "accel_y": 0.0, "accel_z": -1.0,
            "fall_detected": False, "sun_exposure_min": 0,
            "noise_exposure_min": 0, "thermal_exposure_s": 0,
            "timestamp": int(time.time()),
        }
    else:
        state = sm.get_current_state()

    state["latest_gesture"] = lg
    state["gesture_timestamp"] = gt
    state["mayday_active"] = ma
    state["mayday_data"] = md
    with _remote_sensor_lock:
        last_rx = _remote_last_received
    state["armband_connected"] = (last_rx > 0 and (time.time() - last_rx) < 5.0)
    state["emg_connected"] = _emg_connected
    return jsonify(state)


@app.route("/status")
def status():
    with _sm_lock:
        sm = _sm
    if sm is None:
        return jsonify({"heat_tier": "green", "alerts": [],
                        "thermal_exposure_s": 0, "air_remaining_s": 1800})
    state = sm.get_current_state()
    thermal_s = state["thermal_exposure_s"]
    return jsonify({
        "heat_tier": state["heat_tier"],
        "alerts": _active_alerts(state),
        "thermal_exposure_s": thermal_s,
        "air_remaining_s": max(0, 1800 - thermal_s),
    })


@app.route("/fire-config")
def fire_config():
    return jsonify({
        "wind_direction": 225,
        "wind_speed": 15,
        "grid_size": config.FIRE_GRID_SIZE,
        "tick_rate": config.FIRE_TICK_RATE,
    })


# ── Remote sensor ingestion (UNO Q → laptop over WiFi) ───────────────────────

@app.route("/sensor-update", methods=["POST", "OPTIONS"])
def sensor_update():
    if request.method == "OPTIONS":
        return "", 204
    global _remote_last_received
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"status": "error", "msg": "no JSON body"}), 400
    with _remote_sensor_lock:
        _remote_sensor_data.update(payload)
        _remote_last_received = time.time()
    return jsonify({"status": "ok", "received_at": _remote_last_received})


# ── EMG gesture events ───────────────────────────────────────────────────────

@app.route("/emg-event", methods=["POST", "OPTIONS"])
def emg_event_post():
    if request.method == "OPTIONS":
        return "", 204
    body = request.get_json(silent=True) or {}
    gesture = body.get("gesture", "").lower()
    if gesture in ("clench", "half_clench"):
        record_gesture_event(gesture)
        if gesture == "half_clench":
            with _sm_lock:
                sm = _sm
            gps = sm._gps_location if sm else None
            record_mayday({
                "heart_rate": sm.get_current_state().get("heart_rate") if sm else None,
                "spo2": sm.get_current_state().get("spo2") if sm else None,
                "skin_temp_c": sm.get_current_state().get("skin_temp_c") if sm else None,
                "heat_tier": sm.get_current_state().get("heat_tier") if sm else None,
                "gps_lat": gps.lat if gps else None,
                "gps_lng": gps.lng if gps else None,
                "timestamp": int(time.time()),
            })
    return jsonify({"status": "ok"})


@app.route("/emg-events")
def emg_events_get():
    global _emg_events
    with _emg_lock:
        pending = list(_emg_events)
        _emg_events.clear()
    return jsonify({"events": pending})


# ── Fuel analysis ─────────────────────────────────────────────────────────────

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
    logger.info("analyze-fuel received %d bytes", len(image_bytes) if image_bytes else 0)
    if not image_bytes:
        return jsonify({"error": "no_image"})

    # Prefer the injected AIPipeline's already-initialised Gemini client
    with _ai_lock:
        ai = _ai

    gemini_client = None
    if ai is not None and getattr(ai, "gemini_client", None) is not None:
        gemini_client = ai.gemini_client
    elif config.GEMINI_API_KEY:
        try:
            from google import genai
            gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        except Exception as exc:
            logger.warning("Could not create Gemini client: %s", exc)

    if gemini_client is None:
        return jsonify({"error": "no_key"})

    from google.genai import types

    # Try primary model first, fall back to lighter model on 503
    _FALLBACK_MODELS = [config.GEMINI_MODEL, "gemini-2.0-flash", "gemini-1.5-flash"]
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        _FUEL_PROMPT,
    ]
    gen_cfg = types.GenerateContentConfig(response_mime_type="application/json")

    last_exc = None
    for model in _FALLBACK_MODELS:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model, contents=contents, config=gen_cfg,
                )
                raw = response.text or ""
                logger.info("Gemini[%s] raw response (%d chars): %.500s", model, len(raw), raw)
                stripped = raw.strip()
                if stripped.startswith("```"):
                    stripped = stripped.split("\n", 1)[-1]
                    if stripped.endswith("```"):
                        stripped = stripped[: stripped.rfind("```")]
                    stripped = stripped.strip()
                result = json.loads(stripped)
                logger.info("Gemini[%s] returned %d fuel items", model, len(result) if isinstance(result, list) else -1)
                return jsonify(result)
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                if "503" in msg or "UNAVAILABLE" in msg:
                    logger.warning("Gemini[%s] attempt %d unavailable, retrying: %s", model, attempt + 1, msg[:120])
                    time.sleep(1.5 * (attempt + 1))
                else:
                    logger.error("Gemini[%s] error: %s", model, msg)
                    break  # non-transient error, skip to next model

    logger.error("All Gemini models failed: %s", last_exc)
    return jsonify({"error": str(last_exc)})


# ── HUD static files ──────────────────────────────────────────────────────────

def _hud_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hud"))


@app.route("/")
@app.route("/hud")
def hud_index():
    return send_from_directory(_hud_dir(), "index.html")


@app.route("/hud/<path:filename>")
def hud_static(filename):
    return send_from_directory(_hud_dir(), filename)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _active_alerts(state: dict) -> list:
    alerts = []
    tier = state.get("heat_tier", "green")
    if tier == "red":
        alerts.append("DANGER — HEAT STRESS CRITICAL — PULL BACK")
    elif tier == "orange":
        alerts.append("HEAT STRESS WARNING — TAKE SHADE BREAK")
    elif tier == "yellow":
        alerts.append("HEAT STRESS CAUTION — HYDRATE NOW")
    if state.get("heart_rate", 0) > 140:
        alerts.append("HEART RATE CRITICAL")
    if state.get("spo2", 100) < 95:
        alerts.append("LOW BLOOD OXYGEN")
    if state.get("fall_detected"):
        alerts.append("FALL DETECTED")
    return alerts


# ── Entry point ───────────────────────────────────────────────────────────────

def run(host: str = "0.0.0.0", port: int = None,
        ssl_context=None, debug: bool = False):
    port = port or config.SENSOR_SERVER_PORT
    app.run(host=host, port=port, ssl_context=ssl_context,
            debug=debug, use_reloader=False)
