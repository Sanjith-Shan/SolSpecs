"""
Tests for core/mcu_bridge.py — HTTPBridge data formatting.
Run with:  pytest tests/test_mcu_bridge.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.mcu_bridge import HTTPBridge


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bridge():
    """HTTPBridge backed by an in-memory dict; returns (bridge, store)."""
    store = {"data": {}, "ts": 0.0}

    def getter():
        return dict(store["data"]), store["ts"]

    bridge = HTTPBridge(get_data_fn=getter)
    return bridge, store


def _dispatch(raw: dict):
    """Run HTTPBridge._dispatch directly and capture the mcu_frame callback."""
    received = []
    bridge, _ = _make_bridge()
    bridge.on_sensor_data = received.append
    bridge._dispatch(raw)
    return received


def _dispatch_glasses(raw: dict):
    """Run HTTPBridge._dispatch and capture the glasses callback."""
    received = []
    bridge, _ = _make_bridge()
    bridge.on_glasses_data = received.append
    bridge._dispatch(raw)
    return received


# ── MCU frame field mapping ───────────────────────────────────────────────────

class TestHTTPBridgeDispatch:

    def test_heart_rate_passed_through(self):
        frames = _dispatch({"heart_rate": 88, "spo2": 98})
        assert frames[0]["heart_rate"] == 88

    def test_spo2_passed_through(self):
        frames = _dispatch({"heart_rate": 72, "spo2": 95})
        assert frames[0]["spo2"] == 95

    def test_accel_x_passed_through(self):
        frames = _dispatch({"accel_x": 0.12, "accel_y": 0.0, "accel_z": -1.0})
        assert abs(frames[0]["accel_x"] - 0.12) < 1e-6

    def test_accel_z_default_when_missing(self):
        frames = _dispatch({})
        assert frames[0]["accel_z"] == -1.0

    def test_skin_temp_raw_forwarded(self):
        frames = _dispatch({"skin_temp_raw": 630})
        assert frames[0]["skin_temp_raw"] == 630

    def test_skin_temp_raw_defaults_to_132(self):
        frames = _dispatch({})
        assert frames[0]["skin_temp_raw"] == 132

    def test_sweat_raw_forwarded(self):
        frames = _dispatch({"sweat_raw": 200})
        assert frames[0]["sweat_raw"] == 200

    def test_emg_raw_default(self):
        frames = _dispatch({})
        assert frames[0]["emg_raw"] == 512

    def test_heart_rate_defaults_to_zero(self):
        frames = _dispatch({})
        assert frames[0]["heart_rate"] == 0

    def test_spo2_defaults_to_zero(self):
        frames = _dispatch({})
        assert frames[0]["spo2"] == 0

    def test_heart_rate_cast_to_int(self):
        frames = _dispatch({"heart_rate": 72.9})
        assert isinstance(frames[0]["heart_rate"], int)

    def test_accel_values_cast_to_float(self):
        frames = _dispatch({"accel_x": 1, "accel_y": 0, "accel_z": -1})
        assert isinstance(frames[0]["accel_x"], float)

    def test_on_sensor_data_fired_once(self):
        frames = _dispatch({"heart_rate": 70})
        assert len(frames) == 1

    def test_mcu_frame_has_all_required_keys(self):
        frames = _dispatch({})
        required = ("emg_raw", "heart_rate", "spo2", "accel_x", "accel_y",
                    "accel_z", "skin_temp_raw", "sweat_raw")
        for k in required:
            assert k in frames[0], f"Missing key: {k}"

    def test_gsr_raw_not_in_frame(self):
        frames = _dispatch({})
        assert "gsr_raw" not in frames[0]


# ── Glasses ambient data path ─────────────────────────────────────────────────

class TestHTTPBridgeGlassesDispatch:

    def test_no_glasses_callback_when_no_ambient_fields(self):
        frames = _dispatch_glasses({"heart_rate": 72})
        assert frames == []

    def test_glasses_callback_fired_when_ambient_present(self):
        frames = _dispatch_glasses({"ambient_temp_c": 30.0, "ambient_humidity_pct": 65.0})
        assert len(frames) == 1

    def test_ambient_temp_forwarded(self):
        frames = _dispatch_glasses({"ambient_temp_c": 31.5, "ambient_humidity_pct": 70.0})
        assert abs(frames[0]["ambient_temp_c"] - 31.5) < 1e-6

    def test_ambient_humidity_forwarded(self):
        frames = _dispatch_glasses({"ambient_temp_c": 28.0, "ambient_humidity_pct": 55.0})
        assert abs(frames[0]["ambient_humidity_pct"] - 55.0) < 1e-6

    def test_is_direct_sun_default_false(self):
        frames = _dispatch_glasses({"ambient_temp_c": 28.0, "ambient_humidity_pct": 60.0})
        assert frames[0]["is_direct_sun"] is False

    def test_is_direct_sun_forwarded(self):
        frames = _dispatch_glasses({
            "ambient_temp_c": 28.0, "ambient_humidity_pct": 60.0,
            "is_direct_sun": True,
        })
        assert frames[0]["is_direct_sun"] is True

    def test_noise_above_threshold_forwarded(self):
        frames = _dispatch_glasses({
            "ambient_temp_c": 28.0, "ambient_humidity_pct": 60.0,
            "noise_above_threshold": True,
        })
        assert frames[0]["noise_above_threshold"] is True

    def test_glasses_not_fired_when_no_callback(self):
        bridge, _ = _make_bridge()
        # no on_glasses_data set — must not raise
        bridge._dispatch({"ambient_temp_c": 28.0, "ambient_humidity_pct": 60.0})


# ── Poll loop stale-data guard ────────────────────────────────────────────────

class TestHTTPBridgeStaleness:

    def test_same_timestamp_not_dispatched_twice(self):
        import time
        received = []
        ts = time.time()
        store = {"data": {"heart_rate": 70}, "ts": ts}

        def getter():
            return dict(store["data"]), store["ts"]

        bridge = HTTPBridge(get_data_fn=getter)
        bridge.on_sensor_data = received.append
        bridge._last_ts = ts  # already seen this timestamp

        # Simulate one poll iteration
        data, poll_ts = getter()
        if poll_ts > bridge._last_ts and data:
            bridge._last_ts = poll_ts
            bridge._dispatch(data)

        assert received == []

    def test_new_timestamp_dispatched(self):
        import time
        received = []
        ts_old = time.time() - 1.0
        ts_new = time.time()
        store = {"data": {"heart_rate": 80}, "ts": ts_new}

        def getter():
            return dict(store["data"]), store["ts"]

        bridge = HTTPBridge(get_data_fn=getter)
        bridge.on_sensor_data = received.append
        bridge._last_ts = ts_old

        data, poll_ts = getter()
        if poll_ts > bridge._last_ts and data:
            bridge._last_ts = poll_ts
            bridge._dispatch(data)

        assert len(received) == 1
        assert received[0]["heart_rate"] == 80
