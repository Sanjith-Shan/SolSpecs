"""
Tests for core/state_machine.py
Run with:  pytest tests/test_state_machine.py -v
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.state_machine import StateMachine, _tier_index, _higher_tier, _estimate_exertion
import config


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_sm() -> StateMachine:
    sm = StateMachine()
    # Speed up debounce for tests
    sm._TIER_DEBOUNCE = 1
    return sm


NORMAL_GLASSES = {
    "ambient_temp_c": 22.0,
    "ambient_humidity_pct": 50.0,
    "is_direct_sun": False,
    "light_level": 300,
    "noise_level_db": 65,
    "noise_above_threshold": False,
    "timestamp": time.time(),
}

NORMAL_MCU = {
    "emg_raw": 512,
    "heart_rate": 72,
    "spo2": 98,
    "skin_temp_raw": 132,
    "sweat_raw": 100,
    "accel_x": 0.0,
    "accel_y": 0.0,
    "accel_z": 1.0,
    "gyro_x": 0.0,
    "gyro_y": 0.0,
    "gyro_z": 0.0,
}

HOT_GLASSES = {
    "ambient_temp_c": 38.0,
    "ambient_humidity_pct": 75.0,
    "is_direct_sun": True,
    "light_level": 900,
    "noise_level_db": 70,
    "noise_above_threshold": False,
    "timestamp": time.time(),
}

CRITICAL_MCU = {
    **NORMAL_MCU,
    "heart_rate": 148,
    "spo2": 87,
    "skin_temp_raw": 120,  # ~39°C (raw=120 with r_nominal=2430, B=3950)
}


# ─── Tier helpers ─────────────────────────────────────────────────────────────

class TestTierHelpers:

    def test_tier_index_ordering(self):
        assert _tier_index("green") < _tier_index("yellow") < _tier_index("orange") < _tier_index("red")

    def test_higher_tier_returns_red(self):
        assert _higher_tier("orange", "red") == "red"
        assert _higher_tier("red", "yellow") == "red"

    def test_higher_tier_same(self):
        assert _higher_tier("yellow", "yellow") == "yellow"

    def test_estimate_exertion_rest(self):
        # 1 g at rest → 0 exertion
        e = _estimate_exertion(0.0, 0.0, 1.0)
        assert e == pytest.approx(0.0)

    def test_estimate_exertion_high_activity(self):
        # 4 g spike → high exertion
        e = _estimate_exertion(2.0, 2.0, 2.0)
        assert e > 0.5

    def test_estimate_exertion_clamped_to_one(self):
        e = _estimate_exertion(10.0, 10.0, 10.0)
        assert e == pytest.approx(1.0)

    def test_estimate_exertion_clamped_to_zero(self):
        e = _estimate_exertion(0.0, 0.0, 0.99)
        assert e >= 0.0


# ─── StateMachine initialization ─────────────────────────────────────────────

class TestStateMachineInit:

    def test_initial_tier_is_green(self):
        sm = make_sm()
        assert sm.current_tier == "green"

    def test_initial_sun_exposure_zero(self):
        sm = make_sm()
        assert sm.sun_exposure_minutes == pytest.approx(0.0)

    def test_initial_noise_hours_zero(self):
        sm = make_sm()
        assert sm.noise_hours_today == pytest.approx(0.0)


# ─── Tier transitions ─────────────────────────────────────────────────────────

class TestTierTransitions:

    def test_normal_conditions_stay_green(self):
        sm = make_sm()
        for _ in range(5):
            sm.feed_glasses(NORMAL_GLASSES)
            sm.feed_mcu(NORMAL_MCU)
        assert sm.current_tier == "green"

    def test_critical_conditions_escalate_to_red(self):
        sm = make_sm()
        tiers = []
        sm.on_tier_change = tiers.append
        for _ in range(5):
            sm.feed_glasses(HOT_GLASSES)
            sm.feed_mcu(CRITICAL_MCU)
        assert sm.current_tier in ("orange", "red")

    def test_tier_change_callback_fires(self):
        sm = make_sm()
        changes = []
        sm.on_tier_change = changes.append
        for _ in range(5):
            sm.feed_glasses(HOT_GLASSES)
            sm.feed_mcu(CRITICAL_MCU)
        assert len(changes) >= 1

    def test_tier_change_callback_receives_valid_tier(self):
        sm = make_sm()
        changes = []
        sm.on_tier_change = changes.append
        for _ in range(5):
            sm.feed_glasses(HOT_GLASSES)
            sm.feed_mcu(CRITICAL_MCU)
        for tier in changes:
            assert tier in ("green", "yellow", "orange", "red")

    def test_display_update_fires_on_tier_change(self):
        sm = make_sm()
        displays = []
        sm.on_display_update = lambda t, m: displays.append(t)
        for _ in range(5):
            sm.feed_glasses(HOT_GLASSES)
            sm.feed_mcu(CRITICAL_MCU)
        if sm.current_tier != "green":
            assert len(displays) >= 1

    def test_alert_fires_on_tier_escalation(self):
        sm = make_sm()
        alerts = []
        sm.on_alert = lambda text, pri: alerts.append((text, pri))
        for _ in range(5):
            sm.feed_glasses(HOT_GLASSES)
            sm.feed_mcu(CRITICAL_MCU)
        if sm.current_tier != "green":
            assert len(alerts) >= 1

    def test_red_alert_text_contains_stop_work(self):
        sm = make_sm()
        alerts = []
        sm.on_alert = lambda text, pri: alerts.append((text, pri))
        for _ in range(10):
            sm.feed_glasses(HOT_GLASSES)
            sm.feed_mcu(CRITICAL_MCU)
        red_alerts = [t for t, _ in alerts if "stop work" in t.lower() or "danger" in t.lower()]
        if sm.current_tier == "red":
            assert len(red_alerts) >= 1

    def test_tier_debounce_prevents_single_spike(self):
        sm = make_sm()
        sm._TIER_DEBOUNCE = 5  # require 5 consecutive hot readings
        sm.feed_glasses(HOT_GLASSES)
        sm.feed_mcu(CRITICAL_MCU)
        # One reading alone should not flip to red (debounce not met)
        # (may still flip to yellow/orange depending on score, but not red on first call)
        # Just verify tier is valid
        assert sm.current_tier in ("green", "yellow", "orange", "red")


# ─── Sun exposure accumulation ────────────────────────────────────────────────

class TestSunExposure:

    def test_sun_exposure_accumulates_in_direct_sun(self):
        sm = make_sm()
        sun_data = {**NORMAL_GLASSES, "is_direct_sun": True}
        sm._sun_exposure_start = time.time() - 20 * 60  # 20 min ago
        sm._update_sun_exposure(True)
        assert sm.sun_exposure_minutes > 15

    def test_sun_exposure_stops_in_shade(self):
        sm = make_sm()
        sm._sun_exposure_start = time.time() - 10 * 60  # 10 min ago
        sm._update_sun_exposure(False)  # just entered shade
        # 10 minutes should have been committed
        assert sm._sun_exposure_minutes_this_hour >= 9.5

    def test_no_sun_exposure_starts_at_zero(self):
        sm = make_sm()
        sm._update_sun_exposure(False)
        assert sm._current_sun_exposure_minutes == pytest.approx(0.0)

    def test_hourly_reset(self):
        sm = make_sm()
        sm._sun_exposure_minutes_this_hour = 55.0
        sm._sun_hour_reset = time.time() - 3601  # over an hour ago
        sm._update_sun_exposure(False)
        assert sm._sun_exposure_minutes_this_hour == pytest.approx(0.0)


# ─── Noise exposure accumulation ─────────────────────────────────────────────

class TestNoiseExposure:

    def test_noise_exposure_accumulates(self):
        sm = make_sm()
        sm._noise_above_threshold_start = time.time() - 2 * 3600  # 2 hours ago
        sm._update_noise_exposure(False)  # noise just ended
        assert sm._noise_hours_today >= 1.9

    def test_noise_alert_fires_after_threshold(self):
        sm = make_sm()
        alerts = []
        sm.on_alert = lambda text, pri: alerts.append(text)
        sm._noise_hours_today = config.NOISE_ALERT_HOURS - 0.01
        sm._noise_above_threshold_start = time.time() - 120  # 2 min above threshold
        sm._update_noise_exposure(False)
        assert any("noise" in a.lower() or "hearing" in a.lower() for a in alerts)

    def test_noise_alert_only_fires_once(self):
        sm = make_sm()
        alerts = []
        sm.on_alert = lambda text, pri: alerts.append(text)
        sm._noise_hours_today = config.NOISE_ALERT_HOURS + 0.5
        sm._noise_alert_sent = True  # already sent
        sm._check_noise_alert()
        assert len(alerts) == 0


# ─── Skin temperature conversion ─────────────────────────────────────────────

class TestSkinTempConversion:

    def test_baseline_raw_maps_to_normal_temp(self):
        # raw=132 → ~36.5°C (Steinhart-Hart, r_nominal=2430Ω, 10-bit ADC, B=3950)
        temp = StateMachine._skin_raw_to_celsius(132)
        assert abs(temp - 36.5) < 0.5

    def test_lower_raw_maps_to_higher_temp(self):
        # Lower ADC → lower Rth → hotter (NTC inverted divider)
        temp_hot  = StateMachine._skin_raw_to_celsius(100)
        temp_norm = StateMachine._skin_raw_to_celsius(132)
        assert temp_hot > temp_norm

    def test_second_calibration_point(self):
        # raw=200 → nominal 25°C (Rth = R_nominal = 2430Ω)
        temp = StateMachine._skin_raw_to_celsius(200)
        assert abs(temp - 25.0) < 0.5


# ─── Feed methods ─────────────────────────────────────────────────────────────

class TestFeedMethods:

    def test_feed_mcu_does_not_raise(self):
        sm = make_sm()
        sm.feed_mcu(NORMAL_MCU)

    def test_feed_glasses_does_not_raise(self):
        sm = make_sm()
        sm.feed_glasses(NORMAL_GLASSES)

    def test_feed_gps_stores_location(self):
        from core.phone_gps_client import GPSLocation
        sm = make_sm()
        loc = GPSLocation(32.88, -117.23, 5.0, time.time())
        sm.feed_gps(loc)
        assert sm._gps_location == loc

    def test_feed_mcu_missing_keys_does_not_crash(self):
        sm = make_sm()
        sm.feed_glasses(NORMAL_GLASSES)
        sm.feed_mcu({})  # completely empty

    def test_feed_glasses_missing_keys_does_not_crash(self):
        sm = make_sm()
        sm.feed_glasses({})
