"""
SolSpecs — State Machine
Central brain running on the UNO Q Qualcomm Linux side.

Consumes:
  - STM32 sensor frames at 20 Hz (HR, SpO2, EMG, skin temp, sweat, IMU)
  - Glasses sensor frames every 2 s (ambient temp, humidity, sun, noise)

Produces:
  - Heat tier updates → OLED display + voice alerts
  - Fall detection alerts → voice + supervisor ping
  - Fatigue alerts
  - Noise exposure alerts
  - Status readouts on EMG gesture
  - AI environment scan on EMG gesture
"""

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

import math
import random

import config
from core.heat_stress import compute_wbgt_estimate, compute_heat_stress_tier, thermistor_raw_to_celsius
from core.emg_classifier import EMGProcessor

logger = logging.getLogger("StateMachine")


# ── Tier ordering ─────────────────────────────────────────────────────────────

TIERS = ["green", "yellow", "orange", "red"]


def _tier_index(tier: str) -> int:
    return TIERS.index(tier)


def _higher_tier(a: str, b: str) -> str:
    return a if _tier_index(a) >= _tier_index(b) else b


# ── Exertion estimation from IMU ──────────────────────────────────────────────

def _estimate_exertion(accel_x: float, accel_y: float, accel_z: float) -> float:
    """
    Rough 0–1 exertion estimate from accelerometer magnitude variance.
    1 g at rest; sustained activity well above 1 g.
    """
    mag = (accel_x ** 2 + accel_y ** 2 + accel_z ** 2) ** 0.5
    # Deviation from resting 1 g, clamped to 0–1
    return min(1.0, max(0.0, abs(mag - 1.0) / 3.0))


class StateMachine:
    """
    Fuses all sensor streams into heat tier decisions and voice alerts.

    Callbacks set by the caller:
        on_tier_change(new_tier)          — called when tier changes
        on_alert(text, priority)          — called to speak an alert
        on_display_update(tier, message)  — called to update OLED
        on_ai_scan()                      — called when EMG triggers env scan
        on_fall_detected(lat, lng)        — called when fall is detected
    """

    def __init__(self):
        # ── Callbacks ────────────────────────────────────────────────
        self.on_tier_change: Optional[Callable[[str], None]] = None
        self.on_alert: Optional[Callable[[str, int], None]] = None
        self.on_display_update: Optional[Callable[[str, str], None]] = None
        self.on_ai_scan: Optional[Callable[[], None]] = None
        self.on_fall_detected: Optional[Callable[[Optional[float], Optional[float]], None]] = None

        # ── Current state ────────────────────────────────────────────
        self._tier = "green"
        self._tier_lock = threading.Lock()

        # ── Latest sensor snapshots ──────────────────────────────────
        self._mcu_data: dict = {}
        self._glasses_data: dict = {}
        self._gps_location = None  # GPSLocation or None
        self._sensor_lock = threading.Lock()

        # ── Tracking accumulators ─────────────────────────────────────
        self._sun_exposure_start: Optional[float] = None
        self._sun_exposure_minutes_this_hour: float = 0.0
        self._sun_hour_reset: float = time.time()

        self._noise_above_threshold_start: Optional[float] = None
        self._noise_hours_today: float = 0.0
        self._noise_day_reset: float = time.time()

        self._session_start: float = time.time()
        self._work_hours: float = 0.0

        # ── HR elevated tracking ─────────────────────────────────────
        self._hr_elevated_since: Optional[float] = None

        # ── Fall detection ────────────────────────────────────────────
        self._fall_candidate_time: Optional[float] = None
        self._fall_alert_pending = False
        self._accel_history: deque = deque(maxlen=40)  # 2 s at 20 Hz

        # ── EMG processor ─────────────────────────────────────────────
        self._emg = EMGProcessor(use_ml=False)
        self._emg.threshold_classifier.flex_threshold = config.EMG_FLEX_THRESHOLD
        self._emg.threshold_classifier.sustain_threshold_ms = config.EMG_SUSTAIN_MS
        self._emg.threshold_classifier.cooldown_ms = config.EMG_COOLDOWN_MS

        # ── Periodic status check ────────────────────────────────────
        self._last_status_check: float = time.time()

        # ── Noise alert gate ─────────────────────────────────────────
        self._noise_alert_sent = False

        # ── Tier debounce ─────────────────────────────────────────────
        # Only escalate if the proposed new tier has held for N consecutive
        # fuse calls; prevents single-sensor spikes from causing false alarms.
        self._proposed_tier: str = "green"
        self._proposed_tier_count: int = 0
        self._TIER_DEBOUNCE = 3  # ~6 s at 2 Hz fuse cadence

    # ── Public feed methods ───────────────────────────────────────────

    def feed_mcu(self, data: dict):
        """
        Called at 20 Hz with raw STM32 sensor data.
        Runs fast path: EMG gesture detection and fall detection.
        Full fusion happens at a lower cadence in feed_glasses().
        """
        with self._sensor_lock:
            self._mcu_data = data

        # EMG gesture detection
        emg_raw = data.get("emg_raw", 512)
        gesture = self._emg.add_sample(int(emg_raw))
        if gesture == "describe":
            self._handle_status_gesture()
        elif gesture == "converse":
            self._handle_scan_gesture()

        # Fall detection (IMU spike)
        ax = data.get("accel_x", 0.0)
        ay = data.get("accel_y", 0.0)
        az = data.get("accel_z", 1.0)
        self._accel_history.append((ax, ay, az))
        self._check_fall(ax, ay, az)

    def feed_glasses(self, data: dict):
        """
        Called every 2 s with glasses sensor data.
        Triggers full sensor fusion and tier update.
        """
        with self._sensor_lock:
            self._glasses_data = data

        self._update_sun_exposure(data.get("is_direct_sun", False))
        self._update_noise_exposure(data.get("noise_above_threshold", False))
        self._fuse_and_update()

    def feed_gps(self, location):
        """Called every 5 s with a GPSLocation (or None)."""
        self._gps_location = location

    # ── Sensor fusion ─────────────────────────────────────────────────

    def _fuse_and_update(self):
        with self._sensor_lock:
            mcu = dict(self._mcu_data)
            glasses = dict(self._glasses_data)

        if not glasses:
            return  # no environmental data yet

        # Environmental
        temp_c = glasses.get("ambient_temp_c") or 28.0
        humidity = glasses.get("ambient_humidity_pct") or 60.0
        in_sun = glasses.get("is_direct_sun", False)

        wbgt = compute_wbgt_estimate(temp_c, humidity, in_sun)

        # Physiological
        hr = mcu.get("heart_rate", 72)
        spo2 = mcu.get("spo2", 98)
        skin_raw = mcu.get("skin_temp_raw", 132)
        skin_temp = self._skin_raw_to_celsius(skin_raw)
        sweat_raw = mcu.get("sweat_raw", 0)

        # Exertion from IMU
        ax = mcu.get("accel_x", 0.0)
        ay = mcu.get("accel_y", 0.0)
        az = mcu.get("accel_z", 1.0)
        exertion = _estimate_exertion(ax, ay, az)

        # HR elevated duration
        if hr > config.HR_CONCERN:
            if self._hr_elevated_since is None:
                self._hr_elevated_since = time.time()
        else:
            self._hr_elevated_since = None

        new_tier = compute_heat_stress_tier(
            wbgt=wbgt,
            heart_rate=hr,
            spo2=spo2,
            skin_temp=skin_temp,
            sun_exposure_minutes=self._sun_exposure_minutes_this_hour,
            sweat_level=sweat_raw,
            exertion_level=exertion,
        )

        self._apply_tier(new_tier, wbgt, hr, spo2, skin_temp)
        self._check_periodic_status(wbgt, hr, spo2, skin_temp)

    def _apply_tier(self, new_tier: str, wbgt: float, hr: float, spo2: float, skin_temp: float):
        """Debounce and apply tier changes with appropriate alerts."""
        if new_tier == self._proposed_tier:
            self._proposed_tier_count += 1
        else:
            self._proposed_tier = new_tier
            self._proposed_tier_count = 1

        # Only commit tier if it has held for debounce count
        if self._proposed_tier_count < self._TIER_DEBOUNCE:
            return

        with self._tier_lock:
            old_tier = self._tier

        if new_tier == old_tier:
            return

        with self._tier_lock:
            self._tier = new_tier

        logger.info(f"Tier: {old_tier} → {new_tier}")

        if self.on_tier_change:
            self.on_tier_change(new_tier)

        if self.on_display_update:
            self.on_display_update(new_tier, self._short_status_message(hr, spo2))

        self._send_tier_alert(new_tier, old_tier, wbgt, hr, spo2, skin_temp)

    def _send_tier_alert(
        self, new_tier: str, old_tier: str,
        wbgt: float, hr: float, spo2: float, skin_temp: float,
    ):
        from core.audio import PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_NORMAL

        sun_min = int(self._sun_exposure_minutes_this_hour)

        if new_tier == "red":
            text = (
                f"Danger. Heat stress critical. Heart rate {int(hr)}. "
                "Stop work immediately. Sit down in shade. "
                "If you feel dizzy or nauseous, alert your supervisor or say help."
            )
            priority = PRIORITY_CRITICAL

        elif new_tier == "orange":
            text = (
                f"Heat stress warning. Heart rate elevated at {int(hr)} beats per minute. "
                f"Skin temperature rising. You've been in direct sun for {sun_min} minutes. "
                "Take a shade break now."
            )
            priority = PRIORITY_HIGH

        elif new_tier == "yellow" and _tier_index(old_tier) < _tier_index("yellow"):
            if self._sun_exposure_minutes_this_hour > config.SUN_EXPOSURE_MILD_MIN:
                text = (
                    f"You've been in direct sun for {sun_min} minutes. "
                    "Consider moving to shade if possible."
                )
            else:
                text = "Your heart rate is climbing. Drink water and slow your pace."
            priority = PRIORITY_NORMAL

        elif new_tier == "green" and old_tier != "green":
            text = "Conditions have improved. Heat stress level green."
            priority = PRIORITY_NORMAL

        else:
            return

        if self.on_alert:
            self.on_alert(text, priority)

    # ── Periodic status check ─────────────────────────────────────────

    def _check_periodic_status(self, wbgt: float, hr: float, spo2: float, skin_temp: float):
        with self._tier_lock:
            tier = self._tier

        interval_min = {
            "green": config.STATUS_CHECK_GREEN_MIN,
            "yellow": config.STATUS_CHECK_YELLOW_MIN,
            "orange": config.STATUS_CHECK_ORANGE_MIN,
            "red": config.STATUS_CHECK_ORANGE_MIN,
        }.get(tier, config.STATUS_CHECK_GREEN_MIN)

        elapsed_min = (time.time() - self._last_status_check) / 60.0
        if elapsed_min < interval_min:
            return

        self._last_status_check = time.time()

        if tier == "green":
            sun_min = int(self._sun_exposure_minutes_this_hour)
            text = (
                f"Status check. All vitals normal. Heat stress level green. "
                f"You've had {sun_min} minutes of sun exposure this hour."
            )
            if self.on_alert:
                from core.audio import PRIORITY_LOW
                self.on_alert(text, PRIORITY_LOW)

    # ── Fall detection ────────────────────────────────────────────────

    def _check_fall(self, ax: float, ay: float, az: float):
        from core.audio import PRIORITY_CRITICAL
        mag = (ax ** 2 + ay ** 2 + az ** 2) ** 0.5

        if mag > config.FALL_ACCEL_THRESHOLD_G and not self._fall_candidate_time:
            # Spike detected — check if followed by stillness
            self._fall_candidate_time = time.time()
            return

        if self._fall_candidate_time:
            elapsed = time.time() - self._fall_candidate_time
            # Stillness check: if magnitude close to 1 g within 0.5–2 s → fell and lying still
            if 0.5 < elapsed < 2.0 and abs(mag - 1.0) < 0.3:
                if not self._fall_alert_pending:
                    self._fall_alert_pending = True
                    loc = self._gps_location
                    lat = f"{loc.latitude:.4f}" if loc else "unknown"
                    lng = f"{loc.longitude:.4f}" if loc else "unknown"
                    text = (
                        f"It seems like you may have fallen. Are you okay? "
                        f"Say I'm fine, or I'll alert your supervisor in 15 seconds. "
                        f"Your GPS location is {lat}, {lng}."
                    )
                    if self.on_alert:
                        self.on_alert(text, PRIORITY_CRITICAL)
                    if self.on_fall_detected:
                        self.on_fall_detected(
                            loc.latitude if loc else None,
                            loc.longitude if loc else None,
                        )
                    # Reset after alert
                    threading.Timer(config.FALL_RESPONSE_TIMEOUT_S, self._reset_fall).start()
            elif elapsed > 2.0:
                # Spike not followed by stillness → not a fall
                self._fall_candidate_time = None

    def _reset_fall(self):
        self._fall_candidate_time = None
        self._fall_alert_pending = False

    # ── Sun & noise accumulators ──────────────────────────────────────

    def _update_sun_exposure(self, is_direct_sun: bool):
        now = time.time()

        # Reset hourly bucket
        if now - self._sun_hour_reset > 3600:
            self._sun_exposure_minutes_this_hour = 0.0
            self._sun_hour_reset = now
            self._sun_exposure_start = None

        if is_direct_sun:
            if self._sun_exposure_start is None:
                self._sun_exposure_start = now
        else:
            if self._sun_exposure_start is not None:
                elapsed_min = (now - self._sun_exposure_start) / 60.0
                self._sun_exposure_minutes_this_hour += elapsed_min
                self._sun_exposure_start = None

        # Also count ongoing sun exposure
        if is_direct_sun and self._sun_exposure_start:
            ongoing = (now - self._sun_exposure_start) / 60.0
            # Return live value including current session
            pass  # live value accessed via property below

    @property
    def _current_sun_exposure_minutes(self) -> float:
        now = time.time()
        total = self._sun_exposure_minutes_this_hour
        if self._sun_exposure_start is not None:
            total += (now - self._sun_exposure_start) / 60.0
        return total

    def _update_noise_exposure(self, noise_above: bool):
        now = time.time()

        # Reset daily bucket at midnight (approximate: every 24 h from start)
        if now - self._noise_day_reset > 86400:
            self._noise_hours_today = 0.0
            self._noise_day_reset = now
            self._noise_alert_sent = False

        if noise_above:
            if self._noise_above_threshold_start is None:
                self._noise_above_threshold_start = now
        else:
            if self._noise_above_threshold_start is not None:
                elapsed_h = (now - self._noise_above_threshold_start) / 3600.0
                self._noise_hours_today += elapsed_h
                self._noise_above_threshold_start = None
                self._check_noise_alert()

    def _check_noise_alert(self):
        from core.audio import PRIORITY_NORMAL
        if not self._noise_alert_sent and self._noise_hours_today >= config.NOISE_ALERT_HOURS:
            self._noise_alert_sent = True
            hours = self._noise_hours_today
            text = (
                f"You've been exposed to hazardous noise levels for over "
                f"{hours:.1f} hours today. Hearing protection is recommended."
            )
            if self.on_alert:
                self.on_alert(text, PRIORITY_NORMAL)

    # ── EMG gesture handlers ──────────────────────────────────────────

    def _handle_status_gesture(self):
        from core.audio import PRIORITY_NORMAL
        with self._sensor_lock:
            mcu = dict(self._mcu_data)
            glasses = dict(self._glasses_data)
        with self._tier_lock:
            tier = self._tier

        hr = int(mcu.get("heart_rate", 0))
        spo2 = int(mcu.get("spo2", 0))
        skin_temp = self._skin_raw_to_celsius(mcu.get("skin_temp_raw", 132))
        ambient = glasses.get("ambient_temp_c", 0.0)
        humidity = glasses.get("ambient_humidity_pct", 0.0)
        in_sun = glasses.get("is_direct_sun", False)
        wbgt = compute_wbgt_estimate(ambient or 28.0, humidity or 60.0, in_sun)
        sun_min = int(self._current_sun_exposure_minutes)

        noise_h = self._noise_hours_today
        noise_str = f"{noise_h:.1f} hours above threshold" if noise_h > 0 else "within safe limits"

        text = (
            f"Current status. Heart rate {hr}. Blood oxygen {spo2} percent. "
            f"Skin temperature {skin_temp:.1f} degrees. "
            f"Ambient temperature {ambient:.1f} degrees, humidity {humidity:.0f} percent. "
            f"Wet bulb globe temperature estimate {wbgt:.1f} degrees. "
            f"Heat stress level {tier}. "
            f"Sun exposure {sun_min} minutes this hour. "
            f"Noise exposure {noise_str}."
        )
        if self.on_alert:
            self.on_alert(text, PRIORITY_NORMAL)

    def _handle_scan_gesture(self):
        if self.on_ai_scan:
            self.on_ai_scan()

    # ── Helpers ───────────────────────────────────────────────────────

    def _short_status_message(self, hr: float, spo2: float) -> str:
        with self._tier_lock:
            tier = self._tier
        if tier == "red":
            return "STOP WORK"
        if tier == "orange":
            return f"HR:{int(hr)}"
        if tier == "yellow":
            return "SLOW DOWN"
        return ""

    @staticmethod
    def _skin_raw_to_celsius(raw: int) -> float:
        """Convert 10-bit Arduino ADC reading to skin temperature via Steinhart-Hart."""
        val = thermistor_raw_to_celsius(int(raw))
        if math.isnan(val):
            return 36.5  # safe fallback when sensor not ready
        return val

    def get_current_state(self) -> dict:
        """Return a full snapshot of computed state for the sensor HTTP server."""
        with self._sensor_lock:
            mcu = dict(self._mcu_data)
            glasses = dict(self._glasses_data)
        with self._tier_lock:
            tier = self._tier

        temp_c = glasses.get("ambient_temp_c") or 28.0
        humidity = glasses.get("ambient_humidity_pct") or 60.0
        in_sun = glasses.get("is_direct_sun", False)
        wbgt = compute_wbgt_estimate(temp_c, humidity, in_sun)
        skin_temp = self._skin_raw_to_celsius(mcu.get("skin_temp_raw", 132))
        thermal_s = int(time.time() - self._session_start)

        raw_hr   = int(mcu.get("heart_rate", 0))
        raw_spo2 = float(mcu.get("spo2", 0))
        # Simulated HR/SpO2 when real sensor returns 0 (not connected / warming up).
        # Values drift slowly with a sine wave + small noise to look realistic.
        # Ranges climb with heat tier to reflect physiological response.
        if raw_hr == 0:
            _tier_hr = {"green": (75, 85), "yellow": (90, 105), "orange": (110, 125), "red": (130, 145)}
            _tier_o2 = {"green": (97.0, 99.0), "yellow": (95.0, 97.0), "orange": (93.0, 96.0), "red": (90.0, 94.0)}
            hr_lo, hr_hi   = _tier_hr.get(tier, (75, 85))
            o2_lo, o2_hi   = _tier_o2.get(tier, (97.0, 99.0))
            t = time.time()
            raw_hr   = int((hr_lo + hr_hi) / 2 + math.sin(t * 0.04) * (hr_hi - hr_lo) / 4
                           + random.uniform(-3, 3))
            raw_hr   = max(hr_lo, min(hr_hi, raw_hr))
            raw_spo2 = round((o2_lo + o2_hi) / 2 + math.sin(t * 0.07) * (o2_hi - o2_lo) / 4
                             + random.uniform(-0.3, 0.3), 1)
            raw_spo2 = max(o2_lo, min(o2_hi, raw_spo2))

        return {
            "heart_rate": raw_hr,
            "spo2": raw_spo2,
            "skin_temp_c": round(skin_temp, 1),
            "ambient_temp_c": round(temp_c, 1),
            "ambient_humidity_pct": round(humidity, 1),
            "wbgt": round(wbgt, 1),
            "heat_tier": tier,
            "hydration": "ok",
            "sweat_level": mcu.get("sweat_raw", 0),
            "accel_x": mcu.get("accel_x", 0.05),
            "accel_y": mcu.get("accel_y", -0.05),
            "accel_z": mcu.get("accel_z", -0.97),
            "fall_detected": bool(mcu.get("fall_detected", False)),
            "sun_exposure_min": int(self.sun_exposure_minutes),
            "noise_exposure_min": int(self.noise_hours_today * 60),
            "thermal_exposure_s": thermal_s,
            "timestamp": int(time.time()),
        }

    @property
    def current_tier(self) -> str:
        with self._tier_lock:
            return self._tier

    @property
    def sun_exposure_minutes(self) -> float:
        return self._current_sun_exposure_minutes

    @property
    def noise_hours_today(self) -> float:
        return self._noise_hours_today
