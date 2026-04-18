"""
SolSpecs — Interactive Scenario Simulator
End-to-end test harness that runs the full pipeline on a laptop.

Usage:
    python tests/simulate_scenarios.py

All hardware is mocked. Audio prints to console instead of speaking.
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.state_machine import StateMachine
from core.mcu_bridge import SimulatorBridge
from core.glasses_client import MockGlassesClient
from core.phone_gps_client import MockGPSClient
from core.audio import MockAudioManager, PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW
import config


def run_scenario(name: str, mcu: SimulatorBridge, glasses: MockGlassesClient,
                 audio: MockAudioManager, sm: StateMachine, duration: float = 5.0):
    print(f"\n{'─' * 60}")
    print(f"  Scenario: {name}")
    print(f"{'─' * 60}")
    before_count = audio.spoken_count()
    time.sleep(duration)
    new_alerts = audio.spoken[before_count:]
    if new_alerts:
        for pri, text in new_alerts:
            label = ["CRITICAL", "HIGH", "NORMAL", "LOW"][pri]
            print(f"  [{label}] {text[:100]}")
    else:
        print("  (no new alerts)")
    print(f"  Current tier: {sm.current_tier.upper()}")
    print(f"  Sun exposure: {sm.sun_exposure_minutes:.1f} min")


def main():
    print("=" * 60)
    print("  SolSpecs — Scenario Simulator")
    print("=" * 60)

    # Build all subsystems
    sm = StateMachine()
    sm._TIER_DEBOUNCE = 2  # faster for demo
    audio = MockAudioManager()
    mcu = SimulatorBridge(update_hz=20)
    glasses = MockGlassesClient(poll_interval=1.0)
    gps = MockGPSClient(poll_interval=5.0)

    # Wire callbacks
    sm.on_alert = lambda text, pri: audio.speak(text, priority=pri)
    sm.on_tier_change = lambda t: print(f"\n  *** TIER CHANGE → {t.upper()} ***")
    sm.on_display_update = lambda t, m: print(f"  [OLED] {t} | {m!r}")
    sm.on_ai_scan = lambda: print("  [AI SCAN triggered]")
    sm.on_fall_detected = lambda lat, lng: print(f"  [FALL DETECTED] lat={lat} lng={lng}")

    mcu.on_sensor_data = sm.feed_mcu
    glasses.on_sensor_data = sm.feed_glasses

    # Start
    audio.start()
    mcu.start()
    glasses.start()
    gps.start()

    def gps_loop():
        while True:
            sm.feed_gps(gps.location)
            time.sleep(5)
    threading.Thread(target=gps_loop, daemon=True).start()

    time.sleep(0.5)  # let everything spin up

    # ── Scenario 1: Normal conditions ────────────────────────────────
    mcu.simulate_health("normal")
    glasses.set_scenario("normal")
    run_scenario("Normal workday (shade, cool)", mcu, glasses, audio, sm, 3.0)

    # ── Scenario 2: Direct sun, climbing temp ────────────────────────
    glasses.set_scenario("hot_direct_sun")
    run_scenario("Hot direct sun exposure", mcu, glasses, audio, sm, 5.0)

    # ── Scenario 3: Elevated heart rate ─────────────────────────────
    mcu.simulate_health("elevated_hr")
    run_scenario("Elevated HR + hot sun", mcu, glasses, audio, sm, 5.0)

    # ── Scenario 4: Critical — low SpO2 + heat ──────────────────────
    mcu.simulate_health("low_spo2")
    run_scenario("Critical: low SpO2 + extreme heat", mcu, glasses, audio, sm, 5.0)

    # ── Scenario 5: Fall detection ───────────────────────────────────
    mcu.simulate_health("fall")
    run_scenario("Fall detection", mcu, glasses, audio, sm, 4.0)

    # ── Scenario 6: Recovery — return to normal ──────────────────────
    mcu.simulate_health("normal")
    glasses.set_scenario("shade")
    run_scenario("Recovery in shade", mcu, glasses, audio, sm, 5.0)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Total alerts generated: {audio.spoken_count()}")
    print(f"  Final tier: {sm.current_tier.upper()}")
    print(f"  Sun exposure this hour: {sm.sun_exposure_minutes:.1f} min")
    print(f"  Noise hours today: {sm.noise_hours_today:.2f} h")
    print("=" * 60)

    mcu.stop()
    glasses.stop()
    gps.stop()
    audio.stop()


if __name__ == "__main__":
    main()
