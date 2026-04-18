"""
SolSpecs — Entry Point
Runs on the Arduino UNO Q (Qualcomm Linux side).

Modes:
    python main.py              — live mode (real hardware)
    python main.py --simulate   — full simulation on laptop, no hardware
    python main.py --interactive — simulate + keyboard commands for scenario testing
"""

import argparse
import logging
import signal
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("Main")

import config
from core.state_machine import StateMachine
from core.audio import AudioManager, MockAudioManager, PRIORITY_CRITICAL, PRIORITY_NORMAL


def parse_args():
    p = argparse.ArgumentParser(description="SolSpecs wearable heat safety system")
    p.add_argument("--simulate", action="store_true", help="Run without hardware")
    p.add_argument("--interactive", action="store_true", help="Keyboard scenario control")
    p.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def build_mcu_bridge(simulate: bool):
    if simulate:
        from core.mcu_bridge import SimulatorBridge
        bridge = SimulatorBridge(update_hz=20)
        return bridge
    from core.mcu_bridge import SerialBridge
    return SerialBridge(port="/dev/ttyACM0", baud=115200)


def build_glasses_client(simulate: bool):
    if simulate:
        from core.glasses_client import MockGlassesClient
        return MockGlassesClient(poll_interval=2.0)
    from core.glasses_client import GlassesClient
    return GlassesClient(config.GLASSES_URL, poll_interval=config.GLASSES_POLL_INTERVAL_S)


def build_gps_client(simulate: bool):
    if simulate:
        from core.phone_gps_client import MockGPSClient
        return MockGPSClient(poll_interval=5.0)
    from core.phone_gps_client import PhoneGPSClient
    return PhoneGPSClient(config.PHONE_GPS_URL, poll_interval=config.PHONE_GPS_POLL_INTERVAL_S)


def build_audio(simulate: bool):
    if simulate:
        return MockAudioManager()
    return AudioManager()


def build_ai_pipeline():
    try:
        from core.ai_pipeline import AIPipeline
        return AIPipeline()
    except Exception as e:
        logger.warning(f"AI pipeline unavailable: {e}")
        return None


def main():
    args = parse_args()
    simulate = args.simulate or args.interactive or (config.MODE == "simulate")

    logging.getLogger().setLevel(args.loglevel)
    logger.info(f"SolSpecs starting [mode={'simulate' if simulate else 'live'}]")

    # ── Build subsystems ──────────────────────────────────────────────
    sm = StateMachine()
    audio = build_audio(simulate)
    mcu = build_mcu_bridge(simulate)
    glasses = build_glasses_client(simulate)
    gps = build_gps_client(simulate)
    ai = build_ai_pipeline()

    # ── Wire callbacks ────────────────────────────────────────────────

    def on_tier_change(tier: str):
        logger.info(f"Heat tier → {tier.upper()}")

    def on_alert(text: str, priority: int):
        audio.speak(text, priority=priority)

    def on_display_update(tier: str, message: str):
        if glasses.is_connected:
            glasses.send_display(tier, message)

    def on_ai_scan():
        logger.info("AI scan triggered via EMG gesture")
        if not glasses.is_connected:
            audio.speak("Camera not connected.", priority=PRIORITY_NORMAL)
            return
        audio.speak("Capturing environment.", priority=PRIORITY_NORMAL)
        image = glasses.capture()
        if not image:
            audio.speak("Could not capture image.", priority=PRIORITY_NORMAL)
            return
        if ai:
            description = ai.describe_scene(image)
            audio.speak(description, priority=PRIORITY_NORMAL)
        else:
            audio.speak("AI pipeline not available.", priority=PRIORITY_NORMAL)

    def on_fall_detected(lat, lng):
        logger.warning(f"Fall detected at {lat}, {lng}")

    sm.on_tier_change = on_tier_change
    sm.on_alert = on_alert
    sm.on_display_update = on_display_update
    sm.on_ai_scan = on_ai_scan
    sm.on_fall_detected = on_fall_detected

    # ── Wire sensor feeds ─────────────────────────────────────────────

    mcu.on_sensor_data = sm.feed_mcu
    glasses.on_sensor_data = sm.feed_glasses

    def gps_poll_loop():
        while True:
            sm.feed_gps(gps.location)
            time.sleep(config.PHONE_GPS_POLL_INTERVAL_S)

    # ── Start everything ──────────────────────────────────────────────

    audio.start()
    mcu.start()
    glasses.start()
    gps.start()
    threading.Thread(target=gps_poll_loop, daemon=True).start()

    audio.speak("SolSpecs initialized. Monitoring heat stress.", priority=PRIORITY_NORMAL)
    logger.info("All subsystems running")

    # ── Interactive keyboard control ──────────────────────────────────

    if args.interactive:
        _run_interactive(sm, mcu, glasses, audio)
    else:
        _run_until_signal()

    # ── Shutdown ──────────────────────────────────────────────────────

    logger.info("Shutting down...")
    mcu.stop()
    glasses.stop()
    gps.stop()
    audio.stop()
    logger.info("Goodbye.")


def _run_until_signal():
    stop = threading.Event()

    def handle_signal(sig, frame):
        logger.info("Signal received — shutting down")
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    stop.wait()


def _run_interactive(sm, mcu, glasses, audio):
    print("\n=== Interactive Mode ===")
    print("Commands:")
    print("  h  — simulate heat spike (elevated HR + hot day)")
    print("  c  — simulate critical conditions")
    print("  n  — return to normal conditions")
    print("  f  — simulate fall")
    print("  s  — status readout (quick EMG flex)")
    print("  e  — environment scan (sustained EMG flex)")
    print("  q  — quit\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "h":
            print("→ Heat spike: elevated HR, hot glasses data")
            mcu.simulate_health("elevated_hr")
            glasses.set_scenario("hot_direct_sun")
        elif cmd == "c":
            print("→ Critical: very high HR, low SpO2")
            mcu.simulate_health("low_spo2")
            glasses.set_scenario("hot_direct_sun")
        elif cmd == "n":
            print("→ Normal conditions")
            mcu.simulate_health("normal")
            glasses.set_scenario("normal")
        elif cmd == "f":
            print("→ Simulating fall")
            mcu.simulate_health("fall")
        elif cmd == "s":
            print("→ Triggering status readout")
            mcu.simulate_gesture("describe")
        elif cmd == "e":
            print("→ Triggering environment scan")
            mcu.simulate_gesture("converse")
        else:
            print("Unknown command. Type q to quit.")


if __name__ == "__main__":
    main()
