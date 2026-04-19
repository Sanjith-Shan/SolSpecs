"""
SolSpecs — Entry Point
Runs on the Arduino UNO Q (Qualcomm Linux side).

Modes:
    python main.py              — live mode (real hardware)
    python main.py --simulate   — full simulation on laptop, no hardware
    python main.py --interactive — simulate + keyboard commands for scenario testing
    python main.py --https      — serve HUD over HTTPS on port 8443 (required for Quest 3 WebXR)
"""

import argparse
import logging
import signal
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
from core.heat_stress import compute_wbgt_estimate


def parse_args():
    p = argparse.ArgumentParser(description="SolSpecs wearable heat safety system")
    # Mode flags (mutually exclusive; --simulate wins if combined)
    mode_g = p.add_mutually_exclusive_group()
    mode_g.add_argument("--simulate", action="store_true",
                        help="Full simulation — no hardware needed (default)")
    mode_g.add_argument("--live",     action="store_true",
                        help="Live mode — read sensors from STM32 over USB serial")
    mode_g.add_argument("--remote",   action="store_true",
                        help="Remote mode — UNO Q POSTs sensor data over WiFi to /sensor-update")
    p.add_argument("--interactive", action="store_true", help="Keyboard scenario control")
    p.add_argument("--https",       action="store_true",
                   help="Serve HUD over HTTPS on port 8443 (needed for Quest 3 WebXR)")
    p.add_argument("--emg",         action="store_true",
                   help="Enable real Mindrove EMG bridge via LSL (omit for MockEMGBridge)")
    p.add_argument("--loglevel", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


# ── Subsystem builders ────────────────────────────────────────────────────────

def build_mcu_bridge(args, simulate: bool):
    if simulate:
        from core.mcu_bridge import SimulatorBridge
        return SimulatorBridge(update_hz=20)
    if getattr(args, "remote", False):
        from core.mcu_bridge import HTTPBridge
        from core.sensor_server import get_remote_sensor_data
        return HTTPBridge(get_data_fn=get_remote_sensor_data)
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


def build_ai_pipeline(simulate: bool):
    try:
        from core.ai_pipeline import AIPipeline
        return AIPipeline(simulate=simulate)
    except Exception as e:
        logger.warning(f"AI pipeline unavailable: {e}")
        return None


def build_emg_bridge(simulate: bool):
    if simulate:
        from core.emg_bridge import MockEMGBridge
        return MockEMGBridge()
    from core.emg_bridge import EMGBridge
    return EMGBridge()


# ── HTTPS support ─────────────────────────────────────────────────────────────

def _build_ssl_context():
    """
    Generate a temporary self-signed certificate and return an ssl.SSLContext.
    Requires the `cryptography` package (pip install cryptography).
    """
    import ssl
    import ipaddress
    import tempfile
    import os
    from datetime import datetime, timezone, timedelta

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        logger.error(
            "cryptography package required for --https. "
            "Run: pip install cryptography"
        )
        raise

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                x509.IPAddress(ipaddress.IPv4Address("0.0.0.0")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    tmpdir = tempfile.mkdtemp()
    cert_file = os.path.join(tmpdir, "solspecs_cert.pem")
    key_file  = os.path.join(tmpdir, "solspecs_key.pem")

    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_file, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_file, key_file)
    logger.info(f"Self-signed TLS cert written to {tmpdir}")
    return ctx


# ── Sensor server thread ──────────────────────────────────────────────────────

def _start_sensor_server(sm, ai, use_https: bool):
    from core import sensor_server
    sensor_server.set_state_machine(sm)
    if ai is not None:
        sensor_server.set_ai_pipeline(ai)

    if use_https:
        port = 8443
        ssl_ctx = _build_ssl_context()
        logger.info(f"HUD available at https://0.0.0.0:{port}/hud  (accept the self-signed cert warning)")
    else:
        port = config.SENSOR_SERVER_PORT
        ssl_ctx = None
        logger.info(f"HUD available at http://0.0.0.0:{port}/hud")

    t = threading.Thread(
        target=sensor_server.run,
        kwargs={"port": port, "ssl_context": ssl_ctx},
        daemon=True,
        name="SensorServer",
    )
    t.start()
    return t


# ── Vitals snapshot ───────────────────────────────────────────────────────────

def _snapshot_vitals(sm: StateMachine, glasses_data: dict, mcu_data: dict) -> dict:
    temp = glasses_data.get("ambient_temp_c") or 28.0
    humidity = glasses_data.get("ambient_humidity_pct") or 60.0
    in_sun = glasses_data.get("is_direct_sun", False)
    wbgt = compute_wbgt_estimate(temp, humidity, in_sun)
    hr = mcu_data.get("heart_rate", 0)
    spo2 = mcu_data.get("spo2", 0)
    skin_raw = mcu_data.get("skin_temp_raw", 620)
    skin_temp = StateMachine._skin_raw_to_celsius(skin_raw)
    work_hours = (time.time() - sm._session_start) / 3600.0
    return {
        "tier": sm.current_tier,
        "heart_rate": hr,
        "spo2": spo2,
        "skin_temp": skin_temp,
        "ambient_temp": temp,
        "humidity": humidity,
        "wbgt": wbgt,
        "sun_exposure_minutes": sm.sun_exposure_minutes,
        "noise_hours_today": sm.noise_hours_today,
        "work_hours": work_hours,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    simulate = args.simulate or args.interactive
    remote   = getattr(args, "remote", False)

    logging.getLogger().setLevel(args.loglevel)
    if simulate:
        mode_label = "simulate"
    elif remote:
        mode_label = "remote"
    else:
        mode_label = "live"
    logger.info(f"SolSpecs starting [mode={mode_label}]")

    # ── Build subsystems ──────────────────────────────────────────────
    sm      = StateMachine()
    audio   = build_audio(simulate)
    mcu     = build_mcu_bridge(args, simulate)
    glasses = build_glasses_client(simulate or remote)
    gps     = build_gps_client(simulate or remote)
    ai      = build_ai_pipeline(simulate)
    emg     = build_emg_bridge(simulate) if args.emg else None

    _latest_glasses: dict = {}
    _latest_mcu: dict = {}
    _snapshots_lock = threading.Lock()

    # ── Wire callbacks ────────────────────────────────────────────────

    def on_tier_change(tier: str):
        logger.info(f"Heat tier → {tier.upper()}")
        if ai:
            with _snapshots_lock:
                g, m = dict(_latest_glasses), dict(_latest_mcu)
            ai.update_vitals(_snapshot_vitals(sm, g, m))

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

    sm.on_tier_change     = on_tier_change
    sm.on_alert           = on_alert
    sm.on_display_update  = on_display_update
    sm.on_ai_scan         = on_ai_scan
    sm.on_fall_detected   = on_fall_detected

    # ── Wire sensor feeds ─────────────────────────────────────────────

    def on_mcu_data(data: dict):
        with _snapshots_lock:
            _latest_mcu.update(data)
        sm.feed_mcu(data)

    def on_glasses_data(data: dict):
        with _snapshots_lock:
            _latest_glasses.update(data)
        sm.feed_glasses(data)

    mcu.on_sensor_data     = on_mcu_data
    glasses.on_sensor_data = on_glasses_data

    # In remote mode, ambient data also comes from the UNO Q HTTP payload
    if remote and hasattr(mcu, "on_glasses_data"):
        mcu.on_glasses_data = on_glasses_data

    def gps_poll_loop():
        while True:
            sm.feed_gps(gps.location)
            time.sleep(config.PHONE_GPS_POLL_INTERVAL_S)

    def vitals_refresh_loop():
        while True:
            time.sleep(60)
            if ai:
                with _snapshots_lock:
                    g, m = dict(_latest_glasses), dict(_latest_mcu)
                ai.update_vitals(_snapshot_vitals(sm, g, m))

    # ── Start subsystems ──────────────────────────────────────────────

    audio.start()
    mcu.start()
    glasses.start()
    gps.start()
    threading.Thread(target=gps_poll_loop,       daemon=True).start()
    threading.Thread(target=vitals_refresh_loop, daemon=True).start()

    if emg:
        from core.sensor_server import record_gesture_event, record_mayday

        def _on_clench():
            logger.info("EMG: Clench → fuel scan")
            record_gesture_event("clench")   # queued for HUD /emg-events poll
            if sm.on_ai_scan:
                sm.on_ai_scan()

        def _on_half_clench():
            logger.info("EMG: Half-Clench → MAYDAY")
            record_gesture_event("half_clench")
            gps_loc = sm._gps_location
            state = sm.get_current_state()
            record_mayday({
                "heart_rate":  state.get("heart_rate"),
                "spo2":        state.get("spo2"),
                "skin_temp_c": state.get("skin_temp_c"),
                "heat_tier":   state.get("heat_tier"),
                "gps_lat":     gps_loc.lat if gps_loc else None,
                "gps_lng":     gps_loc.lng if gps_loc else None,
                "timestamp":   int(time.time()),
            })

        emg.on_clench      = _on_clench
        emg.on_half_clench = _on_half_clench
        emg.start()
        logger.info("EMG bridge started")

    # ── Start sensor/HUD server ───────────────────────────────────────

    _start_sensor_server(sm, ai, use_https=args.https)

    audio.speak("SolSpecs initialized. Monitoring heat stress.", priority=PRIORITY_NORMAL)
    logger.info("All subsystems running")
    _print_startup_banner(args, mode_label)

    # ── Interactive or signal-wait ────────────────────────────────────

    if args.interactive:
        _run_interactive(sm, mcu, glasses, audio, ai)
    else:
        _run_until_signal()

    # ── Shutdown ──────────────────────────────────────────────────────

    logger.info("Shutting down...")
    mcu.stop()
    glasses.stop()
    gps.stop()
    audio.stop()
    if emg:
        emg.stop()
    logger.info("Goodbye.")


def _print_startup_banner(args, mode_label: str):
    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "0.0.0.0"

    hud_port    = 8443 if args.https else config.SENSOR_SERVER_PORT
    hud_scheme  = "https" if args.https else "http"
    hud_url     = f"{hud_scheme}://{ip}:{hud_port}/hud"
    sensor_url  = f"http://{ip}:{config.SENSOR_SERVER_PORT}/sensor-update"
    emg_url     = f"http://{ip}:{config.SENSOR_SERVER_PORT}/emg-event"
    emg_status  = "--emg (real Mindrove LSL)" if args.emg else "mock (keyboard C/H)"

    print("\n" + "╔" + "═"*46 + "╗")
    print(  "║" + "       SolSpecs ForeSight v1.0".center(46) + "║")
    print(  "╠" + "═"*46 + "╣")
    print(f"║  HUD:        {hud_url:<32}║")
    print(f"║  Sensors:    {sensor_url:<32}║")
    print(f"║  EMG Events: {emg_url:<32}║")
    print(  "║" + " "*46 + "║")
    print(f"║  Mode: --{mode_label:<36}║")
    print(f"║  EMG:  {emg_status:<38}║")
    print(  "╚" + "═"*46 + "╝\n")


def _run_until_signal():
    stop = threading.Event()

    def handle_signal(sig, frame):
        logger.info("Signal received — shutting down")
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    stop.wait()


def _run_interactive(sm, mcu, glasses, audio, ai):
    print("\n=== Interactive Mode ===")
    print("Commands:")
    print("  h  — simulate heat spike (elevated HR + hot day)")
    print("  c  — simulate critical conditions")
    print("  n  — return to normal conditions")
    print("  f  — simulate fall")
    print("  s  — status readout (quick EMG flex)")
    print("  e  — environment scan (sustained EMG flex)")
    print("  a  — ask the AI a question")
    print("  q  — quit\n")

    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if cmd == "q":
            break
        elif cmd == "h":
            mcu.simulate_health("elevated_hr")
            glasses.set_scenario("hot_direct_sun")
        elif cmd == "c":
            mcu.simulate_health("low_spo2")
            glasses.set_scenario("hot_direct_sun")
        elif cmd == "n":
            mcu.simulate_health("normal")
            glasses.set_scenario("normal")
        elif cmd == "f":
            mcu.simulate_health("fall")
        elif cmd == "s":
            mcu.simulate_gesture("describe")
        elif cmd == "e":
            mcu.simulate_gesture("converse")
        elif cmd == "a":
            if ai:
                question = input("  Ask: ").strip()
                if question:
                    reply = ai.chat(question)
                    print(f"  AI: {reply}")
            else:
                print("  AI pipeline not available.")
        else:
            print("Unknown command. Type q to quit.")


if __name__ == "__main__":
    main()
