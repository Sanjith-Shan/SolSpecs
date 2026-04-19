"""
BlindGuide — MCU Bridge
Communication between the Qualcomm Linux side and STM32 MCU side.

On UNO Q: Uses Arduino RPC bridge (serial over internal bus).
On laptop: Uses a simulator that generates fake sensor data for testing.

The STM32 side runs an Arduino sketch that:
    - Reads 3x HC-SR04 ultrasonic sensors
    - Reads EMG ADC input
    - Reads MAX30102 (HR + SpO2) via I2C
    - Reads MPU9250 via I2C
    - Reads thermistor via ADC
    - Reads photoresistor via ADC
    - Reads sound sensor via ADC
    - Outputs EMS PWM signals (2 channels)
    - Controls WS2812B LED strips
    - Controls Modulino Vibro
    - Controls Modulino Buzzer
    
The STM32 sends a packed data frame to Linux at ~20Hz.
Linux sends commands back (EMS intensity, LED mode, etc).
"""

import json
import time
import random
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger("MCUBridge")


# ─── Data Frame Protocol ───────────────────────────────────────────

# STM32 → Linux (sensor readings, ~20Hz)
# JSON for simplicity at hackathon speed. Binary protocol if perf matters.
SENSOR_FRAME_KEYS = [
    "dist_front", "dist_left", "dist_right",     # ultrasonic cm
    "emg_raw",                                     # raw ADC 0-4095
    "heart_rate", "spo2",                          # MAX30102
    "accel_x", "accel_y", "accel_z",              # MPU9250
    "gyro_x", "gyro_y", "gyro_z",
    "skin_temp_raw",                               # thermistor ADC
    "ambient_light",                               # photoresistor ADC
    "ambient_noise",                               # sound sensor ADC
]

# Linux → STM32 (commands)
# {"cmd": "ems", "left": 0.0-1.0, "right": 0.0-1.0}
# {"cmd": "led", "mode": "off|safety|crossing|sos", "color": "amber|red|white"}
# {"cmd": "vibro", "pattern": "none|left_turn|right_turn|alert"}
# {"cmd": "buzzer", "pattern": "none|rapid|warning|sos"}


class MCUBridge:
    """
    Base class for MCU communication.
    Subclass with SerialBridge (real hardware) or SimulatorBridge (testing).
    """

    def __init__(self):
        self.on_sensor_data: Optional[Callable] = None  # callback(data_dict)
        self._running = False

    def start(self):
        raise NotImplementedError

    def stop(self):
        self._running = False

    def send_command(self, command: dict):
        raise NotImplementedError


class SimulatorBridge(MCUBridge):
    """
    Simulates the STM32 sending sensor data.
    Use this on your laptop to test the full pipeline without hardware.
    """

    def __init__(self, update_hz: float = 20):
        super().__init__()
        self.update_hz = update_hz
        self._thread = None

        # Simulation state
        self._obstacle_scenario = "clear"  # "clear", "left", "right", "front", "closing"
        self._health_scenario = "normal"   # "normal", "elevated_hr", "low_spo2", "fall"
        self._emg_gesture_queue = []       # manually trigger gestures for testing

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Simulator bridge started")

    def _run_loop(self):
        tick = 0
        while self._running:
            data = self._generate_frame(tick)
            if self.on_sensor_data:
                self.on_sensor_data(data)
            tick += 1
            time.sleep(1.0 / self.update_hz)

    def _generate_frame(self, tick: int) -> dict:
        """Generate realistic-ish sensor data based on current scenario."""

        # Ultrasonic distances
        if self._obstacle_scenario == "clear":
            dist_f, dist_l, dist_r = 400, 400, 400
        elif self._obstacle_scenario == "left":
            dist_f, dist_l, dist_r = 400, 80 + random.randint(-5, 5), 400
        elif self._obstacle_scenario == "right":
            dist_f, dist_l, dist_r = 400, 400, 60 + random.randint(-5, 5)
        elif self._obstacle_scenario == "front":
            dist_f, dist_l, dist_r = 100 + random.randint(-10, 10), 400, 400
        elif self._obstacle_scenario == "closing":
            # Obstacle getting closer over time
            dist = max(20, 200 - tick * 5)
            dist_f = dist + random.randint(-3, 3)
            dist_l, dist_r = 400, 400
        else:
            dist_f, dist_l, dist_r = 400, 400, 400

        # EMG
        emg_base = 512 + random.randint(-20, 20)  # rest baseline ~512
        if self._emg_gesture_queue:
            gesture_type = self._emg_gesture_queue[0]
            if gesture_type == "describe":
                emg_base = 800 + random.randint(-30, 30)  # brief spike
                if tick % 10 == 0:  # clear after a few frames
                    self._emg_gesture_queue.pop(0)
            elif gesture_type == "converse":
                emg_base = 750 + random.randint(-20, 20)  # sustained

        # Biometrics
        if self._health_scenario == "normal":
            hr, spo2 = 72 + random.randint(-3, 3), 98 + random.randint(-1, 1)
        elif self._health_scenario == "elevated_hr":
            hr, spo2 = 115 + random.randint(-5, 5), 97 + random.randint(-1, 1)
        elif self._health_scenario == "low_spo2":
            hr, spo2 = 80 + random.randint(-3, 3), 87 + random.randint(-2, 2)
        elif self._health_scenario == "fall":
            hr, spo2 = 95 + random.randint(-5, 5), 96
        else:
            hr, spo2 = 72, 98

        # IMU (accelerometer in g, gyro in deg/s)
        if self._health_scenario == "fall":
            ax = random.uniform(-4, 4)  # spike
            ay = random.uniform(-4, 4)
            az = random.uniform(-1, 5)
        else:
            ax = random.uniform(-0.1, 0.1)
            ay = random.uniform(-0.1, 0.1)
            az = random.uniform(0.95, 1.05)  # ~1g downward

        return {
            "dist_front": max(2, dist_f),
            "dist_left": max(2, dist_l),
            "dist_right": max(2, dist_r),
            "emg_raw": emg_base,
            "heart_rate": max(0, hr),
            "spo2": max(0, min(100, spo2)),
            "accel_x": ax, "accel_y": ay, "accel_z": az,
            "gyro_x": random.uniform(-5, 5),
            "gyro_y": random.uniform(-5, 5),
            "gyro_z": random.uniform(-5, 5),
            "skin_temp_raw": 132 + random.randint(-5, 5),  # ~36.5°C
            "ambient_light": 650 + random.randint(-20, 20),
            "ambient_noise": 200 + random.randint(-30, 30),
        }

    def send_command(self, command: dict):
        """Log commands that would be sent to STM32."""
        cmd = command.get("cmd", "unknown")
        if cmd == "ems":
            left, right = command.get("left", 0), command.get("right", 0)
            if left > 0 or right > 0:
                logger.debug(f"→ MCU: EMS left={left:.2f} right={right:.2f}")
        elif cmd == "led":
            logger.debug(f"→ MCU: LED mode={command.get('mode')} color={command.get('color')}")
        elif cmd == "vibro":
            logger.debug(f"→ MCU: VIBRO pattern={command.get('pattern')}")
        elif cmd == "buzzer":
            logger.debug(f"→ MCU: BUZZER pattern={command.get('pattern')}")

    # ─── Simulation controls (for testing) ─────────────────────────

    def simulate_obstacle(self, scenario: str):
        """Set obstacle scenario: clear, left, right, front, closing"""
        self._obstacle_scenario = scenario
        logger.info(f"Simulator: obstacle → {scenario}")

    def simulate_health(self, scenario: str):
        """Set health scenario: normal, elevated_hr, low_spo2, fall"""
        self._health_scenario = scenario
        logger.info(f"Simulator: health → {scenario}")

    def simulate_gesture(self, gesture: str):
        """Queue an EMG gesture: describe, converse"""
        self._emg_gesture_queue.append(gesture)
        logger.info(f"Simulator: queued gesture → {gesture}")


class SerialBridge(MCUBridge):
    """
    Real serial communication with the STM32 via Arduino RPC bridge.
    Use this on the actual UNO Q.
    """

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200):
        super().__init__()
        self.port = port
        self.baud = baud
        self._serial = None

    def start(self):
        import serial
        self._serial = serial.Serial(self.port, self.baud, timeout=0.1)
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        logger.info(f"Serial bridge started on {self.port}")

    def _read_loop(self):
        while self._running:
            try:
                line = self._serial.readline().decode('utf-8').strip()
                if line:
                    data = json.loads(line)
                    if self.on_sensor_data:
                        self.on_sensor_data(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            except Exception as e:
                logger.error(f"Serial read error: {e}")
                time.sleep(0.1)

    def send_command(self, command: dict):
        if self._serial and self._serial.is_open:
            msg = json.dumps(command) + "\n"
            self._serial.write(msg.encode('utf-8'))

    def stop(self):
        super().stop()
        if self._serial:
            self._serial.close()


class HTTPBridge(MCUBridge):
    """
    Receives sensor data posted by the UNO Q armband over WiFi.
    Polls sensor_server's in-memory dict every 100 ms and fires callbacks
    whenever fresh data arrives.

    Accepts an optional on_glasses_data callback for the ambient fields
    (ambient_temp_c, ambient_humidity_pct) that are included in the payload.
    """

    STALE_SECS = 5.0  # seconds before data is considered stale

    def __init__(self, get_data_fn):
        super().__init__()
        self._get_data = get_data_fn   # () -> (dict, float)
        self._thread = None
        self._last_ts: float = 0.0
        self.on_glasses_data = None    # optional callback(dict)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True,
                                        name="HTTPBridge")
        self._thread.start()
        logger.info("HTTP bridge started (polling for UNO Q data)")

    def _poll_loop(self):
        from core.heat_stress import thermistor_raw_to_celsius

        while self._running:
            data, ts = self._get_data()
            if ts > self._last_ts and data:
                self._last_ts = ts
                self._dispatch(data)
            time.sleep(0.1)

    def _dispatch(self, raw: dict):
        from core.heat_stress import thermistor_raw_to_celsius

        # Convert skin_temp_raw → Celsius; fall back to raw if conversion fails
        skin_raw = int(raw.get("skin_temp_raw", 132))
        try:
            skin_temp_raw_val = skin_raw if skin_raw > 0 else 132
        except (TypeError, ValueError):
            skin_temp_raw_val = 132

        mcu_frame = {
            "emg_raw":       int(raw.get("emg_raw",  512)),
            "heart_rate":    int(raw.get("heart_rate", 0)),
            "spo2":          int(raw.get("spo2",       0)),
            "accel_x":       float(raw.get("accel_x",  0.0)),
            "accel_y":       float(raw.get("accel_y",  0.0)),
            "accel_z":       float(raw.get("accel_z", -1.0)),
            "skin_temp_raw": skin_temp_raw_val,
            "sweat_raw":     int(raw.get("sweat_raw",  0)),
        }
        if self.on_sensor_data:
            self.on_sensor_data(mcu_frame)

        # Ambient fields — feed glasses data path if callback provided
        if self.on_glasses_data and (
            "ambient_temp_c" in raw or "ambient_humidity_pct" in raw
        ):
            glasses_frame = {
                "ambient_temp_c":      float(raw.get("ambient_temp_c",      28.0)),
                "ambient_humidity_pct": float(raw.get("ambient_humidity_pct", 60.0)),
                "is_direct_sun":       bool(raw.get("is_direct_sun",        False)),
                "noise_above_threshold": bool(raw.get("noise_above_threshold", False)),
            }
            self.on_glasses_data(glasses_frame)

    def send_command(self, command: dict):
        pass  # HTTP push to UNO Q not yet implemented


# ─── Test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    bridge = SimulatorBridge(update_hz=5)  # slow for readability

    frame_count = 0
    def on_data(data):
        global frame_count
        frame_count += 1
        if frame_count % 5 == 0:
            print(f"  Frame {frame_count}: "
                  f"dist=({data['dist_front']}, {data['dist_left']}, {data['dist_right']}) "
                  f"HR={data['heart_rate']} SpO2={data['spo2']} "
                  f"EMG={data['emg_raw']}")

    bridge.on_sensor_data = on_data
    bridge.start()

    print("=== Simulating scenarios ===")
    time.sleep(2)
    print("\n→ Obstacle approaching from left")
    bridge.simulate_obstacle("left")
    time.sleep(2)
    print("\n→ Obstacle closing from front")
    bridge.simulate_obstacle("closing")
    time.sleep(2)
    print("\n→ Elevated heart rate")
    bridge.simulate_health("elevated_hr")
    time.sleep(2)
    print("\n→ Back to normal")
    bridge.simulate_obstacle("clear")
    bridge.simulate_health("normal")
    time.sleep(1)

    bridge.stop()
    print(f"\nTotal frames received: {frame_count}")
