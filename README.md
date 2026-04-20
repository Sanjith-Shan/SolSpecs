# FireForce — Firefighter Wildfire AR HUD

🏆 1st Place — Hardware & IOT @ UCSD DataHacks

🏆 Best Use of Gemini API

FireForce is a real-time biometric and situational-awareness system for wildland firefighters. A wearable sensor armband (Arduino UNO Q) streams heart rate, SpO₂, skin temperature, sweat, and IMU data over WiFi to a Python state machine that fuses the signals into an OSHA-compliant heat stress tier (green → yellow → orange → red). The computed state is served over HTTP/HTTPS to a Meta Quest 3 browser, which renders a Three.js VR panorama with a fire-spread simulation, live vitals HUD panels, AI-powered fuel classification overlays, Dijkstra evacuation routing, and Web Speech API voice alerts.

---

## Hardware

| Component | Role |
|-----------|------|
| Arduino UNO Q (Qualcomm Snapdragon Linux) | Armband compute, posts sensor data over WiFi |
| STM32 co-processor (Arduino RPC) | Sensor ADC, IMU, MAX30102 at 20 Hz |
| MPU-9250 | Accelerometer / gyroscope (fall detection, exertion) |
| NTC thermistor (10 kΩ, B=3950) | Skin temperature |
| DHT11/22 (on armband) | Ambient temperature + humidity |
| Meta Quest 3 | AR/VR HUD display |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Sanjith-Shan/FireForce.git
cd FireForce
pip install -r requirements.txt
```

EMG support (optional — only needed with real Mindrove hardware):
```bash
pip install torch pandas pylsl
```

### 2. Set API keys

```bash
export GEMINI_API_KEY="your-gemini-key"        # fuel classification + scene analysis
export ELEVENLABS_API_KEY="your-el-key"        # voice synthesis (optional)
export QUALCOMM_AI_API_KEY="your-qualcomm-key" # LLM conversation (optional)
```

All keys are optional for simulation mode — the system falls back gracefully when they are not set.

---

## Run modes

| Flag | Description |
|------|-------------|
| `--simulate` | SimulatorBridge generates fake sensor data (no hardware required) |
| `--remote` | HTTPBridge polls POST `/sensor-update` for real UNO Q data over WiFi |
| `--live` | SerialBridge reads STM32 over USB serial (on-device only) |
| `--emg` | Enable real Mindrove EMG (requires pylsl + calibration CSVs) |
| `--https` | Generate self-signed cert; required for Quest 3 WebXR |
| `--interactive` | CLI scenario keys for testing (h/c/n/f/s/e/a) |
| `--loglevel` | Logging verbosity: DEBUG / INFO / WARNING |

### Laptop simulation (no hardware)

```bash
python main.py --simulate
# HUD at http://localhost:8080/hud
```

### Quest 3 with HTTPS

```bash
python main.py --simulate --https
# HUD at https://<your-ip>:8443/hud
```

Accept the self-signed certificate warning in the Quest browser (**Advanced → Proceed**), then tap **ENTER VR MODE**.

### Real UNO Q armband over WiFi

```bash
# On the laptop:
python main.py --remote --https

# On the UNO Q armband (Arduino sketch or Python):
# POST JSON to http://<laptop-ip>:8080/sensor-update
# Fields: heart_rate, spo2, accel_x/y/z, skin_temp_raw, sweat_raw, gsr_raw,
#         ambient_temp_c, ambient_humidity_pct, is_direct_sun
```

The startup banner printed at launch shows all relevant URLs.

---

## Data flow

```
UNO Q armband
  └─ POST /sensor-update (JSON, ~20 Hz, WiFi)
        │
        ▼
  Laptop (main.py)
  ├── HTTPBridge → StateMachine → heat tier / vitals
  ├── Flask API  → GET /sensors (JSON)
  └── HTTPS server
        │
        ▼
  Meta Quest 3 browser
  └── Three.js VR HUD (polls /sensors every 500 ms)
```

---

## Gesture map

### Mindrove EMG (wrist)

| Gesture | Action |
|---------|--------|
| Clench | Trigger AI fuel scan (captures VR scene → Gemini Vision) |
| Half-Clench | MAYDAY — broadcasts location + vitals alert |

### Quest 3 controllers

| Button | Action |
|--------|--------|
| Trigger (right) | AI fuel scan |
| B | MAYDAY |
| A | Cycle fire time scrub (−30 min / +30 min / reset) |
| Y | Spoken vitals status (Web Speech) |

### Keyboard (simulation / laptop)

| Key | Action |
|-----|--------|
| F | AI fuel scan |
| M | MAYDAY |
| V | Spoken vitals status |
| C | Mock EMG clench (fuel scan) |
| H | Mock EMG half-clench (MAYDAY) |
| h | Heat spike scenario |
| c | Critical SpO₂ scenario |
| n | Return to normal |
| f | Simulate fall |

---

## Project structure

```
FireForce/
├── main.py                   # Entry point — wires all subsystems + Flask server
├── config.py                 # All thresholds, ports, API keys
├── requirements.txt
├── core/
│   ├── state_machine.py      # Sensor fusion, heat tier, fall detection
│   ├── sensor_server.py      # Flask API: /sensors /status /fire-config /sensor-update /emg-event /hud
│   ├── heat_stress.py        # WBGT estimation + heat stress tier scoring
│   ├── ai_pipeline.py        # Gemini Vision + Qualcomm LLM + ElevenLabs TTS
│   ├── mcu_bridge.py         # SimulatorBridge / SerialBridge / HTTPBridge
│   ├── emg_bridge.py         # Mindrove HD classifier + MockEMGBridge
│   ├── classify.py           # Hyperdimensional EMG gesture classifier (torch)
│   ├── glasses_client.py     # HTTP client to glasses ambient sensor
│   ├── phone_gps_client.py   # GPS client from paired phone
│   └── audio.py              # Text-to-speech priority queue
├── hud/
│   ├── index.html            # Three.js VR/AR HUD (all phases)
│   └── fire_simulation.js    # Cellular automata fire engine + Dijkstra evacuation
├── phone/
│   └── gps_server.py         # Flask server running on the paired phone
└── tests/
    ├── test_heat_stress.py
    ├── test_state_machine.py
    ├── test_sensor_server.py  # Routes: /sensors /status /sensor-update /emg-event /emg-events
    ├── test_mcu_bridge.py     # HTTPBridge data formatting + dispatch
    └── test_emg_bridge.py     # EMGBridge / MockEMGBridge callback wiring
```

---

## Running tests

```bash
pytest tests/ -v
```

252 tests, no hardware required.
