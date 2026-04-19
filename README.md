# 🔥 SolSpecs — AI-Powered Firefighter Heads-Up Display

**An AR safety system for wildland firefighters that combines real-time biometric monitoring, AI-driven flammable object detection, predictive fire spread modeling, and intelligent evacuation routing — all delivered through a helmet-mounted heads-up display**

> *Built for DataHacks @ UC San Diego — Theme: Environment, Climate, Energy Sciences*

---

## The Problem

Wildland firefighter fatalities have increased **500%** over three decades — from 2% to 10% of total firefighter deaths — despite fewer total wildfires. Burn-related fatalities rose from 9% to 27%. **60% of entrapment fatalities occur on just 3% of fire weather days**, when sudden wind shifts catch crews off guard.

The primary defense against wildfire is **fireline construction** — clearing vegetation to starve the fire of fuel. Firefighters work in extreme heat, with limited visibility, making life-or-death decisions about what to clear and when to evacuate. They have no real-time intelligence on fire behavior, no physiological monitoring, and no AI assistance.

ForeSight changes that.

---

## What ForeSight Does

### 🌡️ Heat Stress Shield
Continuously monitors the firefighter's physiological state through a sensor armband:
- **Skin temperature** (NTC thermistor)
- **Sweat / hydration level** (water level sensor)
- **Dehydration detection** (galvanic skin response via EMG electrodes)
- **Ambient temperature + humidity** (DHT11 → WBGT calculation)
- **Fall detection + exertion tracking** (Modulino Movement IMU)
- **Heart rate + SpO2** (MAX30102, when available)

Fuses environmental and physiological signals into a **Wet Bulb Globe Temperature (WBGT)** estimate using the Stull (2011) formula — the same metric OSHA uses for occupational heat safety. A multi-signal risk score escalates through four tiers (🟢 Green → 🟡 Yellow → 🟠 Orange → 🔴 Red), with debounced transitions to prevent false alarms. Voice alerts speak through the headset at each escalation.

### 🔥 Wildfire Spread Prediction
A **Rothermel (1972) cellular automata** fire model — the same physics the US Forest Service uses — simulates fire spread across a 64×64 terrain grid. The simulation accounts for:
- Fuel type (grass, brush, forest, urban, water, rock)
- Wind direction and speed
- Terrain slope (fire spreads faster uphill)
- Fuel burn duration by type

The fire map is overlaid on **real satellite imagery** with semi-transparent fire progression. A time scrubber lets the firefighter preview predicted fire positions at +10, +20, and +30 minutes. Teammate positions are displayed as green dots on the map.

### 🌿 AI Fuel Classification for Firebreak Construction
The flagship environmental feature. When the firefighter clenches their fist (detected by EMG) or pulls the controller trigger, the system captures the current scene and sends it to **Gemini Vision** with a specialized fuel assessment prompt.

Gemini returns structured JSON with bounding boxes identifying every visible fuel source:
- **Dead grass / pine needle litter** → Priority 1-2 (🔴 EXTREME)
- **Dead brush / fallen branches** → Priority 3-4 (🟠 HIGH)
- **Chaparral / living brush** → Priority 5-6 (🟡 MODERATE)
- **Small trees / large trees** → Priority 7-8 (🟢 LOW)

Each fuel source is placed as a **3D marker locked in world space** — it stays anchored to its real-world position as the firefighter looks around, thanks to Quest 3's 6DOF tracking. A priority panel ranks all identified fuel sources with clearing instructions. Voice announces the top priority: *"Dead brush, 3 meters at your 2 o'clock. Scrape to mineral soil."*

This priority chain is based on **NWCG (National Wildfire Coordinating Group) fuel model standards**.

### 🗺️ Intelligent Evacuation Routing
When fire approaches the firefighter's position on the simulation map, a **Dijkstra pathfinding algorithm** computes the safest evacuation route, accounting for:
- Current fire positions
- Predicted fire positions 10 minutes ahead
- Terrain obstacles (water, impassable cells)
- Proximity cost (paths near fire are penalized)

The route renders as a green dashed line on the fire map with an arrowhead. A **compass widget** at the top of the HUD shows a directional arrow, cardinal direction, and estimated distance to safety. Five-tier proximity alerts escalate from warning to automatic MAYDAY.

### 💪 EMG Gesture Control
A **Mindrove 4-channel EMG armband** on the forearm classifies muscle gestures using a **hyperdimensional computing model** (random projection → cosine similarity to trained centroids). Two gestures, confirmed after 1 second of consistent classification:
- **Hard clench** → Trigger fuel scan
- **Half clench** → Activate MAYDAY (transmits biometric data + GPS to incident command)

Hands-free, works through gloves, no buttons needed. Quest 3 controllers serve as backup input.

### 🗣️ Voice Intelligence
All alerts are spoken through the Quest 3 headset using the **Web Speech API**:
- Heat tier escalation warnings
- Fire proximity alerts
- MAYDAY activation confirmation
- Periodic status readouts
- Fuel scan results

A two-tone alarm (800/600 Hz) precedes critical alerts. Ambient fire audio (filtered white noise crackling) adds atmosphere in VR mode.

---

## System Architecture

```
┌──────────────────────┐     HTTP POST      ┌──────────────────────────────┐
│   Arduino UNO Q      │    every 500ms     │         Laptop               │
│   (Armband)          │ ──────────────────► │   python main.py             │
│                      │                     │                              │
│ • Thermistor → A1    │                     │ ┌──────────────────────────┐ │
│ • Water level → A2   │                     │ │ State Machine            │ │
│ • GSR → A3           │                     │ │ • WBGT calculation       │ │
│ • DHT11 → Pin 2      │                     │ │ • Heat tier scoring      │ │
│ • Modulino Movement  │                     │ │ • Fall detection         │ │
│   (Qwiic I2C)        │                     │ │ • Alert generation       │ │
│ • MAX30102 (Qwiic)   │                     │ └───────────┬──────────────┘ │
└──────────────────────┘                     │             │                │
                                             │ ┌───────────▼──────────────┐ │
┌──────────────────────┐     LSL Stream      │ │ Flask Server :8080/8443  │ │
│   Mindrove EMG       │ ──────────────────► │ │ • GET /sensors           │ │
│   (4-channel armband)│                     │ │ • POST /sensor-update    │ │
│                      │                     │ │ • POST /analyze-fuel     │ │
│ • HD classifier      │                     │ │ • GET /emg-events        │ │
│ • Clench → scan      │                     │ │ • GET /hud/*             │ │
│ • Half-clench → SOS  │                     │ └───────────┬──────────────┘ │
└──────────────────────┘                     └─────────────│────────────────┘
                                                           │
                                              HTTP GET every 2s
                                                           │
                                             ┌─────────────▼────────────────┐
                                             │      Meta Quest 3            │
                                             │      (Helmet HUD)            │
                                             │                              │
                                             │ • Three.js WebXR VR/AR       │
                                             │ • Panorama forest scene      │
                                             │ • Billboard vegetation       │
                                             │ • Fire simulation + map      │
                                             │ • 3D fuel markers            │
                                             │ • Evacuation routing         │
                                             │ • Voice alerts               │
                                             │ • Controller input           │
                                             └──────────────────────────────┘

All devices connected via phone WiFi hotspot.
```

---

## Hardware

| Component | Role | Connection |
|-----------|------|------------|
| **Arduino UNO Q** | Central MCU — reads all armband sensors | USB-C power, WiFi to hotspot |
| **Modulino Movement** | Fall detection + exertion (IMU) | Qwiic I2C on UNO Q |
| **NTC Thermistor** | Skin temperature | Analog A1 (voltage divider with 10kΩ) |
| **Water Level Sensor** | Sweat detection | Analog A2 |
| **GSR Electrodes** | Dehydration / skin conductance | Analog A3 (10kΩ pull-down) |
| **DHT11** | Ambient temperature + humidity | Digital pin 2 |
| **MAX30102** *(optional)* | Heart rate + SpO2 | Qwiic I2C (address 0x57) |
| **Mindrove 4ch EMG** | Gesture control (clench / half-clench) | Bluetooth → LSL on laptop |
| **Meta Quest 3** | VR/AR heads-up display | WiFi to hotspot, browser-based |
| **Phone** | WiFi hotspot + GPS | Connects all devices |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js (not required — everything is browser-based)
- Gemini API key (free from [aistudio.google.com](https://aistudio.google.com))

### Installation

```bash
git clone https://github.com/YOUR-USERNAME/SolSpecs.git
cd SolSpecs
pip install -r requirements.txt
```

### Set API Keys

```bash
export GEMINI_API_KEY="your-gemini-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"  # optional
```

### Run Modes

**Simulated data (laptop testing):**
```bash
python main.py --simulate --https
```
Open `https://localhost:8443/hud` in Chrome.

**Real armband sensors over WiFi:**
```bash
python main.py --remote --https
```
UNO Q POSTs sensor data to `http://laptop-ip:8080/sensor-update`.

**Real sensors + EMG armband:**
```bash
python main.py --remote --emg --https
```
Requires Mindrove Connect running with LSL enabled and calibration CSVs present.

### Connect Quest 3

1. Turn on phone hotspot
2. Connect laptop, Quest 3, and UNO Q to the same hotspot
3. Find laptop IP: `python -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()"`
4. On Quest 3 browser: `https://LAPTOP-IP:8443/hud`
5. Accept the self-signed certificate warning (Advanced → Proceed)
6. Tap **ENTER VR MODE** or **ENTER AR MODE**

---

## Controls

| Input | Action | Source |
|-------|--------|--------|
| **Hard clench** | Fuel scan — AI identifies vegetation to clear | Mindrove EMG |
| **Call Symbol + Half clench** | MAYDAY — transmit distress + biometrics | Mindrove EMG |
| Right trigger | Fuel scan (backup) | Quest 3 controller |
| Right B button | MAYDAY (backup) | Quest 3 controller |
| Right A button | Cycle fire prediction (+10/20/30 min) | Quest 3 controller |
| Left Y button | Speak full status readout | Quest 3 controller |
| F key | Fuel scan | Keyboard (dev) |
| M key | MAYDAY | Keyboard (dev) |
| V key | Status readout | Keyboard (dev) |

---

## Project Structure

```
SolSpecs/
├── main.py                       # Entry point — starts server + state machine
├── config.py                     # All thresholds, API keys, prompts
├── requirements.txt
│
├── core/
│   ├── state_machine.py          # Sensor fusion, heat tiers, fall detection
│   ├── sensor_server.py          # Flask: /sensors /analyze-fuel /emg-events /hud
│   ├── sensor_server_mock.py     # Standalone mock for testing without hardware
│   ├── heat_stress.py            # WBGT (Stull 2011) + tier scoring
│   ├── ai_pipeline.py            # Gemini Vision + ElevenLabs TTS
│   ├── emg_bridge.py             # Mindrove EMG → gesture callbacks
│   ├── classify.py               # HD computing EMG classifier
│   ├── mcu_bridge.py             # Serial/Simulator/HTTP bridge to STM32
│   ├── phone_gps_client.py       # GPS from paired phone
│   ├── audio.py                  # TTS priority queue
│   └── qualcomm_llm.py           # Qualcomm Cloud AI client
│
├── hud/
│   ├── index.html                # Three.js WebXR VR/AR HUD (~1500 lines)
│   ├── fire_simulation.js        # Cellular automata + Dijkstra routing
│   ├── panorama.jpg              # World Labs equirectangular panorama
│   ├── satellite.jpg             # Satellite imagery for fire map
│   └── textures/                 # Billboard sprite PNGs
│       ├── tree1.png, tree2.png, tree3.png
│       ├── shrub1.png, shrub2.png, shrub3.png
│       ├── dead_brush1.png, dead_brush2.png
│       └── ground.jpg
│
├── phone/
│   └── gps_server.py             # Flask GPS server for phone
│
└── tests/
    ├── test_heat_stress.py        # WBGT, tier scoring
    ├── test_state_machine.py      # Tier transitions, fall detection
    └── test_sensor_server.py      # Flask routes, endpoints
```

---

## Key Algorithms

**Wet Bulb Globe Temperature (Stull 2011):**
Estimates wet bulb temperature from air temperature and relative humidity, then computes WBGT as 0.7·Tw + 0.2·Tg + 0.1·Ta with a solar radiation offset for direct sun exposure.

**Fire Spread (Rothermel 1972 inspired):**
Each burning cell attempts to ignite 8 neighbors. Spread probability = base_rate[fuel] × wind_factor × slope_factor. Wind factor ranges from 0.3 (against wind) to 2.0+ (with wind). Slope factor increases uphill. Seeded PRNG for deterministic behavior.

**Evacuation Routing (Dijkstra):**
Runs on the fire grid with predicted fire state 10 minutes ahead. Cells near fire have cost 10, clear cells cost 1, burning/water cells are impassable. Finds shortest safe path to any map edge.

**EMG Classification (Hyperdimensional Computing):**
4-channel EMG → extract RMS, MAV, zero crossings, waveform length per channel → random projection into 10,000-D hypervectors → cosine similarity to trained class centroids. 1-second temporal confirmation prevents false positives.




---

## Team

Built at DataHacks 2026, UC San Diego by Sanjith Shanmugavel and Hansel Puthenparambil

---

## License

MIT
