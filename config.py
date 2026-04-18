"""
SolSpecs — Configuration
All thresholds, URLs, and settings in one place.
"""

import os

# ─── Network ───────────────────────────────────────────────────────
GLASSES_URL = os.environ.get("GLASSES_URL", "http://solspecs-glasses.local:5001")
PHONE_GPS_URL = os.environ.get("PHONE_GPS_URL", "http://192.168.1.100:5000")
MODE = os.environ.get("SOLSPECS_MODE", "simulate")  # "simulate" or "live"

# ─── API Keys ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
QUALCOMM_AI_API_KEY = os.environ.get("QUALCOMM_AI_API_KEY", "")

# ─── Heat Stress Thresholds ────────────────────────────────────────
# OSHA WBGT thresholds for moderate workload
WBGT_LOW = 24.0       # below this = green
WBGT_MODERATE = 26.0  # yellow territory
WBGT_HIGH = 28.0      # orange territory
WBGT_CRITICAL = 30.0  # red territory

# Tier score thresholds (composite score from all signals)
TIER_YELLOW = 3
TIER_ORANGE = 6
TIER_RED = 10

# ─── Physiological Thresholds ─────────────────────────────────────
HR_MILD = 90
HR_CONCERN = 100
HR_ELEVATED = 120
HR_CRITICAL = 140
HR_ELEVATED_DURATION_S = 30.0

SPO2_MILD = 95
SPO2_CONCERN = 93
SPO2_CRITICAL = 90

SKIN_TEMP_MILD = 37.0
SKIN_TEMP_CONCERN = 37.5
SKIN_TEMP_CRITICAL = 38.5

BREATHING_RATE_HIGH = 25  # breaths per minute
BREATHING_RATE_LOW = 8

# ─── Sun Exposure ─────────────────────────────────────────────────
SUN_EXPOSURE_MILD_MIN = 15
SUN_EXPOSURE_CONCERN_MIN = 30
SUN_EXPOSURE_CRITICAL_MIN = 45
LIGHT_THRESHOLD_DIRECT_SUN = 700  # ADC value, calibrate at venue

# ─── Noise Exposure ───────────────────────────────────────────────
NOISE_THRESHOLD_DB = 85  # OSHA 8-hour limit
NOISE_ALERT_HOURS = 2.0  # alert after this many hours above threshold

# ─── Fall Detection ───────────────────────────────────────────────
FALL_ACCEL_THRESHOLD_G = 3.0
FALL_RESPONSE_TIMEOUT_S = 15.0

# ─── EMG ──────────────────────────────────────────────────────────
EMG_FLEX_THRESHOLD = 300     # ADC, calibrate per user
EMG_SUSTAIN_MS = 800         # hold duration for "scan" gesture
EMG_COOLDOWN_MS = 2000       # min time between gestures

# ─── Polling Intervals ────────────────────────────────────────────
GLASSES_POLL_INTERVAL_S = 2.0
PHONE_GPS_POLL_INTERVAL_S = 5.0
MCU_SENSOR_HZ = 20
STATUS_CHECK_GREEN_MIN = 30
STATUS_CHECK_YELLOW_MIN = 10
STATUS_CHECK_ORANGE_MIN = 5

# ─── Audio ────────────────────────────────────────────────────────
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George
ELEVENLABS_MODEL = "eleven_turbo_v2_5"

# ─── Qualcomm Cloud AI (Cirrascale) ──────────────────────────────
QUALCOMM_AI_BASE_URL = "https://aisuite.cirrascale.com/apis/v2"
QUALCOMM_AI_MODEL = "Llama-3.3-70B"              # default: best quality
QUALCOMM_AI_REASONING_MODEL = "DeepSeek-R1-Distill-Llama-70B"  # trend analysis
QUALCOMM_AI_FAST_MODEL = "Llama-3.1-8B"          # fallback if latency matters
QUALCOMM_AI_TIMEOUT = 30.0                        # seconds

# ─── Gemini ───────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"

SCENE_PROMPT = """You are an AI safety system for an outdoor worker. Analyze this image of their work environment.
Report concisely (under 4 sentences):
1. Sun exposure: Is the worker in direct sunlight or shade?
2. Nearest shade: Where is the closest shaded or covered area, and roughly how far?
3. Hazards: Any visible safety risks — unguarded edges, moving equipment, trip hazards, unstable ground?
4. Hydration: Is water or a cooling station visible?
Speak directly to the worker. Use simple directions like "to your left" or "behind you"."""

CONVERSATION_SYSTEM_PROMPT = """You are an AI safety assistant built into a wearable for outdoor workers.
You help them stay safe in extreme heat conditions and assess their work environment.
Keep responses concise and spoken-word friendly. No markdown, no bullet points.
Use simple directions and be specific about distances.
Always prioritize heat safety and hazard warnings.
If asked about their vitals, reference the most recent sensor data available."""
