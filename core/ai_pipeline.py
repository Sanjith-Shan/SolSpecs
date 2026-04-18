"""
SolSpecs — AI Pipeline
Scene description (Gemini Vision) and voice output (ElevenLabs TTS).
Conversation powered by Qualcomm Cloud AI (Cirrascale / Llama-3.3-70B).

Responsibilities:
  describe_scene()   → Gemini Vision  (image → safety-focused text)
  chat()             → Qualcomm LLM   (conversation with live vitals context)
  speak()            → ElevenLabs TTS (text → MP3 bytes)
  speak_and_play()   → TTS + local playback (for testing)

Can be tested on laptop with API keys set:
    export GEMINI_API_KEY="..."
    export ELEVENLABS_API_KEY="..."
    export QUALCOMM_AI_API_KEY="..."
"""

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import config
from core.qualcomm_llm import QualcommLLM, MockQualcommLLM

logger = logging.getLogger("AIPipeline")

# ── Gemini setup ──────────────────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed. Run: pip install google-genai")

# ── ElevenLabs setup ──────────────────────────────────────────────────────────

try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger.warning("elevenlabs not installed. Run: pip install elevenlabs")


class AIPipeline:
    """
    Coordinates Gemini Vision, Qualcomm LLM, and ElevenLabs TTS.

    Gemini handles image analysis (Cirrascale has no vision model).
    Qualcomm LLM handles all text conversation with live vitals context.
    ElevenLabs converts text to speech.
    """

    def __init__(self, simulate: bool = False):
        self.gemini_client = None
        self.elevenlabs_client = None
        self.last_scene_description = ""

        if simulate:
            self.llm: QualcommLLM = MockQualcommLLM()
            logger.info("AIPipeline: using MockQualcommLLM")
        else:
            self.llm = QualcommLLM()

        self._init_gemini()
        self._init_elevenlabs()

    def _init_gemini(self):
        if not GEMINI_AVAILABLE:
            return
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("No GEMINI_API_KEY set. Scene description unavailable.")
            return
        self.gemini_client = genai.Client(api_key=api_key)
        logger.info("Gemini Vision client initialized")

    def _init_elevenlabs(self):
        if not ELEVENLABS_AVAILABLE:
            return
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("No ELEVENLABS_API_KEY set. Voice output unavailable.")
            return
        self.elevenlabs_client = ElevenLabs(api_key=api_key)
        logger.info("ElevenLabs TTS client initialized")

    # ── Vitals context ────────────────────────────────────────────────

    def update_vitals(self, vitals: dict):
        """
        Push current sensor state into the LLM's system prompt.
        Call this on every tier change or significant vital shift.

        Expected keys: tier, heart_rate, spo2, skin_temp, ambient_temp,
                       humidity, wbgt, sun_exposure_minutes,
                       noise_hours_today, work_hours.
        """
        self.llm.update_vitals(vitals)

    # ── Scene description (Gemini Vision) ────────────────────────────

    def describe_scene(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        """
        Send an image to Gemini Vision with an outdoor worker safety prompt.
        Returns a spoken-word safety assessment of the environment.
        """
        if not self.gemini_client:
            logger.warning("Gemini not available — skipping scene description")
            return "Scene description is not available right now."

        start = time.time()
        try:
            response = self.gemini_client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    config.SCENE_PROMPT,
                ],
            )
            description = response.text
            logger.info(f"Scene described in {time.time()-start:.1f}s: {description[:80]!r}")

            self.last_scene_description = description

            # Add scene to LLM conversation history as context
            self.llm._history_lock.acquire()
            try:
                self.llm._history.append({
                    "role": "user",
                    "content": "[Camera scan of work environment]",
                })
                self.llm._history.append({
                    "role": "assistant",
                    "content": description,
                })
                if len(self.llm._history) > self.llm._max_history:
                    self.llm._history = self.llm._history[-self.llm._max_history:]
            finally:
                self.llm._history_lock.release()

            return description

        except Exception as e:
            logger.error(f"Gemini scene description failed: {e}")
            return "I had trouble analyzing the scene. Please try again."

    def describe_scene_from_file(self, image_path: str) -> str:
        path = Path(image_path)
        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        return self.describe_scene(path.read_bytes(), mime)

    # ── Conversation (Qualcomm LLM) ───────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Conversational response from Llama-3.3-70B on Qualcomm Cloud AI.
        Live vitals are injected into the system prompt automatically.
        """
        return self.llm.chat(user_message)

    def stream_chat(self, user_message: str):
        """Yields text chunks as they arrive from the LLM."""
        return self.llm.stream_chat(user_message)

    def analyze_trend(self, sensor_log: list[dict]) -> str:
        """
        Send rolling sensor history to DeepSeek-R1 for trend reasoning.
        Called periodically in yellow/orange tier — not in the alert hot path.
        """
        return self.llm.analyze_trend(sensor_log)

    # ── Voice output (ElevenLabs) ────────────────────────────────────

    def speak(self, text: str) -> Optional[bytes]:
        """Convert text to speech. Returns MP3 bytes or None."""
        if not self.elevenlabs_client:
            logger.info(f"[WOULD SAY]: {text}")
            return None

        start = time.time()
        try:
            audio = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=config.ELEVENLABS_VOICE_ID,
                model_id=config.ELEVENLABS_MODEL,
                output_format="mp3_22050_32",
            )
            audio_bytes = b"".join(audio)
            logger.info(f"TTS in {time.time()-start:.1f}s, {len(audio_bytes)} bytes")
            return audio_bytes
        except Exception as e:
            logger.error(f"ElevenLabs TTS failed: {e}")
            return None

    def speak_and_play(self, text: str):
        """Synthesize and immediately play audio. Used for testing."""
        audio = self.speak(text)
        if audio:
            self._play_audio(audio)
        else:
            print(f"[WOULD SAY]: {text}")

    def _play_audio(self, audio_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            for player in ["mpv", "afplay", "aplay", "ffplay"]:
                try:
                    cmd = (
                        [player, "--no-video", tmp_path]
                        if player == "mpv"
                        else [player, tmp_path]
                    )
                    subprocess.run(cmd, capture_output=True, timeout=30)
                    return
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            logger.warning("No audio player found. Install mpv or ffplay.")
        finally:
            os.unlink(tmp_path)

    # ── Full pipeline ─────────────────────────────────────────────────

    def capture_describe_speak(self, image_bytes: bytes) -> str:
        """Image → Gemini safety description → ElevenLabs speech."""
        description = self.describe_scene(image_bytes)
        self.speak_and_play(description)
        return description
