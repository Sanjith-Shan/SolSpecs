"""
SolSpecs — Qualcomm Cloud AI Client
Wraps the Cirrascale inference API (OpenAI-compatible) running on
Qualcomm AI 100 Ultra hardware.

Responsibilities:
  - Conversation with live vitals injected into the system prompt
  - Streaming support (SSE) for low time-to-first-token
  - Trend reasoning via DeepSeek-R1 (called on demand, not in hot path)
  - Sync interface so callers don't need to manage an event loop

Gemini Vision is kept for image analysis (Cirrascale has no vision model).
This client handles text-only conversation and reasoning.
"""

import json
import logging
import threading
import time
from typing import Iterator, Optional

import config

logger = logging.getLogger("QualcommLLM")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed. Run: pip install httpx")


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_BASE = config.CONVERSATION_SYSTEM_PROMPT

_VITALS_TEMPLATE = """
Current worker status (live sensor data):
  Heat tier:        {tier}
  Heart rate:       {heart_rate} bpm
  SpO2:             {spo2}%
  Skin temperature: {skin_temp:.1f}°C
  Ambient temp:     {ambient_temp:.1f}°C  |  Humidity: {humidity:.0f}%
  WBGT estimate:    {wbgt:.1f}°C
  Sun exposure:     {sun_exposure_min:.0f} min this hour
  Noise exposure:   {noise_status}
  Work session:     {work_hours:.1f} hours
"""

_TREND_SYSTEM = (
    "You are a heat stress risk analyst for an outdoor worker safety system. "
    "Analyze the provided time-series sensor data and reason about risk trajectory. "
    "Be concise. Use plain language suitable for text-to-speech. No markdown."
)


def _build_system_prompt(vitals: Optional[dict]) -> str:
    if not vitals:
        return _SYSTEM_BASE
    try:
        noise_h = vitals.get("noise_hours_today", 0.0)
        noise_status = (
            f"{noise_h:.1f} hours above {config.NOISE_THRESHOLD_DB} dB threshold"
            if noise_h > 0.1
            else "within safe limits"
        )
        context = _VITALS_TEMPLATE.format(
            tier=vitals.get("tier", "unknown"),
            heart_rate=vitals.get("heart_rate", 0),
            spo2=vitals.get("spo2", 0),
            skin_temp=vitals.get("skin_temp", 0.0),
            ambient_temp=vitals.get("ambient_temp", 0.0),
            humidity=vitals.get("humidity", 0.0),
            wbgt=vitals.get("wbgt", 0.0),
            sun_exposure_min=vitals.get("sun_exposure_minutes", 0.0),
            noise_status=noise_status,
            work_hours=vitals.get("work_hours", 0.0),
        )
        return _SYSTEM_BASE + context
    except Exception:
        return _SYSTEM_BASE


# ── Main client ───────────────────────────────────────────────────────────────

class QualcommLLM:
    """
    Sync client for the Cirrascale Qualcomm Cloud AI endpoint.

    Usage:
        llm = QualcommLLM()
        llm.update_vitals({"tier": "orange", "heart_rate": 118, ...})
        reply = llm.chat("Should I take a break?")
        print(reply)
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ):
        self._api_key = api_key or config.QUALCOMM_AI_API_KEY
        self._model = model or config.QUALCOMM_AI_MODEL
        self._base_url = (base_url or config.QUALCOMM_AI_BASE_URL).rstrip("/")
        self._timeout = config.QUALCOMM_AI_TIMEOUT

        self._vitals: Optional[dict] = None
        self._vitals_lock = threading.Lock()

        self._history: list[dict] = []
        self._history_lock = threading.Lock()
        self._max_history = 20  # keep last N turns

        if not HTTPX_AVAILABLE:
            logger.error("httpx required. Run: pip install httpx")
        if not self._api_key:
            logger.warning("QUALCOMM_AI_API_KEY not set — LLM calls will fail")
        else:
            logger.info(f"QualcommLLM ready (model={self._model})")

    @property
    def available(self) -> bool:
        return HTTPX_AVAILABLE and bool(self._api_key)

    # ── Vitals context ────────────────────────────────────────────────

    def update_vitals(self, vitals: dict):
        """
        Push current sensor state. Injected into every subsequent chat call.
        Call this whenever tier or key vitals change — not every sensor frame.
        """
        with self._vitals_lock:
            self._vitals = dict(vitals)

    # ── Conversation ──────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Send a message and return the full reply (blocking).
        Maintains conversation history across calls.
        """
        if not self.available:
            return self._unavailable_response()

        with self._vitals_lock:
            vitals = dict(self._vitals) if self._vitals else None

        system_prompt = _build_system_prompt(vitals)

        with self._history_lock:
            history = list(self._history)
            history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": system_prompt}] + history

        start = time.time()
        try:
            reply = self._post_chat(messages, stream=False)
            elapsed = time.time() - start
            logger.info(f"Chat response in {elapsed:.1f}s: {reply[:80]!r}")

            with self._history_lock:
                self._history.append({"role": "user", "content": user_message})
                self._history.append({"role": "assistant", "content": reply})
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

            return reply

        except Exception as e:
            logger.error(f"QualcommLLM chat failed: {e}")
            return "I had trouble responding. Please try again."

    def stream_chat(self, user_message: str) -> Iterator[str]:
        """
        Stream the reply token-by-token. Yields text chunks as they arrive.
        Useful for starting TTS on the first sentence before generation finishes.

        Usage:
            for chunk in llm.stream_chat("How am I doing?"):
                print(chunk, end="", flush=True)
        """
        if not self.available:
            yield self._unavailable_response()
            return

        with self._vitals_lock:
            vitals = dict(self._vitals) if self._vitals else None

        system_prompt = _build_system_prompt(vitals)

        with self._history_lock:
            history = list(self._history)
            history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": system_prompt}] + history

        full_reply = []
        try:
            for chunk in self._stream_chat(messages):
                full_reply.append(chunk)
                yield chunk

            reply_text = "".join(full_reply)
            with self._history_lock:
                self._history.append({"role": "user", "content": user_message})
                self._history.append({"role": "assistant", "content": reply_text})
                if len(self._history) > self._max_history:
                    self._history = self._history[-self._max_history:]

        except Exception as e:
            logger.error(f"QualcommLLM stream failed: {e}")
            yield "I had trouble responding."

    def clear_history(self):
        with self._history_lock:
            self._history = []

    # ── Trend reasoning (DeepSeek-R1) ────────────────────────────────

    def analyze_trend(self, sensor_log: list[dict]) -> str:
        """
        Send a rolling sensor history to DeepSeek-R1 for risk trajectory analysis.
        Returns a 1-2 sentence spoken recommendation.

        Args:
            sensor_log: List of dicts, each with keys: timestamp, hr, spo2,
                        skin_temp, wbgt, tier. Typically last 30 minutes.
        """
        if not self.available:
            return self._unavailable_response()

        summary_lines = []
        for entry in sensor_log[-20:]:  # cap at 20 data points
            t = entry.get("timestamp", 0)
            summary_lines.append(
                f"  t={int(t)}s  HR={entry.get('hr', '?')}  "
                f"SpO2={entry.get('spo2', '?')}%  "
                f"skin={entry.get('skin_temp', '?'):.1f}°C  "
                f"WBGT={entry.get('wbgt', '?'):.1f}°C  "
                f"tier={entry.get('tier', '?')}"
            )

        prompt = (
            "Here is 30 minutes of sensor data for an outdoor construction worker:\n"
            + "\n".join(summary_lines)
            + "\n\nIn 1-2 spoken sentences: what is the risk trend and what should "
            "the worker do in the next 15 minutes? Be direct and specific."
        )

        messages = [
            {"role": "system", "content": _TREND_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        try:
            return self._post_chat(
                messages,
                stream=False,
                model_override=config.QUALCOMM_AI_REASONING_MODEL,
            )
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return "Unable to analyze trend at this time."

    # ── HTTP layer ────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _post_chat(
        self,
        messages: list[dict],
        stream: bool = False,
        model_override: Optional[str] = None,
    ) -> str:
        model = model_override or self._model
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": stream,
        }
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def _stream_chat(self, messages: list[dict]) -> Iterator[str]:
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.7,
            "stream": True,
        }
        with httpx.Client(timeout=self._timeout) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk["choices"][0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                yield text
                        except (json.JSONDecodeError, KeyError):
                            continue

    @staticmethod
    def _unavailable_response() -> str:
        return "AI assistant is not available right now."


# ── Mock for testing ──────────────────────────────────────────────────────────

class MockQualcommLLM(QualcommLLM):
    """
    Drop-in mock — no network calls. Returns canned context-aware responses.
    Records all calls for test assertions.
    """

    def __init__(self):
        # Skip parent __init__ network checks
        self._api_key = "mock"
        self._model = config.QUALCOMM_AI_MODEL
        self._base_url = config.QUALCOMM_AI_BASE_URL
        self._timeout = 5.0
        self._vitals: Optional[dict] = None
        self._vitals_lock = threading.Lock()
        self._history: list[dict] = []
        self._history_lock = threading.Lock()
        self._max_history = 20
        self.calls: list[dict] = []  # records every chat call

    @property
    def available(self) -> bool:
        return True

    def chat(self, user_message: str) -> str:
        with self._vitals_lock:
            vitals = dict(self._vitals) if self._vitals else {}

        tier = vitals.get("tier", "green")
        hr = vitals.get("heart_rate", 72)

        # Minimal context-aware canned responses
        if any(w in user_message.lower() for w in ("okay", "ok", "fine", "doing", "status")):
            reply = (
                f"Your heat stress level is {tier}. "
                f"Heart rate is {hr} beats per minute. "
                + ("Take it easy and drink water." if tier in ("yellow", "orange")
                   else "You're within safe limits, keep it up.")
            )
        elif any(w in user_message.lower() for w in ("break", "rest", "stop")):
            reply = "Yes, a 10-minute shade break with water is recommended at this heat level."
        elif any(w in user_message.lower() for w in ("help", "danger", "emergency")):
            reply = "Alert your supervisor immediately and move to the nearest shaded area."
        else:
            reply = f"Current heat tier is {tier}. Stay hydrated and monitor your surroundings."

        self.calls.append({"user": user_message, "reply": reply, "vitals": dict(vitals)})

        with self._history_lock:
            self._history.append({"role": "user", "content": user_message})
            self._history.append({"role": "assistant", "content": reply})

        return reply

    def stream_chat(self, user_message: str) -> Iterator[str]:
        reply = self.chat(user_message)
        # Simulate streaming by yielding word by word
        for word in reply.split():
            yield word + " "

    def analyze_trend(self, sensor_log: list[dict]) -> str:
        reply = "Heart rate has been climbing steadily. A 15-minute shade break is recommended now."
        self.calls.append({"type": "trend", "log_len": len(sensor_log), "reply": reply})
        return reply
