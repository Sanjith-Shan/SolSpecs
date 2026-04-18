"""
SolSpecs — Audio Output Manager
ElevenLabs TTS with a priority queue for voice alerts.

Priority levels:
    0 = CRITICAL  (red alerts — interrupts everything)
    1 = HIGH       (orange alerts — waits for current word boundary)
    2 = NORMAL     (status updates, yellow alerts)
    3 = LOW        (periodic green status checks)

Red alerts preempt any currently playing audio.
Lower-priority alerts are dropped if a higher-priority alert is already queued.
"""

import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("Audio")

try:
    from elevenlabs import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger.warning("elevenlabs not installed. Run: pip install elevenlabs")

import config


PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2
PRIORITY_LOW = 3


@dataclass(order=True)
class _AudioItem:
    priority: int
    sequence: int = field(compare=True)
    text: str = field(compare=False)
    interrupt: bool = field(compare=False, default=False)


class AudioManager:
    """
    Manages a priority queue of TTS alerts and plays them through the
    Bluetooth earbuds via the system audio stack (mpv / afplay / aplay).

    Usage:
        audio = AudioManager()
        audio.start()
        audio.speak("Heat stress warning.", priority=PRIORITY_HIGH)
        audio.speak("Danger. Stop work.", priority=PRIORITY_CRITICAL)
        audio.stop()
    """

    def __init__(self):
        self._client: Optional[ElevenLabs] = None
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._sequence = 0
        self._seq_lock = threading.Lock()

        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._current_proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()

        self._min_queued_priority = PRIORITY_LOW + 1  # sentinel: nothing queued

        self._init_client()

    def _init_client(self):
        if not ELEVENLABS_AVAILABLE:
            return
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            logger.warning("ELEVENLABS_API_KEY not set — audio will log only")
            return
        self._client = ElevenLabs(api_key=api_key)
        logger.info("ElevenLabs client initialized")

    def start(self):
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("Audio manager started")

    def stop(self):
        self._running = False
        self._interrupt_playback()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Audio manager stopped")

    # ── Public API ────────────────────────────────────────────────────

    def speak(self, text: str, priority: int = PRIORITY_NORMAL):
        """
        Queue text for TTS playback.

        Critical priority (0) also sets the interrupt flag so any currently
        playing audio is killed immediately.
        """
        with self._seq_lock:
            seq = self._sequence
            self._sequence += 1

        interrupt = priority == PRIORITY_CRITICAL

        # Drop lower-priority items if something more urgent is already queued
        if priority > self._min_queued_priority:
            logger.debug(f"Dropping lower-priority alert: {text[:40]!r}")
            return

        item = _AudioItem(priority=priority, sequence=seq, text=text, interrupt=interrupt)
        self._queue.put(item)
        self._min_queued_priority = min(self._min_queued_priority, priority)

        if interrupt:
            self._interrupt_playback()

        logger.debug(f"Queued [{priority}]: {text[:60]!r}")

    def is_speaking(self) -> bool:
        with self._proc_lock:
            return self._current_proc is not None and self._current_proc.poll() is None

    def clear_queue(self):
        """Discard all pending alerts."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._min_queued_priority = PRIORITY_LOW + 1

    # ── Internal worker ───────────────────────────────────────────────

    def _worker_loop(self):
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._min_queued_priority = PRIORITY_LOW + 1
                continue

            # Recalculate min queued priority after dequeue
            self._min_queued_priority = PRIORITY_LOW + 1

            self._play_item(item)
            self._queue.task_done()

    def _play_item(self, item: _AudioItem):
        logger.info(f"Speaking [{item.priority}]: {item.text[:80]!r}")
        audio_bytes = self._synthesize(item.text)
        if audio_bytes:
            self._play_audio(audio_bytes)
        else:
            # Fallback: log to console so testing works without API key
            print(f"[AUDIO] {item.text}")

    def _synthesize(self, text: str) -> Optional[bytes]:
        if not self._client:
            return None
        try:
            audio = self._client.text_to_speech.convert(
                text=text,
                voice_id=config.ELEVENLABS_VOICE_ID,
                model_id=config.ELEVENLABS_MODEL,
                output_format="mp3_22050_32",
            )
            return b"".join(audio)
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    def _play_audio(self, audio_bytes: bytes):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            proc = self._launch_player(tmp_path)
            if proc:
                with self._proc_lock:
                    self._current_proc = proc
                proc.wait()
                with self._proc_lock:
                    self._current_proc = None
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _launch_player(self, path: str) -> Optional[subprocess.Popen]:
        players = [
            ["mpv", "--no-video", "--really-quiet", path],
            ["afplay", path],
            ["aplay", path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        ]
        for cmd in players:
            try:
                return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                continue
        logger.warning("No audio player found (mpv/afplay/aplay/ffplay). Install one.")
        return None

    def _interrupt_playback(self):
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                self._current_proc.terminate()
                logger.debug("Interrupted current audio playback")


class MockAudioManager(AudioManager):
    """
    Drop-in audio manager for testing — never calls ElevenLabs or plays audio.
    Records all spoken text for test assertions.
    """

    def __init__(self):
        super().__init__()
        self.spoken: list[tuple[int, str]] = []  # (priority, text)
        self._mock_lock = threading.Lock()

    def _play_item(self, item: _AudioItem):
        with self._mock_lock:
            self.spoken.append((item.priority, item.text))
        logger.info(f"[mock audio {item.priority}] {item.text}")

    def last_spoken(self) -> Optional[str]:
        with self._mock_lock:
            return self.spoken[-1][1] if self.spoken else None

    def spoken_count(self) -> int:
        with self._mock_lock:
            return len(self.spoken)
