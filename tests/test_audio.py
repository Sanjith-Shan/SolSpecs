"""
Tests for core/audio.py
Uses MockAudioManager — no ElevenLabs API key or audio hardware needed.
Run with:  pytest tests/test_audio.py -v
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.audio import (
    AudioManager, MockAudioManager,
    PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW,
)


class TestMockAudioManager:

    def setup_method(self):
        self.audio = MockAudioManager()
        self.audio.start()

    def teardown_method(self):
        self.audio.stop()

    def test_speak_fires_callback(self):
        self.audio.speak("Hello.", priority=PRIORITY_NORMAL)
        time.sleep(0.2)
        assert self.audio.spoken_count() >= 1

    def test_spoken_text_matches(self):
        self.audio.speak("Heat stress warning.", priority=PRIORITY_HIGH)
        time.sleep(0.2)
        texts = [t for _, t in self.audio.spoken]
        assert "Heat stress warning." in texts

    def test_priority_ordering(self):
        """Higher-priority items should be played before lower-priority ones
        when queued simultaneously."""
        # Pause worker briefly to let queue fill, then resume
        self.audio._running = False  # pause
        time.sleep(0.05)

        self.audio.speak("Low priority.", priority=PRIORITY_LOW)
        self.audio.speak("Critical now.", priority=PRIORITY_CRITICAL)
        self.audio.speak("Normal alert.", priority=PRIORITY_NORMAL)

        self.audio._running = True
        # Re-drain the queue manually since we paused the worker
        import queue as q_mod
        items = []
        while True:
            try:
                item = self.audio._queue.get_nowait()
                items.append(item)
            except q_mod.Empty:
                break
        items.sort()
        texts = [i.text for i in items]
        assert texts[0] == "Critical now."

    def test_last_spoken(self):
        self.audio.speak("First.", priority=PRIORITY_NORMAL)
        self.audio.speak("Second.", priority=PRIORITY_NORMAL)
        time.sleep(0.3)
        # last_spoken should be the most recently played item
        assert self.audio.last_spoken() is not None

    def test_clear_queue_empties_pending(self):
        self.audio._running = False  # pause worker
        time.sleep(0.05)
        self.audio.speak("Will be cleared.", priority=PRIORITY_LOW)
        self.audio.clear_queue()
        assert self.audio._queue.empty()
        self.audio._running = True

    def test_spoken_count_increments(self):
        before = self.audio.spoken_count()
        self.audio.speak("One.", priority=PRIORITY_NORMAL)
        self.audio.speak("Two.", priority=PRIORITY_NORMAL)
        time.sleep(0.4)
        assert self.audio.spoken_count() >= before + 2

    def test_stop_does_not_raise(self):
        self.audio.speak("Stopping soon.", priority=PRIORITY_NORMAL)
        time.sleep(0.1)
        self.audio.stop()

    def test_all_priority_levels_accepted(self):
        for p in (PRIORITY_CRITICAL, PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW):
            self.audio.speak(f"Priority {p}.", priority=p)
        time.sleep(0.5)
        assert self.audio.spoken_count() >= 1


class TestAudioManagerPriorityConstants:

    def test_critical_is_lowest_number(self):
        assert PRIORITY_CRITICAL < PRIORITY_HIGH < PRIORITY_NORMAL < PRIORITY_LOW

    def test_priority_values(self):
        assert PRIORITY_CRITICAL == 0
        assert PRIORITY_HIGH == 1
        assert PRIORITY_NORMAL == 2
        assert PRIORITY_LOW == 3


class TestAudioItemOrdering:

    def test_lower_priority_number_sorts_first(self):
        from core.audio import _AudioItem
        a = _AudioItem(priority=PRIORITY_CRITICAL, sequence=10, text="critical")
        b = _AudioItem(priority=PRIORITY_LOW, sequence=1, text="low")
        assert a < b  # critical (0) sorts before low (3)

    def test_same_priority_sequence_breaks_ties(self):
        from core.audio import _AudioItem
        a = _AudioItem(priority=PRIORITY_NORMAL, sequence=1, text="first")
        b = _AudioItem(priority=PRIORITY_NORMAL, sequence=2, text="second")
        assert a < b  # earlier sequence plays first
