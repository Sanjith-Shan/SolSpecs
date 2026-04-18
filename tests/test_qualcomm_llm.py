"""
Tests for core/qualcomm_llm.py and the updated AIPipeline.
All tests use mocks — no real API calls.
Run with:  pytest tests/test_qualcomm_llm.py -v
"""

import sys
import os
import json
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from core.qualcomm_llm import QualcommLLM, MockQualcommLLM, _build_system_prompt
from core.ai_pipeline import AIPipeline
import config


# ─── Prompt builder ───────────────────────────────────────────────────────────

class TestBuildSystemPrompt:

    def test_no_vitals_returns_base_prompt(self):
        result = _build_system_prompt(None)
        assert result == config.CONVERSATION_SYSTEM_PROMPT

    def test_with_vitals_includes_tier(self):
        vitals = {
            "tier": "orange", "heart_rate": 118, "spo2": 97,
            "skin_temp": 37.8, "ambient_temp": 34.0, "humidity": 68.0,
            "wbgt": 29.3, "sun_exposure_minutes": 38.0,
            "noise_hours_today": 0.0, "work_hours": 3.5,
        }
        result = _build_system_prompt(vitals)
        assert "orange" in result
        assert "118" in result
        assert "97" in result
        assert "37.8" in result
        assert "29.3" in result

    def test_with_vitals_includes_noise_status(self):
        vitals = {
            "tier": "yellow", "heart_rate": 95, "spo2": 98,
            "skin_temp": 37.1, "ambient_temp": 30.0, "humidity": 60.0,
            "wbgt": 26.5, "sun_exposure_minutes": 20.0,
            "noise_hours_today": 2.5, "work_hours": 2.0,
        }
        result = _build_system_prompt(vitals)
        assert "2.5" in result
        assert "threshold" in result.lower() or "85" in result

    def test_with_vitals_noise_safe(self):
        vitals = {
            "tier": "green", "heart_rate": 72, "spo2": 98,
            "skin_temp": 36.5, "ambient_temp": 22.0, "humidity": 50.0,
            "wbgt": 20.0, "sun_exposure_minutes": 0.0,
            "noise_hours_today": 0.0, "work_hours": 1.0,
        }
        result = _build_system_prompt(vitals)
        assert "safe" in result.lower()

    def test_bad_vitals_dont_crash(self):
        result = _build_system_prompt({"tier": "green"})  # missing keys
        assert isinstance(result, str)
        assert len(result) > 0


# ─── MockQualcommLLM ──────────────────────────────────────────────────────────

class TestMockQualcommLLM:

    def setup_method(self):
        self.llm = MockQualcommLLM()

    def test_available(self):
        assert self.llm.available is True

    def test_chat_returns_string(self):
        result = self.llm.chat("How am I doing?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_records_call(self):
        self.llm.chat("Am I okay?")
        assert len(self.llm.calls) == 1
        assert self.llm.calls[0]["user"] == "Am I okay?"

    def test_chat_with_vitals_context(self):
        self.llm.update_vitals({"tier": "orange", "heart_rate": 120, "spo2": 97,
                                 "skin_temp": 37.8, "ambient_temp": 34.0,
                                 "humidity": 68.0, "wbgt": 29.0,
                                 "sun_exposure_minutes": 40.0,
                                 "noise_hours_today": 0.0, "work_hours": 3.0})
        result = self.llm.chat("Should I take a break?")
        assert "break" in result.lower() or "shade" in result.lower()

    def test_chat_emergency_keywords(self):
        result = self.llm.chat("I need help")
        assert "supervisor" in result.lower() or "shade" in result.lower()

    def test_chat_builds_history(self):
        self.llm.chat("First message")
        self.llm.chat("Second message")
        with self.llm._history_lock:
            assert len(self.llm._history) == 4  # 2 turns × (user + assistant)

    def test_clear_history(self):
        self.llm.chat("A message")
        self.llm.clear_history()
        with self.llm._history_lock:
            assert len(self.llm._history) == 0

    def test_stream_chat_yields_strings(self):
        chunks = list(self.llm.stream_chat("How am I doing?"))
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert isinstance(full_text, str)
        assert len(full_text) > 0

    def test_analyze_trend_returns_string(self):
        log = [
            {"timestamp": i * 60, "hr": 80 + i * 3, "spo2": 98,
             "skin_temp": 36.5 + i * 0.1, "wbgt": 26.0 + i * 0.5, "tier": "yellow"}
            for i in range(10)
        ]
        result = self.llm.analyze_trend(log)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_update_vitals_stores_data(self):
        vitals = {"tier": "red", "heart_rate": 145}
        self.llm.update_vitals(vitals)
        with self.llm._vitals_lock:
            assert self.llm._vitals["tier"] == "red"


# ─── QualcommLLM with mocked HTTP ─────────────────────────────────────────────

class TestQualcommLLMHTTP:

    def setup_method(self):
        self.llm = QualcommLLM(api_key="test-key", model="Llama-3.3-70B")

    def _mock_response(self, content: str):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "model": "Llama-3.3-70B",
            "usage": {"completion_tokens": 20, "prompt_tokens": 100, "total_tokens": 120},
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_chat_calls_correct_endpoint(self):
        expected_reply = "Your heat stress level is orange. Take a break."
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._mock_response(expected_reply)

        with patch("httpx.Client", return_value=mock_client):
            result = self.llm.chat("How am I doing?")

        assert result == expected_reply
        call_kwargs = mock_client.post.call_args
        assert "chat/completions" in call_kwargs[0][0]

    def test_chat_sends_api_key_header(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._mock_response("OK")

        with patch("httpx.Client", return_value=mock_client):
            self.llm.chat("Hello")

        headers = mock_client.post.call_args[1]["headers"]
        assert "Authorization" in headers
        assert "test-key" in headers["Authorization"]

    def test_chat_includes_vitals_in_system_prompt(self):
        self.llm.update_vitals({
            "tier": "orange", "heart_rate": 118, "spo2": 97,
            "skin_temp": 37.8, "ambient_temp": 34.0, "humidity": 68.0,
            "wbgt": 29.0, "sun_exposure_minutes": 38.0,
            "noise_hours_today": 0.0, "work_hours": 3.0,
        })
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._mock_response("Stay hydrated.")

        with patch("httpx.Client", return_value=mock_client):
            self.llm.chat("Am I okay?")

        payload = mock_client.post.call_args[1]["json"]
        system_msg = payload["messages"][0]["content"]
        assert "orange" in system_msg
        assert "118" in system_msg

    def test_chat_returns_fallback_on_exception(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = Exception("network error")

        with patch("httpx.Client", return_value=mock_client):
            result = self.llm.chat("Hello")

        assert "trouble" in result.lower() or "available" in result.lower()

    def test_chat_unavailable_without_api_key(self):
        llm = QualcommLLM(api_key="")
        result = llm.chat("Hello")
        assert "available" in result.lower() or "not" in result.lower()

    def test_history_trimmed_to_max(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._mock_response("reply")

        with patch("httpx.Client", return_value=mock_client):
            self.llm._max_history = 4  # 2 turns max
            for i in range(5):
                self.llm.chat(f"message {i}")

        with self.llm._history_lock:
            assert len(self.llm._history) <= 4

    def test_analyze_trend_uses_reasoning_model(self):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = self._mock_response("Take a break soon.")

        log = [{"timestamp": i * 60, "hr": 90 + i, "spo2": 97,
                "skin_temp": 37.0 + i * 0.1, "wbgt": 27.0, "tier": "yellow"}
               for i in range(10)]

        with patch("httpx.Client", return_value=mock_client):
            result = self.llm.analyze_trend(log)

        payload = mock_client.post.call_args[1]["json"]
        assert payload["model"] == config.QUALCOMM_AI_REASONING_MODEL
        assert result == "Take a break soon."


# ─── AIPipeline integration ───────────────────────────────────────────────────

class TestAIPipelineWithQualcommLLM:

    def setup_method(self):
        self.pipeline = AIPipeline(simulate=True)  # uses MockQualcommLLM

    def test_chat_delegates_to_llm(self):
        result = self.pipeline.chat("How am I doing?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_update_vitals_reaches_llm(self):
        vitals = {"tier": "yellow", "heart_rate": 95, "spo2": 98,
                  "skin_temp": 37.1, "ambient_temp": 30.0, "humidity": 60.0,
                  "wbgt": 26.5, "sun_exposure_minutes": 20.0,
                  "noise_hours_today": 0.0, "work_hours": 2.0}
        self.pipeline.update_vitals(vitals)
        with self.pipeline.llm._vitals_lock:
            assert self.pipeline.llm._vitals["tier"] == "yellow"

    def test_chat_with_vitals_is_context_aware(self):
        self.pipeline.update_vitals({"tier": "orange", "heart_rate": 120,
                                      "spo2": 97, "skin_temp": 37.9,
                                      "ambient_temp": 35.0, "humidity": 70.0,
                                      "wbgt": 30.0, "sun_exposure_minutes": 45.0,
                                      "noise_hours_today": 0.0, "work_hours": 4.0})
        result = self.pipeline.chat("Should I keep working?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stream_chat_yields_tokens(self):
        chunks = list(self.pipeline.stream_chat("Am I okay?"))
        assert len(chunks) > 0

    def test_analyze_trend_returns_string(self):
        log = [{"timestamp": i * 60, "hr": 85 + i, "spo2": 97,
                "skin_temp": 37.0, "wbgt": 27.0, "tier": "yellow"}
               for i in range(10)]
        result = self.pipeline.analyze_trend(log)
        assert isinstance(result, str)

    def test_llm_is_mock_in_simulate_mode(self):
        assert isinstance(self.pipeline.llm, MockQualcommLLM)

    def test_llm_is_real_in_live_mode(self):
        pipeline = AIPipeline(simulate=False)
        assert isinstance(pipeline.llm, QualcommLLM)
        assert not isinstance(pipeline.llm, MockQualcommLLM)
