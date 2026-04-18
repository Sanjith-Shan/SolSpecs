"""
Tests for core/phone_gps_client.py
Run with:  pytest tests/test_phone_gps_client.py -v
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from core.phone_gps_client import PhoneGPSClient, MockGPSClient, GPSLocation


class TestGPSLocation:

    def test_is_namedtuple(self):
        loc = GPSLocation(latitude=32.88, longitude=-117.23, accuracy=5.0, timestamp=1000.0)
        assert loc.latitude == 32.88
        assert loc.longitude == -117.23
        assert loc.accuracy == 5.0
        assert loc.timestamp == 1000.0


class TestMockGPSClient:

    def setup_method(self):
        self.client = MockGPSClient(poll_interval=0.05)

    def teardown_method(self):
        self.client.stop()

    def test_starts_connected(self):
        self.client.start()
        assert self.client.is_connected is True

    def test_location_populated_after_poll(self):
        self.client.start()
        time.sleep(0.15)
        assert self.client.location is not None

    def test_location_has_valid_coords(self):
        self.client.start()
        time.sleep(0.15)
        loc = self.client.location
        assert loc is not None
        assert 32.0 < loc.latitude < 34.0
        assert -118.0 < loc.longitude < -117.0
        assert loc.accuracy > 0

    def test_location_text_format(self):
        self.client.start()
        time.sleep(0.15)
        text = self.client.location_text
        assert "," in text
        assert "accuracy" in text.lower() or "meters" in text.lower()

    def test_location_text_when_no_fix(self):
        text = self.client.location_text
        assert "unavailable" in text.lower()

    def test_stop_does_not_raise(self):
        self.client.start()
        time.sleep(0.1)
        self.client.stop()


class TestPhoneGPSClientWithMocks:

    def setup_method(self):
        self.client = PhoneGPSClient(
            base_url="http://test-phone.local:5000",
            poll_interval=0.05,
            timeout=1.0,
        )

    def teardown_method(self):
        self.client.stop()

    def test_fetch_location_parses_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"latitude": 32.88, "longitude": -117.23, "accuracy": 5.2}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            loc = self.client._fetch_location()

        assert loc.latitude == 32.88
        assert loc.longitude == -117.23
        assert loc.accuracy == 5.2

    def test_fetch_location_handles_missing_accuracy(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"latitude": 32.88, "longitude": -117.23}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            loc = self.client._fetch_location()

        assert loc.accuracy == 0.0

    def test_poll_loop_updates_location(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"latitude": 32.88, "longitude": -117.23, "accuracy": 4.0}
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            self.client.start()
            time.sleep(0.2)

        assert self.client.location is not None
        assert self.client.is_connected is True

    def test_poll_loop_marks_disconnected_on_failure(self):
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                resp = MagicMock()
                resp.json.return_value = {"latitude": 32.88, "longitude": -117.23, "accuracy": 4.0}
                resp.raise_for_status = MagicMock()
                return resp
            raise Exception("phone offline")

        with patch("requests.get", side_effect=side_effect):
            self.client.start()
            time.sleep(0.4)

        assert self.client.is_connected is False

    def test_location_none_before_first_poll(self):
        assert self.client.location is None

    def test_base_url_trailing_slash_stripped(self):
        client = PhoneGPSClient("http://test.local:5000/")
        assert client.base_url == "http://test.local:5000"

    def test_location_text_unavailable_when_no_fix(self):
        assert "unavailable" in self.client.location_text.lower()
