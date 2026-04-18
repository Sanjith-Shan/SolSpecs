"""
SolSpecs — Phone GPS Client
Polls the phone's GPS endpoint every 5 seconds and stores current coordinates.

The phone runs either:
  - phone/gps_server.py  (Termux Flask server on Android)
  - phone/gps.html       (browser page using Geolocation API)

Both expose:  GET /location → {"latitude": float, "longitude": float, "accuracy": float}
"""

import logging
import threading
import time
from typing import Optional, NamedTuple

logger = logging.getLogger("PhoneGPSClient")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed. Run: pip install requests")


class GPSLocation(NamedTuple):
    latitude: float
    longitude: float
    accuracy: float
    timestamp: float


class PhoneGPSClient:
    """
    Polls the phone GPS server and stores the most recent fix.

    Usage:
        gps = PhoneGPSClient("http://192.168.1.100:5000")
        gps.start()
        ...
        loc = gps.location       # GPSLocation or None
        text = gps.location_text # "32.8801, -117.2340 (±5m)"
        gps.stop()
    """

    def __init__(self, base_url: str, poll_interval: float = 5.0, timeout: float = 4.0):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.timeout = timeout

        self._location: Optional[GPSLocation] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._consecutive_failures = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info(f"GPS client started — polling {self.base_url}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=6)
        logger.info("GPS client stopped")

    @property
    def location(self) -> Optional[GPSLocation]:
        with self._lock:
            return self._location

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def location_text(self) -> str:
        """Human-readable location string for voice readout."""
        loc = self.location
        if loc is None:
            return "GPS location unavailable"
        return (
            f"{loc.latitude:.4f}, {loc.longitude:.4f} "
            f"(accuracy plus or minus {loc.accuracy:.0f} meters)"
        )

    def _poll_loop(self):
        while self._running:
            try:
                loc = self._fetch_location()
                if loc:
                    with self._lock:
                        self._location = loc
                    self._consecutive_failures = 0
                    if not self._connected:
                        logger.info("GPS connected")
                        self._connected = True
            except Exception as e:
                self._consecutive_failures += 1
                if self._connected:
                    logger.warning(f"GPS connection lost: {e}")
                    self._connected = False
                elif self._consecutive_failures == 3:
                    logger.warning(f"GPS unreachable after 3 attempts: {e}")
            time.sleep(self.poll_interval)

    def _fetch_location(self) -> Optional[GPSLocation]:
        if not REQUESTS_AVAILABLE:
            return None
        resp = requests.get(f"{self.base_url}/location", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return GPSLocation(
            latitude=float(data["latitude"]),
            longitude=float(data["longitude"]),
            accuracy=float(data.get("accuracy", 0.0)),
            timestamp=time.time(),
        )


class MockGPSClient(PhoneGPSClient):
    """
    Simulated GPS client for laptop testing.
    Starts at a fixed San Diego construction site location and drifts slowly.
    """

    def __init__(self, poll_interval: float = 5.0):
        super().__init__(base_url="http://mock-phone.local:5000", poll_interval=poll_interval)
        self._base_lat = 32.8801
        self._base_lng = -117.2340
        self._tick = 0

    def start(self):
        self._running = True
        self._connected = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Mock GPS client started")

    def _fetch_location(self) -> GPSLocation:
        import random
        self._tick += 1
        return GPSLocation(
            latitude=self._base_lat + random.uniform(-0.0001, 0.0001),
            longitude=self._base_lng + random.uniform(-0.0001, 0.0001),
            accuracy=round(random.uniform(3.0, 8.0), 1),
            timestamp=time.time(),
        )
