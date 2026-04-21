"""
Tests for Tracker default sensor registration and /detections pipeline.

Verifies that register_default_sensors() bootstraps the three default sensors,
is idempotent, and that a detection with sensor_id=UI-UPLOAD-SWIR flows through
the /detections handler without being dropped as unknown_sensor.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

# Allow imports from the tracker service root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import app, state, TrackerState
from transform import CameraModel
from mock_platform import MockPlatformConfig


EXPECTED_SENSORS = [
    ("AVERA-SAT-01-SWIR", "AVERA-SAT-01"),
    ("AVERA-SAT-02-SWIR", "AVERA-SAT-02"),
    ("UI-UPLOAD-SWIR", "UI-UPLOAD"),
]


# ---------------------------------------------------------------------------
# Unit tests: register_default_sensors()
# ---------------------------------------------------------------------------

class TestRegisterDefaultSensors:
    def setup_method(self):
        """Fresh TrackerState for each test."""
        self.ts = TrackerState()

    def test_registers_all_three_sensors(self):
        self.ts.register_default_sensors()
        for sensor_id, _ in EXPECTED_SENSORS:
            assert sensor_id in self.ts.sensors, f"{sensor_id} not in sensors"
            assert sensor_id in self.ts.transformer.cameras, f"{sensor_id} not in cameras"

    def test_registers_all_three_platforms(self):
        self.ts.register_default_sensors()
        for _, platform_id in EXPECTED_SENSORS:
            assert platform_id in self.ts.platform_generator.platforms, (
                f"{platform_id} not in platforms"
            )

    def test_idempotent_no_duplicates(self):
        self.ts.register_default_sensors()
        sensor_count_1 = len(self.ts.sensors)
        camera_count_1 = len(self.ts.transformer.cameras)
        platform_count_1 = len(self.ts.platform_generator.platforms)

        # Second call — should be a no-op
        self.ts.register_default_sensors()
        assert len(self.ts.sensors) == sensor_count_1
        assert len(self.ts.transformer.cameras) == camera_count_1
        assert len(self.ts.platform_generator.platforms) == platform_count_1

    def test_camera_lookup_succeeds(self):
        self.ts.register_default_sensors()
        for sensor_id, _ in EXPECTED_SENSORS:
            cam = self.ts.transformer.cameras[sensor_id]
            assert isinstance(cam, CameraModel)


# ---------------------------------------------------------------------------
# End-to-end: POST /detections with UI-UPLOAD-SWIR
# ---------------------------------------------------------------------------

class TestDetectionsEndToEnd:
    def setup_method(self):
        """Ensure default sensors are registered before each test."""
        # Reset global state
        state.sensors.clear()
        state.transformer.cameras.clear()
        state.platform_generator.platforms.clear()
        state.detections_processed = 0
        state.register_default_sensors()

    def test_ui_upload_swir_not_dropped(self):
        client = TestClient(app)
        payload = {
            "detections": [
                {
                    "detection_id": "test-det-001",
                    "sensor_id": "UI-UPLOAD-SWIR",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "pixel_u": 512.0,
                    "pixel_v": 384.0,
                    "bbox_x": 500.0,
                    "bbox_y": 370.0,
                    "bbox_w": 25.0,
                    "bbox_h": 28.0,
                    "confidence": 0.87,
                    "object_class": "Debris",
                    "platform_state": None,
                }
            ]
        }
        resp = client.post("/detections", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["transformed"] > 0
        assert body.get("unknown_sensor", 0) == 0
