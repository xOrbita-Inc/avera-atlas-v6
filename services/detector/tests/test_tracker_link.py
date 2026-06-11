"""
Tests for Detector → Tracker pipeline link.

Verifies that the Detector service correctly translates its internal
DetectionFrame format into Tracker's DetectionBatchInput schema and
posts to the correct URL.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio

# Allow imports from the detector service root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import (
    _xyxy_to_xywh,
    _bbox_center,
    build_tracker_payload,
    push_to_tracker,
    Detection,
    DetectionFrame,
    DEFAULT_SENSOR_ID,
    TRACKER_DETECTIONS_URL,
)

# ---------------------------------------------------------------------------
# Vendored subset of Tracker's DetectionBatchInput for schema validation.
# Source of truth: services/tracker/schemas.py :: DetectionInput, DetectionBatchInput
# ---------------------------------------------------------------------------
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class _DetectionInput(BaseModel):
    detection_id: str
    sensor_id: str
    timestamp: datetime
    pixel_u: float
    pixel_v: float
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float
    confidence: float
    object_class: str
    platform_state: Optional[dict] = None


class _DetectionBatchInput(BaseModel):
    detections: list[_DetectionInput]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_detection_frame() -> DetectionFrame:
    return DetectionFrame(
        frame_id="test-frame-1",
        timestamp_utc="2026-04-21T12:00:00Z",
        sensor_id="UI-UPLOAD-SWIR",
        camera_pose=None,
        detections=[
            Detection(
                object_class="debris",
                confidence=0.91,
                bbox=[100.0, 200.0, 150.0, 260.0],
            )
        ],
    )


@pytest.fixture
def multi_detection_frame() -> DetectionFrame:
    return DetectionFrame(
        frame_id="test-frame-2",
        timestamp_utc="2026-04-21T12:01:00Z",
        sensor_id="UI-UPLOAD-SWIR",
        camera_pose=None,
        detections=[
            Detection(object_class="debris", confidence=0.85, bbox=[10.0, 20.0, 50.0, 80.0]),
            Detection(object_class="satellite", confidence=0.93, bbox=[200.0, 300.0, 240.0, 340.0]),
            Detection(object_class="unknown", confidence=0.60, bbox=[500.0, 100.0, 520.0, 130.0]),
        ],
    )


# ---------------------------------------------------------------------------
# Tests: bbox conversion
# ---------------------------------------------------------------------------

class TestBboxConversion:
    def test_xyxy_to_xywh(self):
        result = _xyxy_to_xywh([100.0, 200.0, 150.0, 260.0])
        assert result == {"bbox_x": 100.0, "bbox_y": 200.0, "bbox_w": 50.0, "bbox_h": 60.0}

    def test_bbox_center(self):
        result = _bbox_center([100.0, 200.0, 150.0, 260.0])
        assert result == {"pixel_u": 125.0, "pixel_v": 230.0}


# ---------------------------------------------------------------------------
# Tests: payload structure
# ---------------------------------------------------------------------------

class TestBuildTrackerPayload:
    def test_single_detection_produces_one_record(self, single_detection_frame):
        payload = build_tracker_payload(single_detection_frame)
        assert len(payload["detections"]) == 1

    def test_multi_detection_produces_multiple_records(self, multi_detection_frame):
        payload = build_tracker_payload(multi_detection_frame)
        assert len(payload["detections"]) == 3

    def test_payload_validates_against_tracker_schema(self, single_detection_frame):
        payload = build_tracker_payload(single_detection_frame)
        batch = _DetectionBatchInput(**payload)
        assert len(batch.detections) == 1

    def test_multi_payload_validates_against_tracker_schema(self, multi_detection_frame):
        payload = build_tracker_payload(multi_detection_frame)
        batch = _DetectionBatchInput(**payload)
        assert len(batch.detections) == 3

    def test_bbox_xywh_in_payload(self, single_detection_frame):
        payload = build_tracker_payload(single_detection_frame)
        rec = payload["detections"][0]
        assert rec["bbox_x"] == 100.0
        assert rec["bbox_y"] == 200.0
        assert rec["bbox_w"] == 50.0
        assert rec["bbox_h"] == 60.0

    def test_pixel_center_in_payload(self, single_detection_frame):
        payload = build_tracker_payload(single_detection_frame)
        rec = payload["detections"][0]
        assert rec["pixel_u"] == 125.0
        assert rec["pixel_v"] == 230.0

    def test_platform_state_is_none(self, single_detection_frame):
        payload = build_tracker_payload(single_detection_frame)
        for rec in payload["detections"]:
            assert rec["platform_state"] is None

    def test_sensor_id_propagated(self, single_detection_frame):
        payload = build_tracker_payload(single_detection_frame)
        assert payload["detections"][0]["sensor_id"] == "UI-UPLOAD-SWIR"

    def test_detection_id_is_uuid(self, single_detection_frame):
        import uuid
        payload = build_tracker_payload(single_detection_frame)
        # Should not raise
        uuid.UUID(payload["detections"][0]["detection_id"])
        
    def test_range_confidence_fields_in_payload(self):
        frame = DetectionFrame(
            frame_id="test-frame-range-confidence",
            timestamp_utc="2026-04-21T12:00:00Z",
            sensor_id="UI-UPLOAD-SWIR",
            camera_pose=None,
            detections=[
                Detection(
                    object_class="debris",
                    confidence=0.91,
                    bbox=[100.0, 200.0, 150.0, 260.0],
                    estimated_range_km=45.2,
                    debris_size_class="5cm",
                    range_confidence="HIGH",
                )
            ],
        )

        payload = build_tracker_payload(frame)
        rec = payload["detections"][0]

        assert rec["estimated_range_km"] == 45.2
        assert rec["debris_size_class"] == "5cm"
        assert rec["range_confidence"] == "HIGH"


# ---------------------------------------------------------------------------
# Tests: default sensor_id
# ---------------------------------------------------------------------------

class TestDefaultSensorId:
    def test_default_sensor_id_value(self):
        assert DEFAULT_SENSOR_ID == "UI-UPLOAD-SWIR"


# ---------------------------------------------------------------------------
# Tests: push_to_tracker posts to correct URL
# ---------------------------------------------------------------------------

class TestPushToTracker:
    @pytest.mark.asyncio
    async def test_posts_to_tracker_url(self, single_detection_frame):
        payload = build_tracker_payload(single_detection_frame)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"processed": 1, "iod_ready_ucts": 0}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("main.httpx.AsyncClient", return_value=mock_client):
            await push_to_tracker(payload)

        mock_client.post.assert_called_once_with(TRACKER_DETECTIONS_URL, json=payload)
