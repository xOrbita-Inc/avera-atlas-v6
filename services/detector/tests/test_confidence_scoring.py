import sys
from pathlib import Path

import pytest

# Allow imports from the detector service root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from confidence_scoring import (
    DETECTION_RANGE_KM_BY_SIZE_CLASS,
    calculate_range_confidence,
    get_detection_r_max_km,
)


@pytest.mark.parametrize(
    "debris_size_class,r_max_km",
    [
        ("1cm", 20.0),
        ("5cm", 98.0),
        ("10cm", 195.0),
    ],
)
def test_r_max_constants(debris_size_class, r_max_km):
    assert DETECTION_RANGE_KM_BY_SIZE_CLASS[debris_size_class] == r_max_km
    assert get_detection_r_max_km(debris_size_class) == r_max_km


@pytest.mark.parametrize(
    "debris_size_class,r_max_km",
    [
        ("1cm", 20.0),
        ("5cm", 98.0),
        ("10cm", 195.0),
    ],
)
def test_high_confidence_below_half_rmax(debris_size_class, r_max_km):
    assert calculate_range_confidence(0.5 * r_max_km - 1e-6, debris_size_class) == "HIGH"


@pytest.mark.parametrize(
    "debris_size_class,r_max_km",
    [
        ("1cm", 20.0),
        ("5cm", 98.0),
        ("10cm", 195.0),
    ],
)
def test_medium_confidence_at_half_rmax(debris_size_class, r_max_km):
    assert calculate_range_confidence(0.5 * r_max_km, debris_size_class) == "MEDIUM"


@pytest.mark.parametrize(
    "debris_size_class,r_max_km",
    [
        ("1cm", 20.0),
        ("5cm", 98.0),
        ("10cm", 195.0),
    ],
)
def test_medium_confidence_below_low_boundary(debris_size_class, r_max_km):
    assert calculate_range_confidence(0.85 * r_max_km - 1e-6, debris_size_class) == "MEDIUM"


@pytest.mark.parametrize(
    "debris_size_class,r_max_km",
    [
        ("1cm", 20.0),
        ("5cm", 98.0),
        ("10cm", 195.0),
    ],
)
def test_low_confidence_at_low_boundary(debris_size_class, r_max_km):
    assert calculate_range_confidence(0.85 * r_max_km, debris_size_class) == "LOW"


@pytest.mark.parametrize(
    "debris_size_class,r_max_km",
    [
        ("1cm", 20.0),
        ("5cm", 98.0),
        ("10cm", 195.0),
    ],
)
def test_low_confidence_above_low_boundary(debris_size_class, r_max_km):
    assert calculate_range_confidence(r_max_km, debris_size_class) == "LOW"


def test_unknown_debris_size_class_rejected():
    with pytest.raises(ValueError):
        calculate_range_confidence(10.0, "2cm")


def test_negative_range_rejected():
    with pytest.raises(ValueError):
        calculate_range_confidence(-1.0, "5cm")