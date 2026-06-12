from typing import Literal


DebrisSizeClass = Literal["1cm", "5cm", "10cm"]
RangeConfidenceTier = Literal["HIGH", "MEDIUM", "LOW"]


# ADR-007 median-case detection range parameters.
DETECTION_RANGE_KM_BY_SIZE_CLASS = {
    "1cm": 20.0,
    "5cm": 98.0,
    "10cm": 195.0,
}

HIGH_CONFIDENCE_RANGE_FRACTION = 0.50
LOW_CONFIDENCE_RANGE_FRACTION = 0.85


def get_detection_r_max_km(debris_size_class: str) -> float:
    """Return ADR-007 median-case maximum detection range for a debris size class."""
    try:
        return DETECTION_RANGE_KM_BY_SIZE_CLASS[debris_size_class]
    except KeyError as exc:
        supported = ", ".join(sorted(DETECTION_RANGE_KM_BY_SIZE_CLASS))
        raise ValueError(
            f"Unsupported debris_size_class '{debris_size_class}'. "
            f"Supported values: {supported}"
        ) from exc


def calculate_range_confidence(
    estimated_range_km: float,
    debris_size_class: str,
) -> RangeConfidenceTier:
    """
    Calculate ADR-007 physics-based detection confidence from range/R_max.

    Logic:
    - range < 0.5 * R_max: HIGH
    - 0.5 * R_max <= range < 0.85 * R_max: MEDIUM
    - range >= 0.85 * R_max: LOW
    """
    if estimated_range_km < 0:
        raise ValueError("estimated_range_km must be non-negative")

    r_max_km = get_detection_r_max_km(debris_size_class)

    if estimated_range_km < HIGH_CONFIDENCE_RANGE_FRACTION * r_max_km:
        return "HIGH"

    if estimated_range_km < LOW_CONFIDENCE_RANGE_FRACTION * r_max_km:
        return "MEDIUM"

    return "LOW"