"""
scoring.py – Weighted body-part scoring for plank posture analysis.

Provides a nuanced 0–100 posture quality score by assigning differential
weights to each body region, replacing the old binary pass/fail approach.

This module is part of the **presentation layer only** and does NOT modify
the underlying rule engine or calibration logic.
"""

from typing import Dict, Tuple


# ── Weights (must sum to 1.0) ───────────────────────────────────────────────────────
WEIGHTS: Dict[str, float] = {
    "back":  0.35,   # shoulder_hip_ankle
    "hip":   0.30,   # shoulder_hip_knee
    "neck":  0.15,   # ear_shoulder_hip
    "knee":  0.10,   # hip_knee_ankle  (strict, calibrated range)
    "legs":  0.10,   # hip_knee_ankle  (lenient, fixed range)
}

# ── Body part → angle key ───────────────────────────────────────────────────
PART_ANGLE: Dict[str, str] = {
    "neck": "ear_shoulder_hip",
    "hip":  "shoulder_hip_knee",
    "back": "shoulder_hip_ankle",
    "knee": "hip_knee_ankle",
}

# Fixed (non-calibrated) range for “Leg Alignment”
LEG_FIXED_RANGE: Tuple[float, float] = (160.0, 185.0)

# Degrees outside the acceptable range before a component score reaches 0
DECAY: float = 15.0

# Classification bands (threshold, label)
_CLASSIFICATIONS = [
    (90, "EXCELLENT"),
    (80, "GOOD"),
    (70, "ACCEPTABLE"),
    (0,  "NEEDS CORRECTION"),
]

# Classification → display colour (BGR)
CLASSIFICATION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "EXCELLENT":         (46, 204, 113),
    "GOOD":              (52, 152, 219),
    "ACCEPTABLE":        (241, 196, 15),
    "NEEDS CORRECTION":  (231, 76, 60),
}


def _soft_score(value: float, lo: float, hi: float) -> float:
    """Return a 0–1 component score using linear decay beyond the range."""
    if lo <= value <= hi:
        return 1.0
    deviation = max(lo - value, value - hi, 0.0)
    return max(0.0, 1.0 - deviation / DECAY)


def compute_frame_score(
    angles: Dict[str, float],
    config,
) -> Tuple[float, Dict[str, bool], str]:
    """Compute a weighted 0–100 posture score for one frame.

    Parameters
    ----------
    angles : dict
        ``{angle_key: degrees, ...}`` (smoothed).
    config : PlankRulesConfig
        Calibration configuration with acceptable ranges.

    Returns
    -------
    (score, part_status, classification)
        score           : float in [0, 100]
        part_status     : {part_name: bool}  (True = passing)
        classification : EXCELLENT / GOOD / ACCEPTABLE / NEEDS CORRECTION
    """
    part_scores: Dict[str, float] = {}

    for part, angle_key in PART_ANGLE.items():
        value = angles.get(angle_key, 0.0)
        lo, hi = config.get_range(angle_key)
        part_scores[part] = _soft_score(value, lo, hi)

    # Legs use a lenient fixed range on the same knee angle
    knee_val = angles.get("hip_knee_ankle", 0.0)
    part_scores["legs"] = _soft_score(knee_val, *LEG_FIXED_RANGE)

    # Weighted total scaled to 0–100
    total = sum(part_scores[p] * WEIGHTS[p] for p in WEIGHTS) * 100.0

    # Binary pass/fail per part (component score ≥ 0.5)
    part_status = {p: (part_scores[p] >= 0.5) for p in WEIGHTS}

    # Pick classification
    classification = "NEEDS CORRECTION"
    for threshold, label in _CLASSIFICATIONS:
        if total >= threshold:
            classification = label
            break

    return total, part_status, classification
