"""
posture_rules.py – Rule-based evaluation of plank posture.

Each rule examines one body angle against a learned acceptable range
(loaded from a calibration JSON) and returns a boolean indicating
whether that aspect of the plank is correct, plus a short error label.

Configuration can be loaded from a JSON file produced by ``calibrator.py``
or built manually via ``PlankRulesConfig``.  Hard-coded fallbacks are
provided only for backward compatibility.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from angle_calculator import ANGLE_KEYS

logger = logging.getLogger(__name__)


# ========================================================================
# Configuration
# ========================================================================

# Mapping: angle_key -> (violation_label_if_too_low, violation_label_if_too_high)
# Used by the generic range-check rule to produce meaningful feedback.
_ANGLE_VIOLATION_MAP: Dict[str, Tuple[str, str]] = {
    "ear_shoulder_hip": ("head_forward", "head_backward"),
    "shoulder_hip_knee": ("hips_too_low", "hips_too_high"),
    "shoulder_hip_ankle": ("back_rounded", "back_arched"),
    "hip_knee_ankle": ("knees_bent", "knees_hyperextended"),
}


@dataclass
class PlankRulesConfig:
    """Holds an acceptable [min, max] range for each plank angle.

    The preferred way to create this is via ``from_json(path)`` which
    loads a configuration produced by the calibration module.

    Attributes
    ----------
    angle_ranges : dict
        ``{angle_key: (acceptable_min, acceptable_max), ...}``
    source : str
        Human-readable description of where the config came from.
    """
    angle_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    source: str = "default"

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str) -> "PlankRulesConfig":
        """Load configuration from a JSON file produced by the calibrator.

        The JSON is expected to have the structure::

            {
                "ear_shoulder_hip": {
                    "acceptable_min": 160.0,
                    "acceptable_max": 185.0,
                    ...
                },
                ...
            }

        Parameters
        ----------
        path : str
            Path to the JSON configuration file.

        Returns
        -------
        PlankRulesConfig
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)

        ranges: Dict[str, Tuple[float, float]] = {}
        for key in ANGLE_KEYS:
            if key not in raw:
                logger.warning(
                    "Angle key '%s' missing from config '%s' – using defaults.",
                    key, path,
                )
                ranges[key] = cls._hardcoded_ranges().get(key, (0.0, 180.0))
                continue

            entry = raw[key]
            lo = float(entry["acceptable_min"])
            hi = float(entry["acceptable_max"])
            if lo > hi:
                lo, hi = hi, lo
            ranges[key] = (lo, hi)

        logger.info("Loaded config from %s  (%d angle ranges)", path, len(ranges))
        return cls(angle_ranges=ranges, source=f"json:{path}")

    @classmethod
    def from_dict(
        cls,
        ranges: Dict[str, Tuple[float, float]],
        source: str = "dict",
    ) -> "PlankRulesConfig":
        """Create a config directly from a dictionary of ranges.

        Parameters
        ----------
        ranges : dict
            ``{angle_key: (min, max), ...}``
        source : str
            Description of the source for logging.
        """
        return cls(angle_ranges=dict(ranges), source=source)

    @classmethod
    def _hardcoded_ranges(cls) -> Dict[str, Tuple[float, float]]:
        """Legacy hard-coded ranges used only as a fallback when no
        calibration JSON is available.
        """
        return {
            "ear_shoulder_hip": (155.0, 185.0),
            "shoulder_hip_knee": (150.0, 195.0),
            "shoulder_hip_ankle": (165.0, 185.0),
            "hip_knee_ankle": (165.0, 185.0),
        }

    @classmethod
    def default(cls) -> "PlankRulesConfig":
        """Return a config with legacy hard-coded ranges.

        Kept for backward compatibility.  Prefer ``from_json()`` for
        production use.
        """
        return cls(
            angle_ranges=cls._hardcoded_ranges(),
            source="hardcoded_defaults",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_range(self, angle_key: str) -> Tuple[float, float]:
        """Return (acceptable_min, acceptable_max) for *angle_key*,
        falling back to (0, 180) if the key is unknown.
        """
        return self.angle_ranges.get(angle_key, (0.0, 180.0))


# Default configuration (hard-coded fallback – prefer from_json)
DEFAULT_CONFIG = PlankRulesConfig.default()


# ========================================================================
# Individual rule functions
# ========================================================================

def _check_angle_range(
    angles: Dict[str, float],
    angle_key: str,
    config: PlankRulesConfig,
) -> Tuple[bool, str]:
    """Generic range check for a single angle.

    Parameters
    ----------
    angles : dict
        Per-frame angle values.
    angle_key : str
        Key in ``ANGLE_KEYS`` (and in ``angles``).
    config : PlankRulesConfig

    Returns
    -------
    (bool, str)
        (is_correct, violation_label)
    """
    value = angles.get(angle_key)
    if value is None:
        return True, ""  # cannot assess – skip

    lo, hi = config.get_range(angle_key)
    if lo <= value <= hi:
        return True, ""

    label_low, label_high = _ANGLE_VIOLATION_MAP.get(angle_key, ("bad_form", "bad_form"))
    if value < lo:
        return False, label_low
    return False, label_high


def check_neck_alignment(
    angles: Dict[str, float],
    config: PlankRulesConfig = DEFAULT_CONFIG,
) -> Tuple[bool, str]:
    """Check that the neck is neutral (Ear-Shoulder-Hip within range)."""
    return _check_angle_range(angles, "ear_shoulder_hip", config)


def check_back_straightness(
    angles: Dict[str, float],
    config: PlankRulesConfig = DEFAULT_CONFIG,
) -> Tuple[bool, str]:
    """Check that the back is straight (Shoulder-Hip-Ankle within range)."""
    return _check_angle_range(angles, "shoulder_hip_ankle", config)


def check_hip_alignment(
    angles: Dict[str, float],
    config: PlankRulesConfig = DEFAULT_CONFIG,
) -> Tuple[bool, str]:
    """Check that the hips are neither too low nor too high
    (Shoulder-Hip-Knee within range)."""
    return _check_angle_range(angles, "shoulder_hip_knee", config)


def check_knee_extension(
    angles: Dict[str, float],
    config: PlankRulesConfig = DEFAULT_CONFIG,
) -> Tuple[bool, str]:
    """Check that the knees are extended (Hip-Knee-Ankle within range)."""
    return _check_angle_range(angles, "hip_knee_ankle", config)


# ========================================================================
# Frame-level evaluation
# ========================================================================

def evaluate_frame(
    angles: Dict[str, float],
    config: PlankRulesConfig = DEFAULT_CONFIG,
) -> Dict:
    """Run all rules on a single frame's angles.

    Returns
    -------
    dict
        {
            "is_correct": bool,
            "violations": [str, ...],  # error labels
        }
    """
    rules = [
        check_neck_alignment,
        check_back_straightness,
        check_hip_alignment,
        check_knee_extension,
    ]

    violations: List[str] = []
    for rule in rules:
        passed, label = rule(angles, config)
        if not passed and label:
            violations.append(label)

    return {
        "is_correct": len(violations) == 0,
        "violations": violations,
    }


def evaluate_all_frames(
    all_angles: List[Dict[str, float]],
    config: PlankRulesConfig = DEFAULT_CONFIG,
    correct_threshold: float = 0.80,
) -> Dict:
    """Evaluate every frame and produce a voting summary.

    Parameters
    ----------
    all_angles : list of dict
        Smoothed per-frame angle dictionaries.
    config : PlankRulesConfig
        Rule configuration (loaded from JSON or default).
    correct_threshold : float
        Minimum fraction of correct frames to classify as "Correct Plank".

    Returns
    -------
    dict
        {
            "frame_results": [dict, ...],
            "correct_count": int,
            "total_frames": int,
            "correct_percentage": float,
            "is_correct_plank": bool,
            "violation_counts": {label: count, ...},
            "average_angles": {key: float, ...},
        }
    """
    frame_results = []
    correct_count = 0
    violation_counts: Dict[str, int] = {}

    # Accumulators for average angles
    angle_sums: Dict[str, float] = {k: 0.0 for k in all_angles[0].keys()} if all_angles else {}

    for angles in all_angles:
        result = evaluate_frame(angles, config)
        frame_results.append(result)

        if result["is_correct"]:
            correct_count += 1

        for v in result["violations"]:
            violation_counts[v] = violation_counts.get(v, 0) + 1

        for k, v in angles.items():
            angle_sums[k] += v

    total = len(all_angles)
    correct_pct = (correct_count / total * 100.0) if total > 0 else 0.0
    avg_angles = {k: v / total for k, v in angle_sums.items()} if total > 0 else {}

    summary = {
        "frame_results": frame_results,
        "correct_count": correct_count,
        "total_frames": total,
        "correct_percentage": correct_pct,
        "is_correct_plank": correct_pct >= (correct_threshold * 100),
        "violation_counts": violation_counts,
        "average_angles": avg_angles,
    }

    logger.info(
        "Evaluation: %d/%d correct (%.1f%%) → %s  [config: %s]",
        correct_count, total, correct_pct,
        "CORRECT" if summary["is_correct_plank"] else "INCORRECT",
        config.source,
    )
    return summary
