"""
angle_calculator.py – Compute and smooth plank-relevant body angles.

Provides:
  - compute_frame_angles(): Calculate all key angles for a single frame.
  - smooth_angles(): Apply a temporal moving-average filter across frames.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from utils import calculate_angle

logger = logging.getLogger(__name__)

# Keys for the four required angles
ANGLE_KEYS = [
    "ear_shoulder_hip",
    "shoulder_hip_knee",
    "shoulder_hip_ankle",
    "hip_knee_ankle",
]


def compute_frame_angles(landmarks_xy: Dict[str, np.ndarray]) -> Optional[Dict[str, float]]:
    """Calculate the four plank-relevant angles for one frame.

    Parameters
    ----------
    landmarks_xy : dict
        Mapping of joint names to (x, y) arrays.  Must contain at least:
        ear, shoulder, hip, knee, ankle.

    Returns
    -------
    dict or None
        ``{angle_name: degrees, ...}`` or None if any required landmark is missing.
    """
    required = {"ear", "shoulder", "hip", "knee", "ankle"}
    if not required.issubset(landmarks_xy.keys()):
        return None

    ear = landmarks_xy["ear"]
    shoulder = landmarks_xy["shoulder"]
    hip = landmarks_xy["hip"]
    knee = landmarks_xy["knee"]
    ankle = landmarks_xy["ankle"]

    angles = {
        # Neck alignment: ear -> shoulder -> hip
        "ear_shoulder_hip": calculate_angle(ear, shoulder, hip),
        # Hip alignment: shoulder -> hip -> knee
        "shoulder_hip_knee": calculate_angle(shoulder, hip, knee),
        # Back straightness: shoulder -> hip -> ankle (should be ~180°)
        "shoulder_hip_ankle": calculate_angle(shoulder, hip, ankle),
        # Knee extension: hip -> knee -> ankle (should be ~180°)
        "hip_knee_ankle": calculate_angle(hip, knee, ankle),
    }

    return angles


def smooth_angles(
    all_angles: List[Dict[str, float]],
    window_size: int = 7,
) -> List[Dict[str, float]]:
    """Apply a centred moving-average filter to each angle series.

    Parameters
    ----------
    all_angles : list of dict
        Per-frame angle dictionaries produced by ``compute_frame_angles``.
    window_size : int
        Width of the averaging window (must be odd; will be forced odd).

    Returns
    -------
    list of dict
        Smoothed angle dictionaries with the same keys.
    """
    if not all_angles:
        return []

    n = len(all_angles)

    # Force odd window
    if window_size % 2 == 0:
        window_size += 1
    half_w = window_size // 2

    # Build per-key arrays
    keys = ANGLE_KEYS
    series = {k: np.array([a[k] for a in all_angles]) for k in keys}

    smoothed_series = {}
    kernel = np.ones(window_size) / window_size
    for k in keys:
        arr = series[k].astype(np.float64)
        # Pad edges by repeating the boundary values
        padded = np.pad(arr, (half_w, half_w), mode="edge")
        # Convolution-based moving average (produces exactly n output values)
        moving_avg = np.convolve(padded, kernel, mode="valid")
        smoothed_series[k] = moving_avg[:n]

    # Re-package into list of dicts
    smoothed = [
        {k: float(smoothed_series[k][i]) for k in keys}
        for i in range(n)
    ]

    logger.info("Smoothed %d frames with window_size=%d", n, window_size)
    return smoothed
