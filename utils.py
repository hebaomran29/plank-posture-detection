"""
utils.py – Shared helper functions for the Plank Posture Detection project.

Provides:
  - calculate_angle(): Compute the interior angle (degrees) at vertex B formed by points A-B-C.
  - select_visible_side(): Choose left or right body side based on average landmark visibility.
  - Visualization helpers for drawing pose, angles, status, and feedback on frames.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Pose landmark indices used throughout the project
# ---------------------------------------------------------------------------
LANDMARK_MAP = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Landmarks that define each body "side" (used for visibility comparison)
LEFT_SIDE_LANDMARKS = [7, 11, 13, 15, 23, 25, 27, 29, 31]   # ear -> foot_index
RIGHT_SIDE_LANDMARKS = [8, 12, 14, 16, 24, 26, 28, 30, 32]

# Minimal set of landmarks required to evaluate a plank frame
REQUIRED_LANDMARKS = ["ear", "shoulder", "hip", "knee", "ankle", "heel"]


# =========================================================================
# Geometry helpers
# =========================================================================

def calculate_angle(
    point_a: np.ndarray,
    point_b: np.ndarray,
    point_c: np.ndarray,
) -> float:
    """Return the interior angle (in degrees) at *point_b* for the triple A-B-C.

    Parameters
    ----------
    point_a, point_b, point_c : array-like of shape (2,) or (3,)
        Landmark coordinates (x, y) or (x, y, z).

    Returns
    -------
    float
        Angle in degrees in the range [0, 180].
    """
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    c = np.array(point_c, dtype=np.float64)

    # Vectors BA and BC
    ba = a - b
    bc = c - b

    # Cosine of the angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)


def get_landmark(landmarks, side: str, joint: str) -> Optional[np.ndarray]:
    """Extract (x, y) coordinates for a given side and joint.

    Parameters
    ----------
    landmarks : MediaPipe landmark list
    side : "left" or "right"
    joint : one of the keys in LANDMARK_MAP (e.g. "shoulder", "hip")

    Returns
    -------
    np.ndarray of shape (2,) or None if not found.
    """
    key = f"{side}_{joint}"
    idx = LANDMARK_MAP.get(key)
    if idx is None or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    return np.array([lm.x, lm.y])


def get_visibility(landmarks, side: str, joint: str) -> float:
    """Return the visibility score for a specific landmark."""
    key = f"{side}_{joint}"
    idx = LANDMARK_MAP.get(key)
    if idx is None or idx >= len(landmarks):
        return 0.0
    return float(landmarks[idx].visibility)


# =========================================================================
# Side selection
# =========================================================================

def select_visible_side(landmarks, min_avg_visibility: float = 0.3) -> Optional[str]:
    """Choose "left" or "right" body side based on average landmark visibility.

    If neither side meets *min_avg_visibility*, returns None so the frame
    can be skipped.
    """
    def _avg_vis(indices: List[int]) -> float:
        visibilities = [landmarks[i].visibility for i in indices if i < len(landmarks)]
        return float(np.mean(visibilities)) if visibilities else 0.0

    left_avg = _avg_vis(LEFT_SIDE_LANDMARKS)
    right_avg = _avg_vis(RIGHT_SIDE_LANDMARKS)

    best_avg = max(left_avg, right_avg)
    if best_avg < min_avg_visibility:
        return None

    return "left" if left_avg >= right_avg else "right"


# =========================================================================
# Visualization helpers
# =========================================================================

# Color palette
_COLOR_GREEN = (46, 204, 113)      # correct / ok
_COLOR_RED = (231, 76, 60)         # incorrect / error
_COLOR_YELLOW = (241, 196, 15)     # warning / angles
_COLOR_WHITE = (255, 255, 255)
_COLOR_CYAN = (52, 152, 219)       # skeleton
_COLOR_DARK_BG = (0, 0, 0)         # text background


def draw_skeleton(
    frame: np.ndarray,
    landmarks,
    side: str,
) -> np.ndarray:
    """Draw the plank-relevant skeleton connections on *frame*.

    Only draws the side selected for analysis plus the nose landmark for
    head alignment checks.
    """
    s = side  # "left" or "right"

    # Pairs of (joint_a, joint_b) to connect
    connections = [
        ("shoulder", "elbow"),
        ("elbow", "wrist"),
        ("shoulder", "hip"),
        ("hip", "knee"),
        ("knee", "ankle"),
        ("ankle", "heel"),
        ("heel", "foot_index"),
    ]

    h, w = frame.shape[:2]
    for joint_a, joint_b in connections:
        pt_a = get_landmark(landmarks, s, joint_a)
        pt_b = get_landmark(landmarks, s, joint_b)
        if pt_a is not None and pt_b is not None:
            pa = (int(pt_a[0] * w), int(pt_a[1] * h))
            pb = (int(pt_b[0] * w), int(pt_b[1] * h))
            cv2.line(frame, pa, pb, _COLOR_CYAN, 3, cv2.LINE_AA)

    # Draw individual landmarks as circles
    joints_to_draw = [
        "ear", "shoulder", "elbow", "wrist",
        "hip", "knee", "ankle", "heel", "foot_index",
    ]
    for joint in joints_to_draw:
        pt = get_landmark(landmarks, s, joint)
        if pt is not None:
            center = (int(pt[0] * w), int(pt[1] * h))
            cv2.circle(frame, center, 5, _COLOR_CYAN, -1, cv2.LINE_AA)

    # Draw nose for head alignment
    nose = landmarks[LANDMARK_MAP["nose"]]
    nose_pt = (int(nose.x * w), int(nose.y * h))
    cv2.circle(frame, nose_pt, 5, _COLOR_CYAN, -1, cv2.LINE_AA)
    # Connect nose to ear
    ear_pt = get_landmark(landmarks, s, "ear")
    if ear_pt is not None:
        ear_px = (int(ear_pt[0] * w), int(ear_pt[1] * h))
        cv2.line(frame, nose_pt, ear_px, _COLOR_CYAN, 2, cv2.LINE_AA)

    return frame


def draw_angle_arc(
    frame: np.ndarray,
    vertex: np.ndarray,
    point_a: np.ndarray,
    point_c: np.ndarray,
    angle: float,
    color: Tuple[int, int, int] = _COLOR_YELLOW,
    radius: int = 25,
) -> np.ndarray:
    """Draw an arc at *vertex* showing the calculated angle and its value."""
    h, w = frame.shape[:2]
    vx, vy = int(vertex[0] * w), int(vertex[1] * h)
    ax, ay = int(point_a[0] * w), int(point_a[1] * h)
    cx, cy = int(point_c[0] * w), int(point_c[1] * h)

    # Compute start and end angles (in OpenCV's coordinate system)
    start_angle = np.degrees(np.arctan2(-(ay - vy), ax - vx))
    end_angle = np.degrees(np.arctan2(-(cy - vy), cx - vx))

    # Normalize so arc always sweeps the interior angle
    if end_angle < start_angle:
        start_angle, end_angle = end_angle, start_angle
    if end_angle - start_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360

    cv2.ellipse(
        frame,
        (vx, vy),
        (radius, radius),
        0,
        start_angle,
        end_angle,
        color,
        2,
        cv2.LINE_AA,
    )

    # Angle text near the vertex
    offset_x = int(np.cos(np.radians((start_angle + end_angle) / 2)) * (radius + 14))
    offset_y = int(-np.sin(np.radians((start_angle + end_angle) / 2)) * (radius + 14))
    text_pos = (vx + offset_x, vy + offset_y)
    cv2.putText(
        frame,
        f"{angle:.0f}deg",
        text_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        color,
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_status(
    frame: np.ndarray,
    is_correct: bool,
    score: float,
) -> np.ndarray:
    """Draw the current frame status badge in the top-left corner."""
    label = "CORRECT" if is_correct else "INCORRECT"
    color = _COLOR_GREEN if is_correct else _COLOR_RED

    # Background rectangle for readability
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(
        frame,
        (10, 10),
        (10 + text_size[0] + 16, 10 + text_size[1] + 16),
        _COLOR_DARK_BG,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (18, 10 + text_size[1] + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )

    # Score next to status
    score_text = f"Score: {score:.1f}%"
    score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(
        frame,
        (10, 10 + text_size[1] + 24),
        (10 + score_size[0] + 16, 10 + text_size[1] + 24 + score_size[1] + 12),
        _COLOR_DARK_BG,
        -1,
    )
    cv2.putText(
        frame,
        score_text,
        (18, 10 + text_size[1] + 24 + score_size[1] + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        _COLOR_WHITE,
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_feedback(
    frame: np.ndarray,
    feedback_lines: List[str],
) -> np.ndarray:
    """Draw feedback text lines at the bottom of the frame."""
    if not feedback_lines:
        return frame

    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    padding = 10

    # Calculate total block height
    total_height = len(feedback_lines) * line_height + 2 * padding
    y_start = h - total_height

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, y_start),
        (w, h),
        _COLOR_DARK_BG,
        -1,
    )
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, line in enumerate(feedback_lines):
        y = y_start + padding + i * line_height + line_height
        cv2.putText(
            frame,
            line,
            (padding, y),
            font,
            font_scale,
            _COLOR_RED if "INCORRECT" in line.upper() or "error" in line.lower() else _COLOR_WHITE,
            thickness,
            cv2.LINE_AA,
        )
    return frame
