"""
feedback.py – Translate violation labels into human-readable corrective feedback.

Each violation label (produced by ``posture_rules.py``) is mapped to a short
diagnostic message and a suggested correction.  Multiple violations produce
multiple feedback lines.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Feedback mapping:  violation_label → (diagnosis, correction)
# ---------------------------------------------------------------------------
_FEEDBACK_MAP: Dict[str, Dict[str, str]] = {
    "head_forward": {
        "diagnosis": "Head is positioned too far forward.",
        "correction": "Keep your head and neck aligned with your spine; gaze at the floor just ahead of your hands.",
    },
    "head_backward": {
        "diagnosis": "Head is tilted too far back.",
        "correction": "Tuck your chin slightly and align your neck with your spine.",
    },
    "back_rounded": {
        "diagnosis": "Back is rounded / sagging.",
        "correction": "Engage your core and keep your back straight from shoulders to ankles.",
    },
    "hips_too_low": {
        "diagnosis": "Hips are sagging too low.",
        "correction": "Raise your hips slightly until your body forms a straight line from head to heels.",
    },
    "hips_too_high": {
        "diagnosis": "Hips are piked too high.",
        "correction": "Lower your hips slightly to bring your body into a straight line.",
    },
    "knees_bent": {
        "diagnosis": "Knees are bent.",
        "correction": "Keep your knees fully extended throughout the hold.",
    },
}


def generate_feedback(
    violation_counts: Dict[str, int],
    total_frames: int,
) -> List[str]:
    """Produce a list of human-readable feedback lines ordered by frequency.

    Parameters
    ----------
    violation_counts : dict
        ``{label: count}`` from ``evaluate_all_frames``.
    total_frames : int
        Total number of evaluated frames (used to show percentages).

    Returns
    -------
    list of str
        Each element is a formatted feedback string combining diagnosis
        and correction, ordered from most frequent to least frequent.
    """
    if not violation_counts:
        return ["Great plank form! No significant issues detected."]

    # Sort violations by frequency (descending)
    sorted_violations = sorted(
        violation_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    lines: List[str] = []
    for label, count in sorted_violations:
        entry = _FEEDBACK_MAP.get(label)
        if entry is None:
            continue

        pct = (count / total_frames * 100) if total_frames > 0 else 0
        lines.append(f"[Issue] {entry['diagnosis']} ({pct:.0f}% of frames)")
        lines.append(f"  Fix: {entry['correction']}")

    return lines


def generate_frame_feedback(violations: List[str]) -> List[str]:
    """Generate compact per-frame feedback for on-video overlay.

    Returns short one-line strings suitable for rendering on the video frame.
    """
    lines = []
    for label in violations:
        entry = _FEEDBACK_MAP.get(label)
        if entry is None:
            continue
        # Shorter version for video overlay
        lines.append(f"- {entry['correction']}")
    return lines
