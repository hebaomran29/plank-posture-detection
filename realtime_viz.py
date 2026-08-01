"""
realtime_viz.py – Professional real-time visual feedback for plank posture analysis.

Renders a 1600×900 canvas with the original video on the left (70 %)
and a dark dashboard on the right (30 %) containing:

  - Weighted score (0–100) with classification badge
  - Per-frame status, confidence, and progress bar
  - Body-part pass/fail indicators
  - Live angle readouts
  - Top-3 stabilised corrective feedback
  - Keyboard controls (Q / Space / S)

The annotated output video is saved at the same 1600×900 resolution.

Integration
-----------
Called from ``main.py`` after pose detection, smoothing, and evaluation.
Does NOT modify any detection / calibration / rule-engine logic.
"""

import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from feedback import _FEEDBACK_MAP
from scoring import (
    CLASSIFICATION_COLORS,
    WEIGHTS,
    compute_frame_score,
)
from utils import LANDMARK_MAP, get_landmark, get_visibility

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════════
#  Canvas & layout constants
# ════════════════════════════════════════════════════════════════════════════════

CANVAS_W, CANVAS_H = 1600, 900
VIDEO_RATIO = 0.70
VIDEO_AREA_W = int(CANVAS_W * VIDEO_RATIO)       # 1120
PANEL_X = VIDEO_AREA_W                            # 1120
PANEL_W = CANVAS_W - VIDEO_AREA_W                 # 480
PANEL_MARGIN = 14
PANEL_INNER_W = PANEL_W - 2 * PANEL_MARGIN        # 452
PANEL_GAP = 7

# ── Colour palette (dark theme) ────────────────────────────────────────────────
C_BG           = (18, 20, 26)
C_PANEL_BG     = (30, 33, 42)
C_PANEL_BORDER = (55, 60, 78)
C_TEXT         = (235, 237, 242)
C_TEXT_DIM     = (140, 145, 165)
C_TEXT_ACCENT  = (180, 185, 200)
C_GREEN        = (46, 204, 113)
C_RED          = (231, 76, 60)
C_BLUE         = (80, 140, 240)
C_YELLOW       = (241, 196, 15)
C_WHITE        = (255, 255, 255)
C_BLACK        = (0, 0, 0)
C_BAR_BG       = (50, 54, 68)
C_BAR_FILL     = (46, 204, 113)
C_SEPARATOR    = (45, 48, 62)

# ── Skeleton mappings ─────────────────────────────────────────────────────────

# Each connection: (joint_a, joint_b, body_part)
_CONNECTION_PART: List[Tuple[str, str, str]] = [
    ("ear", "shoulder", "neck"),
    ("shoulder", "hip", "back"),
    ("hip", "knee", "hip"),
    ("knee", "ankle", "knee"),
    ("ankle", "heel", "legs"),
    ("heel", "foot_index", "legs"),
    ("shoulder", "elbow", "arm"),
    ("elbow", "wrist", "arm"),
]

# Body part → skeleton joints to highlight
_PART_JOINTS: Dict[str, List[str]] = {
    "neck": ["nose", "ear"],
    "back": ["shoulder", "hip"],
    "hip":  ["hip", "knee"],
    "knee": ["knee", "ankle"],
    "legs": ["ankle", "heel", "foot_index"],
}

# All joints drawn as dots
_ALL_JOINTS = [
    "ear", "shoulder", "elbow", "wrist",
    "hip", "knee", "ankle", "heel", "foot_index",
]

# Feedback severity (higher = more important)
_SEVERITY: Dict[str, int] = {
    "back_rounded": 3, "back_arched": 3,
    "knees_bent": 2, "knees_hyperextended": 2,
    "hips_too_low": 2, "hips_too_high": 2,
    "head_forward": 1, "head_backward": 1,
}

# Part display names & order
_PART_DISPLAY = [
    ("back",  "Back"),
    ("hip",   "Hip"),
    ("neck",  "Neck"),
    ("knee",  "Knees"),
    ("legs",  "Legs"),
]

# Angle display names & order
_ANGLE_DISPLAY = [
    ("ear_shoulder_hip",    "Neck"),
    ("shoulder_hip_ankle",  "Back"),
    ("shoulder_hip_knee",   "Hip"),
    ("hip_knee_ankle",      "Knee"),
]


# ════════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _rounded_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = -1,
    radius: int = 10,
) -> None:
    """Draw a filled or outlined rounded rectangle."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if r < 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return

    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def _put_text(
    img: np.ndarray,
    text: str,
    pos: Tuple[int, int],
    font_scale: float = 0.5,
    color: Tuple[int, int, int] = C_TEXT,
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> Tuple[int, int]:
    """Put text and return the bottom-right corner for chaining."""
    cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    return (pos[0] + tw, pos[1])


# ════════════════════════════════════════════════════════════════════════════════
#  FeedbackStabilizer
# ════════════════════════════════════════════════════════════════════════════════

class FeedbackStabilizer:
    """Suppresses flickering by requiring a violation to persist for
    *min_consecutive* frames before display, and holding it for
    *clear_after* absent frames before removal."""

    def __init__(self, min_consecutive: int = 5, clear_after: int = 12) -> None:
        self.min_consecutive = min_consecutive
        self.clear_after = clear_after
        self._counters: Dict[str, int] = defaultdict(int)
        self._absent: Dict[str, int] = defaultdict(int)
        self._active: Set[str] = set()

    def update(self, violations: List[str]) -> Set[str]:
        current = set(violations)
        for v in current:
            self._counters[v] += 1
            self._absent[v] = 0
            if self._counters[v] >= self.min_consecutive:
                self._active.add(v)
        for v in list(self._counters):
            if v not in current:
                self._counters[v] = 0
                self._absent[v] += 1
                if self._absent[v] >= self.clear_after:
                    self._active.discard(v)
                    self._absent.pop(v, None)
        return set(self._active)

    def reset(self) -> None:
        self._counters.clear()
        self._absent.clear()
        self._active.clear()


# ════════════════════════════════════════════════════════════════════════════════
#  RealtimeVisualizer
# ════════════════════════════════════════════════════════════════════════════════

class RealtimeVisualizer:
    """Drives the 1600×900 real-time OpenCV display and video writer.

    Usage::

        with RealtimeVisualizer(output_path='result.mp4', fps=30.0,
                                frame_size=(w, h), config=config) as viz:
            viz.run(frame_data, smoothed_angles, evaluation)
    """

    WINDOW_NAME = "Plank Posture Analysis"

    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        config=None,
        stabilizer_min: int = 5,
        stabilizer_clear: int = 12,
        save_dir: str = ".",
    ) -> None:
        self.output_path = output_path
        self.fps = fps
        self.config = config
        self.save_dir = save_dir
        self.orig_w, self.orig_h = frame_size

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (CANVAS_W, CANVAS_H))

        self.stabilizer = FeedbackStabilizer(
            min_consecutive=stabilizer_min,
            clear_after=stabilizer_clear,
        )
        self._paused = False
        self._quit = False

    # ── public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        frame_data: List[Dict],
        smoothed_angles: List[Dict[str, float]],
        evaluation: Dict,
    ) -> str:
        total_video_frames = frame_data[-1]["frame_index"] + 1 if frame_data else 0

        # Pre-compute video scaling
        scale = min(VIDEO_AREA_W / self.orig_w, CANVAS_H / self.orig_h)
        self._sw = int(self.orig_w * scale)
        self._sh = int(self.orig_h * scale)
        self._ox = (VIDEO_AREA_W - self._sw) // 2
        self._oy = (CANVAS_H - self._sh) // 2

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        n = 0

        for i, fd in enumerate(frame_data):
            if self._quit:
                break
            while self._paused and not self._quit:
                key = cv2.waitKey(50) & 0xFF
                if key == ord(" "):
                    self._paused = False
                elif key in (ord("q"), ord("Q")):
                    self._quit = True
                elif key in (ord("s"), ord("S")):
                    self._save_frame(fd["frame"], i)
            if self._quit:
                break

            raw = fd["frame"]
            landmarks = fd["landmarks"]
            side = fd["side"]
            angles = smoothed_angles[i]
            frame_result = evaluation["frame_results"][i]
            violations = frame_result["violations"]

            # Weighted score
            score, part_status, classification = compute_frame_score(angles, self.config) if self.config else (0.0, {}, "NEEDS CORRECTION")

            # Stabilised feedback
            stable = self.stabilizer.update(violations)
            top3 = self._top_corrections(stable)

            # Confidence
            conf = self._avg_confidence(landmarks, side)

            canvas = self._build_canvas(
                frame=raw,
                landmarks=landmarks,
                side=side,
                part_status=part_status,
                score=score,
                classification=classification,
                angles=angles,
                frame_index=fd["frame_index"],
                total_frames=total_video_frames,
                confidence=conf,
                corrections=top3,
            )

            self.writer.write(canvas)
            n += 1
            cv2.imshow(self.WINDOW_NAME, canvas)

            delay = max(1, int(1000 / self.fps))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), ord("Q")):
                self._quit = True
            elif key == ord(" "):
                self._paused = True
            elif key in (ord("s"), ord("S")):
                self._save_frame(canvas, i)

        self.writer.release()
        cv2.destroyAllWindows()
        logger.info("Real-time viz: wrote %d frames to %s", n, self.output_path)
        return self.output_path

    # ── canvas builder ──────────────────────────────────────────────────────────

    def _build_canvas(
        self,
        frame, landmarks, side, part_status,
        score, classification, angles,
        frame_index, total_frames, confidence, corrections,
    ) -> np.ndarray:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), C_BG, dtype=np.uint8)

        # 1. Video (scaled, centred in left area)
        resized = cv2.resize(frame, (self._sw, self._sh))
        canvas[self._oy : self._oy + self._sh, self._ox : self._ox + self._sw] = resized

        # 2. Skeleton on video area
        self._draw_skeleton(canvas, landmarks, side, part_status)

        # 3. Dashboard background
        cv2.rectangle(canvas, (PANEL_X, 0), (CANVAS_W, CANVAS_H), (24, 26, 34), -1)
        cv2.line(canvas, (PANEL_X, 0), (PANEL_X, CANVAS_H), C_SEPARATOR, 2)

        # 4. Dashboard panels
        px = PANEL_X + PANEL_MARGIN
        pw = PANEL_INNER_W
        y = PANEL_MARGIN

        # ─ Title ─
        y = self._draw_title(canvas, px, y, pw)
        y += PANEL_GAP

        # ─ Score ─
        y = self._draw_score_section(canvas, px, y, pw, score, classification)
        y += PANEL_GAP

        # ─ Status info ─
        y = self._draw_status_section(
            canvas, px, y, pw, frame_index, total_frames, confidence, score
        )
        y += PANEL_GAP

        # ─ Body parts ─
        y = self._draw_body_parts_section(canvas, px, y, pw, part_status)
        y += PANEL_GAP

        # ─ Angles ─
        y = self._draw_angles_section(canvas, px, y, pw, angles)
        y += PANEL_GAP

        # ─ Corrections ─
        y = self._draw_corrections_section(canvas, px, y, pw, corrections)
        y += PANEL_GAP

        # ─ Controls ─
        self._draw_controls_section(canvas, px, y, pw)

        return canvas

    # ── skeleton ────────────────────────────────────────────────────────────────

    def _draw_skeleton(
        self, canvas, landmarks, side, part_status: Dict[str, bool]
    ) -> None:
        """Draw colour-coded skeleton on the video area of the canvas.

        Only the body parts flagged as failing are coloured red;
        everything else stays green. Arms are always blue (neutral).
        """
        # Collect failing joints / connections
        bad_joints: Set[str] = set()
        bad_conns: Set[Tuple[str, str]] = set()
        for part, ok in part_status.items():
            if not ok:
                bad_joints.update(_PART_JOINTS.get(part, []))
                for ja, jb, bp in _CONNECTION_PART:
                    if bp == part:
                        bad_conns.add((ja, jb))

        ox, oy, sw, sh = self._ox, self._oy, self._sw, self._sh
        s = side

        # --- connections ---
        for ja, jb, bp in _CONNECTION_PART:
            pt_a = get_landmark(landmarks, s, ja)
            pt_b = get_landmark(landmarks, s, jb)
            if pt_a is None or pt_b is None:
                continue
            pa = (ox + int(pt_a[0] * sw), oy + int(pt_a[1] * sh))
            pb = (ox + int(pt_b[0] * sw), oy + int(pt_b[1] * sh))
            if (ja, jb) in bad_conns:
                color, thick = C_RED, 5
            elif bp == "arm":
                color, thick = C_BLUE, 3
            else:
                color, thick = C_GREEN, 4
            cv2.line(canvas, pa, pb, color, thick, cv2.LINE_AA)

        # --- nose → ear ---
        nose = landmarks[LANDMARK_MAP["nose"]]
        nose_px = (ox + int(nose.x * sw), oy + int(nose.y * sh))
        ear_pt = get_landmark(landmarks, s, "ear")
        if ear_pt is not None:
            ear_px = (ox + int(ear_pt[0] * sw), oy + int(ear_pt[1] * sh))
            nc = C_RED if "neck" in [p for p, ok in part_status.items() if not ok] else C_GREEN
            cv2.line(canvas, nose_px, ear_px, nc, 4, cv2.LINE_AA)

        # --- joint dots ---
        for joint in _ALL_JOINTS:
            pt = get_landmark(landmarks, s, joint)
            if pt is None:
                continue
            cx = ox + int(pt[0] * sw)
            cy = oy + int(pt[1] * sh)
            if joint in bad_joints:
                dc, dr = C_RED, 9
            else:
                dc, dr = C_GREEN, 7
            cv2.circle(canvas, (cx, cy), dr, dc, -1, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), dr, C_WHITE, 1, cv2.LINE_AA)

        # nose dot
        nc = C_RED if "neck" in [p for p, ok in part_status.items() if not ok] else C_GREEN
        nr = 9 if nc == C_RED else 7
        cv2.circle(canvas, nose_px, nr, nc, -1, cv2.LINE_AA)
        cv2.circle(canvas, nose_px, nr, C_WHITE, 1, cv2.LINE_AA)

    # ── dashboard sections ─────────────────────────────────────────────────────

    def _panel_bg(self, canvas, x, y, w, h, alpha=0.80) -> None:
        """Draw a semi-transparent rounded panel background."""
        overlay = canvas.copy()
        _rounded_rect(overlay, (x, y), (x + w, y + h), C_PANEL_BG, -1, 10)
        _rounded_rect(overlay, (x, y), (x + w, y + h), C_PANEL_BORDER, 1, 10)
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # ── Title ──────────────────────────────────────────────────────────────────

    def _draw_title(self, canvas, x, y, w) -> int:
        h = 38
        _put_text(canvas, "PLANK ANALYSIS", (x + 8, y + 26),
                  font_scale=0.62, color=C_TEXT_ACCENT, thickness=2)
        return y + h

    # ── Score ───────────────────────────────────────────────────────────────────

    def _draw_score_section(self, canvas, x, y, w, score, classification) -> int:
        h = 108
        self._panel_bg(canvas, x, y, w, h)

        color = CLASSIFICATION_COLORS.get(classification, C_RED)

        # Big number
        score_text = f"{score:.1f}"
        (tw, th), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)
        sx = x + (w - tw) // 2
        _put_text(canvas, score_text, (sx, y + 52),
                  font_scale=2.0, color=color, thickness=3)

        # Classification badge
        (cw, ch), _ = cv2.getTextSize(classification, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cx = x + (w - cw) // 2
        _put_text(canvas, classification, (cx, y + 88),
                  font_scale=0.52, color=color, thickness=1)
        return y + h

    # ── Status info ─────────────────────────────────────────────────────────────

    def _draw_status_section(self, canvas, x, y, w,
                             frame_index, total_frames, confidence, score) -> int:
        h = 128
        self._panel_bg(canvas, x, y, w, h)
        lx = x + 14
        ry = y + 14
        line_h = 26

        # Current Status
        status_label = "CORRECT PLANK" if score >= 70 else "INCORRECT PLANK"
        status_color = C_GREEN if score >= 70 else C_RED
        _put_text(canvas, "Current Status", (lx, ry), 0.42, C_TEXT_DIM)
        _put_text(canvas, status_label, (lx + 130, ry), 0.48, status_color, 1)
        ry += line_h

        # Frame number
        _put_text(canvas, f"Frame  {frame_index + 1} / {total_frames}",
                  (lx, ry), 0.45, C_TEXT)
        ry += line_h

        # Pose Confidence
        _put_text(canvas, f"Pose Confidence  {confidence * 100:.0f}%",
                  (lx, ry), 0.45, C_TEXT)
        ry += line_h + 4

        # Progress bar
        progress = (frame_index + 1) / max(total_frames, 1)
        bar_x, bar_y, bar_w, bar_h = lx, ry, w - 28, 14
        _rounded_rect(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      C_BAR_BG, -1, 7)
        fill_w = max(0, int(bar_w * progress))
        if fill_w > 2:
            _rounded_rect(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          C_BAR_FILL, -1, 7)
        pct_text = f"{progress * 100:.0f}%"
        (ptw, _), _ = cv2.getTextSize(pct_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(canvas, pct_text,
                    (bar_x + (bar_w - ptw) // 2, bar_y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_WHITE, 1, cv2.LINE_AA)

        return y + h

    # ── Body parts ─────────────────────────────────────────────────────────────

    def _draw_body_parts_section(self, canvas, x, y, w,
                                 part_status: Dict[str, bool]) -> int:
        h = 168
        self._panel_bg(canvas, x, y, w, h)
        lx = x + 14
        ry = y + 14
        _put_text(canvas, "BODY PARTS", (lx, ry), 0.45, C_TEXT_ACCENT, 1)
        ry += 28

        for part_key, part_label in _PART_DISPLAY:
            ok = part_status.get(part_key, True)
            indicator_color = C_GREEN if ok else C_RED
            # Label
            _put_text(canvas, part_label, (lx + 4, ry + 18), 0.48, C_TEXT, 1)
            # Indicator circle
            ix = x + w - 34
            cv2.circle(canvas, (ix, ry + 12), 8, indicator_color, -1, cv2.LINE_AA)
            if ok:
                # Checkmark
                cv2.line(canvas, (ix - 4, ry + 12), (ix - 1, ry + 16), C_WHITE, 2, cv2.LINE_AA)
                cv2.line(canvas, (ix - 1, ry + 16), (ix + 5, ry + 7), C_WHITE, 2, cv2.LINE_AA)
            else:
                # X mark
                cv2.line(canvas, (ix - 4, ry + 7), (ix + 4, ry + 17), C_WHITE, 2, cv2.LINE_AA)
                cv2.line(canvas, (ix - 4, ry + 17), (ix + 4, ry + 7), C_WHITE, 2, cv2.LINE_AA)
            ry += 26

        return y + h

    # ── Angles ──────────────────────────────────────────────────────────────────

    def _draw_angles_section(self, canvas, x, y, w,
                             angles: Dict[str, float]) -> int:
        h = 148
        self._panel_bg(canvas, x, y, w, h)
        lx = x + 14
        ry = y + 14
        _put_text(canvas, "CURRENT ANGLES", (lx, ry), 0.45, C_TEXT_ACCENT, 1)
        ry += 28

        for angle_key, label in _ANGLE_DISPLAY:
            val = angles.get(angle_key, 0.0)
            _put_text(canvas, label, (lx + 4, ry + 18), 0.48, C_TEXT, 1)
            val_text = f"{val:.1f} deg"
            (vtw, _), _ = cv2.getTextSize(val_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            vx = x + w - 14 - vtw
            _put_text(canvas, val_text, (vx, ry + 18), 0.48, C_WHITE, 1)
            ry += 28

        return y + h

    # ── Corrections ─────────────────────────────────────────────────────────────

    def _draw_corrections_section(self, canvas, x, y, w,
                                  corrections: List[str]) -> int:
        h = 170
        self._panel_bg(canvas, x, y, w, h)
        lx = x + 14
        ry = y + 14
        _put_text(canvas, "CORRECTIONS", (lx, ry), 0.45, C_TEXT_ACCENT, 1)
        ry += 28

        if not corrections:
            _put_text(canvas, "No issues detected.", (lx + 4, ry + 16),
                      0.42, C_TEXT_DIM, 1)
            return y + h

        for idx, text in enumerate(corrections):
            _put_text(canvas, f"{idx + 1}.", (lx + 4, ry + 16), 0.44, C_RED, 1)
            _put_text(canvas, text, (lx + 24, ry + 16), 0.42, C_TEXT, 1)
            ry += 38

        return y + h

    # ── Controls ────────────────────────────────────────────────────────────────

    def _draw_controls_section(self, canvas, x, y, w) -> None:
        hints = "[Q] Quit    [Space] Pause    [S] Save Frame"
        _put_text(canvas, hints, (x + 8, y + 22), 0.36, C_TEXT_DIM, 1)

    # ── helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _top_corrections(stable_violations: Set[str], max_items: int = 3) -> List[str]:
        """Return up to *max_items* correction strings, ordered by severity."""
        sorted_v = sorted(
            stable_violations,
            key=lambda v: (-_SEVERITY.get(v, 0), v),
        )
        lines: List[str] = []
        for v in sorted_v[:max_items]:
            entry = _FEEDBACK_MAP.get(v)
            if entry and entry["correction"] not in lines:
                lines.append(entry["correction"])
        return lines

    @staticmethod
    def _avg_confidence(landmarks, side: str) -> float:
        joints = ["ear", "shoulder", "hip", "knee", "ankle", "heel"]
        vals = [get_visibility(landmarks, side, j) for j in joints]
        return float(np.mean(vals)) if vals else 0.0

    def _save_frame(self, frame: np.ndarray, idx: int) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(path, frame)
        logger.info("Saved screenshot: %s", path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.release()
        cv2.destroyAllWindows()
        return False

