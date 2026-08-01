"""
pose_detector.py – MediaPipe Pose detection wrapper.

Responsibilities:
  - Initialise and configure MediaPipe Pose.
  - Process video frames and extract pose landmarks.
  - Select the most visible body side per frame.
  - Filter out low-confidence frames.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from utils import LANDMARK_MAP, select_visible_side, REQUIRED_LANDMARKS, get_visibility

logger = logging.getLogger(__name__)

# Minimal visibility threshold for each required landmark in a single frame
_DEFAULT_MIN_LANDMARK_VISIBILITY = 0.3


class PoseDetector:
    """Wraps MediaPipe Pose for frame-by-frame landmark extraction."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 2,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialise the MediaPipe Pose model.

        Parameters
        ----------
        static_image_mode : bool
            If False (default), the detector treats input as a video stream
            and uses tracking to reduce latency.
        model_complexity : int
            0 (lite), 1 (full), or 2 (heavy).  2 gives the best accuracy.
        smooth_landmarks : bool
            Apply internal landmark smoothing.
        min_detection_confidence : float
            Minimum confidence for initial pose detection.
        min_tracking_confidence : float
            Minimum confidence for landmark tracking across frames.
        """
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame: np.ndarray) -> Optional[mp.solutions.pose.PoseLandmark]:
        """Run pose detection on a single BGR frame.

        Returns the landmarks list, or None if no pose is detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks is None:
            return None
        return results.pose_landmarks.landmark

    def process_video(
        self,
        video_path: str,
        min_landmark_visibility: float = _DEFAULT_MIN_LANDMARK_VISIBILITY,
    ) -> List[Dict]:
        """Process every frame of a video and return per-frame pose data.

        For each frame that passes quality checks the returned dict contains::

            {
                "frame_index": int,
                "frame": np.ndarray,          # original BGR frame
                "landmarks": list,            # MediaPipe landmark objects
                "side": str,                  # "left" or "right"
                "landmarks_xy": dict,         # {joint_name: np.array([x, y])}
            }

        Frames with no detected pose or poor visibility are skipped.

        Parameters
        ----------
        min_landmark_visibility : float
            Minimum visibility score (0-1) for each required landmark.
            Lower values accept more frames but may be less accurate.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info("Processing video: %s (%d frames)", video_path, total_frames)

        frame_data: List[Dict] = []
        frame_idx = 0
        skipped = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.process_frame(frame)
            if landmarks is None:
                skipped += 1
                frame_idx += 1
                continue

            # Select the visible body side
            side = select_visible_side(landmarks)
            if side is None:
                skipped += 1
                frame_idx += 1
                continue

            # Verify every required landmark has sufficient visibility
            required_ok = True
            for joint in REQUIRED_LANDMARKS:
                vis = get_visibility(landmarks, side, joint)
                if vis < min_landmark_visibility:
                    required_ok = False
                    break

            if not required_ok:
                skipped += 1
                frame_idx += 1
                continue

            # Extract (x, y) coordinates for the chosen side
            landmarks_xy = {}
            for joint in REQUIRED_LANDMARKS:
                key = f"{side}_{joint}"
                idx = LANDMARK_MAP.get(key)
                if idx is not None and idx < len(landmarks):
                    lm = landmarks[idx]
                    landmarks_xy[joint] = np.array([lm.x, lm.y])

            # Also grab the nose for head alignment
            nose_idx = LANDMARK_MAP["nose"]
            landmarks_xy["nose"] = np.array([landmarks[nose_idx].x, landmarks[nose_idx].y])

            frame_data.append({
                "frame_index": frame_idx,
                "frame": frame,
                "landmarks": landmarks,
                "side": side,
                "landmarks_xy": landmarks_xy,
            })

            frame_idx += 1

        cap.release()
        logger.info(
            "Processed %d valid frames (skipped %d)",
            len(frame_data),
            skipped,
        )
        return frame_data

    def close(self) -> None:
        """Release the underlying MediaPipe Pose model."""
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
