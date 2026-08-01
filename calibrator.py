"""
calibrator.py – Data-driven calibration of plank posture rules.

Processes one or more videos of CORRECT plank posture, extracts smoothed
angle measurements, computes descriptive statistics, and writes a JSON
configuration file whose ranges drive the rule engine.

Usage
-----
    from calibrator import Calibrator

    cal = Calibrator(smooth_window=7)
    cal.add_video("correct_plank_1.mp4")
    cal.add_video("correct_plank_2.mp4")
    config_path = cal.save("plank_rules.json")

The produced JSON can then be loaded by ``PlankRulesConfig.from_json()``.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from angle_calculator import ANGLE_KEYS, compute_frame_angles, smooth_angles
from pose_detector import PoseDetector

logger = logging.getLogger(__name__)

# Default path for the generated config
DEFAULT_CONFIG_PATH = "plank_rules.json"

# Number of standard deviations used to widen the acceptable range
# beyond the observed percentile bounds.
DEFAULT_STD_PADDING = 0.0  # set >0 to add extra tolerance


class Calibrator:
    """Collects angle measurements from correct-plank videos and
    builds a JSON rule configuration from the aggregated data."""

    def __init__(self, smooth_window: int = 7) -> None:
        self.smooth_window = smooth_window
        # Accumulator: angle_key -> list of per-frame values
        self._measurements: Dict[str, List[float]] = {k: [] for k in ANGLE_KEYS}
        # Track per-video frame counts for reporting
        self._video_frame_counts: List[int] = []

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_video(self, video_path: str) -> "Calibrator":
        """Process a single video of correct plank posture and accumulate
        its smoothed angle measurements.

        Parameters
        ----------
        video_path : str
            Path to a video file showing a correct plank (side view).

        Returns
        -------
        Calibrator
            Self, for method chaining.
        """
        logger.info("Calibrating from video: %s", video_path)

        # Step 1: Detect poses
        with PoseDetector() as detector:
            frame_data = detector.process_video(video_path)

        if not frame_data:
            logger.warning("No valid poses in '%s' – skipping.", video_path)
            return self

        # Step 2: Compute raw angles
        raw_angles: List[Dict[str, float]] = []
        for fd in frame_data:
            angles = compute_frame_angles(fd["landmarks_xy"])
            if angles is not None:
                raw_angles.append(angles)

        if not raw_angles:
            logger.warning("No angle data extracted from '%s' – skipping.", video_path)
            return self

        # Step 3: Smooth
        smoothed = smooth_angles(raw_angles, window_size=self.smooth_window)

        # Step 4: Accumulate
        for angles in smoothed:
            for k in ANGLE_KEYS:
                self._measurements[k].append(angles[k])

        self._video_frame_counts.append(len(smoothed))
        logger.info(
            "Collected %d frames from '%s'.  Total frames so far: %d",
            len(smoothed), video_path, self.total_frames,
        )
        return self

    @property
    def total_frames(self) -> int:
        """Total number of accumulated (smoothed) frames across all videos."""
        return len(self._measurements[ANGLE_KEYS[0]]) if self._measurements else 0

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute descriptive statistics for every angle key.

        Returns
        -------
        dict
            ``{
                angle_key: {
                    "mean": float,
                    "median": float,
                    "std": float,
                    "min": float,
                    "max": float,
                    "p5": float,   # 5th percentile
                    "p95": float,  # 95th percentile
                },
                ...
            }``
        """
        if self.total_frames == 0:
            raise ValueError("No measurements collected. Add at least one video first.")

        stats: Dict[str, Dict[str, float]] = {}
        for k in ANGLE_KEYS:
            arr = np.array(self._measurements[k], dtype=np.float64)
            stats[k] = {
                "mean": round(float(np.mean(arr)), 2),
                "median": round(float(np.median(arr)), 2),
                "std": round(float(np.std(arr, ddof=1)), 2),
                "min": round(float(np.min(arr)), 2),
                "max": round(float(np.max(arr)), 2),
                "p5": round(float(np.percentile(arr, 5)), 2),
                "p95": round(float(np.percentile(arr, 95)), 2),
            }
        return stats

    # ------------------------------------------------------------------
    # Config generation
    # ------------------------------------------------------------------

    def generate_config(
        self,
        std_padding: float = DEFAULT_STD_PADDING,
    ) -> Dict[str, Dict[str, float]]:
        """Build the full rule configuration from collected statistics.

        For each angle the acceptable range is derived from the 5th and
        95th percentiles of the calibration data, optionally widened by
        ``std_padding`` standard deviations on each side.

        Parameters
        ----------
        std_padding : float
            Extra tolerance in standard-deviation units.  0 = use only
            the percentile bounds.

        Returns
        -------
        dict
            JSON-serialisable configuration ready to be saved.
        """
        stats = self.compute_statistics()

        config: Dict[str, Dict[str, float]] = {}
        for k, s in stats.items():
            # Base range from percentiles
            lo = s["p5"]
            hi = s["p95"]

            # Optionally widen with std
            if std_padding > 0:
                lo -= std_padding * s["std"]
                hi += std_padding * s["std"]

            config[k] = {
                "mean": s["mean"],
                "std": s["std"],
                "acceptable_min": round(lo, 2),
                "acceptable_max": round(hi, 2),
            }

            # Carry the full statistics for reference
            for stat_key in ("median", "min", "max", "p5", "p95"):
                config[k][stat_key] = s[stat_key]

        return config

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        output_path: str = DEFAULT_CONFIG_PATH,
        std_padding: float = DEFAULT_STD_PADDING,
    ) -> str:
        """Generate the config and write it to a JSON file.

        Parameters
        ----------
        output_path : str
            Destination path for the JSON file.
        std_padding : float
            Extra tolerance (see ``generate_config``).

        Returns
        -------
        str
            Absolute path to the written file.
        """
        config = self.generate_config(std_padding=std_padding)

        path = Path(output_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        logger.info(
            "Saved calibration config to %s (%d videos, %d frames)",
            path, len(self._video_frame_counts), self.total_frames,
        )
        return str(path)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Print a human-readable calibration summary to stdout."""
        if self.total_frames == 0:
            print("No measurements collected yet.")
            return

        stats = self.compute_statistics()
        config = self.generate_config()

        print()
        print("=" * 64)
        print("  PLANK POSTURE CALIBRATION REPORT")
        print("=" * 64)
        print(f"  Calibration videos:  {len(self._video_frame_counts)}")
        print(f"  Total frames:        {self.total_frames}")
        print()

        header = f"  {'Angle':<24s} {'Mean':>7s} {'Std':>6s} {'Min':>7s} {'Max':>7s} {'P5':>7s} {'P95':>7s}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for k in ANGLE_KEYS:
            s = stats[k]
            label = k.replace("_", " ").title()
            print(
                f"  {label:<24s} {s['mean']:>7.1f} {s['std']:>6.1f}"
                f" {s['min']:>7.1f} {s['max']:>7.1f}"
                f" {s['p5']:>7.1f} {s['p95']:>7.1f}"
            )

        print()
        print("  Generated Acceptable Ranges:")
        for k in ANGLE_KEYS:
            c = config[k]
            label = k.replace("_", " ").title()
            print(
                f"    {label:<24s}  [{c['acceptable_min']:.1f}, {c['acceptable_max']:.1f}]"
            )
        print()
        print("=" * 64)
