"""
main.py – Orchestration entry point for Plank Posture Detection.

Workflow:
  1. Load a video file.
  2. Detect pose landmarks for every frame (pose_detector).
  3. Calculate body angles per frame (angle_calculator).
  4. Smooth the angle series over time (angle_calculator).
  5. Evaluate each frame against plank rules (posture_rules).
  6. Generate corrective feedback (feedback).
  7. Render annotated video (visualization in utils).
  8. Print a structured summary to the console.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List
import cv2
from angle_calculator import compute_frame_angles, smooth_angles, ANGLE_KEYS
from feedback import generate_feedback
from pose_detector import PoseDetector
from calibrator import Calibrator
from posture_rules import evaluate_all_frames, PlankRulesConfig
from realtime_viz import RealtimeVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ========================================================================
# Constants
# ========================================================================

DEFAULT_VIDEO_PATH = "sample_videos/plank.mp4"
DEFAULT_OUTPUT_PATH = "output_plank_analysis.mp4"
DEFAULT_CONFIG_PATH = "plank_rules.json"
DEFAULT_SMOOTH_WINDOW = 7
DEFAULT_CORRECT_THRESHOLD = 0.80


# ========================================================================
# Console summary
# ========================================================================

def print_summary(evaluation: Dict, avg_angles: Dict[str, float], config_source: str = "unknown") -> None:
    """Print a clean text summary to the console."""
    correct_pct = evaluation["correct_percentage"]
    is_correct = evaluation["is_correct_plank"]
    violations = evaluation["violation_counts"]
    total = evaluation["total_frames"]
    correct = evaluation["correct_count"]

    print()
    print("=" * 60)
    print("  PLANK POSTURE ANALYSIS REPORT")
    print("=" * 60)
    print()
    print(f"  Config Source:           {config_source}")
    print(f"  Overall Score:           {correct_pct:.1f}%")
    print(f"  Verdict:                 {'CORRECT PLANK' if is_correct else 'INCORRECT PLANK'}")
    print(f"  Correct Frames:          {correct} / {total}")
    print()
    print("  Average Angles:")
    for key in ANGLE_KEYS:
        label = key.replace("_", " ").title()
        print(f"    {label:30s}  {avg_angles[key]:.1f} deg")
    print()

    if violations:
        print("  Detected Errors:")
        for label, count in sorted(violations.items(), key=lambda x: -x[1]):
            pct = count / total * 100 if total > 0 else 0
            print(f"    - {label:25s}  in {pct:.0f}% of frames")
        print()

        print("  Suggested Corrections:")
        feedback_lines = generate_feedback(violations, total)
        for line in feedback_lines:
            print(f"    {line}")
    else:
        print("  No errors detected. Great form!")
    print()
    print("=" * 60)


# ========================================================================
# Main pipeline
# ========================================================================

def run_analysis(
    video_path: str,
    output_path: str,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    correct_threshold: float = DEFAULT_CORRECT_THRESHOLD,
    config: PlankRulesConfig = PlankRulesConfig(),
) -> str:
    """Execute the full plank-posture analysis pipeline.

    Parameters
    ----------
    video_path : str
        Path to the input plank video.
    output_path : str
        Path for the annotated output video.
    smooth_window : int
        Moving-average window size for angle smoothing.
    correct_threshold : float
        Fraction of correct frames required for overall "Correct".
    config : PlankRulesConfig
        Tunable rule thresholds.

    Returns
    -------
    str
        Path to the saved output video.
    """
    # -----------------------------------------------------------------
    # Step 1: Pose detection
    # -----------------------------------------------------------------
    logger.info("Step 1: Detecting poses in video...")
    with PoseDetector() as detector:
        frame_data = detector.process_video(video_path)

    if not frame_data:
        logger.error("No valid poses detected in the video.")
        sys.exit(1)

    logger.info("Detected poses in %d frames.", len(frame_data))

    # -----------------------------------------------------------------
    # Step 2: Calculate raw angles per frame
    # -----------------------------------------------------------------
    logger.info("Step 2: Calculating body angles...")
    raw_angles: List[Dict[str, float]] = []
    for fd in frame_data:
        angles = compute_frame_angles(fd["landmarks_xy"])
        if angles is not None:
            raw_angles.append(angles)
        else:
            # Fallback: zeroes (will be filtered by evaluation)
            raw_angles.append({k: 0.0 for k in ANGLE_KEYS})

    # Trim frame_data to match the frames we have angles for
    # (they should be the same length, but let's be safe)
    min_len = min(len(frame_data), len(raw_angles))
    frame_data = frame_data[:min_len]
    raw_angles = raw_angles[:min_len]

    # -----------------------------------------------------------------
    # Step 3: Temporal smoothing
    # -----------------------------------------------------------------
    logger.info("Step 3: Smoothing angles (window=%d)...", smooth_window)
    smoothed_angles = smooth_angles(raw_angles, window_size=smooth_window)

    # -----------------------------------------------------------------
    # Step 4: Rule-based evaluation + frame voting
    # -----------------------------------------------------------------
    logger.info("Step 4: Evaluating posture against rules...")
    evaluation = evaluate_all_frames(
        smoothed_angles,
        config=config,
        correct_threshold=correct_threshold,
    )

    # -----------------------------------------------------------------
    # Step 5: Real-time visualisation & video writing
    # -----------------------------------------------------------------
    logger.info("Step 5: Starting real-time visualisation...")

    # Get video properties from the first available frame
    first_frame = frame_data[0]["frame"]
    vid_h, vid_w = first_frame.shape[:2]

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    with RealtimeVisualizer(
        output_path=output_path,
        fps=fps,
        frame_size=(vid_w, vid_h),
        config=config,
    ) as viz:
        viz.run(frame_data, smoothed_angles, evaluation)

    # -----------------------------------------------------------------
    # Step 6: Print console summary
    # -----------------------------------------------------------------
    print_summary(evaluation, evaluation["average_angles"], config_source=config.source)

    return output_path


# ========================================================================
# CLI
# ========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plank Posture Detection – Analyse a side-view plank video using MediaPipe Pose.",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- analyse sub-command (default behaviour) ---
    analyse_parser = subparsers.add_parser("analyse", help="Analyse a plank video (default)")
    analyse_parser.add_argument(
        "--video",
        type=str,
        default=DEFAULT_VIDEO_PATH,
        help=f"Path to input plank video (default: {DEFAULT_VIDEO_PATH})",
    )
    analyse_parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for annotated output video (default: {DEFAULT_OUTPUT_PATH})",
    )
    analyse_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to calibration JSON config (default: {DEFAULT_CONFIG_PATH})",
    )
    analyse_parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help=f"Moving-average window for angle smoothing (default: {DEFAULT_SMOOTH_WINDOW})",
    )
    analyse_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CORRECT_THRESHOLD,
        help=f"Min fraction of correct frames for 'Correct Plank' (default: {DEFAULT_CORRECT_THRESHOLD})",
    )

    # --- calibrate sub-command ---
    cal_parser = subparsers.add_parser(
        "calibrate",
        help="Calibrate rules from videos of CORRECT plank posture.",
    )
    cal_parser.add_argument(
        "videos",
        nargs="+",
        help="One or more paths to correct-plank video files.",
    )
    cal_parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Output JSON config path (default: {DEFAULT_CONFIG_PATH})",
    )
    cal_parser.add_argument(
        "--smooth-window",
        type=int,
        default=DEFAULT_SMOOTH_WINDOW,
        help=f"Moving-average window (default: {DEFAULT_SMOOTH_WINDOW})",
    )
    cal_parser.add_argument(
        "--std-padding",
        type=float,
        default=0.0,
        help="Extra tolerance in std-dev units (default: 0.0)",
    )

    # --- Backward compatibility ---
    # When no sub-command is given, these top-level flags let the old
    # CLI style (python main.py --video X --output Y) keep working.
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CORRECT_THRESHOLD)

    return parser.parse_args()


def _load_config(config_path: str) -> PlankRulesConfig:
    """Load calibration config from JSON if it exists, otherwise fall back
    to hard-coded defaults and warn the user."""
    if os.path.isfile(config_path):
        logger.info("Loading calibration config from: %s", config_path)
        return PlankRulesConfig.from_json(config_path)

    logger.warning(
        "No calibration config found at '%s'. "
        "Falling back to hard-coded defaults. "
        "Run 'python main.py calibrate --videos vid1.mp4 vid2.mp4' first.",
        config_path,
    )
    return PlankRulesConfig.default()


def main() -> None:
    args = parse_args()

    # ---- calibrate mode ----
    if args.command == "calibrate":
        if not args.videos:
            logger.error("Please provide at least one video path.")
            sys.exit(1)

        for vp in args.videos:
            if not os.path.isfile(vp):
                logger.error("Video file not found: %s", vp)
                sys.exit(1)

        cal = Calibrator(smooth_window=args.smooth_window)
        for vp in args.videos:
            cal.add_video(vp)

        cal.print_report()
        saved = cal.save(output_path=args.output, std_padding=args.std_padding)
        print(f"\n  Config saved to: {saved}")
        return

    # ---- analyse mode (default) ----
    video_path = args.video
    if not os.path.isfile(video_path):
        logger.error("Video file not found: %s", video_path)
        sys.exit(1)

    config = _load_config(args.config)
    run_analysis(
        video_path=video_path,
        output_path=args.output,
        smooth_window=args.smooth_window,
        correct_threshold=args.threshold,
        config=config,
    )


if __name__ == "__main__":
    main()
