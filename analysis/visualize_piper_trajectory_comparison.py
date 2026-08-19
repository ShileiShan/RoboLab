#!/usr/bin/env python3
"""Create validated blue/orange overlays for two open-loop Piper replays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


BASELINE_COLOR = (220, 116, 45)  # BGR blue
CANDIDATE_COLOR = (44, 130, 246)  # BGR orange
TEXT_COLOR = (245, 245, 245)
HEADER_HEIGHT = 54


class OverlayVideoWriter:
    """Write a browser-compatible MP4 when ffmpeg is available."""

    def __init__(self, path: Path, fps: float):
        self.path = path
        self.fps = fps
        self._writer = None
        # OpenCV's mp4v output is broadly readable by desktop players but is
        # not consistently supported by browsers.  Stage it first, then use
        # H.264/yuv420p for the final requested filename.
        self._use_ffmpeg = shutil.which("ffmpeg") is not None
        self._staging_path = (
            path.with_name(f"{path.stem}.raw{path.suffix}") if self._use_ffmpeg else path
        )

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            height, width = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                str(self._staging_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open overlay video for writing: {self._staging_path}")
        # The compositor works in BGR; VideoWriter expects the same order.
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if not self._use_ffmpeg or not self._staging_path.exists():
            return
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-v", "error", "-i", str(self._staging_path),
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            # Preserve a valid mp4v artifact rather than discarding a completed
            # comparison when a host's ffmpeg build lacks libx264.
            self._staging_path.replace(self.path)
            print(f"Warning: H.264 browser transcode failed; kept mp4v overlay: {exc}")
        else:
            self._staging_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-video", type=Path, required=True)
    parser.add_argument("--candidate-video", type=Path, required=True)
    parser.add_argument("--baseline-mask-video", type=Path)
    parser.add_argument("--candidate-mask-video", type=Path)
    parser.add_argument("--baseline-object-mask-video", "--baseline-breakfast-objects-mask-video",
                        dest="baseline_object_mask_video", type=Path,
                        help="Optional binary mask for tracked comparison objects (e.g. kettle/cup or bread/toaster).")
    parser.add_argument("--candidate-object-mask-video", "--candidate-breakfast-objects-mask-video",
                        dest="candidate_object_mask_video", type=Path,
                        help="Optional binary mask for tracked comparison objects (e.g. kettle/cup or bread/toaster).")
    parser.add_argument("--frames", type=int, default=8,
                        help="Number of equally spaced contact-sheet frames (default: 8).")
    parser.add_argument("--output", type=Path, required=True, help="Contact-sheet PNG output.")
    parser.add_argument("--overlay-video", type=Path,
                        help="Per-frame blue/orange overlay MP4 (required for --layout overlay).")
    parser.add_argument("--background-output", type=Path,
                        help="Shared static background PNG (default: beside --output).")
    parser.add_argument("--layout", choices=("side-by-side", "overlay"), default="overlay")
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--max-frame-width", type=int, default=512)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    parser.add_argument("--initial-frame-mae-max", type=float, default=2.0,
                        help="Maximum permitted RGB MAE at t=0 (default: 2.0/255; renderer-tolerant).")
    parser.add_argument("--initial-mask-iou-min", type=float, default=1.0,
                        help="Minimum permitted robot-mask IoU at t=0 (default: exact overlap).")
    return parser.parse_args()


class VideoReader:
    def __init__(self, path: Path):
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.frame_count < 1 or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"Video has no readable frames or FPS: {path}")

    def read_index(self, index: int) -> np.ndarray:
        index = min(self.frame_count - 1, max(0, index))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {index} from {self.path}")
        return frame

    def release(self) -> None:
        self.capture.release()


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    if frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    return cv2.resize(frame, (max_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def normalize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    # Mask videos are encoded as white-on-black RGB.  A midpoint threshold is
    # robust to H.264 ringing without turning compression noise into robot pixels.
    return mask >= 128


def add_header(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    header = np.full((HEADER_HEIGHT, image.shape[1], 3), 28, dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (8, HEADER_HEIGHT), color, thickness=-1)
    cv2.putText(header, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT_COLOR, 2, cv2.LINE_AA)
    return np.vstack((header, image))


def alpha_tint(background: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.82) -> np.ndarray:
    result = background.copy()
    if mask.any():
        color_image = np.empty_like(background)
        color_image[:] = color
        result[mask] = cv2.addWeighted(background[mask], 1.0 - alpha, color_image[mask], alpha, 0)
    return result


def make_overlay(background: np.ndarray, baseline_mask: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    """Tint robot pixels while retaining a single shared, robot-free background."""
    baseline_only = baseline_mask & ~candidate_mask
    candidate_only = candidate_mask & ~baseline_mask
    overlap = baseline_mask & candidate_mask
    image = alpha_tint(background, baseline_only, BASELINE_COLOR)
    image = alpha_tint(image, candidate_only, CANDIDATE_COLOR)
    # Coincident geometry is deliberately a 50/50 blue/orange blend.  This
    # makes exact t=0 overlap visible without arbitrarily drawing one run over
    # the other.
    overlap_color = tuple(round((blue + orange) / 2) for blue, orange in zip(BASELINE_COLOR, CANDIDATE_COLOR))
    return alpha_tint(image, overlap, overlap_color)


def make_contact_sheet(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    rows = math.ceil(len(tiles) / columns)
    tile_height = max(tile.shape[0] for tile in tiles)
    tile_width = max(tile.shape[1] for tile in tiles)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 18, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        y, x = row * tile_height, col * tile_width
        canvas[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
    return canvas


def initial_mask_iou(baseline_mask: np.ndarray, candidate_mask: np.ndarray) -> float:
    union = baseline_mask | candidate_mask
    if not union.any():
        raise RuntimeError("Both t=0 robot masks are empty; semantic robot rendering is not working.")
    return float((baseline_mask & candidate_mask).sum() / union.sum())


def main() -> None:
    args = parse_args()
    if args.frames < 1 or args.columns < 1 or args.max_frame_width < 1:
        raise ValueError("--frames, --columns, and --max-frame-width must be positive.")
    if args.initial_frame_mae_max < 0 or not 0 <= args.initial_mask_iou_min <= 1:
        raise ValueError("Initial-frame thresholds must be non-negative; IoU must be in [0, 1].")
    if bool(args.baseline_mask_video) != bool(args.candidate_mask_video):
        raise ValueError("Provide both robot mask videos or neither.")
    if bool(args.baseline_object_mask_video) != bool(args.candidate_object_mask_video):
        raise ValueError("Provide both tracked-object mask videos or neither.")
    if args.baseline_object_mask_video and not args.baseline_mask_video:
        raise ValueError("Tracked-object masks require both robot mask videos.")
    if args.layout == "overlay" and not args.baseline_mask_video:
        raise ValueError("--layout overlay requires both robot mask videos.")
    if args.layout == "overlay" and args.overlay_video is None:
        raise ValueError("--layout overlay requires --overlay-video.")

    baseline_reader = VideoReader(args.baseline_video)
    candidate_reader = VideoReader(args.candidate_video)
    baseline_mask_reader = VideoReader(args.baseline_mask_video) if args.baseline_mask_video else None
    candidate_mask_reader = VideoReader(args.candidate_mask_video) if args.candidate_mask_video else None
    baseline_breakfast_mask_reader = (
        VideoReader(args.baseline_object_mask_video)
        if args.baseline_object_mask_video else None
    )
    candidate_breakfast_mask_reader = (
        VideoReader(args.candidate_object_mask_video)
        if args.candidate_object_mask_video else None
    )
    try:
        readers = [baseline_reader, candidate_reader]
        if baseline_mask_reader and candidate_mask_reader:
            readers.extend((baseline_mask_reader, candidate_mask_reader))
        if baseline_breakfast_mask_reader and candidate_breakfast_mask_reader:
            readers.extend((baseline_breakfast_mask_reader, candidate_breakfast_mask_reader))
        if max(reader.fps for reader in readers) - min(reader.fps for reader in readers) > 1.0e-6:
            raise RuntimeError("RGB and mask videos must have the same FPS for physical-time synchronization.")
        common_frames = min(reader.frame_count for reader in readers)
        fps = baseline_reader.fps

        initial_baseline = baseline_reader.read_index(0)
        initial_candidate = candidate_reader.read_index(0)
        initial_candidate = cv2.resize(initial_candidate, (initial_baseline.shape[1], initial_baseline.shape[0]))
        initial_mae = float(np.mean(np.abs(initial_baseline.astype(np.int16) - initial_candidate.astype(np.int16))))

        initial_baseline_mask = normalize_mask(
            baseline_mask_reader.read_index(0), initial_baseline.shape[:2]
        ) if baseline_mask_reader else None
        initial_candidate_mask = normalize_mask(
            candidate_mask_reader.read_index(0), initial_baseline.shape[:2]
        ) if candidate_mask_reader else None
        initial_baseline_breakfast_mask = normalize_mask(
            baseline_breakfast_mask_reader.read_index(0), initial_baseline.shape[:2]
        ) if baseline_breakfast_mask_reader else None
        initial_candidate_breakfast_mask = normalize_mask(
            candidate_breakfast_mask_reader.read_index(0), initial_baseline.shape[:2]
        ) if candidate_breakfast_mask_reader else None
        initial_iou = (
            initial_mask_iou(initial_baseline_mask, initial_candidate_mask)
            if initial_baseline_mask is not None and initial_candidate_mask is not None else None
        )
        initial_breakfast_iou = (
            initial_mask_iou(initial_baseline_breakfast_mask, initial_candidate_breakfast_mask)
            if initial_baseline_breakfast_mask is not None and initial_candidate_breakfast_mask is not None else None
        )
        if initial_mae > args.initial_frame_mae_max:
            raise RuntimeError(
                f"t=0 RGB frames differ (MAE={initial_mae:.6f}, allowed={args.initial_frame_mae_max:.6f}). "
                "The two replays do not share an identical initial state."
            )
        if initial_iou is not None and initial_iou < args.initial_mask_iou_min:
            raise RuntimeError(
                f"t=0 robot masks do not fully overlap (IoU={initial_iou:.6f}, "
                f"required={args.initial_mask_iou_min:.6f})."
            )
        if initial_breakfast_iou is not None and initial_breakfast_iou < args.initial_mask_iou_min:
            raise RuntimeError(
                f"t=0 breakfast-object masks do not fully overlap (IoU={initial_breakfast_iou:.6f}, "
                f"required={args.initial_mask_iou_min:.6f})."
            )

        background_path = args.background_output or args.output.with_name("static_background.png")
        background_path.parent.mkdir(parents=True, exist_ok=True)
        if args.layout == "overlay":
            # Inpaint coincident t=0 comparison geometry once.  This produces
            # one static scene background; later frames add only the blue /
            # orange robot silhouettes and, when requested, breakfast objects.
            initial_comparison_mask = initial_baseline_mask | initial_candidate_mask
            if initial_baseline_breakfast_mask is not None:
                initial_comparison_mask |= (
                    initial_baseline_breakfast_mask | initial_candidate_breakfast_mask
                )
            background = cv2.inpaint(
                initial_baseline,
                (initial_comparison_mask.astype(np.uint8) * 255),
                5,
                cv2.INPAINT_TELEA,
            )
            if not cv2.imwrite(str(background_path), background):
                raise RuntimeError(f"Could not write static background: {background_path}")
        else:
            background = None

        overlay_writer = None
        if args.overlay_video:
            args.overlay_video.parent.mkdir(parents=True, exist_ok=True)
            overlay_writer = OverlayVideoWriter(args.overlay_video, fps)

        sample_indices = [0] if args.frames == 1 else np.linspace(0, common_frames - 1, args.frames).round().astype(int).tolist()
        samples: dict[int, np.ndarray] = {}
        for index in range(common_frames):
            baseline = baseline_reader.read_index(index)
            candidate = candidate_reader.read_index(index)
            candidate = cv2.resize(candidate, (baseline.shape[1], baseline.shape[0]), interpolation=cv2.INTER_AREA)
            if args.layout == "overlay":
                baseline_mask = normalize_mask(baseline_mask_reader.read_index(index), baseline.shape[:2])
                candidate_mask = normalize_mask(candidate_mask_reader.read_index(index), baseline.shape[:2])
                if baseline_breakfast_mask_reader is not None:
                    baseline_mask |= normalize_mask(
                        baseline_breakfast_mask_reader.read_index(index), baseline.shape[:2]
                    )
                    candidate_mask |= normalize_mask(
                        candidate_breakfast_mask_reader.read_index(index), baseline.shape[:2]
                    )
                frame = make_overlay(background, baseline_mask, candidate_mask)
            else:
                frame = np.hstack((baseline, candidate))
            if overlay_writer is not None:
                overlay_writer.write(frame)
            if index in sample_indices:
                samples[index] = frame

        if overlay_writer is not None:
            overlay_writer.release()

        tiles = []
        for index in sample_indices:
            frame = resize_frame(samples[index], args.max_frame_width)
            timestamp_s = index / fps
            if args.layout == "overlay":
                compared = "robot + tracked objects" if baseline_breakfast_mask_reader else "robot"
                caption = f"{args.baseline_label} (blue) + {args.candidate_label} (orange): {compared}"
                tile = add_header(frame, caption, (150, 150, 150))
            else:
                tile = add_header(frame, f"{args.baseline_label} | {args.candidate_label}", (150, 150, 150))
            cv2.putText(tile, f"t = {timestamp_s:.2f} s", (14, tile.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, TEXT_COLOR, 2, cv2.LINE_AA)
            tiles.append(tile)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), make_contact_sheet(tiles, args.columns)):
            raise RuntimeError(f"Could not write contact sheet: {args.output}")

        manifest = {
            "baseline_video": str(args.baseline_video),
            "candidate_video": str(args.candidate_video),
            "baseline_mask_video": str(args.baseline_mask_video) if args.baseline_mask_video else None,
            "candidate_mask_video": str(args.candidate_mask_video) if args.candidate_mask_video else None,
            "baseline_object_mask_video": (
                str(args.baseline_object_mask_video)
                if args.baseline_object_mask_video else None
            ),
            "candidate_object_mask_video": (
                str(args.candidate_object_mask_video)
                if args.candidate_object_mask_video else None
            ),
            "overlay_video": str(args.overlay_video) if args.overlay_video else None,
            "static_background": str(background_path) if background is not None else None,
            "layout": args.layout,
            "frames": args.frames,
            "fps": fps,
            "common_frames": common_frames,
            "common_duration_s": round((common_frames - 1) / fps, 6),
            "timestamps_s": [round(index / fps, 6) for index in sample_indices],
            "initial_frame_rgb_mae": round(initial_mae, 6),
            "initial_robot_mask_iou": round(initial_iou, 6) if initial_iou is not None else None,
            "initial_object_mask_iou": (
                round(initial_breakfast_iou, 6) if initial_breakfast_iou is not None else None
            ),
            "robot_masks_used": bool(args.baseline_mask_video),
            "object_masks_used": bool(args.baseline_object_mask_video),
        }
        manifest_path = args.output.with_suffix(".json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote synchronized contact sheet: {args.output}")
        print(f"Wrote blue/orange overlay video: {args.overlay_video}")
        print(f"Wrote static background: {background_path}")
        print(f"Wrote comparison manifest: {manifest_path}")
    finally:
        if 'overlay_writer' in locals() and overlay_writer is not None:
            overlay_writer.release()
        baseline_reader.release()
        candidate_reader.release()
        if baseline_mask_reader:
            baseline_mask_reader.release()
        if candidate_mask_reader:
            candidate_mask_reader.release()
        if baseline_breakfast_mask_reader:
            baseline_breakfast_mask_reader.release()
        if candidate_breakfast_mask_reader:
            candidate_breakfast_mask_reader.release()


if __name__ == "__main__":
    main()
