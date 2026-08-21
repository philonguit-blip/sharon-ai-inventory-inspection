"""Extract sharp, diverse video frames for Foundation pre-annotation.

This module is intentionally UI-agnostic so the Streamlit Test Lab and future
batch tools use exactly the same sampling, quality filtering and audit metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


@dataclass(frozen=True)
class VideoFrameExtractionConfig:
    target_fps: float = 2.0
    blur_threshold: float = 60.0
    similarity_threshold: int = 6
    max_dimension: int = 1920
    jpeg_quality: int = 92
    min_brightness: float | None = None
    max_brightness: float | None = None
    max_frames: int = 200

    def validate(self) -> None:
        if not np.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be greater than 0.")
        if self.blur_threshold < 0:
            raise ValueError("blur_threshold must be non-negative.")
        if not 0 <= self.similarity_threshold <= 64:
            raise ValueError("similarity_threshold must be between 0 and 64.")
        if self.max_dimension < 0:
            raise ValueError("max_dimension must be non-negative.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100.")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be greater than 0.")
        for name, value in (
            ("min_brightness", self.min_brightness),
            ("max_brightness", self.max_brightness),
        ):
            if value is not None and not 0 <= float(value) <= 255:
                raise ValueError(f"{name} must be between 0 and 255.")
        if (
            self.min_brightness is not None
            and self.max_brightness is not None
            and self.min_brightness >= self.max_brightness
        ):
            raise ValueError("min_brightness must be below max_brightness.")


@dataclass(frozen=True)
class ExtractedVideoFrame:
    file_name: str
    jpeg_bytes: bytes
    metadata: dict[str, Any]


@dataclass(frozen=True)
class VideoFrameExtractionResult:
    frames: list[ExtractedVideoFrame]
    stats: dict[str, Any]


def _quality_gray(frame_bgr: np.ndarray, max_width: int = 1280) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    measured = frame_bgr
    if width > max_width:
        scale = max_width / float(width)
        measured = cv2.resize(
            frame_bgr,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(measured, cv2.COLOR_BGR2GRAY)


def _perceptual_hash(frame_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
    low = cv2.dct(gray)[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    value = 0
    for bit in low > median:
        value = (value << 1) | int(bool(bit))
    return value


def _resize_for_export(frame_bgr: np.ndarray, max_dimension: int) -> np.ndarray:
    if max_dimension <= 0:
        return frame_bgr
    height, width = frame_bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return frame_bgr
    scale = max_dimension / float(longest)
    return cv2.resize(
        frame_bgr,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def extract_video_frames(
    video_path: str | Path,
    *,
    source_name: str | None = None,
    config: VideoFrameExtractionConfig | None = None,
) -> VideoFrameExtractionResult:
    """Extract in-memory JPEG frames and an auditable per-video summary."""
    active = config or VideoFrameExtractionConfig()
    active.validate()
    path = Path(video_path)
    display_name = str(source_name or path.name)
    safe_stem = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in Path(display_name).stem
    ).strip("_") or "video"

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open uploaded video: {display_name}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise RuntimeError(
            f"Uploaded video has invalid source FPS ({source_fps}): {display_name}"
        )

    effective_target_fps = min(float(active.target_fps), source_fps)
    frame_stride = max(1, int(round(source_fps / effective_target_fps)))
    saved_hashes: list[int] = []
    extracted: list[ExtractedVideoFrame] = []
    counters = {
        "sampled_candidates": 0,
        "rejected_blur": 0,
        "rejected_dark": 0,
        "rejected_bright": 0,
        "rejected_duplicate": 0,
    }
    frame_index = 0
    stopped_at_limit = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride != 0:
                frame_index += 1
                continue
            counters["sampled_candidates"] += 1

            measured = _quality_gray(frame)
            blur_score = float(cv2.Laplacian(measured, cv2.CV_64F).var())
            brightness = float(measured.mean())
            if blur_score < active.blur_threshold:
                counters["rejected_blur"] += 1
                frame_index += 1
                continue
            if (
                active.min_brightness is not None
                and brightness < active.min_brightness
            ):
                counters["rejected_dark"] += 1
                frame_index += 1
                continue
            if (
                active.max_brightness is not None
                and brightness > active.max_brightness
            ):
                counters["rejected_bright"] += 1
                frame_index += 1
                continue

            export_frame = _resize_for_export(frame, active.max_dimension)
            phash = _perceptual_hash(export_frame)
            nearest_distance = (
                min((phash ^ old).bit_count() for old in saved_hashes)
                if saved_hashes
                else None
            )
            if (
                nearest_distance is not None
                and nearest_distance <= active.similarity_threshold
            ):
                counters["rejected_duplicate"] += 1
                frame_index += 1
                continue

            timestamp_ms = int(round(frame_index / source_fps * 1000.0))
            sequence = len(extracted) + 1
            file_name = (
                f"{safe_stem}_frame_{sequence:04d}_"
                f"src{frame_index:07d}_{timestamp_ms:09d}ms.jpg"
            )
            encoded_ok, encoded = cv2.imencode(
                ".jpg",
                export_frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(active.jpeg_quality)],
            )
            if not encoded_ok:
                raise RuntimeError(f"Cannot encode extracted frame: {file_name}")
            saved_hashes.append(phash)
            extracted.append(
                ExtractedVideoFrame(
                    file_name=file_name,
                    jpeg_bytes=encoded.tobytes(),
                    metadata={
                        "source_type": "video_frame",
                        "source_video": display_name,
                        "source_frame_idx": int(frame_index),
                        "timestamp_ms": timestamp_ms,
                        "source_width": source_width,
                        "source_height": source_height,
                        "saved_width": int(export_frame.shape[1]),
                        "saved_height": int(export_frame.shape[0]),
                        "blur_score": round(blur_score, 3),
                        "brightness": round(brightness, 3),
                        "phash": f"{phash:016x}",
                        "nearest_saved_phash_distance": nearest_distance,
                    },
                )
            )
            frame_index += 1
            if len(extracted) >= active.max_frames:
                stopped_at_limit = True
                break
    finally:
        capture.release()

    stats: dict[str, Any] = {
        "video": display_name,
        "source_fps": source_fps,
        "total_frames": total_frames,
        "duration_seconds": (
            round(total_frames / source_fps, 3) if total_frames > 0 else 0.0
        ),
        "source_width": source_width,
        "source_height": source_height,
        "frame_stride": frame_stride,
        "saved": len(extracted),
        "max_frames": int(active.max_frames),
        "stopped_at_limit": stopped_at_limit,
        **counters,
    }
    return VideoFrameExtractionResult(frames=extracted, stats=stats)
