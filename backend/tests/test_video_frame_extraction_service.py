from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.video_frame_extraction_service import (
    VideoFrameExtractionConfig,
    extract_video_frames,
)


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float = 2.0) -> None:
        self.frames = [frame.copy() for frame in frames]
        self.fps = fps
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.frames[0].shape[1])
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.frames[0].shape[0])
        return 0.0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index].copy()
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def _pattern(kind: int) -> np.ndarray:
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    if kind == 0:
        cv2.rectangle(image, (8, 8), (55, 70), (255, 255, 255), -1)
    elif kind == 1:
        cv2.circle(image, (75, 40), 28, (255, 255, 255), -1)
    else:
        cv2.line(image, (0, 0), (119, 79), (255, 255, 255), 12)
    return image


def test_extracts_diverse_frames_with_lineage_and_limit(monkeypatch) -> None:
    frames = [_pattern(0), _pattern(0), _pattern(1), _pattern(2)]
    fake = _FakeCapture(frames)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: fake)

    result = extract_video_frames(
        "ignored.mp4",
        source_name="bakery sample.mp4",
        config=VideoFrameExtractionConfig(
            target_fps=2.0,
            blur_threshold=0.0,
            similarity_threshold=0,
            max_dimension=64,
            jpeg_quality=90,
            max_frames=2,
        ),
    )

    assert fake.released is True
    assert len(result.frames) == 2
    assert result.stats["saved"] == 2
    assert result.stats["rejected_duplicate"] == 1
    assert result.stats["stopped_at_limit"] is True
    assert result.frames[0].file_name.startswith("bakery_sample_frame_0001_")
    assert result.frames[0].metadata["source_video"] == "bakery sample.mp4"
    assert result.frames[0].metadata["source_frame_idx"] == 0
    assert result.frames[1].metadata["source_frame_idx"] == 2
    assert max(
        result.frames[0].metadata["saved_width"],
        result.frames[0].metadata["saved_height"],
    ) == 64
    decoded = cv2.imdecode(
        np.frombuffer(result.frames[0].jpeg_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None


def test_rejects_blurry_uniform_frame(monkeypatch) -> None:
    fake = _FakeCapture([np.full((60, 80, 3), 127, dtype=np.uint8)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: fake)

    result = extract_video_frames(
        "ignored.mp4",
        config=VideoFrameExtractionConfig(
            blur_threshold=1.0,
            max_frames=5,
        ),
    )

    assert result.frames == []
    assert result.stats["rejected_blur"] == 1


@pytest.mark.parametrize(
    "config",
    [
        VideoFrameExtractionConfig(target_fps=0),
        VideoFrameExtractionConfig(similarity_threshold=65),
        VideoFrameExtractionConfig(max_frames=0),
        VideoFrameExtractionConfig(min_brightness=200, max_brightness=100),
    ],
)
def test_rejects_unsafe_configuration(config) -> None:
    with pytest.raises(ValueError):
        config.validate()
