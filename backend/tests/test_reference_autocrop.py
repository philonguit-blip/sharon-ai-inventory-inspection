from __future__ import annotations

import cv2
import numpy as np

from app.services.reference_autocrop_service import (
    analyse_reference_mask,
    auto_crop_reference,
    build_colour_foreground_masks,
    build_reference_prompt_points,
    composite_reference_on_white,
    deduplicate_reference_candidates,
    refine_reference_foreground_mask,
    select_reference_foreground_candidate,
    segment_tight_reference_foreground,
)


class _ArrayProxy:
    def __init__(self, values: np.ndarray):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Masks:
    def __init__(self, values: np.ndarray):
        self.data = _ArrayProxy(values)


class _SamResult:
    def __init__(self, values: np.ndarray):
        self.masks = _Masks(values)


class _FakeSam:
    def __init__(self, masks: np.ndarray):
        self.masks = masks
        self.points = None
        self.bboxes = None

    def predict(self, **kwargs):
        if kwargs.get("points") is not None:
            self.points = kwargs.get("points")
        if kwargs.get("bboxes") is not None:
            self.bboxes = kwargs.get("bboxes")
        return [_SamResult(self.masks)]


class _ColourEncoder:
    def encode_rgb(self, images, batch_size=16):
        del batch_size
        rows = []
        for image in images:
            mean = image.reshape(-1, 3).mean(axis=0)
            rows.append([mean[0], mean[1]])
        return np.asarray(rows, dtype=np.float32)


def test_large_product_mask_is_accepted_but_whole_frame_is_rejected():
    product = np.zeros((100, 100), dtype=bool)
    product[25:80, 20:80] = True
    candidate = analyse_reference_mask(product)

    assert candidate is not None
    assert candidate.box_ratio == 0.33
    assert candidate.fill_ratio == 1.0
    assert analyse_reference_mask(np.ones((100, 100), dtype=bool)) is None


def test_candidate_touching_two_borders_is_rejected():
    mask = np.zeros((100, 160), dtype=bool)
    mask[0:55, 40:160] = True

    assert analyse_reference_mask(mask) is None


def test_colour_foreground_proposal_finds_warm_product_on_neutral_background():
    image = np.full((120, 180, 3), 170, dtype=np.uint8)
    image[30:100, 45:145] = (35, 95, 170)

    candidates = [
        analyse_reference_mask(mask, proposal_source="colour")
        for mask in build_colour_foreground_masks(image)
    ]

    assert any(candidate is not None for candidate in candidates)


def test_prompt_grid_searches_beyond_the_old_central_region():
    points = build_reference_prompt_points(1000, 500, grid_size=7)
    xs = points[:, 0, 0]
    ys = points[:, 0, 1]

    assert len(points) == 49
    assert xs.min() < 100
    assert xs.max() > 900
    assert ys.min() < 50
    assert ys.max() > 450


def test_candidate_deduplication_keeps_the_stronger_overlapping_mask():
    strong_mask = np.zeros((100, 100), dtype=bool)
    strong_mask[20:80, 20:80] = True
    weak_mask = np.zeros((100, 100), dtype=bool)
    weak_mask[21:79, 21:79] = True
    strong = analyse_reference_mask(strong_mask)
    weak = analyse_reference_mask(weak_mask)

    kept = deduplicate_reference_candidates([weak, strong], iou_threshold=0.80)

    assert len(kept) == 1
    assert kept[0].geometry_score == max(
        strong.geometry_score,
        weak.geometry_score,
    )


def test_existing_sku_dino_ranking_selects_the_semantic_product_crop():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:80, 5:45] = (0, 0, 255)  # red in BGR
    image[20:80, 55:95] = (0, 255, 0)

    left = np.zeros((100, 100), dtype=np.float32)
    left[20:80, 5:45] = 1.0
    right = np.zeros((100, 100), dtype=np.float32)
    right[20:80, 55:95] = 1.0
    sam = _FakeSam(np.stack([right, left]))

    result = auto_crop_reference(
        image,
        sam_model=sam,
        max_side=640,
        padding=0.0,
        device="cpu",
        encoder=_ColourEncoder(),
        target_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    assert result.box_xyxy == [5, 20, 45, 80]
    assert result.metadata["method"] == "sam_geometry+dino_target"
    assert result.metadata["target_similarity"] > 0.99
    assert result.foreground_mask is not None
    assert result.foreground_mask.shape == result.crop_bgr.shape[:2]
    assert result.foreground_mask.all()


def test_white_background_composite_preserves_product_and_fills_mask_holes():
    image = np.full((60, 80, 3), (20, 40, 60), dtype=np.uint8)
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[10:50, 15:65] = 1
    mask[25:35, 30:40] = 0  # surface detail must not become a white hole

    result, metadata = composite_reference_on_white(
        image,
        mask,
        feather_radius=0,
    )

    assert np.array_equal(result[0, 0], [255, 255, 255])
    assert np.array_equal(result[30, 35], [20, 40, 60])
    assert np.array_equal(result[20, 20], [20, 40, 60])
    assert metadata["mode"] == "white"
    assert 0.35 < metadata["foreground_coverage"] < 0.50


def test_white_background_rejects_an_empty_or_full_frame_mask():
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    for mask in (
        np.zeros((20, 20), dtype=np.uint8),
        np.ones((20, 20), dtype=np.uint8),
    ):
        try:
            composite_reference_on_white(image, mask, feather_radius=0)
        except ValueError:
            continue
        raise AssertionError("Unsafe mask should have been rejected")


def test_tight_reference_segmentation_keeps_the_existing_frame():
    image = np.full((80, 120, 3), 180, dtype=np.uint8)
    image[10:70, 15:105] = (30, 90, 160)
    product = np.zeros((80, 120), dtype=np.float32)
    product[10:70, 15:105] = 1.0
    sam = _FakeSam(np.stack([product]))

    mask, metadata = segment_tight_reference_foreground(
        image,
        sam_model=sam,
        max_side=640,
        device="cpu",
    )

    assert mask.shape == image.shape[:2]
    assert mask[30, 40]
    assert not mask[0, 0]
    assert metadata["method"] == "sam_multi_prompt_tight_reference+grabcut"
    assert sam.bboxes is not None
    assert sam.points is not None
    assert metadata["refinement"] == "aligned_sam_silhouette"


def test_grabcut_refinement_removes_connected_neutral_tray_leakage():
    image = np.full((160, 220, 3), 185, dtype=np.uint8)
    initial = np.zeros((160, 220), dtype=np.uint8)

    cv2.ellipse(image, (95, 80), (68, 55), 0, 0, 360, (35, 90, 165), -1)
    cv2.ellipse(initial, (95, 80), (68, 55), 0, 0, 360, 1, -1)
    # Simulate the exact failure class: a connected steel/shadow polygon is
    # included below/right of the product by the raw SAM mask.
    leak = np.asarray([[120, 118], [205, 142], [185, 105]], dtype=np.int32)
    cv2.fillPoly(initial, [leak], 1)
    cv2.fillPoly(image, [leak], (145, 145, 145))

    refined, metadata = refine_reference_foreground_mask(
        image,
        initial,
        iterations=5,
    )

    assert refined[80, 95]
    assert not refined[132, 190]
    assert metadata["removed_mask_ratio"] > 0.03
    assert metadata["retained_mask_ratio"] > 0.60


def test_aligned_sam_mask_replaces_coarse_colour_mask_for_white_background():
    colour_mask = np.zeros((100, 120), dtype=np.uint8)
    colour_mask[15:90, 15:105] = 1
    sam_mask = np.zeros((100, 120), dtype=np.uint8)
    sam_mask[20:85, 20:100] = 1
    unrelated = np.zeros((100, 120), dtype=np.uint8)
    unrelated[5:35, 75:110] = 1

    selected = analyse_reference_mask(
        colour_mask,
        proposal_source="colour",
    )
    aligned_sam = analyse_reference_mask(sam_mask, proposal_source="sam")
    other_sam = analyse_reference_mask(unrelated, proposal_source="sam")

    assert selected is not None
    assert aligned_sam is not None
    assert other_sam is not None
    foreground = select_reference_foreground_candidate(
        selected,
        [selected, other_sam, aligned_sam],
    )

    assert foreground is aligned_sam
