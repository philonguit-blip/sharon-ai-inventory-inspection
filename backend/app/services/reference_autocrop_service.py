"""Safe SAM2 auto-cropping for Foundation reference images.

The reference manager receives full one-product photos whose product can be
large or off-centre.  This module deliberately keeps crop selection separate
from production instance segmentation: it searches the whole image, rejects
container/background masks, deduplicates overlapping proposals and can use
existing DINO reference vectors to select the proposal for a known SKU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np


class ReferenceEncoder(Protocol):
    def encode_rgb(
        self,
        images: list[np.ndarray],
        batch_size: int = 16,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ReferenceCropCandidate:
    box_xyxy: tuple[int, int, int, int]
    mask_ratio: float
    box_ratio: float
    fill_ratio: float
    centre_x: float
    centre_y: float
    aspect: float
    border_touches: int
    geometry_score: float
    proposal_source: str = "sam"


@dataclass(frozen=True)
class ReferenceCropResult:
    crop_bgr: np.ndarray
    box_xyxy: list[int]
    mask_ratio: float
    metadata: dict[str, Any]
    foreground_mask: np.ndarray | None = None


def composite_reference_on_white(
    image_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    feather_radius: float = 1.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Place one segmented reference product on an opaque white background.

    The largest connected component is retained and its internal holes are
    filled so seeds, scoring marks and crumb texture are not accidentally
    painted white. A small optional feather keeps object boundaries natural
    without introducing transparency into the saved reference image.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Reference image is empty.")
    mask = np.asarray(foreground_mask)
    if mask.ndim != 2:
        raise ValueError("Foreground mask must be a 2D array.")
    height, width = image_bgr.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        raise ValueError("Foreground mask is empty.")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        raise ValueError("Foreground mask has no connected component.")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cleaned = (labels == largest).astype(np.uint8)

    # Close only tiny boundary gaps, then fill true internal holes by drawing
    # the external contour. This preserves the product silhouette while
    # preventing its surface details from turning into background pixels.
    kernel_size = max(3, int(round(min(height, width) * 0.008)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("Foreground mask has no usable contour.")
    filled = np.zeros_like(cleaned)
    cv2.drawContours(
        filled,
        [max(contours, key=cv2.contourArea)],
        -1,
        1,
        thickness=cv2.FILLED,
    )
    coverage = float(filled.mean())
    if not 0.01 <= coverage <= 0.995:
        raise ValueError(
            f"Unsafe foreground coverage for white background: {coverage:.3f}"
        )

    radius = max(0.0, float(feather_radius))
    alpha = filled.astype(np.float32)
    if radius > 0:
        sigma = max(0.1, radius)
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
        alpha = np.clip(alpha, 0.0, 1.0)
    alpha = alpha[:, :, None]
    white = np.full_like(image_bgr, 255)
    composited = np.rint(
        image_bgr.astype(np.float32) * alpha
        + white.astype(np.float32) * (1.0 - alpha)
    ).astype(np.uint8)
    return composited, {
        "mode": "white",
        "foreground_coverage": coverage,
        "feather_radius": radius,
        "component_count": int(count - 1),
    }


def _largest_mask_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(binary, dtype=bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def refine_reference_foreground_mask(
    image_bgr: np.ndarray,
    initial_mask: np.ndarray,
    *,
    iterations: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove tray/shadow leakage from a SAM foreground mask with GrabCut.

    SAM occasionally joins a low-texture tray or cast shadow to a bakery item.
    The initial SAM mask remains a hard outer boundary; a colour-distant eroded
    core is used as definite foreground while its uncertain edge is allowed to
    contract. The result must retain the dominant component and a safe fraction
    of the original object, otherwise the operation fails for manual review.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Reference image is empty.")
    height, width = image_bgr.shape[:2]
    mask = np.asarray(initial_mask)
    if mask.ndim != 2:
        raise ValueError("Initial foreground mask must be 2D.")
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    initial = _largest_mask_component(mask)
    initial_coverage = float(initial.mean())
    if not 0.01 <= initial_coverage <= 0.95:
        raise ValueError(
            f"Unsafe initial foreground coverage: {initial_coverage:.3f}"
        )

    min_side = min(height, width)
    erosion_size = max(3, int(round(min_side * 0.025)))
    if erosion_size % 2 == 0:
        erosion_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (erosion_size, erosion_size),
    )
    eroded = cv2.erode(initial.astype(np.uint8), kernel, iterations=1) > 0
    outside = ~initial
    if float(outside.mean()) < 0.02 or not eroded.any():
        raise ValueError("Foreground mask leaves insufficient refinement seeds.")

    # CIE Lab separates neutral steel/plastic from warm bakery surfaces more
    # reliably than raw BGR. Scale channels to comparable perceptual ranges.
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    background_pixels = lab[outside]
    background_centre = np.median(background_pixels, axis=0)
    colour_distance = np.sqrt(
        ((lab[:, :, 0] - background_centre[0]) / 1.8) ** 2
        + (lab[:, :, 1] - background_centre[1]) ** 2
        + (lab[:, :, 2] - background_centre[2]) ** 2
    )
    chroma_distance = np.sqrt(
        (lab[:, :, 1] - background_centre[1]) ** 2
        + (lab[:, :, 2] - background_centre[2]) ** 2
    )
    core_distances = colour_distance[eroded]
    foreground_cut = float(np.percentile(core_distances, 38))
    sure_foreground = eroded & (colour_distance >= foreground_cut)
    if float(sure_foreground.mean()) < 0.008:
        sure_foreground = eroded

    grabcut_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[initial] = cv2.GC_PR_FGD
    grabcut_mask[sure_foreground] = cv2.GC_FGD

    # Pixels near the SAM edge that look like the observed background are
    # explicitly made probable background. This is the key tray/shadow guard.
    boundary = initial & ~eroded
    background_distance_limit = float(
        np.percentile(colour_distance[outside], 92)
    )
    background_chroma_limit = float(
        np.percentile(chroma_distance[outside], 95) + 4.0
    )
    grabcut_mask[
        initial
        & (chroma_distance <= background_chroma_limit)
        & ~sure_foreground
    ] = cv2.GC_PR_BGD
    grabcut_mask[boundary & (colour_distance <= background_distance_limit)] = (
        cv2.GC_PR_BGD
    )

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            image_bgr,
            grabcut_mask,
            None,
            bg_model,
            fg_model,
            max(1, int(iterations)),
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        raise RuntimeError(f"GrabCut foreground refinement failed: {exc}") from exc

    refined = np.isin(grabcut_mask, [cv2.GC_FGD, cv2.GC_PR_FGD]) & initial
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        refined.astype(np.uint8), 8
    )
    if count <= 1:
        raise RuntimeError("Foreground refinement removed every product component.")
    overlaps = [
        int(np.logical_and(labels == label, sure_foreground).sum())
        for label in range(1, count)
    ]
    selected_label = 1 + int(np.argmax(overlaps))
    refined = labels == selected_label

    close_size = max(3, int(round(min_side * 0.008)))
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    refined = cv2.morphologyEx(
        refined.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
    ).astype(bool)
    refined_coverage = float(refined.mean())
    retained_ratio = refined_coverage / max(initial_coverage, 1e-9)
    if not 0.01 <= refined_coverage <= 0.95 or not 0.48 <= retained_ratio <= 1.02:
        raise RuntimeError(
            "Unsafe foreground refinement: "
            f"coverage={refined_coverage:.3f}, retained={retained_ratio:.3f}."
        )

    return refined, {
        "refinement": "grabcut_lab_guarded",
        "initial_foreground_coverage": initial_coverage,
        "foreground_coverage": refined_coverage,
        "retained_mask_ratio": retained_ratio,
        "removed_mask_ratio": 1.0 - retained_ratio,
        "foreground_seed_ratio": float(sure_foreground.mean()),
        "background_distance_limit": background_distance_limit,
        "background_chroma_limit": background_chroma_limit,
        "foreground_distance_cut": foreground_cut,
        "grabcut_iterations": max(1, int(iterations)),
    }


def _tight_mask_candidate(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    source: str,
) -> tuple[float, np.ndarray, dict[str, Any]] | None:
    mask = _largest_mask_component(mask)
    if not mask.any():
        return None
    height, width = mask.shape
    ys, xs = np.where(mask)
    coverage = float(mask.mean())
    if not 0.015 <= coverage <= 0.92:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(
        ys.max() + 1
    )
    box_ratio = float((x2 - x1) * (y2 - y1) / max(1, height * width))
    fill_ratio = coverage / max(box_ratio, 1e-9)
    border_touches = (
        int(x1 <= 1)
        + int(y1 <= 1)
        + int(x2 >= width - 1)
        + int(y2 >= height - 1)
    )
    centre_patch = mask[
        int(height * 0.30) : max(int(height * 0.70), int(height * 0.30) + 1),
        int(width * 0.30) : max(int(width * 0.70), int(width * 0.30) + 1),
    ]
    centre_overlap = float(centre_patch.mean()) if centre_patch.size else 0.0

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    outside = ~mask
    colour_separation = 0.0
    if outside.any():
        inside_centre = np.median(lab[mask], axis=0)
        outside_centre = np.median(lab[outside], axis=0)
        colour_separation = float(
            np.linalg.norm(
                np.asarray(
                    [
                        (inside_centre[0] - outside_centre[0]) / 1.8,
                        inside_centre[1] - outside_centre[1],
                        inside_centre[2] - outside_centre[2],
                    ],
                    dtype=np.float32,
                )
            )
        )
    # SAM normally follows the physical product edge more closely than a
    # coarse colour blob. Colour remains a fallback for translucent boxes, but
    # must not win merely because a cast shadow enlarged its silhouette.
    source_bonus = {"point": 0.12, "box": 0.07, "colour": 0.0}.get(source, 0.0)
    score = (
        0.34 * min(1.0, fill_ratio)
        + 0.23 * centre_overlap
        + 0.18 * (1.0 - min(1.0, abs(coverage - 0.48) / 0.48))
        + 0.15 * min(1.0, colour_separation / 32.0)
        + source_bonus
        - 0.08 * border_touches
    )
    return float(score), mask, {
        "proposal_source": source,
        "coverage": coverage,
        "box_ratio": box_ratio,
        "fill_ratio": fill_ratio,
        "border_touches": border_touches,
        "centre_overlap": centre_overlap,
        "colour_separation": colour_separation,
    }


def segment_tight_reference_foreground(
    image_bgr: np.ndarray,
    *,
    sam_model: Any,
    max_side: int,
    device: str,
    margin_ratio: float = 0.03,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Segment the dominant product in an already-approved tight crop.

    Unlike :func:`auto_crop_reference`, this path does not search for or change
    the crop box. A nearly full-frame box prompt asks SAM only for the product
    silhouette. This is the appropriate operation when normalising the
    background of reference images that were already reviewed by an operator.
    """

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Reference image is empty.")
    source_height, source_width = image_bgr.shape[:2]
    scale = min(
        1.0,
        max(640, int(max_side)) / float(max(source_height, source_width)),
    )
    resized = cv2.resize(
        image_bgr,
        (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    height, width = resized.shape[:2]
    margin = min(0.12, max(0.01, float(margin_ratio)))
    prompt_box = [
        float(width) * margin,
        float(height) * margin,
        float(width) * (1.0 - margin),
        float(height) * (1.0 - margin),
    ]
    box_results = sam_model.predict(
        source=resized,
        bboxes=[prompt_box],
        conf=0.0,
        device=device,
        save=False,
        verbose=False,
        retina_masks=True,
    )
    points = build_reference_prompt_points(width, height, grid_size=5, margin_ratio=0.12)
    point_results = sam_model.predict(
        source=resized,
        points=points,
        labels=np.ones((len(points), 1), dtype=np.int32),
        conf=0.10,
        device=device,
        save=False,
        verbose=False,
        retina_masks=True,
    )
    candidate_inputs: list[tuple[str, np.ndarray]] = []
    for source, results in (("box", box_results), ("point", point_results)):
        if not results or results[0].masks is None:
            continue
        for raw_mask in results[0].masks.data.detach().cpu().numpy() > 0.5:
            candidate_inputs.append((source, raw_mask))
    for colour_mask in build_colour_foreground_masks(resized):
        candidate_inputs.append(("colour", colour_mask))

    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for source, raw_mask in candidate_inputs:
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        candidate = _tight_mask_candidate(resized, mask, source=source)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError(
            "SAM/colour proposals produced no safe tight-reference foreground mask."
        )
    score, selected, candidate_metadata = max(candidates, key=lambda item: item[0])
    if candidate_metadata.get("proposal_source") == "colour":
        refined, refinement_metadata = refine_reference_foreground_mask(
            resized,
            selected,
        )
    else:
        refined = selected
        coverage = float(refined.mean())
        refinement_metadata = {
            "refinement": "aligned_sam_silhouette",
            "initial_foreground_coverage": coverage,
            "foreground_coverage": coverage,
            "retained_mask_ratio": 1.0,
            "removed_mask_ratio": 0.0,
        }
    source_mask = cv2.resize(
        refined.astype(np.uint8),
        (source_width, source_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    return source_mask, {
        "method": "sam_multi_prompt_tight_reference+grabcut",
        "candidate_count": int(len(candidates)),
        "foreground_coverage": float(source_mask.mean()),
        "selection_score": float(score),
        "prompt_margin_ratio": margin,
        **candidate_metadata,
        **refinement_metadata,
    }


def build_reference_prompt_points(
    width: int,
    height: int,
    *,
    grid_size: int = 7,
    margin_ratio: float = 0.08,
) -> np.ndarray:
    """Return independent positive SAM prompts spanning the whole frame."""

    grid_size = max(3, int(grid_size))
    margin_ratio = min(0.20, max(0.02, float(margin_ratio)))
    xs = np.linspace(margin_ratio, 1.0 - margin_ratio, grid_size)
    ys = np.linspace(margin_ratio, 1.0 - margin_ratio, grid_size)
    return np.asarray(
        [[[float(width) * x, float(height) * y]] for y in ys for x in xs],
        dtype=np.float32,
    )


def analyse_reference_mask(
    mask: np.ndarray,
    *,
    proposal_source: str = "sam",
) -> ReferenceCropCandidate | None:
    """Validate and score one SAM mask as a possible single product."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not mask.any():
        return None

    height, width = mask.shape
    ys, xs = np.where(mask)
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max() + 1), int(ys.max() + 1)
    box_width = x2 - x1
    box_height = y2 - y1
    image_area = max(1, width * height)

    mask_ratio = float(mask.mean())
    box_ratio = float(box_width * box_height / image_area)
    fill_ratio = float(mask_ratio / max(box_ratio, 1e-12))
    centre_x = float((x1 + x2) / (2 * width))
    centre_y = float((y1 + y2) / (2 * height))
    aspect = float(box_width / max(1, box_height))
    border_touches = int(x1 <= 1) + int(y1 <= 1) + int(x2 >= width - 1) + int(
        y2 >= height - 1
    )

    # Whole-frame masks are normally the translucent container or background.
    # The old 0.18 box limit rejected valid large loaves, so large foreground
    # objects are now allowed while fill/border checks reject broad backgrounds.
    if not (
        0.004 <= mask_ratio <= 0.68
        and 0.008 <= box_ratio <= 0.72
        and 0.04 <= centre_x <= 0.96
        and 0.04 <= centre_y <= 0.96
        and 0.25 <= aspect <= 4.0
        and fill_ratio >= 0.30
        and border_touches <= 1
    ):
        return None
    if box_ratio > 0.55 and fill_ratio < 0.46:
        return None
    # A reference product may legitimately be close to one edge, but a mask
    # touching two edges is either clipped or is usually the tray/container.
    # Keeping such a crop silently pollutes the reference bank with background.
    if border_touches >= 2:
        return None

    centre_distance = min(
        1.0,
        float(np.hypot(centre_x - 0.5, centre_y - 0.5) / np.hypot(0.5, 0.5)),
    )
    centre_score = 1.0 - centre_distance
    size_score = min(1.0, float(np.sqrt(box_ratio / 0.28)))
    if box_ratio > 0.58:
        size_score *= max(0.0, 1.0 - (box_ratio - 0.58) / 0.14)
    border_score = 1.0 - 0.32 * border_touches
    geometry_score = (
        0.48 * min(1.0, fill_ratio)
        + 0.27 * size_score
        + 0.15 * centre_score
        + 0.10 * max(0.0, border_score)
    )

    return ReferenceCropCandidate(
        box_xyxy=(x1, y1, x2, y2),
        mask_ratio=mask_ratio,
        box_ratio=box_ratio,
        fill_ratio=fill_ratio,
        centre_x=centre_x,
        centre_y=centre_y,
        aspect=aspect,
        border_touches=border_touches,
        geometry_score=float(geometry_score),
        proposal_source=str(proposal_source),
    )


def build_colour_foreground_masks(image_bgr: np.ndarray) -> list[np.ndarray]:
    """Build coarse foreground proposals for cases where SAM follows a tray.

    Translucent bakery containers can cause point-prompt SAM to segment the
    plastic/background rather than the product. These conservative colour and
    chroma proposals are only candidates; existing-SKU DINO ranking still
    decides whether they resemble the requested product.
    """

    height, width = image_bgr.shape[:2]
    image_area = max(1, height * width)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    saturation_floor = max(34, int(np.percentile(saturation, 48)))
    chroma_floor = max(44, int(np.percentile(saturation, 67)))
    raw_masks = [
        (((hue <= 42) | (hue >= 168)) & (saturation >= saturation_floor)),
        (saturation >= chroma_floor),
        (((hue <= 48) | (hue >= 162)) & (saturation >= 28) & (value <= 210)),
    ]

    # Seed/crumb texture leaves holes across a loaf surface. A relatively wide
    # close operation reconnects those warm/chromatic islands into one product
    # proposal before the convex hull is formed.
    close_size = max(7, int(round(min(height, width) * 0.035)))
    if close_size % 2 == 0:
        close_size += 1
    open_size = max(3, int(round(min(height, width) * 0.008)))
    if open_size % 2 == 0:
        open_size += 1
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_size, close_size),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (open_size, open_size),
    )

    proposals: list[np.ndarray] = []
    for raw_mask in raw_masks:
        cleaned = (raw_mask.astype(np.uint8) * 255)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)
        contours, _ = cv2.findContours(
            cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        useful = [
            contour
            for contour in contours
            if cv2.contourArea(contour) / image_area >= 0.004
        ]
        for contour in sorted(useful, key=cv2.contourArea, reverse=True)[:4]:
            hull_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(
                hull_mask,
                [cv2.convexHull(contour)],
                -1,
                1,
                thickness=cv2.FILLED,
            )
            proposals.append(hull_mask.astype(bool))
    return proposals


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def select_reference_foreground_candidate(
    selected_crop_candidate: ReferenceCropCandidate,
    candidates: list[ReferenceCropCandidate],
) -> ReferenceCropCandidate:
    """Prefer an aligned SAM silhouette over a coarse colour crop mask.

    Colour proposals help localise warm bakery items when a translucent box
    confuses SAM, but their connected components can absorb a cast shadow on a
    metal tray. If SAM produced a strong mask for the same object, use that
    tighter silhouette only for white compositing while preserving the crop.
    """

    if selected_crop_candidate.proposal_source != "colour":
        return selected_crop_candidate
    compatible = [
        candidate
        for candidate in candidates
        if candidate.proposal_source == "sam"
        and _box_iou(
            candidate.box_xyxy,
            selected_crop_candidate.box_xyxy,
        )
        >= 0.55
        and candidate.mask_ratio
        >= selected_crop_candidate.mask_ratio * 0.60
        and candidate.mask_ratio
        <= selected_crop_candidate.mask_ratio * 1.18
        and candidate.fill_ratio >= 0.58
        and candidate.border_touches <= selected_crop_candidate.border_touches
    ]
    if not compatible:
        return selected_crop_candidate
    return max(compatible, key=lambda candidate: candidate.geometry_score)


def deduplicate_reference_candidates(
    candidates: list[ReferenceCropCandidate],
    *,
    iou_threshold: float = 0.88,
    limit: int = 12,
) -> list[ReferenceCropCandidate]:
    kept: list[ReferenceCropCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.geometry_score,
        reverse=True,
    ):
        if any(
            _box_iou(candidate.box_xyxy, existing.box_xyxy) >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(candidate)
        if len(kept) >= max(1, int(limit)):
            break
    return kept


def _padded_source_box(
    candidate: ReferenceCropCandidate,
    *,
    scale: float,
    padding: float,
    source_width: int,
    source_height: int,
) -> list[int]:
    x1, y1, x2, y2 = [value / max(scale, 1e-12) for value in candidate.box_xyxy]
    pad_x = (x2 - x1) * max(0.0, float(padding))
    pad_y = (y2 - y1) * max(0.0, float(padding))
    return [
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(source_width, int(round(x2 + pad_x))),
        min(source_height, int(round(y2 + pad_y))),
    ]


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def _semantic_localisation_box(
    image_bgr: np.ndarray,
    *,
    encoder: ReferenceEncoder,
    target_embeddings: np.ndarray,
    batch_size: int,
) -> tuple[list[int], float, int, bool]:
    """Localise a known SKU when point-prompt SAM follows the background."""

    height, width = image_bgr.shape[:2]
    window_width = max(1, int(round(width * 0.52)))
    window_height = max(1, int(round(height * 0.42)))
    x_positions = np.linspace(0, max(0, width - window_width), 5).round().astype(int)
    y_positions = np.linspace(0, max(0, height - window_height), 5).round().astype(int)
    boxes: list[list[int]] = []
    windows: list[np.ndarray] = []
    for y1 in y_positions:
        for x1 in x_positions:
            x2 = min(width, int(x1) + window_width)
            y2 = min(height, int(y1) + window_height)
            boxes.append([int(x1), int(y1), x2, y2])
            windows.append(cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))

    embeddings = encoder.encode_rgb(windows, batch_size=max(1, int(batch_size)))
    if len(embeddings) != len(windows):
        raise RuntimeError("DINO returned an unexpected localisation embedding count.")
    similarities = (
        _normalise_rows(embeddings) @ _normalise_rows(target_embeddings).T
    ).max(axis=1)
    best_similarity = float(similarities.max())
    ranked = np.argsort(similarities)[::-1]
    selected_indices = [
        int(index)
        for index in ranked
        if float(similarities[index]) >= best_similarity - 0.04
    ][:6]
    selected_boxes = [boxes[index] for index in selected_indices]
    x1 = min(box[0] for box in selected_boxes)
    y1 = min(box[1] for box in selected_boxes)
    x2 = max(box[2] for box in selected_boxes)
    y2 = max(box[3] for box in selected_boxes)

    # DINO is strongest on discriminative texture and may localise only part of
    # a loaf. Expand the connected high-similarity band so the saved reference
    # contains the complete product rather than a visually attractive fragment.
    pad_x = int(round((x2 - x1) * 0.25))
    pad_y = int(round((y2 - y1) * 0.25))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_y)

    minimum_width = int(round(width * 0.62))
    minimum_height = int(round(height * 0.68))
    centre_x = (x1 + x2) // 2
    centre_y = (y1 + y2) // 2
    if x2 - x1 < minimum_width:
        x1 = max(0, centre_x - minimum_width // 2)
        x2 = min(width, x1 + minimum_width)
        x1 = max(0, x2 - minimum_width)
    if y2 - y1 < minimum_height:
        y1 = max(0, centre_y - minimum_height // 2)
        y2 = min(height, y1 + minimum_height)
        y1 = max(0, y2 - minimum_height)
    refined_by_chroma = False
    localisation_roi = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(localisation_roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    chroma_floor = max(42, int(np.percentile(saturation, 52)))
    warm_product = (
        ((hue <= 48) | (hue >= 165))
        & (saturation >= chroma_floor)
        & (value <= 238)
    )
    warm_y, warm_x = np.where(warm_product)
    coverage = float(len(warm_x) / max(1, warm_product.size))
    if len(warm_x) and 0.03 <= coverage <= 0.70:
        rx1, rx2 = np.percentile(warm_x, [8, 92])
        ry1, ry2 = np.percentile(warm_y, [8, 92])
        span_x = max(1.0, float(rx2 - rx1))
        span_y = max(1.0, float(ry2 - ry1))
        refined_x1 = max(x1, int(round(x1 + rx1 - span_x * 0.18)))
        refined_y1 = max(y1, int(round(y1 + ry1 - span_y * 0.18)))
        refined_x2 = min(x2, int(round(x1 + rx2 + span_x * 0.18)))
        refined_y2 = min(y2, int(round(y1 + ry2 + span_y * 0.18)))
        if (
            refined_x2 - refined_x1 >= width * 0.42
            and refined_y2 - refined_y1 >= height * 0.46
        ):
            x1, y1, x2, y2 = (
                refined_x1,
                refined_y1,
                refined_x2,
                refined_y2,
            )
            refined_by_chroma = True
    return (
        [x1, y1, x2, y2],
        best_similarity,
        len(selected_indices),
        refined_by_chroma,
    )


def auto_crop_reference(
    image_bgr: np.ndarray,
    *,
    sam_model: Any,
    max_side: int,
    padding: float,
    device: str,
    encoder: ReferenceEncoder | None = None,
    target_embeddings: np.ndarray | None = None,
    batch_size: int = 16,
    grid_size: int = 7,
    candidate_limit: int = 12,
) -> ReferenceCropResult:
    """Crop one reference with SAM geometry and optional target-SKU DINO ranking."""

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Reference image is empty.")

    source_height, source_width = image_bgr.shape[:2]
    scale = min(1.0, max(640, int(max_side)) / float(max(source_height, source_width)))
    resized = cv2.resize(
        image_bgr,
        (
            max(1, int(round(source_width * scale))),
            max(1, int(round(source_height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )
    height, width = resized.shape[:2]
    points = build_reference_prompt_points(
        width,
        height,
        grid_size=grid_size,
    )
    result = sam_model.predict(
        source=resized,
        points=points,
        labels=np.ones((len(points), 1), dtype=np.int32),
        conf=0.15,
        device=device,
        save=False,
        verbose=False,
    )[0]
    if result.masks is None:
        raise RuntimeError("SAM returned no masks.")

    raw_masks = result.masks.data.detach().cpu().numpy() > 0.5
    colour_masks = build_colour_foreground_masks(resized)
    candidate_masks: dict[int, np.ndarray] = {}
    analysed: list[ReferenceCropCandidate | None] = []
    for mask in raw_masks:
        candidate = analyse_reference_mask(mask)
        analysed.append(candidate)
        if candidate is not None:
            candidate_masks[id(candidate)] = np.asarray(mask, dtype=bool)
    for mask in colour_masks:
        candidate = analyse_reference_mask(mask, proposal_source="colour")
        analysed.append(candidate)
        if candidate is not None:
            candidate_masks[id(candidate)] = np.asarray(mask, dtype=bool)
    all_candidates = [candidate for candidate in analysed if candidate is not None]
    plausible = deduplicate_reference_candidates(
        all_candidates,
        limit=candidate_limit,
    )
    if not plausible:
        raise RuntimeError(
            "SAM found no safe product crop after whole-frame search. "
            "Use a tighter photo or the already-cropped input mode."
        )

    boxes = [
        _padded_source_box(
            candidate,
            scale=scale,
            padding=padding,
            source_width=source_width,
            source_height=source_height,
        )
        for candidate in plausible
    ]
    crops = [image_bgr[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in boxes]
    valid_indices = [index for index, crop in enumerate(crops) if crop.size]
    if not valid_indices:
        raise RuntimeError("Auto-crop produced only empty images.")
    plausible = [plausible[index] for index in valid_indices]
    boxes = [boxes[index] for index in valid_indices]
    crops = [crops[index] for index in valid_indices]

    method = "sam_geometry"
    semantic_scores: np.ndarray | None = None
    target_embeddings_array = np.asarray(
        target_embeddings if target_embeddings is not None else [],
        dtype=np.float32,
    )
    if (
        encoder is not None
        and target_embeddings_array.ndim == 2
        and len(target_embeddings_array) > 0
    ):
        candidate_embeddings = encoder.encode_rgb(
            [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops],
            batch_size=max(1, int(batch_size)),
        )
        if len(candidate_embeddings) != len(crops):
            raise RuntimeError("DINO returned an unexpected auto-crop embedding count.")
        semantic_scores = (
            _normalise_rows(candidate_embeddings)
            @ _normalise_rows(target_embeddings_array).T
        ).max(axis=1)
        combined_scores = np.asarray(
            [candidate.geometry_score for candidate in plausible],
            dtype=np.float32,
        ) * 0.28 + semantic_scores * 0.72
        selected_index = int(np.argmax(combined_scores))
        method = "sam_geometry+dino_target"
    else:
        combined_scores = np.asarray(
            [candidate.geometry_score for candidate in plausible],
            dtype=np.float32,
        )
        selected_index = int(np.argmax(combined_scores))

    selected = plausible[selected_index]
    selected_crop = crops[selected_index]
    selected_box = boxes[selected_index]
    if selected_crop.size == 0:
        raise RuntimeError("Auto-crop selected an empty image.")

    metadata: dict[str, Any] = {
        "method": method,
        "sam_mask_count": int(len(raw_masks)),
        "colour_mask_count": int(len(colour_masks)),
        "plausible_candidate_count": int(len(plausible)),
        "proposal_source": selected.proposal_source,
        "geometry_score": float(selected.geometry_score),
        "combined_score": float(combined_scores[selected_index]),
        "box_ratio": float(selected.box_ratio),
        "fill_ratio": float(selected.fill_ratio),
        "border_touches": int(selected.border_touches),
    }
    if semantic_scores is not None:
        metadata["target_similarity"] = float(semantic_scores[selected_index])

    weak_sam_selection = (
        selected.border_touches > 0
        or selected.fill_ratio < 0.58
        or selected.geometry_score < 0.68
    )
    if weak_sam_selection and semantic_scores is not None and encoder is not None:
        (
            localisation_box,
            localisation_similarity,
            localisation_windows,
            localisation_refined,
        ) = (
            _semantic_localisation_box(
                image_bgr,
                encoder=encoder,
                target_embeddings=target_embeddings_array,
                batch_size=batch_size,
            )
        )
        lx1, ly1, lx2, ly2 = localisation_box
        localisation_crop = image_bgr[ly1:ly2, lx1:lx2].copy()
        if localisation_crop.size:
            metadata.update(
                {
                    "method": "dino_window_fallback",
                    "fallback_reason": "low_quality_sam_candidate",
                    "localisation_similarity": float(localisation_similarity),
                    "localisation_window_count": int(localisation_windows),
                    "localisation_refined_by_chroma": bool(localisation_refined),
                }
            )
            return ReferenceCropResult(
                crop_bgr=localisation_crop,
                box_xyxy=localisation_box,
                mask_ratio=float(selected.mask_ratio),
                metadata=metadata,
                # The fallback box is selected by DINO windows rather than the
                # chosen SAM proposal. Reusing that unrelated mask would erase
                # valid product pixels, so white-background export must ask for
                # manual review instead of silently producing a bad reference.
                foreground_mask=None,
            )

    if weak_sam_selection and semantic_scores is None:
        raise RuntimeError(
            "Auto-crop quality is too low for a new SKU. Add one tight reference "
            "first, then retry auto-crop with target-SKU DINO verification."
        )

    foreground_candidate = select_reference_foreground_candidate(
        selected,
        all_candidates,
    )
    metadata.update(
        {
            "foreground_proposal_source": foreground_candidate.proposal_source,
            "foreground_geometry_score": float(
                foreground_candidate.geometry_score
            ),
            "foreground_replaced_colour_mask": bool(
                foreground_candidate is not selected
            ),
        }
    )
    selected_mask = candidate_masks.get(id(foreground_candidate))
    foreground_mask = None
    if selected_mask is not None:
        source_mask = cv2.resize(
            selected_mask.astype(np.uint8),
            (source_width, source_height),
            interpolation=cv2.INTER_NEAREST,
        )
        x1, y1, x2, y2 = selected_box
        cropped_mask = source_mask[y1:y2, x1:x2]
        if cropped_mask.shape == selected_crop.shape[:2] and cropped_mask.any():
            foreground_mask = cropped_mask.astype(bool)
            initial_crop_coverage = float(foreground_mask.mean())
            # The selected SAM proposal can contain a connected steel tray or
            # cast-shadow wedge. Refining after the padded source crop keeps
            # the mask perfectly aligned with the preview/export image and
            # gives GrabCut real background pixels around the product.
            if (
                foreground_candidate.proposal_source == "colour"
                and 0.01 <= initial_crop_coverage <= 0.95
            ):
                try:
                    foreground_mask, refinement_metadata = (
                        refine_reference_foreground_mask(
                            selected_crop,
                            foreground_mask,
                        )
                    )
                    metadata.update(refinement_metadata)
                    metadata["auto_crop_mask_refined"] = True
                except (ValueError, RuntimeError) as exc:
                    # Auto-cropping to the original background remains useful,
                    # but an uncertain mask must never generate a misleading
                    # white-background reference silently.
                    metadata.update(
                        {
                            "auto_crop_mask_refined": False,
                            "mask_refinement_error": str(exc),
                        }
                    )
                    foreground_mask = None
            elif foreground_candidate.proposal_source == "colour":
                metadata.update(
                    {
                        "auto_crop_mask_refined": False,
                        "mask_refinement_skipped": (
                            "crop mask is already effectively full-frame"
                        ),
                    }
                )
            else:
                metadata.update(
                    {
                        "refinement": "aligned_sam_silhouette",
                        "initial_foreground_coverage": initial_crop_coverage,
                        "foreground_coverage": initial_crop_coverage,
                        "retained_mask_ratio": 1.0,
                        "removed_mask_ratio": 0.0,
                        "auto_crop_mask_refined": False,
                    }
                )
            if foreground_mask is not None:
                metadata["foreground_coverage"] = float(foreground_mask.mean())

    return ReferenceCropResult(
        crop_bgr=selected_crop,
        box_xyxy=selected_box,
        mask_ratio=float(selected.mask_ratio),
        metadata=metadata,
        foreground_mask=foreground_mask,
    )
