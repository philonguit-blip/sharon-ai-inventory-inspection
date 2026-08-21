"""Safe router combining production YOLO with the Foundation fallback."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.config import (
    HYBRID_BOX_RESCUE_ENABLED,
    HYBRID_BOX_RESCUE_EXISTING_COVERAGE,
    HYBRID_BOX_RESCUE_EXPANSION_RATIO,
    HYBRID_BOX_RESCUE_MAX_CANDIDATES,
    HYBRID_BOX_RESCUE_MIN_CONFIDENCE,
    HYBRID_BOX_RESCUE_MIN_THRESHOLD_RATIO,
    HYBRID_DEFAULT_MODE,
    HYBRID_ENABLED,
    HYBRID_YOLO_FALLBACK_MARGIN,
)
from app.services.hybrid_errors import (
    FoundationInferenceError,
    FoundationNotReadyError,
)

if TYPE_CHECKING:
    from app.services.foundation_inference_service import FoundationInferenceService


VALID_MODES = {"AUTO", "YOLO", "FOUNDATION", "COMPARE"}


class HybridInferenceService:
    def __init__(
        self,
        yolo_service: Any,
        foundation_service: "FoundationInferenceService | None" = None,
        *,
        enabled: bool = HYBRID_ENABLED,
        default_mode: str = HYBRID_DEFAULT_MODE,
        fallback_margin: float = HYBRID_YOLO_FALLBACK_MARGIN,
        box_rescue_enabled: bool = HYBRID_BOX_RESCUE_ENABLED,
        box_rescue_min_threshold_ratio: float = (
            HYBRID_BOX_RESCUE_MIN_THRESHOLD_RATIO
        ),
        box_rescue_min_confidence: float = HYBRID_BOX_RESCUE_MIN_CONFIDENCE,
        box_rescue_max_candidates: int = HYBRID_BOX_RESCUE_MAX_CANDIDATES,
        box_rescue_expansion_ratio: float = HYBRID_BOX_RESCUE_EXPANSION_RATIO,
        box_rescue_existing_coverage: float = HYBRID_BOX_RESCUE_EXISTING_COVERAGE,
    ) -> None:
        self.yolo = yolo_service
        if foundation_service is None:
            from app.services.foundation_inference_service import FoundationInferenceService

            foundation_service = FoundationInferenceService()
        self.foundation = foundation_service
        self.enabled = bool(enabled)
        self.default_mode = self.normalise_mode(default_mode)
        self.fallback_margin = float(fallback_margin)
        self.box_rescue_enabled = bool(box_rescue_enabled)
        self.box_rescue_min_threshold_ratio = max(
            0.05, min(1.0, float(box_rescue_min_threshold_ratio))
        )
        self.box_rescue_min_confidence = max(
            0.0, min(1.0, float(box_rescue_min_confidence))
        )
        self.box_rescue_max_candidates = max(1, int(box_rescue_max_candidates))
        self.box_rescue_expansion_ratio = max(
            0.0, min(0.50, float(box_rescue_expansion_ratio))
        )
        self.box_rescue_existing_coverage = max(
            0.0, min(1.0, float(box_rescue_existing_coverage))
        )

    @staticmethod
    def normalise_mode(mode: str | None) -> str:
        value = str(mode or "AUTO").strip().upper()
        if value not in VALID_MODES:
            raise ValueError(f"Unsupported inference mode: {value}")
        return value

    def health(self) -> dict[str, Any]:
        yolo_health = self.yolo.health()
        foundation_health = self.foundation.health()
        return {
            "ready": bool(yolo_health.get("ready")),
            "type": "HYBRID",
            "enabled": self.enabled,
            "default_mode": self.default_mode,
            "available_modes": sorted(VALID_MODES),
            "box_rescue": {
                "enabled": self.box_rescue_enabled,
                "min_threshold_ratio": self.box_rescue_min_threshold_ratio,
                "min_confidence": self.box_rescue_min_confidence,
                "max_candidates": self.box_rescue_max_candidates,
                "expansion_ratio": self.box_rescue_expansion_ratio,
                "existing_box_coverage": self.box_rescue_existing_coverage,
            },
            "yolo": yolo_health,
            "foundation": foundation_health,
        }

    @staticmethod
    def _decision(result: dict[str, Any]) -> dict[str, Any]:
        value = result.get("decision")
        return value if isinstance(value, dict) else {}

    def _yolo_is_uncertain(self, result: dict[str, Any]) -> bool:
        decision = self._decision(result)
        if str(decision.get("decision") or "").upper() not in {"DIRECT", "FAMILY"}:
            return True
        average = float(decision.get("avg_confidence") or 0.0)
        threshold = float(decision.get("confidence_threshold") or 0.0)
        return average < threshold + self.fallback_margin

    @staticmethod
    def _identity(decision: dict[str, Any]) -> str:
        if decision.get("product_code"):
            return str(decision["product_code"]).casefold()
        return str(decision.get("dominant_class") or "").casefold()

    @staticmethod
    def _decision_choices(
        decision: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Return every business SKU represented by a decision, with evidence."""
        decision_type = str(decision.get("decision") or "").upper()
        raw_choices: list[dict[str, Any]] = []
        if decision_type == "DIRECT" and decision.get("product_code"):
            raw_choices = [decision]
        elif decision_type == "FAMILY":
            raw_choices = [item for item in (decision.get("members") or []) if isinstance(item, dict)]
        elif decision_type == "REVIEW":
            raw_choices = [item for item in (decision.get("candidates") or []) if isinstance(item, dict)]
        choices: dict[str, dict[str, Any]] = {}
        for item in raw_choices:
            code = str(item.get("product_code") or "").strip()
            if not code:
                continue
            candidate = {
                "product_code": code,
                "product_name": str(item.get("product_name") or item.get("display_name") or code),
                "display_name": str(item.get("display_name") or item.get("product_name") or code),
                "visual_class": item.get("visual_class") or decision.get("dominant_class"),
                "detected_quantity": int(item.get("detected_quantity") or decision.get("dominant_count") or 0),
                "avg_confidence": float(item.get("avg_confidence") or decision.get("avg_confidence") or 0.0),
            }
            existing = choices.get(code.casefold())
            if existing is None or (candidate["detected_quantity"], candidate["avg_confidence"]) > (int(existing.get("detected_quantity") or 0), float(existing.get("avg_confidence") or 0.0)):
                choices[code.casefold()] = candidate
        return choices

    @classmethod
    def _decision_is_confirmable(cls, decision: dict[str, Any]) -> bool:
        return bool(
            str(decision.get("decision") or "").upper()
            in {"DIRECT", "FAMILY", "REVIEW"}
            and int(decision.get("count") or 0) > 0
            and cls._decision_choices(decision)
        )

    @staticmethod
    def _tag(
        result: dict[str, Any],
        *,
        requested: str,
        selected: str,
        reason: str,
        compared: bool = False,
    ) -> dict[str, Any]:
        result["engine"] = selected
        result["hybrid"] = {
            "requested_mode": requested,
            "selected_engine": selected,
            "reason": reason,
            "compared": compared,
        }
        return result

    @staticmethod
    def _carry_yolo_rescue_pool(
        selected: dict[str, Any],
        yolo: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep YOLO low-confidence candidates when Foundation becomes output."""
        if selected is yolo:
            return selected
        candidates = yolo.get("candidate_objects")
        if isinstance(candidates, list):
            selected["yolo_candidate_objects"] = candidates
        box_rescue = yolo.get("foundation_box_rescue")
        if isinstance(box_rescue, dict):
            selected["foundation_box_rescue"] = box_rescue
        return selected

    @staticmethod
    def _box_coverage(first: list[float], second: list[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(value) for value in first]
        bx1, by1, bx2, by2 = [float(value) for value in second]
        intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0.0, min(ay2, by2) - max(ay1, by1)
        )
        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        smaller = min(first_area, second_area)
        return intersection / smaller if smaller > 0.0 else 0.0

    def _box_rescue_candidates(
        self,
        result: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Select unique, sub-threshold YOLO proposals for Foundation."""
        raw_candidates = result.get("candidate_objects")
        existing = [
            item
            for item in (result.get("objects") or [])
            if isinstance(item, dict) and len(item.get("box_xyxy") or []) == 4
        ]
        stats = {
            "available": 0,
            "below_rescue_floor": 0,
            "overlaps_existing": 0,
            "truncated": 0,
        }
        if not self.box_rescue_enabled or not isinstance(raw_candidates, list):
            return [], stats

        eligible: list[dict[str, Any]] = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                continue
            box = raw_candidate.get("box_xyxy") or []
            if len(box) != 4:
                continue
            confidence = float(raw_candidate.get("confidence") or 0.0)
            threshold = float(raw_candidate.get("confidence_threshold") or 0.0)
            if threshold <= 0.0 or confidence >= threshold:
                continue
            stats["available"] += 1
            rescue_floor = max(
                self.box_rescue_min_confidence,
                threshold * self.box_rescue_min_threshold_ratio,
            )
            if confidence < rescue_floor:
                stats["below_rescue_floor"] += 1
                continue
            if any(
                self._box_coverage(list(box), list(item["box_xyxy"]))
                >= self.box_rescue_existing_coverage
                for item in existing
            ):
                stats["overlaps_existing"] += 1
                continue
            eligible.append(dict(raw_candidate))

        eligible.sort(
            key=lambda item: float(item.get("confidence") or 0.0),
            reverse=True,
        )
        if len(eligible) > self.box_rescue_max_candidates:
            stats["truncated"] = len(eligible) - self.box_rescue_max_candidates
            eligible = eligible[: self.box_rescue_max_candidates]
        return eligible, stats

    def _attempt_foundation_box_rescue(
        self,
        yolo: dict[str, Any],
        *,
        raw_bytes: bytes,
        annotated_path: Path | str | None,
    ) -> tuple[dict[str, Any], bool, bool]:
        candidates, selection = self._box_rescue_candidates(yolo)
        if not candidates:
            return yolo, False, False
        verify = getattr(self.foundation, "verify_yolo_candidates", None)
        apply_rescue = getattr(self.yolo, "apply_foundation_box_rescue", None)
        if not callable(verify) or not callable(apply_rescue):
            return yolo, False, False
        verification = verify(
            raw_bytes,
            candidates,
            expansion_ratio=self.box_rescue_expansion_ratio,
        )
        updated = apply_rescue(
            yolo,
            verification,
            raw_bytes=raw_bytes,
            annotated_path=annotated_path,
        )
        metadata = updated.setdefault("foundation_box_rescue", {})
        metadata["selection"] = selection
        disagreement = int(verification.get("disagreements") or 0) > 0
        rescued = int(metadata.get("kept_after_deduplication") or 0) > 0
        return updated, disagreement, rescued

    @staticmethod
    def _review_preference(
        result: dict[str, Any], decision: dict[str, Any], source: str
    ) -> tuple[int, int, int, float, int]:
        """Rank safe disagreement candidates for the review count/preview.

        DIRECT is more specific than FAMILY. A safe candidate with an explicit
        product code is preferred, followed by the amount of retained object
        evidence. The job remains REVIEW; this only chooses the most useful
        pre-filled count and matching annotation for the operator.
        """
        decision_type = str(decision.get("decision") or "").upper()
        return (
            max(0, int(decision.get("physical_count") or decision.get("count") or 0)),
            2 if decision_type == "DIRECT" else 1 if decision_type == "FAMILY" else 0,
            1 if decision.get("product_code") else 0,
            float(decision.get("avg_confidence") or 0.0),
            1 if source == "YOLO" else 0,
        )

    @staticmethod
    def _foundation_annotation_path(
        annotated_path: Path | str | None,
    ) -> Path | None:
        if annotated_path is None:
            return None
        canonical = Path(annotated_path).expanduser().resolve()
        return canonical.with_name(
            f"{canonical.stem}.foundation-candidate{canonical.suffix or '.jpg'}"
        )

    @staticmethod
    def _finalize_compared_annotation(
        result: dict[str, Any],
        *,
        annotated_path: Path | str | None,
        foundation_annotated_path: Path | None,
    ) -> dict[str, Any]:
        """Make the visible annotation match the result used for count."""
        if annotated_path is None or foundation_annotated_path is None:
            return result
        canonical = Path(annotated_path).expanduser().resolve()
        decision = HybridInferenceService._decision(result)
        selected_engine = str((result.get("hybrid") or {}).get("selected_engine") or "")
        use_foundation = bool(
            selected_engine == "FOUNDATION"
            or str(decision.get("preferred_source") or "").upper() == "FOUNDATION"
        )
        if use_foundation and foundation_annotated_path.is_file():
            canonical.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(foundation_annotated_path, canonical)
            result["annotated_path"] = str(canonical)
        elif canonical.is_file():
            result["annotated_path"] = str(canonical)
        try:
            foundation_annotated_path.unlink(missing_ok=True)
        except OSError:
            pass
        return result

    def rescue_result_for_class(
        self,
        payload: dict[str, Any],
        raw_class: str,
        *,
        raw_bytes: bytes | None = None,
        annotated_path: Path | str | None = None,
    ) -> dict[str, Any] | None:
        """Proxy same-batch rescue to the YOLO service without a new forward pass."""
        working = dict(payload)
        if not isinstance(working.get("candidate_objects"), list):
            yolo_candidates = working.get("yolo_candidate_objects")
            if isinstance(yolo_candidates, list):
                working["candidate_objects"] = yolo_candidates

        rescue = getattr(self.yolo, "rescue_result_for_class", None)
        if not callable(rescue):
            return None

        rescued = rescue(
            working,
            raw_class,
            raw_bytes=raw_bytes,
            annotated_path=annotated_path,
        )
        if rescued is None:
            return None

        requested = str(
            (payload.get("hybrid") or {}).get("requested_mode")
            or self.default_mode
        ).upper()
        rescued["hybrid_candidates"] = payload.get("hybrid_candidates")
        return self._tag(
            rescued,
            requested=requested,
            selected="YOLO_BATCH_RESCUE",
            reason=(
                "A failed image was recovered from low-confidence YOLO candidates "
                "after the other images in the same job agreed on one visual class."
            ),
            compared=bool((payload.get("hybrid") or {}).get("compared")),
        )

    def _compare(
        self,
        yolo: dict[str, Any],
        foundation: dict[str, Any],
        requested: str,
    ) -> dict[str, Any]:
        yd = self._decision(yolo)
        fd = self._decision(foundation)
        y_ok = str(yd.get("decision") or "").upper() in {"DIRECT", "FAMILY"}
        f_ok = str(fd.get("decision") or "").upper() in {"DIRECT", "FAMILY"}
        y_confirmable = self._decision_is_confirmable(yd)
        f_confirmable = self._decision_is_confirmable(fd)
        same_identity = self._identity(yd) and self._identity(yd) == self._identity(fd)
        same_count = int(yd.get("count") or 0) == int(fd.get("count") or 0)
        if y_ok and f_ok and same_identity and same_count:
            yolo["decision"]["message"] = (
                "YOLO and Foundation independently agree. Confirm the product and count."
            )
            yolo["hybrid_candidates"] = {"yolo": yd, "foundation": fd}
            return self._tag(
                yolo,
                requested=requested,
                selected="CONSENSUS",
                reason="Both engines agree on SKU/family and count.",
                compared=True,
            )
        if y_ok and not f_ok:
            return self._tag(
                yolo,
                requested=requested,
                selected="YOLO",
                reason="Foundation did not produce a safe decision.",
                compared=True,
            )
        if f_ok and not y_ok:
            foundation = self._carry_yolo_rescue_pool(foundation, yolo)
            return self._tag(
                foundation,
                requested=requested,
                selected="FOUNDATION",
                reason="YOLO was uncertain and Foundation produced a safe decision.",
                compared=True,
            )
        if not y_ok and not f_ok:
            if y_confirmable != f_confirmable:
                selected_source, selected, selected_decision = (
                    ("YOLO", yolo, yd)
                    if y_confirmable
                    else ("FOUNDATION", foundation, fd)
                )
            elif y_confirmable and f_confirmable:
                selected_source, selected, selected_decision = max(
                    (("YOLO", yolo, yd), ("FOUNDATION", foundation, fd)),
                    key=lambda item: self._review_preference(
                        item[1], item[2], item[0]
                    ),
                )
            else:
                selected_source, selected, selected_decision = (
                    ("FOUNDATION", foundation, fd)
                    if int(foundation.get("total_detections") or 0)
                    > int(yolo.get("total_detections") or 0)
                    else ("YOLO", yolo, yd)
                )
            selected = self._carry_yolo_rescue_pool(selected, yolo)
            selected["hybrid_candidates"] = {"yolo": yd, "foundation": fd}
            if self._decision_is_confirmable(selected_decision):
                return self._tag(
                    selected,
                    requested=requested,
                    selected=selected_source,
                    reason=(
                        f"{selected_source.title()} produced a countable review "
                        "with closed-catalog choices; operator confirmation is "
                        "required."
                    ),
                    compared=True,
                )
            return self._tag(
                selected,
                requested=requested,
                selected=selected_source,
                reason="Neither engine produced a safe decision; retake/manual offline review is required.",
                compared=True,
            )

        # A disagreement is never auto-written. Use the strongest safe engine
        # only to pre-fill count/preview, then expose candidates for confirmation.
        preferred_source, selected, preferred_decision = max(
            (("YOLO", yolo, yd), ("FOUNDATION", foundation, fd)),
            key=lambda item: self._review_preference(item[1], item[2], item[0]),
        )
        selected = self._carry_yolo_rescue_pool(selected, yolo)
        selected["hybrid_candidates"] = {"yolo": yd, "foundation": fd}
        candidate_by_code: dict[str, dict[str, Any]] = {}
        for source, decision in (("YOLO", yd), ("FOUNDATION", fd)):
            for code_key, choice in self._decision_choices(decision).items():
                candidate = candidate_by_code.setdefault(
                    code_key,
                    {
                        **choice,
                        "sources": [],
                        "count": int(decision.get("count") or 0),
                    },
                )
                candidate["sources"].append(source)
                candidate["count"] = max(
                    int(candidate.get("count") or 0),
                    int(decision.get("count") or 0),
                )
        candidate_codes = []
        for candidate in candidate_by_code.values():
            sources = list(dict.fromkeys(candidate.pop("sources", [])))
            candidate_codes.append(
                {**candidate, "source": "+".join(sources)}
            )
        selected["decision"] = {
            "decision": "REVIEW",
            "display_name": "Hybrid disagreement",
            "dominant_class": preferred_decision.get("dominant_class"),
            "count": int(preferred_decision.get("physical_count") or preferred_decision.get("count") or 0),
            "physical_count": int(preferred_decision.get("physical_count") or preferred_decision.get("count") or 0),
            "dominant_count": int(preferred_decision.get("dominant_count") or 0),
            "total_detections": int(preferred_decision.get("total_detections") or preferred_decision.get("count") or 0),
            "purity": 0.0,
            "classification_purity": 0.0,
            "avg_confidence": float(preferred_decision.get("avg_confidence") or 0.0),
            "candidates": candidate_codes,
            "preferred_source": preferred_source,
            "preferred_product_code": preferred_decision.get("product_code"),
            "requires_user_selection": True,
            "requires_confirmation": True,
            "message": "YOLO and Foundation disagree. Select the correct SKU and verify quantity before creating a draft receipt.",
        }
        return self._tag(
            selected,
            requested=requested,
            selected="REVIEW",
            reason="The engines disagree; operator review is mandatory.",
            compared=True,
        )

    def _route_auto_result(
        self,
        yolo: dict[str, Any],
        *,
        raw_bytes: bytes,
        image_name: str,
        annotated_path: Path | str | None,
    ) -> dict[str, Any]:
        """Run box-level rescue first, then full-image Foundation if required."""
        initially_uncertain = self._yolo_is_uncertain(yolo)
        rescue_candidates, _ = self._box_rescue_candidates(yolo)
        if not initially_uncertain and not rescue_candidates:
            return self._tag(
                yolo,
                requested="AUTO",
                selected="YOLO",
                reason="YOLO decision cleared the production safety gates.",
            )

        if not self.foundation.is_ready():
            return self._tag(
                yolo,
                requested="AUTO",
                selected="YOLO",
                reason=(
                    "Foundation assets are not ready; AUTO safely retained the "
                    "YOLO result."
                ),
            )

        disagreement = False
        rescued = False
        if rescue_candidates:
            try:
                yolo, disagreement, rescued = self._attempt_foundation_box_rescue(
                    yolo,
                    raw_bytes=raw_bytes,
                    annotated_path=annotated_path,
                )
            except FoundationInferenceError as exc:
                yolo["foundation_box_rescue"] = {
                    "attempted": len(rescue_candidates),
                    "accepted": 0,
                    "error": str(exc),
                }
                return self._tag(
                    yolo,
                    requested="AUTO",
                    selected="YOLO",
                    reason=(
                        "Foundation box verification failed; AUTO retained the "
                        "YOLO result without promoting any candidate."
                    ),
                )

        if not disagreement and not self._yolo_is_uncertain(yolo):
            if rescued:
                return self._tag(
                    yolo,
                    requested="AUTO",
                    selected="FOUNDATION_BOX_RESCUE",
                    reason=(
                        "Foundation independently verified same-class, "
                        "sub-threshold YOLO boxes."
                    ),
                )
            return self._tag(
                yolo,
                requested="AUTO",
                selected="YOLO",
                reason=(
                    "YOLO remained safe after Foundation rejected or skipped "
                    "the sub-threshold proposals."
                ),
            )

        foundation_annotated_path = self._foundation_annotation_path(annotated_path)
        try:
            foundation = self.foundation.infer_bytes(
                raw_bytes,
                image_name=image_name,
                annotated_path=foundation_annotated_path,
            )
        except FoundationInferenceError as exc:
            yolo.setdefault("foundation_full_fallback", {})["error"] = str(exc)
            return self._tag(
                yolo,
                requested="AUTO",
                selected="YOLO",
                reason=(
                    "Full-image Foundation fallback failed; AUTO safely retained "
                    "the YOLO result."
                ),
            )
        compared = self._compare(yolo, foundation, "AUTO")
        return self._finalize_compared_annotation(
            compared,
            annotated_path=annotated_path,
            foundation_annotated_path=foundation_annotated_path,
        )

    def infer_bytes(
        self,
        raw_bytes: bytes,
        *,
        image_name: str,
        annotated_path: Path | str | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        requested = self.normalise_mode(mode or self.default_mode)
        if not self.enabled:
            requested = "YOLO"

        if requested == "YOLO":
            result = self.yolo.infer_bytes(
                raw_bytes, image_name=image_name, annotated_path=annotated_path
            )
            return self._tag(result, requested="YOLO", selected="YOLO", reason="Explicit fast-path request.")

        if requested == "FOUNDATION":
            result = self.foundation.infer_bytes(
                raw_bytes, image_name=image_name, annotated_path=annotated_path
            )
            return self._tag(result, requested=requested, selected="FOUNDATION", reason="Explicit Foundation request.")

        yolo = self.yolo.infer_bytes(
            raw_bytes, image_name=image_name, annotated_path=annotated_path
        )
        if requested == "AUTO":
            return self._route_auto_result(
                yolo,
                raw_bytes=raw_bytes,
                image_name=image_name,
                annotated_path=annotated_path,
            )

        if not self.foundation.is_ready():
            raise FoundationNotReadyError(
                "COMPARE requires SAM2, DINOv2 and reference embeddings. "
                "See HYBRID_OPERATIONS_GUIDE.md."
            )

        foundation_annotated_path = self._foundation_annotation_path(annotated_path)
        foundation = self.foundation.infer_bytes(
            raw_bytes,
            image_name=image_name,
            annotated_path=foundation_annotated_path,
        )
        compared = self._compare(yolo, foundation, requested)
        return self._finalize_compared_annotation(
            compared,
            annotated_path=annotated_path,
            foundation_annotated_path=foundation_annotated_path,
        )

    def infer_batch(
        self,
        items: list[dict[str, Any]],
        *,
        mode: str | None = None,
    ) -> list[dict[str, Any]]:
        """Batch the YOLO fast path and fall back per image only when needed."""
        if not items:
            return []
        requested = self.normalise_mode(mode or self.default_mode)
        if not self.enabled:
            requested = "YOLO"
        if requested == "FOUNDATION":
            return [
                self.infer_bytes(
                    bytes(item["raw_bytes"]),
                    image_name=str(item["image_name"]),
                    annotated_path=item.get("annotated_path"),
                    mode="FOUNDATION",
                )
                for item in items
            ]

        yolo_results = self.yolo.infer_batch(items)
        output: list[dict[str, Any]] = []
        for item, yolo in zip(items, yolo_results, strict=True):
            if requested == "YOLO":
                output.append(
                    self._tag(
                        yolo,
                        requested="YOLO",
                        selected="YOLO",
                        reason="Explicit batched fast-path request.",
                    )
                )
                continue
            if requested == "AUTO":
                output.append(
                    self._route_auto_result(
                        yolo,
                        raw_bytes=bytes(item["raw_bytes"]),
                        image_name=str(item["image_name"]),
                        annotated_path=item.get("annotated_path"),
                    )
                )
                continue
            if not self.foundation.is_ready():
                raise FoundationNotReadyError(
                    "COMPARE requires SAM2, DINOv2 and reference embeddings. "
                    "See HYBRID_OPERATIONS_GUIDE.md."
                )
            foundation_annotated_path = self._foundation_annotation_path(
                item.get("annotated_path")
            )
            foundation = self.foundation.infer_bytes(
                bytes(item["raw_bytes"]),
                image_name=str(item["image_name"]),
                annotated_path=foundation_annotated_path,
            )
            compared = self._compare(yolo, foundation, requested)
            output.append(
                self._finalize_compared_annotation(
                    compared,
                    annotated_path=item.get("annotated_path"),
                    foundation_annotated_path=foundation_annotated_path,
                )
            )
        return output
