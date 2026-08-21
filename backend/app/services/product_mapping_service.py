"""Validated visual-class to business-SKU mapping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ProductMappingError(RuntimeError):
    pass


class ProductMappingService:
    """Resolve visual classes and expose the complete operator SKU catalog."""

    VALID_TYPES = {"direct", "family"}

    def __init__(
        self,
        mapping_path: Path | str,
        confidence_overrides: dict[str, float] | None = None,
    ):
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        overrides = {
            str(raw_class): float(value)
            for raw_class, value in (confidence_overrides or {}).items()
        }
        payload = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        classes = payload.get("classes")
        if not isinstance(classes, dict) or not classes:
            raise ProductMappingError("Product mapping is empty or invalid.")

        self.schema_version = int(payload.get("schema_version") or 1)
        self.business_sku_count = int(payload.get("business_sku_count") or 0)
        self.supported_product_count = int(
            payload.get("supported_product_count") or self.business_sku_count
        )
        self.visual_class_count = int(payload.get("visual_class_count") or len(classes))
        self._classes: dict[str, dict[str, Any]] = {}
        self._products_by_code: dict[str, dict[str, Any]] = {}

        for raw_class, raw_item in classes.items():
            if not isinstance(raw_item, dict):
                raise ProductMappingError(f"Mapping for {raw_class!r} must be an object.")
            mapping_type = str(raw_item.get("type") or "direct").strip().lower()
            if mapping_type not in self.VALID_TYPES:
                raise ProductMappingError(
                    f"Invalid mapping type for {raw_class!r}: {mapping_type!r}."
                )
            threshold = float(
                overrides.get(
                    str(raw_class), raw_item.get("confidence_threshold", 0.25)
                )
            )
            if not 0.0 < threshold <= 1.0:
                raise ProductMappingError(
                    f"Invalid confidence threshold for {raw_class!r}: {threshold}."
                )
            display_name = str(
                raw_item.get("display_name") or raw_item.get("product_name") or raw_class
            ).strip()
            item: dict[str, Any] = {
                "raw_class": str(raw_class),
                "type": mapping_type,
                "display_name": display_name,
                "confidence_threshold": threshold,
                "purchase_price": float(raw_item.get("purchase_price", 0)),
            }

            if mapping_type == "direct":
                code = str(raw_item.get("product_code") or "").strip()
                name = str(raw_item.get("product_name") or display_name).strip()
                if not code or not name:
                    raise ProductMappingError(
                        f"Direct class {raw_class!r} must have product_code/product_name."
                    )
                item.update({"product_code": code, "product_name": name, "members": []})
                self._register_business_product(
                    code=code,
                    name=name,
                    display_name=display_name,
                    raw_class=str(raw_class),
                    source_type="direct",
                    family_name=None,
                )
            else:
                raw_members = raw_item.get("members")
                if not isinstance(raw_members, list) or not raw_members:
                    raise ProductMappingError(
                        f"Family class {raw_class!r} must contain members."
                    )
                members: list[dict[str, str]] = []
                seen: set[str] = set()
                for raw_member in raw_members:
                    if not isinstance(raw_member, dict):
                        raise ProductMappingError(f"Invalid member in {raw_class!r}.")
                    code = str(raw_member.get("product_code") or "").strip()
                    name = str(
                        raw_member.get("product_name")
                        or raw_member.get("display_name")
                        or ""
                    ).strip()
                    member_display = str(raw_member.get("display_name") or name).strip()
                    if not code or not name:
                        raise ProductMappingError(
                            f"Family {raw_class!r} contains a member without code/name."
                        )
                    key = code.casefold()
                    if key in seen:
                        raise ProductMappingError(
                            f"Duplicate product code {code!r} in family {raw_class!r}."
                        )
                    seen.add(key)
                    member = {
                        "product_code": code,
                        "product_name": name,
                        "display_name": member_display,
                    }
                    members.append(member)
                    self._register_business_product(
                        code=code,
                        name=name,
                        display_name=member_display,
                        raw_class=str(raw_class),
                        source_type="family_member",
                        family_name=display_name,
                    )
                item["members"] = members

            self._classes[str(raw_class)] = item

        unknown_overrides = sorted(set(overrides) - set(self._classes))
        if unknown_overrides:
            raise ProductMappingError(
                "Threshold overrides contain unknown classes: "
                + ", ".join(unknown_overrides)
            )

        visual_product_count = len(self._products_by_code)
        if self.visual_class_count and self.visual_class_count != len(self._classes):
            raise ProductMappingError(
                f"visual_class_count={self.visual_class_count} but mapping has "
                f"{len(self._classes)} classes."
            )
        if self.business_sku_count and self.business_sku_count != visual_product_count:
            raise ProductMappingError(
                f"business_sku_count={self.business_sku_count} but mapping resolves "
                f"{visual_product_count} visual-model business SKUs."
            )

        catalog_products = payload.get("catalog_products") or []
        if not isinstance(catalog_products, list):
            raise ProductMappingError("catalog_products must be a list.")
        for raw_product in catalog_products:
            if not isinstance(raw_product, dict):
                raise ProductMappingError("Every catalog product must be an object.")
            code = str(raw_product.get("product_code") or "").strip()
            name = str(raw_product.get("product_name") or "").strip()
            display_name = str(raw_product.get("display_name") or name).strip()
            visual_class = str(raw_product.get("visual_class") or code).strip()
            source_type = str(
                raw_product.get("source_type") or "foundation_reference"
            ).strip()
            family_name = str(raw_product.get("family_name") or "").strip() or None
            if not code or not name:
                raise ProductMappingError(
                    "Catalog products must contain product_code/product_name."
                )
            self._register_business_product(
                code=code,
                name=name,
                display_name=display_name,
                raw_class=visual_class,
                source_type=source_type,
                family_name=family_name,
            )

        if (
            self.supported_product_count
            and self.supported_product_count != len(self._products_by_code)
        ):
            raise ProductMappingError(
                f"supported_product_count={self.supported_product_count} but the "
                f"operator catalog contains {len(self._products_by_code)} products."
            )

    def _register_business_product(
        self,
        *,
        code: str,
        name: str,
        display_name: str,
        raw_class: str,
        source_type: str,
        family_name: str | None,
    ) -> None:
        key = code.casefold()
        product = {
            "product_code": code,
            "product_name": name,
            "display_name": display_name,
            "visual_class": raw_class,
            "source_type": source_type,
            "family_name": family_name,
        }
        existing = self._products_by_code.get(key)
        if existing is not None and existing != product:
            raise ProductMappingError(f"Conflicting business product code: {code!r}.")
        self._products_by_code[key] = product

    def resolve(self, raw_class: str) -> dict[str, Any]:
        item = self._classes.get(str(raw_class))
        if item is None:
            raise ProductMappingError(f"Unmapped model class: {raw_class}")
        resolved = dict(item)
        resolved["members"] = [dict(member) for member in item.get("members", [])]
        return resolved

    def resolve_family_member(self, raw_class: str, product_code: str) -> dict[str, str]:
        mapping = self.resolve(raw_class)
        if mapping["type"] != "family":
            raise ProductMappingError(f"Class {raw_class!r} is not a family.")
        target = str(product_code).strip().casefold()
        for member in mapping["members"]:
            if str(member["product_code"]).casefold() == target:
                return dict(member)
        raise ProductMappingError(
            f"Product code {product_code!r} is not a member of {raw_class!r}."
        )

    def resolve_product_code(self, product_code: str) -> dict[str, Any]:
        target = str(product_code).strip().casefold()
        item = self._products_by_code.get(target)
        if item is None:
            raise ProductMappingError(
                f"Product code {product_code!r} is not in the trained product catalog."
            )
        return dict(item)

    def all_business_products(self) -> list[dict[str, Any]]:
        """Return the complete searchable operator correction catalog."""
        return sorted(
            (dict(item) for item in self._products_by_code.values()),
            key=lambda item: (
                str(item.get("display_name") or item["product_name"]).casefold(),
                item["product_code"].casefold(),
            ),
        )

    def minimum_threshold(self) -> float:
        return min(float(item["confidence_threshold"]) for item in self._classes.values())

    def class_names(self) -> list[str]:
        return list(self._classes)

    def class_settings(self) -> list[dict[str, Any]]:
        return [self.resolve(raw_class) for raw_class in self.class_names()]
