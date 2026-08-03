import json
from pathlib import Path
from typing import Any


class ProductMappingError(RuntimeError):
    pass


class ProductMappingService:
    def __init__(self, mapping_path: Path):
        payload = json.loads(
            mapping_path.read_text(encoding="utf-8")
        )

        classes = payload.get("classes")

        if not isinstance(classes, dict) or not classes:
            raise ProductMappingError(
                "Product mapping is empty or invalid."
            )

        self._classes = classes

    def resolve(self, raw_class: str) -> dict[str, Any]:
        product = self._classes.get(raw_class)

        if product is None:
            raise ProductMappingError(
                f"Unmapped model class: {raw_class}"
            )

        return {
            "raw_class": raw_class,
            "product_code": str(product["product_code"]),
            "product_name": str(product["product_name"]),
            "confidence_threshold": float(
                product.get("confidence_threshold", 0.55)
            ),
            "purchase_price": float(
                product.get("purchase_price", 0)
            ),
        }

    def minimum_threshold(self) -> float:
        return min(
            float(item.get("confidence_threshold", 0.55))
            for item in self._classes.values()
        )
