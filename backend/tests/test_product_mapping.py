from __future__ import annotations

import unittest
from pathlib import Path

from app.services.product_mapping_service import ProductMappingService


MAPPING_PATH = Path(__file__).resolve().parents[1] / "config" / "product_mapping.json"
NEW_CLASS = "BR-SD-0000127-750_SourdoughSharonMultiseed_Pc"


class ProductMappingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ProductMappingService(MAPPING_PATH)

    def test_new_sourdough_class_uses_requested_threshold(self):
        product = self.service.resolve(NEW_CLASS)

        self.assertEqual(product["product_code"], "BR-SD-0000127-750")
        self.assertEqual(product["product_name"], "Sourdough Sharon Multiseed (Pc)")
        self.assertEqual(product["confidence_threshold"], 0.45)

    def test_production_thresholds_match_current_mapping(self):
        self.assertEqual(len(self.service.class_names()), 27)
        self.assertEqual(self.service.resolve("COMMON_BaguetteMini")["confidence_threshold"], 0.30)
        self.assertEqual(self.service.resolve("COMMON_BurgerNoSesame")["confidence_threshold"], 0.50)
        self.assertTrue(all(0 < self.service.resolve(name)["confidence_threshold"] <= 1 for name in self.service.class_names()))

    def test_new_class_sets_minimum_candidate_threshold(self):
        self.assertEqual(self.service.minimum_threshold(), 0.25)

    def test_foundation_only_product_is_available_for_operator_confirmation(self):
        product = self.service.resolve_product_code("CA-GF-0000169")

        self.assertEqual(
            product["product_name"],
            "CakeMiniVanilamuffinglutenfree_40g",
        )
        self.assertEqual(product["source_type"], "direct")
        self.assertEqual(len(self.service.class_names()), 27)
        self.assertEqual(
            len(self.service.all_business_products()),
            self.service.supported_product_count,
        )

    def test_common_cookies_catalog_contains_two_family_members(self):
        earl_grey = self.service.resolve_product_code("CA-COO-0000040")
        chocolate = self.service.resolve_product_code("CA-COO-0000112")

        self.assertEqual(earl_grey["family_name"], "COMMON Cookies")
        self.assertEqual(chocolate["family_name"], "COMMON Cookies")
        self.assertEqual(earl_grey["source_type"], "family_member")
        self.assertEqual(chocolate["source_type"], "family_member")
        self.assertEqual(self.service.supported_product_count, 50)


if __name__ == "__main__":
    unittest.main()
