from __future__ import annotations

import unittest

from app.routes.local_jobs import _auto_create_kiotviet_draft


PRODUCTS = [
    {
        "product_code": "PA-CRO-0000054",
        "product_name": "Bánh nhiều lớp - Viennoiserie - Mini Croissant (Baked)",
        "quantity": 2,
        "purchase_price": 0,
    }
]


class FakeKiotVietService:
    def __init__(self, *, is_draft: bool = True, create_error: str = "") -> None:
        self.is_draft = is_draft
        self.create_error = create_error
        self.preview_calls = 0
        self.create_calls = 0

    def preview_purchase_receipt(self, products, job_id):
        self.preview_calls += 1
        return {
            "dry_run": True,
            "validation": {
                "branch": {"id": 266664, "name": "Warehouse"},
                "products": products,
                "is_draft": self.is_draft,
            },
        }

    def create_purchase_receipt(self, products, job_id):
        self.create_calls += 1
        if self.create_error:
            raise RuntimeError(self.create_error)
        return {
            "dry_run": False,
            "validation": {"is_draft": True},
            "receipt": {"id": 123, "code": "PN-TEST"},
        }


class AutoCreateKiotVietDraftTests(unittest.TestCase):
    def test_creates_only_after_draft_validation_passes(self):
        service = FakeKiotVietService()

        result = _auto_create_kiotviet_draft("a" * 32, PRODUCTS, service)

        self.assertEqual(service.preview_calls, 1)
        self.assertEqual(service.create_calls, 1)
        self.assertTrue(result["created"])
        self.assertEqual(result["validation_status"], "PASSED")
        self.assertEqual(result["receipt"]["code"], "PN-TEST")

    def test_blocks_automatic_creation_when_not_configured_as_draft(self):
        service = FakeKiotVietService(is_draft=False)

        result = _auto_create_kiotviet_draft("b" * 32, PRODUCTS, service)

        self.assertEqual(service.preview_calls, 1)
        self.assertEqual(service.create_calls, 0)
        self.assertFalse(result["created"])
        self.assertEqual(result["validation_status"], "FAILED")

    def test_keeps_validation_result_when_creation_fails(self):
        service = FakeKiotVietService(create_error="simulated API failure")

        result = _auto_create_kiotviet_draft("c" * 32, PRODUCTS, service)

        self.assertEqual(service.create_calls, 1)
        self.assertFalse(result["created"])
        self.assertEqual(result["validation_status"], "PASSED")
        self.assertIn("simulated API failure", result["error"])


if __name__ == "__main__":
    unittest.main()
