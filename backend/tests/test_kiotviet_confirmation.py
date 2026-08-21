from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException

from app.routes import local_jobs
from app.services.kiotviet_service import (
    KiotVietError,
    KiotVietService,
    VIETNAM_TIMEZONE,
)
from app.services.product_mapping_service import ProductMappingService


class FakeKiotVietService:
    create_as_draft = True

    def __init__(self, recovered_receipt=None, *, merged=False) -> None:
        self.create_calls = 0
        self.find_calls = 0
        self.recovered_receipt = recovered_receipt
        self.merged = merged

    def get_product_by_code(self, product_code):
        return {"id": 123, "code": product_code, "isActive": True}

    def resolve_branch(self):
        return {"id": 266664, "branchName": "Warehouse"}

    def create_purchase_receipt(self, products, job_id):
        self.create_calls += 1
        return {
            "dry_run": False,
            "action": "UPDATED" if self.merged else "CREATED",
            "merged_into_daily_receipt": self.merged,
            "recovered": False,
            "validation": {"is_draft": True},
            "receipt": {"id": 456, "code": "PN-TEST"},
        }

    def find_purchase_receipt_by_job_id(self, job_id):
        self.find_calls += 1
        return self.recovered_receipt


class FakeManufacturingService:
    def __init__(self) -> None:
        self.create_calls = []
        self.recovered = None

    def create_manufacturing_receipt(self, product, job_id, branch_id):
        self.create_calls.append((product, job_id, branch_id))
        return {
            "dry_run": False,
            "action": "CREATED",
            "document_type": "MANUFACTURING",
            "recovered": False,
            "validation": {"rpa": True},
            "receipt": {"code": "MS-TEST", "request_id": job_id},
        }

    def reconcile_by_job_id(self, job_id):
        return self.recovered


class KiotVietConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_jobs_root = local_jobs.JOBS_ROOT
        local_jobs.JOBS_ROOT = Path(self.temporary.name).resolve()

    def tearDown(self):
        local_jobs.JOBS_ROOT = self.previous_jobs_root
        self.temporary.cleanup()

    def _request(self, service, manufacturing_service=None):
        return SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    kiotviet_service=service,
                    manufacturing_service=manufacturing_service,
                    excel_service=None,
                    r2_storage_service=None,
                )
            )
        )

    def _write_awaiting_job(self, job_id):
        directory = local_jobs._job_directory(job_id)
        directory.mkdir(parents=True)
        local_jobs._write_json_atomic(
            local_jobs._job_state_path(job_id),
            {
                "job_id": job_id,
                "status": "AWAITING_CONFIRMATION",
                "decision": {
                    "decision": "DIRECT",
                    "product_code": "PA-CRO-0000054",
                    "product_name": "Mini Croissant (Baked)",
                    "count": 2,
                },
                "r2_objects": [],
            },
        )

    def test_confirmation_creates_exactly_one_draft_receipt(self):
        job_id = "a" * 32
        service = FakeKiotVietService()
        self._write_awaiting_job(job_id)

        result = local_jobs.confirm_job(
            job_id,
            local_jobs.ConfirmJobRequest(confirm=True),
            self._request(service),
        )
        repeated = local_jobs.confirm_job(
            job_id,
            local_jobs.ConfirmJobRequest(confirm=True),
            self._request(service),
        )

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(repeated["status"], "COMPLETED")
        self.assertEqual(service.create_calls, 1)
        self.assertTrue(result["kiotviet"]["created"])

    def test_confirmation_reports_daily_receipt_merge(self):
        job_id = "d" * 32
        service = FakeKiotVietService(merged=True)
        self._write_awaiting_job(job_id)

        result = local_jobs.confirm_job(
            job_id,
            local_jobs.ConfirmJobRequest(confirm=True),
            self._request(service),
        )

        self.assertEqual(result["kiotviet"]["action"], "UPDATED")
        self.assertTrue(result["kiotviet"]["merged_into_daily_receipt"])
        self.assertIn("today's KiotViet purchase receipt", result["message"])

    def test_operator_can_create_manufacturing_receipt(self):
        job_id = "e" * 32
        service = FakeKiotVietService()
        manufacturing = FakeManufacturingService()
        self._write_awaiting_job(job_id)

        result = local_jobs.confirm_job(
            job_id,
            local_jobs.ConfirmJobRequest(
                confirm=True,
                document_type="MANUFACTURING",
            ),
            self._request(service, manufacturing),
        )

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["document_type"], "MANUFACTURING")
        self.assertEqual(result["kiotviet"]["document_type"], "MANUFACTURING")
        self.assertEqual(result["kiotviet"]["receipt"]["code"], "MS-TEST")
        self.assertEqual(len(manufacturing.create_calls), 1)
        product, created_job_id, branch_id = manufacturing.create_calls[0]
        self.assertEqual(created_job_id, job_id)
        self.assertEqual(branch_id, 266664)
        self.assertEqual(product["quantity"], 2)
        self.assertEqual(service.create_calls, 0)

    def test_interrupted_confirmation_reconciles_without_second_post(self):
        job_id = "b" * 32
        service = FakeKiotVietService(
            recovered_receipt={
                "id": 789,
                "code": "PN-RECOVERED",
                "description": f"AI inventory inspection job {job_id}",
            }
        )
        directory = local_jobs._job_directory(job_id)
        directory.mkdir(parents=True)
        local_jobs._write_json_atomic(
            local_jobs._job_state_path(job_id),
            {
                "job_id": job_id,
                "status": "CONFIRMING",
                "confirmed_product": {
                    "product_code": "PA-CRO-0000054",
                    "product_name": "Mini Croissant (Baked)",
                    "product_id": 123,
                    "quantity": 2,
                    "purchase_price": 0,
                },
                "r2_objects": [],
            },
        )

        result = local_jobs.confirm_job(
            job_id,
            local_jobs.ConfirmJobRequest(confirm=True),
            self._request(service),
        )

        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(service.find_calls, 1)
        self.assertEqual(service.create_calls, 0)
        self.assertTrue(result["kiotviet"]["recovered"])

    def test_interrupted_confirmation_without_receipt_is_not_reposted(self):
        job_id = "c" * 32
        service = FakeKiotVietService()
        directory = local_jobs._job_directory(job_id)
        directory.mkdir(parents=True)
        local_jobs._write_json_atomic(
            local_jobs._job_state_path(job_id),
            {
                "job_id": job_id,
                "status": "CONFIRMING",
                "confirmed_product": {
                    "product_code": "PA-CRO-0000054",
                    "product_name": "Mini Croissant (Baked)",
                    "product_id": 123,
                    "quantity": 2,
                    "purchase_price": 0,
                },
                "r2_objects": [],
            },
        )

        with self.assertRaises(HTTPException) as context:
            local_jobs.confirm_job(
                job_id,
                local_jobs.ConfirmJobRequest(confirm=True),
                self._request(service),
            )

        self.assertEqual(context.exception.status_code, 409)
        self.assertEqual(service.create_calls, 0)

    def test_operator_can_correct_to_a_supported_foundation_product(self):
        mapping_path = Path(__file__).resolve().parents[1] / "config" / "product_mapping.json"
        mapping = ProductMappingService(mapping_path)
        decision = {
            "decision": "REVIEW",
            "count": 11,
            "candidates": [
                {"product_code": "PA-CRO-0000054", "product_name": "Croissant"}
            ],
        }

        product = local_jobs._confirmed_product_from_decision(
            decision,
            "BR-GF-00000155",
            12,
            mapping,
        )

        self.assertEqual(product["product_code"], "BR-GF-00000155")
        self.assertEqual(product["quantity"], 12)

    def test_operator_correction_rejects_unknown_product_code(self):
        mapping_path = Path(__file__).resolve().parents[1] / "config" / "product_mapping.json"
        mapping = ProductMappingService(mapping_path)
        decision = {"decision": "REVIEW", "count": 3, "candidates": []}

        with self.assertRaises(HTTPException) as context:
            local_jobs._confirmed_product_from_decision(
                decision,
                "NOT-A-REAL-SKU",
                3,
                mapping,
            )

        self.assertEqual(context.exception.status_code, 409)


class KiotVietDailyMergeTests(unittest.TestCase):
    def _service(self, existing):
        service = object.__new__(KiotVietService)
        service.merge_daily_drafts = True
        service.create_as_draft = True
        service.replace_completed_on_update_failure = True
        service.supplier_code = ""
        service.find_daily_purchase_receipt = lambda **kwargs: existing
        service.build_purchase_receipt_payload = lambda products, job_id: (
            {
                "purchaseDate": "2026-08-11T16:30:00+07:00",
                "branchId": 266664,
                "description": f"AI inventory inspection job {job_id}",
                "isDraft": 1,
                "discount": 0,
                "discountRatio": 0,
                "paidAmount": 0,
                "paymentMethod": "Cash",
                "isApplyPurchaseTax": False,
                "purchaseOrderDetails": [
                    {
                        "productCode": item["product_code"],
                        "description": f"AI inventory inspection job {job_id}",
                        "quantity": item["quantity"],
                        "price": 0,
                        "discount": 0,
                        "discountRatio": 0,
                    }
                    for item in products
                ],
            },
            {"is_draft": True},
        )
        service.requests = []
        service.get_product_by_code = lambda code: {
            "id": 1001 if code == "SKU-A" else 1002,
            "code": code,
            "isActive": True,
        }

        def request(method, path, **kwargs):
            service.requests.append((method, path, kwargs))
            body = kwargs.get("json_body") or {}
            return {
                "id": 456,
                "code": "PN-DAILY",
                "status": 1,
                "description": body.get("description"),
                "purchaseOrderDetails": body.get("purchaseOrderDetails", []),
            }

        service._request = request
        service._purchase_receipt_detail = lambda receipt_id: existing
        return service

    def test_same_day_update_adds_existing_sku_and_appends_new_sku(self):
        first_job = "1" * 32
        second_job = "2" * 32
        existing = {
            "id": 456,
            "code": "PN-DAILY",
            "status": 1,
            "purchaseDate": "2026-08-11T16:00:00+07:00",
            "description": f"AI inventory inspection job {first_job}",
            "discount": 0,
            "discountRatio": 0,
            "totalPayment": 0,
            "payments": [],
            "purchaseOrderDetails": [
                {
                    "productCode": "SKU-A",
                    "quantity": 3,
                    "price": 0,
                    "discount": 0,
                    "discountRatio": 0,
                }
            ],
        }
        service = self._service(existing)

        result = service.create_purchase_receipt(
            [
                {"product_code": "SKU-A", "product_name": "A", "quantity": 2},
                {"product_code": "SKU-B", "product_name": "B", "quantity": 4},
            ],
            second_job,
        )

        self.assertEqual(result["action"], "UPDATED")
        self.assertTrue(result["merged_into_daily_receipt"])
        self.assertEqual(len(service.requests), 1)
        method, path, kwargs = service.requests[0]
        self.assertEqual((method, path), ("PUT", "/purchaseorders/456"))
        body = kwargs["json_body"]
        self.assertEqual(body["purchaseDate"], existing["purchaseDate"])
        quantities = {
            item["productCode"]: item["quantity"]
            for item in body["purchaseOrderDetails"]
        }
        self.assertEqual(quantities, {"SKU-A": 5, "SKU-B": 4})
        self.assertIn(KiotVietService._job_marker(first_job), body["description"])
        self.assertIn(KiotVietService._job_marker(second_job), body["description"])

    def test_first_job_of_day_creates_one_daily_receipt(self):
        job_id = "4" * 32
        service = self._service(None)

        result = service.create_purchase_receipt(
            [{"product_code": "SKU-A", "product_name": "A", "quantity": 2}],
            job_id,
        )

        self.assertEqual(result["action"], "CREATED")
        self.assertFalse(result["merged_into_daily_receipt"])
        self.assertEqual(len(service.requests), 1)
        method, path, kwargs = service.requests[0]
        self.assertEqual((method, path), ("POST", "/purchaseorders"))
        self.assertTrue(
            kwargs["json_body"]["description"].startswith(
                "AI inventory inspection daily "
            )
        )
        self.assertIn(
            KiotVietService._job_marker(job_id),
            kwargs["json_body"]["description"],
        )

    def test_daily_lookup_ignores_manual_receipts(self):
        service = object.__new__(KiotVietService)
        service.supplier_code = ""
        current = local_jobs.datetime.now(VIETNAM_TIMEZONE)
        day = current.date()
        ai_receipt = {
            "id": 20,
            "code": "PN-AI",
            "status": 1,
            "purchaseDate": f"{day.isoformat()}T16:00:00+07:00",
            "description": f"AI inventory inspection job {'5' * 32}",
        }
        service._purchase_receipts_for_date = lambda business_date: [
            {
                "id": 10,
                "code": "PN-MANUAL",
                "status": 1,
                "purchaseDate": f"{day.isoformat()}T15:00:00+07:00",
                "description": "Manual warehouse receipt",
            },
            ai_receipt,
        ]

        selected = service.find_daily_purchase_receipt(now=current)

        self.assertEqual(selected["id"], ai_receipt["id"])

    def test_retry_with_existing_job_marker_does_not_add_quantity_twice(self):
        job_id = "3" * 32
        today = local_jobs.datetime.now(VIETNAM_TIMEZONE).date()
        existing = {
            "id": 456,
            "code": "PN-DAILY",
            "status": 1,
            "purchaseDate": f"{today.isoformat()}T16:00:00+07:00",
            "description": KiotVietService._daily_description("", job_id, today),
            "purchaseOrderDetails": [
                {"productCode": "SKU-A", "quantity": 5, "price": 0}
            ],
        }
        service = self._service(existing)

        result = service.create_purchase_receipt(
            [{"product_code": "SKU-A", "product_name": "A", "quantity": 2}],
            job_id,
        )

        self.assertEqual(result["action"], "REUSED")
        self.assertTrue(result["recovered"])
        self.assertEqual(service.requests, [])

    def test_completed_daily_receipt_is_updated_and_kept_completed(self):
        job_id = "6" * 32
        existing = {
            "id": 456,
            "code": "PN-COMPLETE",
            "status": 3,
            "purchaseDate": "2026-08-11T16:00:00+07:00",
            "description": f"AI inventory inspection job {'5' * 32}",
            "discount": 0,
            "discountRatio": 0,
            "totalPayment": 0,
            "payments": [],
            "purchaseOrderDetails": [
                {"productCode": "SKU-A", "quantity": 3, "price": 0}
            ],
        }
        service = self._service(existing)
        service.create_as_draft = False

        result = service.create_purchase_receipt(
            [{"product_code": "SKU-A", "product_name": "A", "quantity": 2}],
            job_id,
        )

        self.assertEqual(result["action"], "UPDATED")
        method, path, kwargs = service.requests[0]
        self.assertEqual((method, path), ("PUT", "/purchaseorders/456"))
        self.assertEqual(kwargs["json_body"]["isDraft"], 0)
        self.assertEqual(
            kwargs["json_body"]["purchaseOrderDetails"][0]["quantity"], 5
        )

    def test_business_rejection_replaces_completed_receipt_then_cancels_old(self):
        job_id = "7" * 32
        existing = {
            "id": 456,
            "code": "PN-OLD",
            "status": 3,
            "purchaseDate": "2026-08-11T16:00:00+07:00",
            "description": f"AI inventory inspection job {'5' * 32}",
            "discount": 0,
            "discountRatio": 0,
            "totalPayment": 0,
            "payments": [],
            "purchaseOrderDetails": [
                {"productCode": "SKU-A", "quantity": 3, "price": 0}
            ],
        }
        service = self._service(existing)
        service.create_as_draft = False
        service.replace_completed_on_update_failure = True
        service.cancelled = []

        def request(method, path, **kwargs):
            service.requests.append((method, path, kwargs))
            if method == "PUT":
                raise KiotVietError("completed receipt is locked", status_code=420)
            if method == "POST":
                body = kwargs["json_body"]
                return {
                    "id": 789,
                    "code": "PN-NEW",
                    "status": 3,
                    "description": body["description"],
                    "purchaseOrderDetails": body["purchaseOrderDetails"],
                }
            raise AssertionError((method, path))

        service._request = request
        service._delete_purchase_receipt = lambda receipt_id: service.cancelled.append(
            int(receipt_id)
        )

        result = service.create_purchase_receipt(
            [{"product_code": "SKU-B", "product_name": "B", "quantity": 4}],
            job_id,
        )

        self.assertEqual(result["action"], "REPLACED")
        self.assertEqual(result["receipt"]["code"], "PN-NEW")
        self.assertEqual(service.cancelled, [456])
        self.assertIn("AIR:456", result["receipt"]["description"])


if __name__ == "__main__":
    unittest.main()
