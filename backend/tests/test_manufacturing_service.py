from __future__ import annotations

import unittest

import httpx

from app.services.manufacturing_service import (
    ManufacturingError,
    ManufacturingService,
)


class _Client:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        status_code, payload = self.responses.pop(0)
        request = httpx.Request(method, url)
        return httpx.Response(status_code, json=payload, request=request)


class ManufacturingServiceTests(unittest.TestCase):
    def _service(self, responses):
        service = ManufacturingService("http://127.0.0.1:8000", "secret-token")
        service.client = _Client(responses)
        return service

    def test_create_sends_idempotency_key_and_exact_quantity(self):
        service = self._service(
            [
                (
                    200,
                    {
                        "status": "SUCCESS",
                        "created": True,
                        "save_mode": "DRAFT",
                        "receipt": {"code": "MS001"},
                    },
                )
            ]
        )
        job_id = "a" * 32

        result = service.create_manufacturing_receipt(
            {
                "product_code": "SKU-A",
                "product_name": "Product A",
                "quantity": 12,
            },
            job_id,
            266664,
        )

        self.assertEqual(result["action"], "CREATED")
        self.assertEqual(result["document_type"], "MANUFACTURING")
        method, url, kwargs = service.client.calls[0]
        self.assertEqual((method, url), ("POST", "http://127.0.0.1:8000/run-manufacture"))
        self.assertEqual(kwargs["headers"]["X-Internal-Token"], "secret-token")
        self.assertEqual(kwargs["json"]["requestId"], job_id)
        self.assertEqual(kwargs["json"]["branchId"], 266664)
        self.assertEqual(kwargs["json"]["items"][0]["targetThreshold"], 12)

    def test_reconcile_returns_successful_cached_result(self):
        service = self._service(
            [
                (
                    200,
                    {
                        "status": "SUCCESS",
                        "created": True,
                        "reused": True,
                        "receipt": {"code": "MS001"},
                    },
                )
            ]
        )
        result = service.reconcile_by_job_id("b" * 32)
        self.assertEqual(result["action"], "REUSED")
        self.assertTrue(result["recovered"])

    def test_uncertain_result_blocks_automatic_retry(self):
        service = self._service(
            [(200, {"status": "UNCERTAIN", "message": "Inspect KiotViet."})]
        )
        with self.assertRaises(ManufacturingError) as context:
            service.reconcile_by_job_id("c" * 32)
        self.assertEqual(context.exception.status_code, 409)

    def test_failure_before_save_is_marked_safe_to_retry(self):
        service = self._service(
            [
                (
                    200,
                    {
                        "status": "ERROR",
                        "created": False,
                        "safe_to_retry": True,
                    },
                )
            ]
        )
        result = service.reconcile_by_job_id("d" * 32)
        self.assertTrue(result["retry_safe"])


if __name__ == "__main__":
    unittest.main()
