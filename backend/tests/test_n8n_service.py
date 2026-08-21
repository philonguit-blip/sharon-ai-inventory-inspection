from __future__ import annotations

import base64
import unittest

import httpx

from app.services.n8n_service import N8nOrchestratorError, N8nOrchestratorService


class N8nOrchestratorServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_basic_auth_and_json(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            expected = base64.b64encode(b"gateway:secret").decode("ascii")
            self.assertEqual(request.headers["Authorization"], f"Basic {expected}")
            self.assertEqual(request.url.path, "/webhook/bakery-upload-init")
            self.assertEqual(request.method, "POST")
            return httpx.Response(
                200,
                json={
                    "job_id": "a" * 32,
                    "uploads": [{"upload_url": "https://upload.example/signed"}],
                },
            )

        service = N8nOrchestratorService(
            "https://n8n.example/webhook",
            "gateway",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await service.prepare_uploads({"files": []})
        finally:
            await service.close()
        self.assertEqual(response["job_id"], "a" * 32)

    async def test_waits_for_outbound_worker_presign_result(self):
        request_id = "b" * 32
        status_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal status_calls
            if request.url.path.endswith("bakery-upload-init"):
                return httpx.Response(
                    200,
                    json={"request_id": request_id, "status": "WAITING_FOR_WORKER"},
                )
            status_calls += 1
            return httpx.Response(
                200,
                json={
                    "job_id": request_id,
                    "status": "READY",
                    "uploads": [{"upload_url": "signed"}],
                },
            )

        service = N8nOrchestratorService(
            "https://n8n.example/webhook",
            "gateway",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await service.prepare_uploads({"files": [{"filename": "a.jpg"}]})
        finally:
            await service.close()
        self.assertEqual(response["job_id"], request_id)
        self.assertEqual(status_calls, 1)

    async def test_forwards_client_error_without_exposing_credentials(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "Invalid job ID."})

        service = N8nOrchestratorService(
            "https://n8n.example/webhook",
            "gateway",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaises(N8nOrchestratorError) as context:
                await service.job_status("bad")
        finally:
            await service.close()
        self.assertEqual(context.exception.status_code, 422)
        self.assertEqual(context.exception.detail, "Invalid job ID.")

    async def test_rejects_workflow_error_returned_with_http_200(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"status": "ERROR", "detail": "Worker is unavailable."},
            )

        service = N8nOrchestratorService(
            "https://n8n.example/webhook",
            "gateway",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaises(N8nOrchestratorError) as context:
                await service.submit_job({"job_id": "a" * 32, "files": []})
        finally:
            await service.close()
        self.assertEqual(context.exception.status_code, 422)
        self.assertEqual(context.exception.detail, "Worker is unavailable.")

    async def test_rejects_empty_success_response(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        service = N8nOrchestratorService(
            "https://n8n.example/webhook",
            "gateway",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            with self.assertRaises(N8nOrchestratorError) as context:
                await service.health()
        finally:
            await service.close()
        self.assertEqual(context.exception.status_code, 502)

    async def test_confirm_job_posts_job_id_and_confirmation(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/webhook/bakery-confirm")
            self.assertEqual(
                request.read(),
                b'{"confirm":true,"product_code":"SKU-1","job_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
            )
            return httpx.Response(200, json={"status": "CONFIRMING"})

        service = N8nOrchestratorService(
            "https://n8n.example/webhook",
            "gateway",
            "secret",
            transport=httpx.MockTransport(handler),
        )
        try:
            response = await service.confirm_job(
                "a" * 32,
                {"confirm": True, "product_code": "SKU-1"},
            )
        finally:
            await service.close()
        self.assertEqual(response["status"], "CONFIRMING")


if __name__ == "__main__":
    unittest.main()
