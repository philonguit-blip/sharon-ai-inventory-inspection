from __future__ import annotations

import unittest

import httpx

from app.queue_worker import BakeryRelayWorker


class BakeryRelayWorkerTests(unittest.TestCase):
    def test_presign_task_uses_stable_job_id_and_reports_result(self):
        remote_requests: list[tuple[str, str, dict]] = []
        local_requests: list[tuple[str, str, dict]] = []
        job_id = "a" * 32

        def remote_handler(request: httpx.Request) -> httpx.Response:
            body = __import__("json").loads(request.content or b"{}")
            remote_requests.append((request.method, request.url.path, body))
            if request.url.path.endswith("bakery-worker-heartbeat"):
                return httpx.Response(200, json={"ok": True})
            if request.url.path.endswith("bakery-worker-next"):
                return httpx.Response(
                    200,
                    json={
                        "task": {
                            "task_id": job_id,
                            "job_id": job_id,
                            "task_type": "PRESIGN",
                            "payload": {
                                "files": [
                                    {
                                        "filename": "tray.jpg",
                                        "content_type": "image/jpeg",
                                        "size_bytes": 123,
                                    }
                                ]
                            },
                        }
                    },
                )
            if request.url.path.endswith("bakery-worker-result"):
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(404)

        def local_handler(request: httpx.Request) -> httpx.Response:
            body = __import__("json").loads(request.content or b"{}")
            local_requests.append((request.method, request.url.path, body))
            if request.url.path.endswith("/health"):
                return httpx.Response(
                    200,
                    json={"ready": True, "r2_configured": True, "model": {}},
                )
            if request.url.path.endswith("/uploads/presign"):
                return httpx.Response(
                    200,
                    json={"job_id": job_id, "uploads": [{"upload_url": "signed"}]},
                )
            return httpx.Response(404)

        worker = BakeryRelayWorker(
            worker_id="test-worker",
            remote_client=httpx.Client(
                base_url="https://n8n.example/webhook/",
                transport=httpx.MockTransport(remote_handler),
            ),
            local_client=httpx.Client(
                base_url="http://local/api/v1/bakery/",
                transport=httpx.MockTransport(local_handler),
            ),
        )
        try:
            self.assertTrue(worker.run_once())
        finally:
            worker.close()

        presign = next(item for item in local_requests if item[1].endswith("presign"))
        self.assertEqual(presign[2]["job_id"], job_id)
        report = next(item for item in remote_requests if item[1].endswith("result"))
        self.assertTrue(report[2]["ok"])
        self.assertTrue(report[2]["final"])
        self.assertEqual(report[2]["result"]["job_id"], job_id)

    def test_empty_queue_returns_without_work(self):
        def remote_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("bakery-worker-heartbeat"):
                return httpx.Response(200, json={"ok": True})
            return httpx.Response(200, json={"task": None})

        def local_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ready": True})

        worker = BakeryRelayWorker(
            remote_client=httpx.Client(
                base_url="https://n8n.example/webhook/",
                transport=httpx.MockTransport(remote_handler),
            ),
            local_client=httpx.Client(
                base_url="http://local/api/v1/bakery/",
                transport=httpx.MockTransport(local_handler),
            ),
        )
        try:
            self.assertFalse(worker.run_once())
        finally:
            worker.close()

    def test_process_task_refreshes_heartbeat_while_waiting(self):
        remote_requests: list[tuple[str, dict]] = []
        job_id = "b" * 32
        job_status_calls = 0

        def remote_handler(request: httpx.Request) -> httpx.Response:
            body = __import__("json").loads(request.content or b"{}")
            remote_requests.append((request.url.path, body))
            return httpx.Response(200, json={"ok": True})

        def local_handler(request: httpx.Request) -> httpx.Response:
            nonlocal job_status_calls
            if request.url.path.endswith("/jobs/from-r2"):
                return httpx.Response(
                    202,
                    json={"job_id": job_id, "status": "QUEUED", "total_images": 2},
                )
            if request.url.path.endswith(f"/jobs/{job_id}/links"):
                return httpx.Response(
                    200,
                    json={"job_id": job_id, "status": "COMPLETED", "total_images": 2},
                )
            if request.url.path.endswith(f"/jobs/{job_id}"):
                job_status_calls += 1
                if job_status_calls == 1:
                    return httpx.Response(
                        200,
                        json={
                            "job_id": job_id,
                            "status": "PROCESSING",
                            "total_images": 2,
                            "processed_images": 1,
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "job_id": job_id,
                        "status": "COMPLETED",
                        "total_images": 2,
                        "processed_images": 2,
                    },
                )
            if request.url.path.endswith("/health"):
                return httpx.Response(
                    200,
                    json={"ready": True, "r2_configured": True, "model": {}},
                )
            return httpx.Response(404)

        worker = BakeryRelayWorker(
            worker_id="test-worker",
            remote_client=httpx.Client(
                base_url="https://n8n.example/webhook/",
                transport=httpx.MockTransport(remote_handler),
            ),
            local_client=httpx.Client(
                base_url="http://local/api/v1/bakery/",
                transport=httpx.MockTransport(local_handler),
            ),
            job_poll_seconds=0.2,
        )
        try:
            worker._run_process(
                {
                    "task_id": f"process-{job_id}",
                    "job_id": job_id,
                    "task_type": "PROCESS",
                    "payload": {"job_id": job_id, "files": []},
                }
            )
        finally:
            worker.close()

        heartbeat_calls = [
            path for path, _ in remote_requests if path.endswith("bakery-worker-heartbeat")
        ]
        self.assertGreaterEqual(len(heartbeat_calls), 1)
        progress_updates = [
            body
            for path, body in remote_requests
            if path.endswith("bakery-worker-result")
            and body.get("final") is False
            and body.get("result", {}).get("processed_images") == 1
        ]
        self.assertEqual(len(progress_updates), 1)


if __name__ == "__main__":
    unittest.main()
