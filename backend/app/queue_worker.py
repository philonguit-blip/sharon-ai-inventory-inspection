"""Outbound-only relay worker for the n8n bakery job queue.

The worker never opens a public port. It polls authenticated n8n webhooks,
executes the requested operation against the local FastAPI service, and posts
the result back to n8n. Image bytes continue to travel directly through R2.
"""

from __future__ import annotations

import argparse
import socket
import time
from typing import Any

import httpx

from app.config import (
    APP_AUTH_PASSWORD,
    APP_AUTH_USERNAME,
    N8N_WEBHOOK_BASE,
    N8N_WEBHOOK_PASSWORD,
    N8N_WEBHOOK_USERNAME,
)


class RelayWorkerError(RuntimeError):
    pass


class BakeryRelayWorker:
    def __init__(
        self,
        *,
        remote_base: str = N8N_WEBHOOK_BASE,
        local_base: str = "http://127.0.0.1:8080/api/v1/bakery",
        worker_id: str | None = None,
        poll_seconds: float = 2.0,
        job_poll_seconds: float = 1.2,
        heartbeat_seconds: float = 10.0,
        job_timeout_seconds: float = 3600,
        remote_client: httpx.Client | None = None,
        local_client: httpx.Client | None = None,
    ) -> None:
        self.worker_id = worker_id or f"bakery-{socket.gethostname().lower()}"
        self.poll_seconds = max(float(poll_seconds), 0.2)
        self.job_poll_seconds = max(float(job_poll_seconds), 0.2)
        self.heartbeat_seconds = max(float(heartbeat_seconds), 2.0)
        self.job_timeout_seconds = max(float(job_timeout_seconds), 10)
        self.remote = remote_client or httpx.Client(
            base_url=f"{remote_base.rstrip('/')}/",
            auth=httpx.BasicAuth(N8N_WEBHOOK_USERNAME, N8N_WEBHOOK_PASSWORD),
            timeout=httpx.Timeout(45),
        )
        self.local = local_client or httpx.Client(
            base_url=f"{local_base.rstrip('/')}/",
            auth=httpx.BasicAuth(APP_AUTH_USERNAME, APP_AUTH_PASSWORD),
            # A 160 MB batch can take several minutes to verify and download
            # from R2 before FastAPI returns the accepted job.
            timeout=httpx.Timeout(600),
        )

    def close(self) -> None:
        self.remote.close()
        self.local.close()

    @staticmethod
    def _json(response: httpx.Response, operation: str) -> dict[str, Any]:
        try:
            payload = response.json() if response.content else {}
        except ValueError as exc:
            raise RelayWorkerError(
                f"{operation} returned invalid JSON (HTTP {response.status_code})."
            ) from exc
        if not response.is_success:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise RelayWorkerError(
                str(detail or f"{operation} failed (HTTP {response.status_code}).")
            )
        if not isinstance(payload, dict):
            raise RelayWorkerError(f"{operation} returned an invalid payload.")
        return payload

    def _remote(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        response = self.remote.request(method, path, json=json_body, params=params)
        return self._json(response, f"n8n {path}")

    def _local(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.local.request(method, path, json=json_body)
        return self._json(response, f"local API {path}")

    def heartbeat(self) -> dict[str, Any]:
        health = self._local("GET", "health")
        return self._remote(
            "POST",
            "bakery-worker-heartbeat",
            json_body={
                "worker_id": self.worker_id,
                "ready": bool(health.get("ready")),
                "r2_configured": bool(health.get("r2_configured")),
                "kiotviet_configured": bool(health.get("kiotviet_configured")),
                "kiotviet_auto_create_draft": bool(
                    health.get("kiotviet_auto_create_draft")
                ),
                "model": health.get("model"),
                "max_images_per_job": health.get("max_images_per_job", 50),
                "max_image_size_mb": health.get("max_image_size_mb", 50),
                "max_job_upload_size_mb": health.get(
                    "max_job_upload_size_mb", 160
                ),
                "allowed_image_extensions": health.get(
                    "allowed_image_extensions", []
                ),
            },
        )

    def next_task(self) -> dict[str, Any] | None:
        payload = self._remote(
            "GET",
            "bakery-worker-next",
            params={"worker_id": self.worker_id},
        )
        task = payload.get("task")
        return task if isinstance(task, dict) else None

    def report(
        self,
        task: dict[str, Any],
        *,
        ok: bool,
        final: bool,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        response = self._remote(
            "POST",
            "bakery-worker-result",
            json_body={
                "worker_id": self.worker_id,
                "task_id": str(task["task_id"]),
                "job_id": str(task.get("job_id") or task["task_id"]),
                "task_type": str(task["task_type"]),
                "ok": ok,
                "final": final,
                "result": result or {},
                "error": error,
            },
        )
        if response.get("ok") is not True:
            raise RelayWorkerError(
                str(response.get("detail") or "n8n rejected the worker result.")
            )
        return response

    def _run_presign(self, task: dict[str, Any]) -> None:
        payload = dict(task.get("payload") or {})
        payload["job_id"] = str(task.get("job_id") or task["task_id"])
        result = self._local("POST", "uploads/presign", json_body=payload)
        self.report(task, ok=True, final=True, result=result)

    def _run_process(self, task: dict[str, Any]) -> None:
        payload = dict(task.get("payload") or {})
        accepted = self._local("POST", "jobs/from-r2", json_body=payload)
        try:
            self.report(task, ok=True, final=False, result=accepted)
        except RelayWorkerError as exc:
            # Losing a progress update must not turn an already accepted local
            # job into an error. The final result remains authoritative.
            print(f"[WORKER] Could not publish progress: {exc}")

        job_id = str(accepted["job_id"])
        deadline = time.monotonic() + self.job_timeout_seconds
        next_heartbeat = 0.0
        last_reported_processed = int(accepted.get("processed_images") or 0)
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_heartbeat:
                try:
                    self.heartbeat()
                except RelayWorkerError as exc:
                    # A temporary heartbeat failure must not cancel an AI job
                    # that is already running locally.
                    print(f"[WORKER] Could not refresh heartbeat: {exc}")
                next_heartbeat = now + self.heartbeat_seconds
            state = self._local("GET", f"jobs/{job_id}")
            status = str(state.get("status") or "")
            processed_images = int(state.get("processed_images") or 0)
            if processed_images != last_reported_processed:
                progress = {
                    "job_id": job_id,
                    "status": status or "PROCESSING",
                    "total_images": int(state.get("total_images") or 0),
                    "processed_images": processed_images,
                    "error": str(state.get("error") or ""),
                }
                try:
                    self.report(task, ok=True, final=False, result=progress)
                    last_reported_processed = processed_images
                except RelayWorkerError as exc:
                    # Progress is advisory; the final result remains
                    # authoritative and must still be delivered.
                    print(f"[WORKER] Could not publish image progress: {exc}")
            if status == "COMPLETED":
                linked = self._local("GET", f"jobs/{job_id}/links")
                self.report(task, ok=True, final=True, result=linked)
                return
            if status == "ERROR":
                self.report(
                    task,
                    ok=False,
                    final=True,
                    result=state,
                    error=str(state.get("error") or "AI job failed."),
                )
                return
            time.sleep(self.job_poll_seconds)
        raise RelayWorkerError(f"Job {job_id} exceeded the worker timeout.")

    def run_once(self) -> bool:
        self.heartbeat()
        task = self.next_task()
        if task is None:
            return False
        try:
            task_type = str(task.get("task_type") or "")
            if task_type == "PRESIGN":
                self._run_presign(task)
            elif task_type == "PROCESS":
                self._run_process(task)
            else:
                raise RelayWorkerError(f"Unsupported task type: {task_type or 'empty'}.")
        except Exception as exc:
            try:
                self.report(task, ok=False, final=True, error=str(exc))
            except Exception:
                pass
            raise
        return True

    def run_forever(self) -> None:
        while True:
            try:
                worked = self.run_once()
                if not worked:
                    time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                return
            except Exception as exc:
                print(f"[WORKER] {exc}")
                time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sharon Bakery outbound n8n worker")
    parser.add_argument("--once", action="store_true", help="Poll and process once")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    arguments = parser.parse_args()
    worker = BakeryRelayWorker(poll_seconds=arguments.poll_seconds)
    try:
        if arguments.once:
            worker.run_once()
        else:
            worker.run_forever()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
