"""Outbound-only relay worker for the n8n bakery job queue.

The worker never opens a public port. It polls authenticated n8n webhooks,
executes the requested operation against the local FastAPI service, and posts
the result back to n8n. Image bytes continue to travel directly through R2.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import socket
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
from typing import Any

import httpx

from app.config import (
    APP_AUTH_PASSWORD,
    APP_AUTH_USERNAME,
    N8N_WEBHOOK_BASE,
    N8N_WEBHOOK_PASSWORD,
    N8N_WEBHOOK_USERNAME,
    RUNTIME_PATH,
)


class RelayWorkerError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code

    @property
    def transient(self) -> bool:
        return self.status_code in {408, 425, 429, 500, 502, 503, 504}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _compact(value: Any, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _acquire_single_worker_lock(
    lock_path: Path = RUNTIME_PATH / "outbound-worker.lock",
) -> BinaryIO | None:
    """Acquire a process-wide, automatically released worker lock."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.write(str(os.getpid()).encode("ascii"))
        handle.truncate()
        handle.flush()
        return handle
    except (OSError, BlockingIOError):
        handle.close()
        return None


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
        debug: bool | None = None,
        log_file: Path | str | None = None,
    ) -> None:
        self.worker_id = worker_id or (
            f"bakery-{socket.gethostname().lower()}-{os.getpid()}"
        )
        self.poll_seconds = max(float(poll_seconds), 0.2)
        self.job_poll_seconds = max(float(job_poll_seconds), 0.2)
        self.heartbeat_seconds = max(float(heartbeat_seconds), 2.0)
        self.job_timeout_seconds = max(float(job_timeout_seconds), 10)
        self.debug = _env_bool("SHARON_WORKER_DEBUG", False) if debug is None else bool(debug)
        default_log = RUNTIME_PATH / "logs" / f"outbound-worker-{datetime.now():%Y%m%d}.log"
        configured_log = str(os.getenv("SHARON_WORKER_LOG_FILE", "")).strip()
        self.log_file = Path(log_file or configured_log or default_log).expanduser().resolve()
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat_log = 0.0
        self._last_health_signature: tuple[Any, ...] | None = None
        # Heartbeats used to be sent once per queue poll (about every 2 s),
        # which created unnecessary n8n executions and DB/static-data writes.
        # Keep a single scheduler for idle and in-job heartbeats instead.
        self._next_heartbeat_due = 0.0
        self._heartbeat_failure_count = 0
        self._last_heartbeat_warning = 0.0
        self.remote = remote_client or httpx.Client(
            base_url=f"{remote_base.rstrip('/')}/",
            auth=httpx.BasicAuth(N8N_WEBHOOK_USERNAME, N8N_WEBHOOK_PASSWORD),
            timeout=httpx.Timeout(45),
        )
        self.local = local_client or httpx.Client(
            base_url=f"{local_base.rstrip('/')}/",
            auth=httpx.BasicAuth(APP_AUTH_USERNAME, APP_AUTH_PASSWORD),
            # A high-resolution tray image can take time to verify and download
            # from R2 before FastAPI returns the accepted job.
            timeout=httpx.Timeout(600),
        )
        self._emit("INFO", "=" * 72)
        self._emit("INFO", f"Worker started | id={self.worker_id} | pid={os.getpid()} | debug={self.debug}")
        self._emit("INFO", f"Remote n8n={remote_base.rstrip('/')} | Local API={local_base.rstrip('/')}")
        self._emit(
            "INFO",
            f"Poll={self.poll_seconds:.1f}s | Job poll={self.job_poll_seconds:.1f}s | "
            f"Heartbeat={self.heartbeat_seconds:.1f}s | Timeout={self.job_timeout_seconds:.0f}s",
        )
        self._emit("INFO", f"Persistent log: {self.log_file}")
        self._emit("INFO", "=" * 72)

    def _emit(self, level: str, message: str, *, debug_only: bool = False) -> None:
        if debug_only and not self.debug:
            return
        prefix = f"[{_timestamp()}] [{level.upper():5}]"
        lines = str(message).splitlines() or [""]
        rendered = "\n".join(f"{prefix} {line}" for line in lines)
        print(rendered, flush=True)
        try:
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(rendered + "\n")
        except OSError:
            # Logging must never stop inference.
            pass

    def _debug(self, message: str) -> None:
        self._emit("DEBUG", message, debug_only=True)

    @staticmethod
    def _task_context(task: dict[str, Any]) -> str:
        return (
            f"task={task.get('task_id', '-')} "
            f"job={task.get('job_id') or task.get('task_id', '-')} "
            f"type={task.get('task_type', '-')}"
        )

    @staticmethod
    def _decision_summary(payload: dict[str, Any]) -> str:
        decision = payload.get("decision") if isinstance(payload, dict) else None
        decision = decision if isinstance(decision, dict) else {}
        hybrid = payload.get("hybrid") if isinstance(payload, dict) else None
        hybrid = hybrid if isinstance(hybrid, dict) else {}
        return (
            f"status={payload.get('status', '-')} "
            f"decision={decision.get('decision', '-')} "
            f"class={decision.get('display_name') or decision.get('dominant_class') or '-'} "
            f"count={decision.get('count', 0)} "
            f"purity={float(decision.get('purity') or 0.0):.3f} "
            f"conf={float(decision.get('avg_confidence') or 0.0):.3f} "
            f"engine={hybrid.get('selected_engine') or payload.get('engine') or '-'}"
        )

    def close(self) -> None:
        self._emit("INFO", "Closing outbound worker clients.")
        self.remote.close()
        self.local.close()

    @staticmethod
    def _json(response: httpx.Response, operation: str) -> dict[str, Any]:
        try:
            payload = response.json() if response.content else {}
        except ValueError as exc:
            raise RelayWorkerError(
                f"{operation} returned invalid JSON (HTTP {response.status_code}).",
                status_code=response.status_code,
            ) from exc
        if not response.is_success:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise RelayWorkerError(
                str(detail or f"{operation} failed (HTTP {response.status_code})."),
                status_code=response.status_code,
            )
        if not isinstance(payload, dict):
            raise RelayWorkerError(f"{operation} returned an invalid payload.")
        return payload

    @staticmethod
    def _is_transient_remote_error(exc: Exception) -> bool:
        if isinstance(exc, RelayWorkerError):
            return exc.transient
        return isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        )

    def _remote(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        quiet: bool = False,
        transient_retries: int = 0,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if not quiet:
            self._debug(f"[HTTP][n8n] -> {method.upper()} {path}")

        attempts = max(1, int(transient_retries) + 1)
        for attempt in range(1, attempts + 1):
            try:
                response = self.remote.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                )
                payload = self._json(response, f"n8n {path}")
                elapsed = (time.perf_counter() - started) * 1000
                if not quiet:
                    self._debug(
                        f"[HTTP][n8n] <- {response.status_code} {method.upper()} "
                        f"{path} ({elapsed:.0f} ms)"
                    )
                return payload
            except Exception as exc:
                can_retry = (
                    attempt < attempts
                    and self._is_transient_remote_error(exc)
                )
                if can_retry:
                    delay = min(0.5 * (2 ** (attempt - 1)), 2.0)
                    self._emit(
                        "WARN",
                        f"[HTTP][n8n] transient failure for {path}: {exc}; "
                        f"retry {attempt}/{attempts - 1} in {delay:.1f}s",
                    )
                    time.sleep(delay)
                    continue

                elapsed = (time.perf_counter() - started) * 1000
                self._emit(
                    "ERROR",
                    f"[HTTP][n8n] {method.upper()} {path} failed after "
                    f"{elapsed:.0f} ms: {type(exc).__name__}: {exc}",
                )
                raise

        raise RelayWorkerError(f"n8n {path} failed without a response.")

    def _local(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        quiet: bool = False,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if not quiet:
            self._debug(f"[HTTP][local] -> {method.upper()} {path}")
        try:
            response = self.local.request(method, path, json=json_body)
        except Exception as exc:
            elapsed = (time.perf_counter() - started) * 1000
            self._emit("ERROR", f"[HTTP][local] {method.upper()} {path} failed after {elapsed:.0f} ms: {type(exc).__name__}: {exc}")
            raise
        elapsed = (time.perf_counter() - started) * 1000
        if not quiet:
            self._debug(f"[HTTP][local] <- {response.status_code} {method.upper()} {path} ({elapsed:.0f} ms)")
        try:
            return self._json(response, f"local API {path}")
        except Exception as exc:
            self._emit("ERROR", f"[HTTP][local] invalid/error response for {path}: {exc}")
            raise

    def heartbeat(self) -> dict[str, Any]:
        health = self._local("GET", "health", quiet=True)
        result = self._remote(
            "POST",
            "bakery-worker-heartbeat",
            quiet=True,
            transient_retries=2,
            json_body={
                "worker_id": self.worker_id,
                "ready": bool(health.get("ready")),
                "r2_configured": bool(health.get("r2_configured")),
                "kiotviet_configured": bool(health.get("kiotviet_configured")),
                "manufacturing_configured": bool(
                    health.get("manufacturing_configured")
                ),
                "kiotviet_auto_create_draft": bool(
                    health.get("kiotviet_auto_create_draft")
                ),
                "model": health.get("model"),
                "max_images_per_job": health.get("max_images_per_job", 50),
                "max_image_size_mb": health.get("max_image_size_mb", 50),
                "max_job_upload_size_mb": health.get(
                    "max_job_upload_size_mb", 200
                ),
                "allowed_image_extensions": health.get(
                    "allowed_image_extensions", []
                ),
            },
        )
        model = health.get("model") if isinstance(health.get("model"), dict) else {}
        yolo = model.get("yolo") if isinstance(model.get("yolo"), dict) else model
        signature = (
            bool(health.get("ready")),
            bool(health.get("r2_configured")),
            bool(health.get("kiotviet_configured")),
            bool(health.get("manufacturing_configured")),
            str(yolo.get("model_path") or ""),
            int(yolo.get("image_size") or 0),
            float(yolo.get("iou") or 0.0),
        )
        now = time.monotonic()
        if signature != self._last_health_signature or now - self._last_heartbeat_log >= 30:
            model_name = Path(str(yolo.get("model_path") or "-")).name
            self._emit(
                "INFO",
                "[HEARTBEAT] "
                f"ready={health.get('ready')} r2={health.get('r2_configured')} "
                f"kiotviet={health.get('kiotviet_configured')} "
                f"manufacturing={health.get('manufacturing_configured')} model={model_name} "
                f"imgsz={yolo.get('image_size', '-')} iou={yolo.get('iou', '-')}",
            )
            self._last_health_signature = signature
            self._last_heartbeat_log = now
        return result

    def _heartbeat_if_due(self, *, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now < self._next_heartbeat_due:
            return True

        try:
            self.heartbeat()
        except Exception as exc:
            self._heartbeat_failure_count += 1
            # Back off heartbeat retries so a degraded n8n/proxy is not hammered.
            delay = min(
                self.heartbeat_seconds
                * (2 ** min(self._heartbeat_failure_count - 1, 3)),
                60.0,
            )
            self._next_heartbeat_due = time.monotonic() + delay
            warn_now = time.monotonic()
            if (
                self._heartbeat_failure_count == 1
                or warn_now - self._last_heartbeat_warning >= 30.0
            ):
                self._emit(
                    "WARN",
                    "[HEARTBEAT] temporary failure; worker will keep running "
                    f"and retry in {delay:.1f}s: {type(exc).__name__}: {exc}",
                )
                self._last_heartbeat_warning = warn_now
            return False

        recovered = self._heartbeat_failure_count > 0
        self._heartbeat_failure_count = 0
        self._next_heartbeat_due = time.monotonic() + self.heartbeat_seconds
        if recovered:
            self._emit("INFO", "[HEARTBEAT] n8n heartbeat recovered.")
        return True

    def next_task(self) -> dict[str, Any] | None:
        payload = self._remote(
            "GET",
            "bakery-worker-next",
            params={"worker_id": self.worker_id},
            quiet=True,
            transient_retries=1,
        )
        task = payload.get("task")
        if isinstance(task, dict):
            self._emit("INFO", f"[QUEUE] leased {self._task_context(task)} attempt={task.get('attempts', '-')}")
            return task
        return None

    def report(
        self,
        task: dict[str, Any],
        *,
        ok: bool,
        final: bool,
        result: dict[str, Any] | None = None,
        error: str = "",
    ) -> dict[str, Any]:
        report_body = {
            "worker_id": self.worker_id,
            "task_id": str(task["task_id"]),
            "job_id": str(task.get("job_id") or task["task_id"]),
            "task_type": str(task["task_type"]),
            "ok": ok,
            "final": final,
            "result": result or {},
            "error": error,
        }
        summary = self._decision_summary(result or {}) if isinstance(result, dict) else ""
        self._debug(
            f"[REPORT] {self._task_context(task)} ok={ok} final={final} "
            f"{summary} error={_compact(error, 120) if error else '-'}"
        )
        response = self._remote(
            "POST",
            "bakery-worker-result",
            json_body=report_body,
            quiet=not final,
        )
        if (
            response.get("ok") is not True
            and "lease does not belong" in str(response.get("detail") or "").lower()
        ):
            # Re-register this process and retry once. The n8n workflow only
            # allows its latest heartbeat worker to recover a stale lease.
            self.heartbeat()
            response = self._remote(
                "POST",
                "bakery-worker-result",
                json_body=report_body,
            )
        if response.get("ok") is not True:
            raise RelayWorkerError(
                str(response.get("detail") or "n8n rejected the worker result.")
            )
        return response

    def _run_presign(self, task: dict[str, Any]) -> None:
        started = time.perf_counter()
        payload = dict(task.get("payload") or {})
        payload["job_id"] = str(task.get("job_id") or task["task_id"])
        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        file_preview = ", ".join(str(item.get("filename") or "?") for item in files[:5] if isinstance(item, dict))
        if len(files) > 5:
            file_preview += f", +{len(files) - 5} more"
        self._emit(
            "INFO",
            f"[PRESIGN] START {self._task_context(task)} files={len(files)} "
            f"mode={payload.get('inference_mode', 'AUTO')} [{_compact(file_preview, 220)}]",
        )
        result = self._local("POST", "uploads/presign", json_body=payload)
        uploads = result.get("uploads") if isinstance(result.get("uploads"), list) else []
        elapsed = time.perf_counter() - started
        self._emit(
            "INFO",
            f"[PRESIGN] READY job={result.get('job_id', payload['job_id'])} "
            f"urls={len(uploads)} elapsed={elapsed:.2f}s",
        )
        self.report(task, ok=True, final=True, result=result)

    def _run_process(self, task: dict[str, Any]) -> None:
        task_started = time.perf_counter()
        payload = dict(task.get("payload") or {})
        files = payload.get("files") if isinstance(payload.get("files"), list) else []
        self._emit(
            "INFO",
            f"[PROCESS] START {self._task_context(task)} files={len(files)} "
            f"mode={payload.get('inference_mode', 'AUTO')}",
        )
        # Creating an R2 job should normally return immediately. Keep the
        # heartbeat alive even when an older backend still downloads the batch
        # synchronously before returning the accepted response.
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                self._local, "POST", "jobs/from-r2", json_body=payload
            )
            while not pending.done():
                self._heartbeat_if_due()
                time.sleep(min(self.heartbeat_seconds, 2.0))
            accepted = pending.result()
        self._emit(
            "INFO",
            f"[PROCESS] ACCEPTED job={accepted.get('job_id', task.get('job_id'))} "
            f"status={accepted.get('status', '-')} total={accepted.get('total_images', len(files))} "
            f"message={_compact(accepted.get('message'), 180)}",
        )
        try:
            self.report(task, ok=True, final=False, result=accepted)
        except RelayWorkerError as exc:
            # Losing a progress update must not turn an already accepted local
            # job into an error. The final result remains authoritative.
            print(f"[WORKER] Could not publish progress: {exc}")

        job_id = str(accepted["job_id"])
        deadline = time.monotonic() + self.job_timeout_seconds
        last_reported_processed = int(accepted.get("processed_images") or 0)
        last_reported_status = str(accepted.get("status") or "")
        last_reported_message = str(accepted.get("message") or "")
        while time.monotonic() < deadline:
            self._heartbeat_if_due()
            state = self._local("GET", f"jobs/{job_id}")
            status = str(state.get("status") or "")
            processed_images = int(state.get("processed_images") or 0)
            message = str(state.get("message") or "")
            if (
                processed_images != last_reported_processed
                or status != last_reported_status
                or message != last_reported_message
            ):
                self._emit(
                    "INFO",
                    f"[JOB] job={job_id} status={status or 'PROCESSING'} "
                    f"processed={processed_images}/{int(state.get('total_images') or 0)} "
                    f"message={_compact(message, 220)}",
                )
            # Terminal states must be reported with the COMPLETE linked payload
            # before any lightweight progress update. Otherwise n8n can briefly
            # expose AWAITING_CONFIRMATION/NEEDS_RETAKE without `decision`,
            # `images` or `product_catalog`, and the browser renders a false
            # AMBIGUOUS result.
            if status in {
                "AWAITING_CONFIRMATION",
                "NEEDS_RETAKE",
                "COMPLETED",
            }:
                linked = self._local("GET", f"jobs/{job_id}/links")
                self._emit(
                    "INFO",
                    f"[PROCESS] TERMINAL job={job_id} {self._decision_summary(linked)} "
                    f"images={len(linked.get('images') or [])} r2_objects={len(linked.get('r2_objects') or [])} "
                    f"elapsed={time.perf_counter() - task_started:.2f}s",
                )
                self.report(task, ok=True, final=True, result=linked)
                return
            if status == "ERROR":
                self._emit(
                    "ERROR",
                    f"[PROCESS] FAILED job={job_id} processed={processed_images} "
                    f"error={_compact(state.get('error'), 300)} elapsed={time.perf_counter() - task_started:.2f}s",
                )
                self.report(
                    task,
                    ok=False,
                    final=True,
                    result=state,
                    error=str(state.get("error") or "AI job failed."),
                )
                return

            if (
                processed_images != last_reported_processed
                or status != last_reported_status
                or message != last_reported_message
            ):
                progress = {
                    "job_id": job_id,
                    "status": status or "PROCESSING",
                    "total_images": int(state.get("total_images") or 0),
                    "processed_images": processed_images,
                    "message": message,
                    "error": str(state.get("error") or ""),
                }
                try:
                    self.report(task, ok=True, final=False, result=progress)
                    last_reported_processed = processed_images
                    last_reported_status = status
                    last_reported_message = message
                except RelayWorkerError as exc:
                    # Progress is advisory; the final result remains
                    # authoritative and must still be delivered.
                    print(f"[WORKER] Could not publish image progress: {exc}")
            time.sleep(self.job_poll_seconds)
        raise RelayWorkerError(f"Job {job_id} exceeded the worker timeout.")

    def _run_confirm(self, task: dict[str, Any]) -> None:
        started = time.perf_counter()
        payload = dict(task.get("payload") or {})
        job_id = str(task.get("job_id") or payload.get("job_id") or "")
        if not job_id:
            raise RelayWorkerError("CONFIRM task has no job_id.")

        payload.pop("job_id", None)
        self._emit(
            "INFO",
            f"[CONFIRM] START job={job_id} product_code={payload.get('product_code') or '-'} "
            f"quantity={payload.get('quantity') or '-'} "
            f"document_type={payload.get('document_type') or 'PURCHASE_RECEIPT'} "
            f"confirm={payload.get('confirm', True)}",
        )
        confirm_response = self._local(
            "POST",
            f"jobs/{job_id}/confirm",
            json_body=payload,
        )
        self._debug(
            f"[CONFIRM] local response job={job_id} status={confirm_response.get('status', '-')} "
            f"message={_compact(confirm_response.get('message'), 180)}"
        )

        # After confirmation the backend may create/upload Excel and other
        # artifacts. Report the linked state so remote web clients receive
        # presigned R2 download URLs instead of local-only relative paths.
        linked = self._local(
            "GET",
            f"jobs/{job_id}/links",
        )
        receipt = linked.get("kiotviet") if isinstance(linked.get("kiotviet"), dict) else {}
        receipt_data = receipt.get("receipt") if isinstance(receipt.get("receipt"), dict) else {}
        self._emit(
            "INFO",
            f"[CONFIRM] DONE job={job_id} status={linked.get('status', '-')} "
            f"product={linked.get('confirmed_product', {}).get('product_code', payload.get('product_code', '-')) if isinstance(linked.get('confirmed_product'), dict) else payload.get('product_code', '-')} "
            f"qty={linked.get('total_quantity', payload.get('quantity', '-'))} "
            f"kiotviet_created={receipt.get('created', False)} receipt={receipt_data.get('code', '-')} "
            f"elapsed={time.perf_counter() - started:.2f}s",
        )
        self.report(
            task,
            ok=True,
            final=True,
            result=linked,
        )

    def _run_developer_settings(self, task: dict[str, Any]) -> None:
        payload = dict(task.get("payload") or {})
        action = str(payload.get("action") or "GET").strip().upper()
        developer_key = str(payload.get("developer_key") or "")
        if action == "GET":
            result = self._local(
                "POST",
                "developer/settings/query",
                json_body={"developer_key": developer_key},
            )
        elif action == "UPDATE":
            result = self._local(
                "PUT",
                "developer/settings",
                json_body={
                    "developer_key": developer_key,
                    "active_model": str(payload.get("active_model") or ""),
                    "thresholds": payload.get("thresholds") or {},
                },
            )
        else:
            raise RelayWorkerError(f"Unsupported developer action: {action}")
        self._emit(
            "INFO",
            f"[DEV] {action} completed model={result.get('active_model', '-')}",
        )
        self.report(task, ok=True, final=True, result=result)

    def run_once(self) -> bool:
        # Heartbeat is advisory liveness data. A temporary n8n 502/503/504 must
        # not turn into a worker loop failure or interrupt local inference.
        self._heartbeat_if_due()
        try:
            task = self.next_task()
        except Exception as exc:
            if self._is_transient_remote_error(exc):
                self._emit(
                    "WARN",
                    f"[QUEUE] n8n temporarily unavailable; poll will retry: "
                    f"{type(exc).__name__}: {exc}",
                )
                return False
            raise
        if task is None:
            return False
        started = time.perf_counter()
        try:
            task_type = str(task.get("task_type") or "")
            self._emit("INFO", f"[TASK] BEGIN {self._task_context(task)}")
            if task_type == "PRESIGN":
                self._run_presign(task)
            elif task_type == "PROCESS":
                self._run_process(task)
            elif task_type == "CONFIRM":
                self._run_confirm(task)
            elif task_type == "DEVELOPER_SETTINGS":
                self._run_developer_settings(task)
            else:
                raise RelayWorkerError(f"Unsupported task type: {task_type or 'empty'}.")
            self._emit(
                "INFO",
                f"[TASK] END {self._task_context(task)} elapsed={time.perf_counter() - started:.2f}s",
            )
        except Exception as exc:
            self._emit(
                "ERROR",
                f"[TASK] FAILED {self._task_context(task)} after {time.perf_counter() - started:.2f}s: "
                f"{type(exc).__name__}: {exc}",
            )
            if self.debug:
                self._emit("TRACE", traceback.format_exc(), debug_only=True)
            try:
                self.report(task, ok=False, final=True, error=str(exc))
            except Exception as report_exc:
                self._emit("ERROR", f"[REPORT] Could not publish failure to n8n: {report_exc}")
            raise
        return True

    def run_forever(self) -> None:
        self._emit(
            "INFO",
            "[QUEUE] Waiting for PRESIGN / PROCESS / CONFIRM / DEVELOPER_SETTINGS tasks...",
        )
        while True:
            try:
                worked = self.run_once()
                if not worked:
                    time.sleep(self.poll_seconds)
            except KeyboardInterrupt:
                self._emit("INFO", "Keyboard interrupt received; worker is stopping.")
                return
            except Exception as exc:
                self._emit("ERROR", f"[WORKER] loop error: {type(exc).__name__}: {exc}")
                if self.debug:
                    self._emit("TRACE", traceback.format_exc(), debug_only=True)
                time.sleep(self.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sharon Bakery outbound n8n worker")
    parser.add_argument("--once", action="store_true", help="Poll and process once")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--debug",
        action="store_true",
        default=_env_bool("SHARON_WORKER_DEBUG", False),
        help="Print detailed task, HTTP timing, state-transition and traceback diagnostics.",
    )
    parser.add_argument(
        "--no-debug",
        action="store_false",
        dest="debug",
        help="Disable verbose debug diagnostics even when SHARON_WORKER_DEBUG=1.",
    )
    parser.add_argument(
        "--log-file",
        default=os.getenv("SHARON_WORKER_LOG_FILE", ""),
        help="Optional persistent log file path.",
    )
    arguments = parser.parse_args()
    worker_lock = _acquire_single_worker_lock()
    if worker_lock is None:
        print("[WORKER] Mot outbound worker khac dang chay; khong khoi dong ban trung.", flush=True)
        return
    worker = BakeryRelayWorker(
        poll_seconds=arguments.poll_seconds,
        debug=arguments.debug,
        log_file=arguments.log_file or None,
    )
    try:
        if arguments.once:
            worker.run_once()
        else:
            worker.run_forever()
    finally:
        worker.close()
        worker_lock.close()


if __name__ == "__main__":
    main()
