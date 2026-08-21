"""Authenticated client for the local KiotViet manufacturing RPA service."""

from __future__ import annotations

from typing import Any

import httpx


class ManufacturingError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ManufacturingService:
    """Create one manufacturing receipt after explicit operator confirmation.

    KiotViet's public purchase-order API does not cover the manufacturing UI
    used by Sharon Bakery. A separate, localhost-only Selenium service performs
    that browser action and persists the job ID as an idempotency key.
    """

    def __init__(
        self,
        base_url: str,
        internal_token: str,
        timeout_seconds: float = 180.0,
    ) -> None:
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.internal_token = str(internal_token or "").strip()
        if not self.base_url:
            raise ManufacturingError("MANUFACTURING_RPA_BASE_URL is not configured.")
        if not self.internal_token:
            raise ManufacturingError("MANUFACTURING_RPA_INTERNAL_TOKEN is not configured.")
        self.client = httpx.Client(timeout=max(10.0, float(timeout_seconds)))

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Internal-Token": self.internal_token}

    @staticmethod
    def _message(payload: Any, fallback: str) -> str:
        if not isinstance(payload, dict):
            return fallback
        return str(
            payload.get("detail")
            or payload.get("message")
            or payload.get("error")
            or fallback
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            response = self.client.request(
                method,
                url,
                headers=self.headers,
                json=json_body,
            )
        except httpx.RequestError as exc:
            raise ManufacturingError(
                f"Manufacturing RPA is unreachable at {self.base_url}: {exc}"
            ) from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if response.is_error:
            raise ManufacturingError(
                self._message(
                    payload,
                    f"Manufacturing RPA returned HTTP {response.status_code}.",
                ),
                status_code=response.status_code,
            )
        if not isinstance(payload, dict):
            raise ManufacturingError("Manufacturing RPA returned invalid JSON.")
        return payload

    @staticmethod
    def _normalized_result(payload: dict[str, Any], job_id: str) -> dict[str, Any]:
        status = str(payload.get("status") or "").strip().upper()
        if status not in {"SUCCESS", "COMPLETED"}:
            raise ManufacturingError(
                ManufacturingService._message(
                    payload,
                    "Manufacturing RPA did not confirm receipt creation.",
                )
            )
        created = bool(payload.get("created", True))
        if not created:
            raise ManufacturingError("Manufacturing RPA did not create a receipt.")
        receipt = payload.get("receipt")
        if not isinstance(receipt, dict):
            receipt = {}
        receipt = {
            "request_id": job_id,
            "code": str(receipt.get("code") or payload.get("receipt_code") or "").strip(),
            "url": str(receipt.get("url") or payload.get("receipt_url") or "").strip(),
            **receipt,
        }
        return {
            "dry_run": False,
            "action": "REUSED" if bool(payload.get("reused")) else "CREATED",
            "document_type": "MANUFACTURING",
            "recovered": bool(payload.get("reused")),
            "validation": {
                "rpa": True,
                "request_id": job_id,
                "save_mode": str(payload.get("save_mode") or "COMPLETED").upper(),
            },
            "receipt": receipt,
        }

    def create_manufacturing_receipt(
        self,
        product: dict[str, Any],
        job_id: str,
        branch_id: int,
    ) -> dict[str, Any]:
        quantity = int(product.get("quantity") or 0)
        if quantity <= 0:
            raise ManufacturingError("Manufacturing quantity must be positive.")
        payload = self._request(
            "POST",
            "/run-manufacture",
            json_body={
                "requestId": job_id,
                "branchId": int(branch_id),
                "items": [
                    {
                        "productCode": str(product.get("product_code") or "").strip(),
                        "productName": str(product.get("product_name") or "").strip(),
                        "targetThreshold": quantity,
                    }
                ],
            },
        )
        return self._normalized_result(payload, job_id)

    def reconcile_by_job_id(self, job_id: str) -> dict[str, Any] | None:
        try:
            payload = self._request("GET", f"/manufacture/{job_id}")
        except ManufacturingError as exc:
            if exc.status_code == 404:
                return None
            raise
        status = str(payload.get("status") or "").strip().upper()
        if status in {"SUCCESS", "COMPLETED"}:
            return self._normalized_result(payload, job_id)
        if status in {"PROCESSING", "UNCERTAIN"}:
            raise ManufacturingError(
                self._message(
                    payload,
                    "Manufacturing result is uncertain; inspect KiotViet before retrying.",
                ),
                status_code=409,
            )
        if status == "ERROR" and bool(payload.get("safe_to_retry")):
            return {
                "document_type": "MANUFACTURING",
                "retry_safe": True,
                "validation": {"request_id": job_id, "rpa": True},
            }
        return None

    def check_connection(self) -> dict[str, Any]:
        payload = self._request("GET", "/healthz")
        return {
            "ready": bool(payload.get("ready")),
            "service": "kiotviet-manufacturing-rpa",
            "detail": payload,
        }
