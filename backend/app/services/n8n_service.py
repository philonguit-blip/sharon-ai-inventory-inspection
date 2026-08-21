"""Server-side client for the Sharon Bakery n8n orchestration webhooks."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class N8nOrchestratorError(RuntimeError):
    def __init__(self, detail: str, status_code: int = 502) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class N8nOrchestratorService:
    """Keep n8n credentials on the server and expose only same-origin APIs."""

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        timeout_seconds: float = 30,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("N8N_WEBHOOK_BASE is required.")
        if not username or not password:
            raise ValueError("n8n webhook Basic Auth credentials are required.")
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            auth=httpx.BasicAuth(username, password),
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=False,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = await self._client.request(
                method,
                f"{self.base_url}/{path.lstrip('/')}",
                json=json_body,
                headers={"Accept": "application/json"},
            )
        except httpx.TimeoutException as exc:
            raise N8nOrchestratorError("n8n did not respond before timeout.", 504) from exc
        except httpx.RequestError as exc:
            raise N8nOrchestratorError("Cannot connect to n8n orchestration.") from exc

        try:
            payload = response.json() if response.content else {}
        except ValueError as exc:
            raise N8nOrchestratorError(
                f"n8n returned an invalid response (HTTP {response.status_code})."
            ) from exc

        if not response.is_success:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            message = str(detail or f"n8n request failed (HTTP {response.status_code}).")
            forwarded_status = (
                response.status_code if 400 <= response.status_code < 500 else 502
            )
            raise N8nOrchestratorError(message, forwarded_status)
        if not isinstance(payload, dict) or not payload:
            raise N8nOrchestratorError("n8n returned an empty response.")
        return payload

    async def health(self) -> dict[str, Any]:
        return await self._request_json("GET", "bakery-health")

    @staticmethod
    def _raise_workflow_error(payload: dict[str, Any]) -> None:
        if str(payload.get("status") or "").upper() != "ERROR":
            return
        raise N8nOrchestratorError(
            str(
                payload.get("detail")
                or payload.get("error")
                or "n8n could not complete the request."
            ),
            422,
        )

    async def prepare_uploads(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request_json(
            "POST", "bakery-upload-init", json_body=payload
        )
        self._raise_workflow_error(response)
        if response.get("uploads"):
            return response
        request_id = str(response.get("request_id") or response.get("job_id") or "")
        if not request_id:
            raise N8nOrchestratorError("n8n did not return an upload request ID.")
        for _ in range(90):
            await asyncio.sleep(0.5)
            status = await self._request_json(
                "GET", f"bakery-request-status?request_id={request_id}"
            )
            self._raise_workflow_error(status)
            if status.get("status") in {"READY", "COMPLETED"} and status.get(
                "uploads"
            ):
                return status
        raise N8nOrchestratorError("AI worker did not prepare upload URLs in time.", 504)

    async def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._request_json("POST", "bakery-submit", json_body=payload)
        self._raise_workflow_error(response)
        return response

    async def job_status(self, job_id: str) -> dict[str, Any]:
        return await self._request_json(
            "GET", f"bakery-job-status?job_id={job_id}"
        )

    async def confirm_job(
        self,
        job_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        request_payload = dict(payload)
        request_payload["job_id"] = job_id
        response = await self._request_json(
            "POST",
            "bakery-confirm",
            json_body=request_payload,
        )
        self._raise_workflow_error(response)
        return response
