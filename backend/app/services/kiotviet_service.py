"""KiotViet Retail Public API client for draft purchase receipts."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import httpx

from app.config import (
    KIOTVIET_BRANCH_NAME,
    KIOTVIET_CLIENT_ID,
    KIOTVIET_CLIENT_SECRET,
    KIOTVIET_CREATE_AS_DRAFT,
    KIOTVIET_DEFAULT_PURCHASE_PRICE,
    KIOTVIET_PURCHASE_BY_USERNAME,
    KIOTVIET_RETAILER,
    KIOTVIET_SUPPLIER_CODE,
)


TOKEN_URL = "https://id.kiotviet.vn/connect/token"
API_BASE_URL = "https://public.kiotapi.com"
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))


class KiotVietError(RuntimeError):
    pass


class KiotVietService:
    def __init__(self, timeout_seconds: float = 30.0) -> None:
        missing = [
            name
            for name, value in {
                "KIOTVIET_RETAILER": KIOTVIET_RETAILER,
                "KIOTVIET_CLIENT_ID": KIOTVIET_CLIENT_ID,
                "KIOTVIET_CLIENT_SECRET": KIOTVIET_CLIENT_SECRET,
                "KIOTVIET_BRANCH_NAME": KIOTVIET_BRANCH_NAME,
            }.items()
            if not value
        ]
        if missing:
            raise KiotVietError(
                "Missing KiotViet configuration: " + ", ".join(missing)
            )
        self.retailer = KIOTVIET_RETAILER
        self.branch_name = KIOTVIET_BRANCH_NAME
        self.supplier_code = KIOTVIET_SUPPLIER_CODE
        self.create_as_draft = KIOTVIET_CREATE_AS_DRAFT
        self.purchase_price = float(KIOTVIET_DEFAULT_PURCHASE_PRICE)
        self.purchase_by_username = KIOTVIET_PURCHASE_BY_USERNAME
        self.client_id = KIOTVIET_CLIENT_ID
        self.client_secret = KIOTVIET_CLIENT_SECRET
        self.client = httpx.Client(timeout=timeout_seconds)
        self._token = ""
        self._token_expires_at = 0.0
        self._token_lock = threading.Lock()

    def _access_token(self) -> str:
        if self._token and time.monotonic() < self._token_expires_at:
            return self._token
        with self._token_lock:
            if self._token and time.monotonic() < self._token_expires_at:
                return self._token
            try:
                response = self.client.post(
                    TOKEN_URL,
                    data={
                        "scopes": "PublicApi.Access",
                        "grant_type": "client_credentials",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                raise KiotVietError("KiotViet authentication failed.") from exc

            token = str(payload.get("access_token") or "")
            if not token:
                raise KiotVietError("KiotViet token response has no access_token.")
            expires_in = max(60, int(payload.get("expires_in", 86400)))
            self._token = token
            self._token_expires_at = time.monotonic() + expires_in - 60
            return token

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        headers = {
            "Retailer": self.retailer,
            "Authorization": f"Bearer {self._access_token()}",
            "Accept": "application/json",
        }
        try:
            response = self.client.request(
                method,
                f"{API_BASE_URL}{path}",
                params=params,
                json=json_body,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            detail = exc.response.text[:500]
            raise KiotVietError(
                f"KiotViet API returned HTTP {status_code}: {detail}"
            ) from exc
        except Exception as exc:
            raise KiotVietError(f"KiotViet API request failed: {method} {path}") from exc

    def resolve_branch(self) -> dict[str, Any]:
        payload = self._request("GET", "/branches", params={"pageSize": 100})
        branches = payload.get("data", []) if isinstance(payload, dict) else []
        matches = [
            branch
            for branch in branches
            if str(branch.get("branchName", "")).strip().casefold()
            == self.branch_name.casefold()
        ]
        if len(matches) != 1:
            raise KiotVietError(
                f"Expected exactly one KiotViet branch named {self.branch_name!r}; "
                f"found {len(matches)}."
            )
        return matches[0]

    def get_product_by_code(self, product_code: str) -> dict[str, Any]:
        payload = self._request(
            "GET", f"/products/code/{quote(product_code, safe='')}"
        )
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            payload = payload["data"]
        if not isinstance(payload, dict):
            raise KiotVietError(f"Product not found in KiotViet: {product_code}")
        actual_code = str(payload.get("code") or "").strip()
        if actual_code.casefold() != product_code.casefold():
            raise KiotVietError(
                f"KiotViet returned the wrong product for code {product_code}."
            )
        return payload

    def resolve_purchase_user(self) -> dict[str, Any] | None:
        if not self.purchase_by_username:
            return None
        payload = self._request("GET", "/users", params={"pageSize": 100})
        users = payload.get("data", []) if isinstance(payload, dict) else []
        target = self.purchase_by_username.casefold()
        matches = [
            user
            for user in users
            if str(user.get("userName") or "").strip().casefold() == target
            or str(user.get("givenName") or "").strip().casefold() == target
        ]
        if len(matches) != 1:
            raise KiotVietError(
                "Expected exactly one KiotViet purchase user named "
                f"{self.purchase_by_username!r}; found {len(matches)}."
            )
        return matches[0]

    def build_purchase_receipt_payload(
        self, products: list[dict[str, Any]], job_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not products:
            raise KiotVietError("Cannot create a purchase receipt without products.")
        branch = self.resolve_branch()
        purchase_user = self.resolve_purchase_user()
        validated_products: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []

        for item in products:
            product_code = str(item.get("product_code") or "").strip()
            expected_name = str(item.get("product_name") or "").strip()
            quantity = int(item.get("quantity") or 0)
            if not product_code or not expected_name or quantity <= 0:
                raise KiotVietError("Job contains invalid product data.")
            live_product = self.get_product_by_code(product_code)
            live_name = str(
                live_product.get("fullName") or live_product.get("name") or ""
            ).strip()
            if live_name.casefold() != expected_name.casefold():
                raise KiotVietError(
                    f"Product name mismatch for {product_code}: "
                    f"job={expected_name!r}, KiotViet={live_name!r}."
                )
            validated_products.append(
                {
                    "id": live_product.get("id"),
                    "code": product_code,
                    "name": live_name,
                    "quantity": quantity,
                }
            )
            details.append(
                {
                    "productCode": product_code,
                    "description": f"AI inventory inspection job {job_id}",
                    "quantity": quantity,
                    "price": self.purchase_price,
                    "discount": 0,
                    "discountRatio": 0,
                }
            )

        body: dict[str, Any] = {
            "purchaseDate": datetime.now(VIETNAM_TIMEZONE).isoformat(),
            "branchId": int(branch["id"]),
            "description": f"AI inventory inspection job {job_id}",
            "isDraft": 1 if self.create_as_draft else 0,
            "discount": 0,
            "discountRatio": 0,
            "paidAmount": 0,
            "paymentMethod": "Cash",
            "isApplyPurchaseTax": False,
            "purchaseOrderDetails": details,
        }
        if self.supplier_code:
            body["supplier"] = {"code": self.supplier_code}
        if purchase_user is not None:
            body["purchaseById"] = int(purchase_user["id"])

        validation = {
            "retailer": self.retailer,
            "branch": {
                "id": int(branch["id"]),
                "name": str(branch.get("branchName") or ""),
                "code": str(branch.get("branchCode") or ""),
            },
            "products": validated_products,
            "supplier_code": self.supplier_code,
            "purchase_by": (
                {
                    "id": int(purchase_user["id"]),
                    "username": str(purchase_user.get("userName") or ""),
                    "name": str(purchase_user.get("givenName") or ""),
                }
                if purchase_user is not None
                else None
            ),
            "is_draft": self.create_as_draft,
        }
        return body, validation

    def preview_purchase_receipt(
        self, products: list[dict[str, Any]], job_id: str
    ) -> dict[str, Any]:
        body, validation = self.build_purchase_receipt_payload(products, job_id)
        return {"dry_run": True, "validation": validation, "payload": body}

    def create_purchase_receipt(
        self, products: list[dict[str, Any]], job_id: str
    ) -> dict[str, Any]:
        body, validation = self.build_purchase_receipt_payload(products, job_id)
        response = self._request("POST", "/purchaseorders", json_body=body)
        if not isinstance(response, dict):
            raise KiotVietError("KiotViet returned an invalid purchase receipt response.")
        return {
            "dry_run": False,
            "validation": validation,
            "receipt": response,
        }

    def check_connection(self) -> dict[str, Any]:
        branch = self.resolve_branch()
        return {
            "ready": True,
            "retailer": self.retailer,
            "branch_id": int(branch["id"]),
            "branch_name": str(branch.get("branchName") or ""),
            "is_draft": self.create_as_draft,
        }
