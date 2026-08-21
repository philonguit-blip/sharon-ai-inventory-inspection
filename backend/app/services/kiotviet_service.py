"""KiotViet Retail Public API client for daily purchase receipts."""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
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
    KIOTVIET_MERGE_DAILY_DRAFTS,
    KIOTVIET_PURCHASE_BY_USERNAME,
    KIOTVIET_REPLACE_COMPLETED_ON_UPDATE_FAILURE,
    KIOTVIET_RETAILER,
    KIOTVIET_SUPPLIER_CODE,
)


TOKEN_URL = "https://id.kiotviet.vn/connect/token"
API_BASE_URL = "https://public.kiotapi.com"
VIETNAM_TIMEZONE = timezone(timedelta(hours=7))
# KiotViet validates purchaseDate against its current retailer-local time.
# Keep a small safety margin so minor clock skew cannot make a newly-created
# receipt appear to be in the future.
KIOTVIET_CLOCK_SAFETY_SECONDS = 5
LOGGER = logging.getLogger(__name__)
DAILY_DESCRIPTION_PREFIX = "AI inventory inspection daily"
LEGACY_JOB_DESCRIPTION_PREFIX = "AI inventory inspection job"
DAILY_JOB_MARKER_LIMIT = 10
# KiotViet purchase-order status values: 1 = draft, 2 = cancelled,
# 3 = completed. Both draft and completed daily receipts are candidates for
# same-day merging; cancelled receipts must never be reused.
ACTIVE_PURCHASE_RECEIPT_STATUSES = {1, 3}
REPLACEMENT_MARKER_PATTERN = re.compile(r"AIR:([0-9]+)")


def normalize_product_name(value: str | None) -> str:
    """Normalize a product name for non-blocking comparison only."""
    if not value:
        return ""

    normalized = unicodedata.normalize(
        "NFKC",
        str(value),
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip().casefold()


class KiotVietError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class KiotVietService:
    def __init__(
        self,
        timeout_seconds: float = 30.0,
    ) -> None:
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
                "Missing KiotViet configuration: "
                + ", ".join(missing)
            )

        self.retailer = KIOTVIET_RETAILER
        self.branch_name = KIOTVIET_BRANCH_NAME
        self.supplier_code = KIOTVIET_SUPPLIER_CODE
        self.create_as_draft = KIOTVIET_CREATE_AS_DRAFT
        self.merge_daily_drafts = KIOTVIET_MERGE_DAILY_DRAFTS
        self.replace_completed_on_update_failure = (
            KIOTVIET_REPLACE_COMPLETED_ON_UPDATE_FAILURE
        )
        self.purchase_price = float(
            KIOTVIET_DEFAULT_PURCHASE_PRICE
        )
        self.purchase_by_username = (
            KIOTVIET_PURCHASE_BY_USERNAME
        )
        self.client_id = KIOTVIET_CLIENT_ID
        self.client_secret = KIOTVIET_CLIENT_SECRET

        self.client = httpx.Client(
            timeout=timeout_seconds
        )

        self._token = ""
        self._token_expires_at = 0.0
        self._token_lock = threading.Lock()

    @staticmethod
    def _unwrap_data(payload: Any) -> Any:
        if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
            return payload["data"]
        return payload

    @staticmethod
    def _current_purchase_datetime() -> str:
        """Return a KiotViet-safe Vietnam local datetime string.

        The purchase-order API works with retailer-local ``datetime`` values.
        Sending an offset-aware timestamp very close to the current instant can
        trigger ``KvValidatePurchaseOrderException: Vượt quá thời gian hiện tại``
        on some KiotViet deployments.  Use a timezone-free Vietnam wall-clock
        value and backdate it by a few seconds to absorb harmless clock skew.
        """
        current = datetime.now(VIETNAM_TIMEZONE) - timedelta(
            seconds=KIOTVIET_CLOCK_SAFETY_SECONDS
        )
        return current.replace(tzinfo=None).isoformat(timespec="seconds")

    @staticmethod
    def _purchase_date(value: Any) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=VIETNAM_TIMEZONE)
        return parsed.astimezone(VIETNAM_TIMEZONE)

    @staticmethod
    def _job_marker(job_id: str) -> str:
        compact = re.sub(r"[^0-9a-f]", "", str(job_id).casefold())[:16]
        if len(compact) != 16:
            raise KiotVietError("Invalid job ID for KiotViet receipt marker.")
        return f"AIJ:{compact}"

    @classmethod
    def _description_contains_job(cls, description: Any, job_id: str) -> bool:
        text = str(description or "")
        return (
            f"{LEGACY_JOB_DESCRIPTION_PREFIX} {job_id}" in text
            or cls._job_marker(job_id) in text
        )

    @classmethod
    def _daily_description(
        cls,
        existing_description: Any,
        job_id: str,
        business_date: Any,
    ) -> str:
        text = str(existing_description or "")
        markers = [
            f"AIJ:{item.casefold()}"
            for item in re.findall(r"AIJ:([0-9a-f]{16})", text, flags=re.IGNORECASE)
        ]
        legacy_jobs = re.findall(
            rf"{re.escape(LEGACY_JOB_DESCRIPTION_PREFIX)}\s+([0-9a-f]{{32}})",
            text,
            flags=re.IGNORECASE,
        )
        markers.extend(f"AIJ:{item[:16].casefold()}" for item in legacy_jobs)
        markers.append(cls._job_marker(job_id))
        unique = list(dict.fromkeys(markers))[-DAILY_JOB_MARKER_LIMIT:]
        return (
            f"{DAILY_DESCRIPTION_PREFIX} {business_date.isoformat()} | "
            + " ".join(unique)
        )

    def _access_token(self) -> str:
        if (
            self._token
            and time.monotonic()
            < self._token_expires_at
        ):
            return self._token

        with self._token_lock:
            if (
                self._token
                and time.monotonic()
                < self._token_expires_at
            ):
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
                    headers={
                        "Content-Type": (
                            "application/"
                            "x-www-form-urlencoded"
                        )
                    },
                )

                response.raise_for_status()
                payload = response.json()

            except Exception as exc:
                raise KiotVietError(
                    "KiotViet authentication failed."
                ) from exc

            token = str(
                payload.get("access_token") or ""
            )

            if not token:
                raise KiotVietError(
                    "KiotViet token response "
                    "has no access_token."
                )

            expires_in = max(
                60,
                int(
                    payload.get(
                        "expires_in",
                        86400,
                    )
                ),
            )

            self._token = token

            self._token_expires_at = (
                time.monotonic()
                + expires_in
                - 60
            )

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
            "Authorization": (
                f"Bearer {self._access_token()}"
            ),
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
                "KiotViet API returned "
                f"HTTP {status_code}: {detail}",
                status_code=status_code,
            ) from exc

        except Exception as exc:
            raise KiotVietError(
                "KiotViet API request failed: "
                f"{method} {path}"
            ) from exc

    def resolve_branch(
        self,
    ) -> dict[str, Any]:
        payload = self._request(
            "GET",
            "/branches",
            params={
                "pageSize": 100,
            },
        )

        branches = (
            payload.get("data", [])
            if isinstance(payload, dict)
            else []
        )

        matches = [
            branch
            for branch in branches
            if (
                str(
                    branch.get(
                        "branchName",
                        "",
                    )
                )
                .strip()
                .casefold()
                == self.branch_name.casefold()
            )
        ]

        if len(matches) != 1:
            raise KiotVietError(
                "Expected exactly one KiotViet "
                f"branch named {self.branch_name!r}; "
                f"found {len(matches)}."
            )

        return matches[0]

    def get_product_by_code(
        self,
        product_code: str,
    ) -> dict[str, Any]:
        payload = self._request(
            "GET",
            (
                "/products/code/"
                f"{quote(product_code, safe='')}"
            ),
        )

        if (
            isinstance(payload, dict)
            and isinstance(
                payload.get("data"),
                dict,
            )
        ):
            payload = payload["data"]

        if not isinstance(payload, dict):
            raise KiotVietError(
                "Product not found in KiotViet: "
                f"{product_code}"
            )

        actual_code = str(
            payload.get("code") or ""
        ).strip()

        if (
            actual_code.casefold()
            != product_code.casefold()
        ):
            raise KiotVietError(
                "KiotViet returned the wrong "
                f"product for code {product_code}."
            )

        return payload

    def resolve_purchase_user(
        self,
    ) -> dict[str, Any] | None:
        if not self.purchase_by_username:
            return None

        payload = self._request(
            "GET",
            "/users",
            params={
                "pageSize": 100,
            },
        )

        users = (
            payload.get("data", [])
            if isinstance(payload, dict)
            else []
        )

        target = (
            self.purchase_by_username.casefold()
        )

        matches = [
            user
            for user in users
            if (
                str(
                    user.get("userName") or ""
                )
                .strip()
                .casefold()
                == target
            )
            or (
                str(
                    user.get("givenName") or ""
                )
                .strip()
                .casefold()
                == target
            )
        ]

        if len(matches) != 1:
            raise KiotVietError(
                "Expected exactly one KiotViet "
                "purchase user named "
                f"{self.purchase_by_username!r}; "
                f"found {len(matches)}."
            )

        return matches[0]

    def build_purchase_receipt_payload(
        self,
        products: list[dict[str, Any]],
        job_id: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
    ]:
        if not products:
            raise KiotVietError(
                "Cannot create a purchase "
                "receipt without products."
            )

        branch = self.resolve_branch()
        purchase_user = (
            self.resolve_purchase_user()
        )

        validated_products: list[
            dict[str, Any]
        ] = []

        validation_warnings: list[str] = []
        details: list[dict[str, Any]] = []

        for item in products:
            product_code = str(
                item.get("product_code") or ""
            ).strip()

            expected_name = str(
                item.get("product_name") or ""
            ).strip()

            quantity = int(
                item.get("quantity") or 0
            )

            if (
                not product_code
                or not expected_name
                or quantity <= 0
            ):
                raise KiotVietError(
                    "Job contains invalid product data."
                )

            # Resolve the live KiotViet product by
            # product code.
            #
            # Product code and product ID are strict
            # identifiers.
            #
            # Display-name differences are warnings
            # only because names may be edited in
            # KiotViet without changing the
            # underlying SKU.
            live_product = (
                self.get_product_by_code(
                    product_code
                )
            )

            live_product_id = (
                live_product.get("id")
            )

            try:
                live_product_id = int(live_product_id)
            except (TypeError, ValueError) as exc:
                raise KiotVietError(
                    "KiotViet product "
                    f"{product_code} "
                    "has an invalid product ID."
                ) from exc

            if live_product_id <= 0:
                raise KiotVietError(
                    "KiotViet product "
                    f"{product_code} "
                    "has an invalid product ID: "
                    f"{live_product_id}."
                )

            if (
                live_product.get(
                    "isActive",
                    True,
                )
                is False
            ):
                raise KiotVietError(
                    "KiotViet product is inactive: "
                    f"{product_code}."
                )

            live_name = str(
                live_product.get("fullName")
                or live_product.get("name")
                or ""
            ).strip()

            if not live_name:
                raise KiotVietError(
                    "KiotViet product "
                    f"{product_code} "
                    "has no product name."
                )

            name_matches = (
                normalize_product_name(
                    live_name
                )
                == normalize_product_name(
                    expected_name
                )
            )

            if not name_matches:
                warning = (
                    "Product name differs for "
                    f"{product_code}: "
                    f"job={expected_name!r}, "
                    f"KiotViet={live_name!r}. "
                    "Receipt creation continues "
                    "because the product code "
                    "matched."
                )

                validation_warnings.append(
                    warning
                )

                LOGGER.warning(warning)

            validated_products.append(
                {
                    "id": int(
                        live_product_id
                    ),
                    "code": product_code,
                    "job_name": expected_name,
                    "name": live_name,
                    "name_matches": (
                        name_matches
                    ),
                    "quantity": quantity,
                }
            )

            details.append(
                {
                    # KiotViet purchase-order validation requires the
                    # live product identifier as well as the product code.
                    # Sending only productCode can be normalized by some
                    # KiotViet deployments to ProductId=0 and rejected with
                    # KvValidateProductException.
                    "productId": live_product_id,
                    "productCode": (
                        product_code
                    ),
                    "description": (
                        "AI inventory inspection "
                        f"job {job_id}"
                    ),
                    "quantity": quantity,
                    "price": (
                        self.purchase_price
                    ),
                    "discount": 0,
                    "discountRatio": 0,
                }
            )

        body: dict[str, Any] = {
            "purchaseDate": self._current_purchase_datetime(),
            "branchId": int(
                branch["id"]
            ),
            "description": (
                "AI inventory inspection "
                f"job {job_id}"
            ),
            "isDraft": (
                1
                if self.create_as_draft
                else 0
            ),
            "discount": 0,
            "discountRatio": 0,
            "paidAmount": 0,
            "paymentMethod": "Cash",
            "isApplyPurchaseTax": False,
            "purchaseOrderDetails": details,
        }

        if self.supplier_code:
            body["supplier"] = {
                "code": self.supplier_code
            }

        if purchase_user is not None:
            body["purchaseById"] = int(
                purchase_user["id"]
            )

        validation = {
            "retailer": self.retailer,
            "branch": {
                "id": int(
                    branch["id"]
                ),
                "name": str(
                    branch.get(
                        "branchName"
                    )
                    or ""
                ),
                "code": str(
                    branch.get(
                        "branchCode"
                    )
                    or ""
                ),
            },
            "products": validated_products,
            "warnings": validation_warnings,
            "supplier_code": (
                self.supplier_code
            ),
            "purchase_by": (
                {
                    "id": int(
                        purchase_user["id"]
                    ),
                    "username": str(
                        purchase_user.get(
                            "userName"
                        )
                        or ""
                    ),
                    "name": str(
                        purchase_user.get(
                            "givenName"
                        )
                        or ""
                    ),
                }
                if purchase_user is not None
                else None
            ),
            "is_draft": (
                self.create_as_draft
            ),
        }

        return body, validation

    def preview_purchase_receipt(
        self,
        products: list[dict[str, Any]],
        job_id: str,
    ) -> dict[str, Any]:
        body, validation = (
            self.build_purchase_receipt_payload(
                products,
                job_id,
            )
        )

        return {
            "dry_run": True,
            "validation": validation,
            "payload": body,
            "daily_merge_enabled": bool(self.merge_daily_drafts),
        }

    def _purchase_receipt_detail(self, receipt_id: Any) -> dict[str, Any]:
        if receipt_id in (None, ""):
            raise KiotVietError("KiotViet purchase receipt has no ID.")
        payload = self._request("GET", f"/purchaseorders/{int(receipt_id)}")
        detail = self._unwrap_data(payload)
        if not isinstance(detail, dict):
            raise KiotVietError("KiotViet returned an invalid purchase receipt detail.")
        return detail

    def _purchase_receipts_for_date(self, business_date: Any) -> list[dict[str, Any]]:
        branch = self.resolve_branch()
        params: dict[str, Any] = {
            "branchIds": [int(branch["id"])],
            "status": sorted(ACTIVE_PURCHASE_RECEIPT_STATUSES),
            "fromPurchaseDate": business_date.isoformat(),
            "toPurchaseDate": (business_date + timedelta(days=1)).isoformat(),
            "pageSize": 100,
            "currentItem": 0,
        }
        details: list[dict[str, Any]] = []
        for _ in range(10):
            payload = self._request("GET", "/purchaseorders", params=params)
            records = payload.get("data", []) if isinstance(payload, dict) else []
            if not isinstance(records, list):
                raise KiotVietError("KiotViet returned an invalid purchase receipt list.")
            for record in records:
                if not isinstance(record, dict) or record.get("id") in (None, ""):
                    continue
                detail = self._purchase_receipt_detail(record["id"])
                detail = {**record, **detail}
                purchase_date = self._purchase_date(detail.get("purchaseDate"))
                if purchase_date is None or purchase_date.date() != business_date:
                    continue
                if int(detail.get("branchId") or 0) != int(branch["id"]):
                    continue
                if int(detail.get("status") or 0) not in ACTIVE_PURCHASE_RECEIPT_STATUSES:
                    continue
                details.append(detail)

            params["currentItem"] = int(params["currentItem"]) + len(records)
            total = int(payload.get("total") or 0) if isinstance(payload, dict) else 0
            if not records or int(params["currentItem"]) >= total:
                break
        return details

    def find_daily_purchase_receipt(
        self,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Return the one active AI-managed receipt for today's quantities."""
        current = now or datetime.now(VIETNAM_TIMEZONE)
        if current.tzinfo is None:
            current = current.replace(tzinfo=VIETNAM_TIMEZONE)
        business_date = current.astimezone(VIETNAM_TIMEZONE).date()
        expected_daily_prefix = f"{DAILY_DESCRIPTION_PREFIX} {business_date.isoformat()}"
        daily: list[dict[str, Any]] = []
        legacy: list[dict[str, Any]] = []

        for receipt in self._purchase_receipts_for_date(business_date):
            if self.supplier_code:
                supplier_code = str(receipt.get("supplierCode") or "").strip()
                if supplier_code.casefold() != self.supplier_code.casefold():
                    continue
            description = str(receipt.get("description") or "").strip()
            if description.startswith(expected_daily_prefix):
                daily.append(receipt)
            elif description.startswith(f"{LEGACY_JOB_DESCRIPTION_PREFIX} "):
                legacy.append(receipt)

        if len(daily) > 1:
            codes = ", ".join(str(item.get("code") or item.get("id")) for item in daily)
            raise KiotVietError(
                f"Multiple active AI daily receipts exist for {business_date}: {codes}."
            )
        if daily:
            return daily[0]

        # Adopt the earliest legacy AI receipt so deployments made during a day
        # continue using an existing receipt instead of creating another one.
        if legacy:
            legacy.sort(
                key=lambda item: (
                    self._purchase_date(item.get("purchaseDate"))
                    or datetime.max.replace(tzinfo=VIETNAM_TIMEZONE),
                    int(item.get("id") or 0),
                )
            )
            if len(legacy) > 1:
                LOGGER.warning(
                    "Found %s legacy AI receipts for %s; adopting the earliest (%s).",
                    len(legacy),
                    business_date,
                    legacy[0].get("code") or legacy[0].get("id"),
                )
            return legacy[0]
        return None

    @staticmethod
    def _request_quantity(value: Any) -> int | float:
        quantity = float(value or 0)
        if quantity <= 0:
            raise KiotVietError("Purchase receipt contains a non-positive quantity.")
        return int(quantity) if quantity.is_integer() else quantity

    def _merge_purchase_order_details(
        self,
        existing_details: Any,
        incoming_details: Any,
    ) -> list[dict[str, Any]]:
        """Merge daily receipt lines without ever dropping KiotViet productId.

        Older AI-created receipt details may contain only ``productCode``.
        Before PUT/POST, resolve any missing/zero productId from live KiotViet
        so the API never receives ProductId=0.
        """
        merged: dict[str, dict[str, Any]] = {}

        def resolve_product_id(item: dict[str, Any], code: str) -> int:
            raw_product_id = item.get("productId")
            if raw_product_id in (None, ""):
                raw_product_id = item.get("ProductId")
            if raw_product_id in (None, ""):
                raw_product_id = item.get("product_id")

            if raw_product_id not in (None, ""):
                try:
                    product_id = int(raw_product_id)
                except (TypeError, ValueError):
                    product_id = 0
                if product_id > 0:
                    return product_id

            # Existing purchase-order lines from older backend versions can
            # lack productId. Resolve it again by the strict business code.
            live_product = self.get_product_by_code(code)
            live_product_id = live_product.get("id")
            try:
                product_id = int(live_product_id)
            except (TypeError, ValueError) as exc:
                raise KiotVietError(
                    f"KiotViet product {code} has an invalid product ID."
                ) from exc

            if product_id <= 0:
                raise KiotVietError(
                    f"KiotViet product {code} has an invalid product ID: "
                    f"{product_id}."
                )

            if live_product.get("isActive", True) is False:
                raise KiotVietError(
                    f"KiotViet product is inactive: {code}."
                )

            return product_id

        def add(item: Any, *, existing: bool) -> None:
            if not isinstance(item, dict):
                raise KiotVietError(
                    "KiotViet purchase receipt contains an invalid line."
                )

            code = str(
                item.get("productCode")
                or item.get("ProductCode")
                or ""
            ).strip()

            if not code:
                raise KiotVietError(
                    "KiotViet purchase receipt line has no product code."
                )

            if existing and item.get("purchaseOrderDetailTaxes"):
                raise KiotVietError(
                    "Cannot merge an AI job into a taxed purchase receipt detail."
                )

            product_id = resolve_product_id(item, code)
            quantity = self._request_quantity(item.get("quantity"))
            key = code.casefold()

            if key in merged:
                existing_product_id = int(merged[key]["productId"])
                if existing_product_id != product_id:
                    raise KiotVietError(
                        "KiotViet returned conflicting product IDs for "
                        f"{code}: {existing_product_id} vs {product_id}."
                    )

                merged[key]["quantity"] = self._request_quantity(
                    float(merged[key]["quantity"]) + float(quantity)
                )
                return

            merged[key] = {
                "productId": product_id,
                "productCode": code,
                "description": str(item.get("description") or ""),
                "quantity": quantity,
                "price": float(item.get("price") or 0),
                "discount": float(item.get("discount") or 0),
                "discountRatio": float(item.get("discountRatio") or 0),
            }

            if isinstance(item.get("productTax"), dict):
                merged[key]["productTax"] = dict(item["productTax"])

        for detail in existing_details or []:
            add(detail, existing=True)

        for detail in incoming_details or []:
            add(detail, existing=False)

        if not merged:
            raise KiotVietError(
                "Cannot update an empty KiotViet purchase receipt."
            )

        # Final defensive gate: never send ProductId=0 to KiotViet.
        for detail in merged.values():
            if int(detail.get("productId") or 0) <= 0:
                raise KiotVietError(
                    "Refusing to send KiotViet purchase receipt detail "
                    f"without a valid productId: {detail.get('productCode')}."
                )

        return list(merged.values())

    def _daily_update_body(
        self,
        existing: dict[str, Any],
        incoming: dict[str, Any],
        job_id: str,
        business_date: Any,
    ) -> dict[str, Any]:
        if float(existing.get("discount") or 0) != 0 or float(existing.get("discountRatio") or 0) != 0:
            raise KiotVietError("Cannot merge into a discounted AI purchase receipt.")
        if float(existing.get("totalPayment") or existing.get("paidAmount") or 0) != 0:
            raise KiotVietError("Cannot merge into an AI purchase receipt that has been paid.")
        if existing.get("payments"):
            raise KiotVietError("Cannot merge into an AI purchase receipt with payments.")

        body = dict(incoming)
        # The official update contract keeps the existing purchaser and does
        # not accept purchaseById, even though the create contract does.
        body.pop("purchaseById", None)
        body["purchaseDate"] = str(existing.get("purchaseDate") or incoming["purchaseDate"])
        body["isDraft"] = 1 if self.create_as_draft else 0
        body["description"] = self._daily_description(
            existing.get("description"), job_id, business_date
        )
        body["purchaseOrderDetails"] = self._merge_purchase_order_details(
            existing.get("purchaseOrderDetails"),
            incoming.get("purchaseOrderDetails"),
        )
        return body

    @staticmethod
    def _replacement_marker(receipt_id: Any) -> str:
        if receipt_id in (None, ""):
            raise KiotVietError("Cannot replace a KiotViet receipt without an ID.")
        return f"AIR:{int(receipt_id)}"

    @staticmethod
    def _replacement_source_id(description: Any) -> int | None:
        match = REPLACEMENT_MARKER_PATTERN.search(str(description or ""))
        return int(match.group(1)) if match else None

    def _delete_purchase_receipt(self, receipt_id: Any) -> None:
        receipt_id = int(receipt_id)
        try:
            self._request(
                "DELETE",
                "/purchaseorders",
                params={"id": receipt_id, "IsVoidPayment": True},
            )
        except KiotVietError as delete_error:
            try:
                remaining = self._purchase_receipt_detail(receipt_id)
            except KiotVietError:
                return
            if int(remaining.get("status") or 0) not in ACTIVE_PURCHASE_RECEIPT_STATUSES:
                return
            raise delete_error

        try:
            remaining = self._purchase_receipt_detail(receipt_id)
        except KiotVietError:
            return
        if int(remaining.get("status") or 0) in ACTIVE_PURCHASE_RECEIPT_STATUSES:
            raise KiotVietError(
                f"KiotViet receipt {receipt_id} is still active after cancellation."
            )

    def _replace_completed_purchase_receipt(
        self,
        existing: dict[str, Any],
        replacement_body: dict[str, Any],
        validation: dict[str, Any],
        job_id: str,
    ) -> dict[str, Any]:
        old_id = int(existing["id"])
        replacement_body = dict(replacement_body)
        replacement_body["isDraft"] = 0
        replacement_body["description"] = (
            f"{replacement_body.get('description') or ''} "
            f"{self._replacement_marker(old_id)}"
        ).strip()

        try:
            response = self._request(
                "POST", "/purchaseorders", json_body=replacement_body
            )
            replacement = self._unwrap_data(response)
        except KiotVietError as create_error:
            # A timeout can happen after KiotViet committed the POST. Reconcile
            # by the new job marker before ever attempting another creation.
            replacement = self.find_purchase_receipt_by_job_id(job_id)
            if replacement is None or int(replacement.get("id") or 0) == old_id:
                raise create_error

        if not isinstance(replacement, dict) or replacement.get("id") in (None, ""):
            replacement = self.find_purchase_receipt_by_job_id(job_id)
        if not isinstance(replacement, dict) or replacement.get("id") in (None, ""):
            raise KiotVietError(
                "KiotViet replacement receipt could not be verified; the old "
                "receipt was left active."
            )

        try:
            self._delete_purchase_receipt(old_id)
        except KiotVietError as exc:
            raise KiotVietError(
                "The replacement receipt was created, but the previous receipt "
                f"{existing.get('code') or old_id} is still active. Retry the "
                "same job to finish reconciliation."
            ) from exc

        return {
            "dry_run": False,
            "action": "REPLACED",
            "merged_into_daily_receipt": True,
            "recovered": False,
            "replaced_receipt": {
                "id": old_id,
                "code": existing.get("code"),
            },
            "validation": {
                **validation,
                "daily_merge": True,
                "completed_receipt_replaced": True,
            },
            "receipt": replacement,
        }

    def reconcile_purchase_receipt_by_job_id(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        """Recover a write and finish cancellation of a replaced receipt."""
        receipt = self.find_purchase_receipt_by_job_id(job_id)
        if receipt is None:
            return None
        replaced_id = self._replacement_source_id(receipt.get("description"))
        if replaced_id is not None and replaced_id != int(receipt.get("id") or 0):
            self._delete_purchase_receipt(replaced_id)
        return receipt

    def find_purchase_receipt_by_job_id(
        self,
        job_id: str,
        *,
        lookback_days: int = 30,
    ) -> dict[str, Any] | None:
        """Find a previously created receipt after an interrupted confirmation.

        KiotViet does not expose a client-supplied idempotency key for purchase
        receipts. We therefore store a stable job marker in ``description`` and
        inspect recent receipt details only when recovering a CONFIRMING job.
        Normal confirmations do not pay this additional API cost.
        """
        branch = self.resolve_branch()
        now = datetime.now(VIETNAM_TIMEZONE)
        params: dict[str, Any] = {
            "branchIds": [int(branch["id"])],
            "fromPurchaseDate": (now - timedelta(days=max(1, lookback_days))).date().isoformat(),
            "toPurchaseDate": (now + timedelta(days=1)).date().isoformat(),
            "pageSize": 100,
            "currentItem": 0,
        }
        matches: list[dict[str, Any]] = []

        for _ in range(10):
            payload = self._request("GET", "/purchaseorders", params=params)
            records = payload.get("data", []) if isinstance(payload, dict) else []
            if not isinstance(records, list):
                raise KiotVietError("KiotViet returned an invalid purchase receipt list.")

            for record in records:
                if not isinstance(record, dict) or record.get("id") in (None, ""):
                    continue
                detail = self._purchase_receipt_detail(record["id"])
                if self._description_contains_job(detail.get("description"), job_id):
                    matches.append(detail)

            params["currentItem"] = int(params["currentItem"]) + len(records)
            total = int(payload.get("total") or 0) if isinstance(payload, dict) else 0
            if not records or int(params["currentItem"]) >= total:
                break

        if len(matches) > 1:
            codes = ", ".join(str(item.get("code") or item.get("id")) for item in matches)
            raise KiotVietError(
                f"Multiple KiotViet receipts reference job {job_id}: {codes}."
            )
        return matches[0] if matches else None

    def create_purchase_receipt(
        self,
        products: list[dict[str, Any]],
        job_id: str,
    ) -> dict[str, Any]:
        body, validation = (
            self.build_purchase_receipt_payload(
                products,
                job_id,
            )
        )

        current = datetime.now(VIETNAM_TIMEZONE)
        business_date = current.date()
        existing = None
        if self.merge_daily_drafts:
            existing = self.find_daily_purchase_receipt(now=current)

        if existing is not None and self._description_contains_job(
            existing.get("description"), job_id
        ):
            return {
                "dry_run": False,
                "action": "REUSED",
                "merged_into_daily_receipt": True,
                "recovered": True,
                "validation": {**validation, "daily_merge": True},
                "receipt": existing,
            }

        if existing is not None:
            receipt_id = existing.get("id")
            update_body = self._daily_update_body(
                existing, body, job_id, business_date
            )
            try:
                response = self._request(
                    "PUT",
                    f"/purchaseorders/{int(receipt_id)}",
                    json_body=update_body,
                )
            except KiotVietError as update_error:
                may_replace = bool(
                    self.replace_completed_on_update_failure
                    and not self.create_as_draft
                    and int(existing.get("status") or 0) == 3
                    # KiotViet uses the non-standard HTTP 420 status for
                    # KvValidatePurchaseOrderException when a completed receipt
                    # is locked. Keep the standard business-conflict statuses
                    # for compatibility with other retailer configurations.
                    and update_error.status_code in {400, 409, 420, 422}
                )
                if not may_replace:
                    raise
                return self._replace_completed_purchase_receipt(
                    existing,
                    update_body,
                    validation,
                    job_id,
                )
            receipt = self._unwrap_data(response)
            if not isinstance(receipt, dict) or receipt.get("id") in (None, ""):
                receipt = self._purchase_receipt_detail(receipt_id)
            return {
                "dry_run": False,
                "action": "UPDATED",
                "merged_into_daily_receipt": True,
                "recovered": False,
                "validation": {**validation, "daily_merge": True},
                "receipt": receipt,
            }

        if self.merge_daily_drafts:
            body["description"] = self._daily_description("", job_id, business_date)
        response = self._request("POST", "/purchaseorders", json_body=body)

        receipt = self._unwrap_data(response)
        if not isinstance(receipt, dict):
            raise KiotVietError(
                "KiotViet returned an invalid "
                "purchase receipt response."
            )

        return {
            "dry_run": False,
            "action": "CREATED",
            "merged_into_daily_receipt": False,
            "recovered": False,
            "validation": {
                **validation,
                "daily_merge": bool(self.merge_daily_drafts),
            },
            "receipt": receipt,
        }

    def check_connection(
        self,
    ) -> dict[str, Any]:
        branch = self.resolve_branch()

        return {
            "ready": True,
            "retailer": self.retailer,
            "branch_id": int(
                branch["id"]
            ),
            "branch_name": str(
                branch.get("branchName")
                or ""
            ),
            "is_draft": (
                self.create_as_draft
            ),
        }
