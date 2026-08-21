"""Cloudflare R2 storage client for bakery job artifacts."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from app.config import (
    R2_ACCESS_KEY_ID,
    R2_ACCOUNT_ID,
    R2_BUCKET_NAME,
    R2_PUBLIC_BASE_URL,
    R2_SECRET_ACCESS_KEY,
)

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None
    Config = None
    BotoCoreError = None
    ClientError = None


class StorageError(RuntimeError):
    pass


class R2StorageService:
    def __init__(self) -> None:
        if boto3 is None or Config is None:
            raise StorageError("boto3 is not installed; install backend/requirements.txt.")
        missing = [
            name
            for name, value in {
                "CLOUDFLARE_ACCOUNT_ID": R2_ACCOUNT_ID,
                "R2_ACCESS_KEY_ID": R2_ACCESS_KEY_ID,
                "R2_SECRET_ACCESS_KEY": R2_SECRET_ACCESS_KEY,
                "R2_BUCKET_NAME": R2_BUCKET_NAME,
            }.items()
            if not value
        ]
        if missing:
            raise StorageError("Missing R2 configuration: " + ", ".join(missing))

        self.bucket = R2_BUCKET_NAME
        self.public_base_url = R2_PUBLIC_BASE_URL
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )

    @staticmethod
    def job_key(job_id: str, category: str, filename: str) -> str:
        safe_category = category.strip("/")
        safe_filename = Path(filename).name
        if not safe_category or not safe_filename:
            raise StorageError("Invalid R2 artifact key.")
        return f"purchase-intake/{job_id}/{safe_category}/{safe_filename}"

    def upload_file(
        self,
        local_path: Path | str,
        object_key: str,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        path = Path(local_path).expanduser().resolve()
        if not path.is_file():
            raise StorageError(f"Artifact not found: {path}")
        if not object_key.startswith("purchase-intake/"):
            raise StorageError("R2 object key must start with purchase-intake/.")

        media_type = content_type or mimetypes.guess_type(path.name)[0]
        media_type = media_type or "application/octet-stream"
        try:
            self.client.upload_file(
                str(path),
                self.bucket,
                object_key,
                ExtraArgs={"ContentType": media_type},
            )
        except Exception as exc:
            raise StorageError(f"R2 upload failed for {object_key}.") from exc

        result: dict[str, Any] = {
            "bucket": self.bucket,
            "object_key": object_key,
            "content_type": media_type,
            "size": path.stat().st_size,
            "uri": f"s3://{self.bucket}/{object_key}",
        }
        if self.public_base_url:
            result["public_url"] = f"{self.public_base_url}/{object_key}"
        return result

    def presign_upload(
        self,
        object_key: str,
        content_type: str,
        expires_in: int = 900,
    ) -> str:
        if not object_key.startswith("purchase-intake/"):
            raise StorageError("R2 object key must start with purchase-intake/.")
        try:
            return self.client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": self.bucket,
                    "Key": object_key,
                    "ContentType": content_type,
                },
                ExpiresIn=expires_in,
            )
        except Exception as exc:
            raise StorageError("Cannot create R2 upload URL.") from exc

    def object_info(self, object_key: str) -> dict[str, Any]:
        if not object_key.startswith("purchase-intake/"):
            raise StorageError("R2 object key must start with purchase-intake/.")
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=object_key)
        except Exception as exc:
            if ClientError is not None and isinstance(exc, ClientError):
                error = exc.response.get("Error", {})
                error_code = str(error.get("Code") or "")
                status_code = int(
                    exc.response.get("ResponseMetadata", {}).get(
                        "HTTPStatusCode", 0
                    )
                    or 0
                )
                if error_code in {"404", "NoSuchKey", "NotFound"} or status_code == 404:
                    raise StorageError(
                        f"R2 object not found: {object_key}."
                    ) from exc
            raise StorageError(
                f"Cannot inspect R2 object metadata: {object_key}."
            ) from exc
        return {
            "object_key": object_key,
            "size": int(response.get("ContentLength") or 0),
            "content_type": str(response.get("ContentType") or ""),
            "etag": str(response.get("ETag") or "").strip('"'),
        }

    def download_file(self, object_key: str, local_path: Path | str) -> Path:
        if not object_key.startswith("purchase-intake/"):
            raise StorageError("R2 object key must start with purchase-intake/.")
        destination = Path(local_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, object_key, str(destination))
        except Exception as exc:
            if destination.exists():
                destination.unlink()
            raise StorageError(f"R2 download failed for {object_key}.") from exc
        return destination

    def presign_download(self, object_key: str, expires_in: int = 3600) -> str:
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": object_key},
                ExpiresIn=expires_in,
            )
        except Exception as exc:
            raise StorageError("Cannot create R2 download URL.") from exc

    def check_connection(self) -> dict[str, Any]:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception as exc:
            raise StorageError("Cannot access the configured R2 bucket.") from exc
        return {"ready": True, "bucket": self.bucket}
