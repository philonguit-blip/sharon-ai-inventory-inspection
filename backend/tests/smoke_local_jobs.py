"""End-to-end smoke test for the bakery upload/job/download API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routes.local_jobs import router
from app.services.bakery_inference_service import BakeryInferenceService
from app.services.excel_service import ExcelService


def run(image_path: Path) -> dict[str, object]:
    app = FastAPI()
    app.state.bakery_inference_service = BakeryInferenceService()
    app.state.excel_service = ExcelService()
    app.state.bakery_startup_error = ""
    app.include_router(router)

    with TestClient(app) as client:
        health = client.get("/api/v1/bakery/health")
        health.raise_for_status()
        assert health.json()["ready"] is True

        with image_path.open("rb") as image_file:
            response = client.post(
                "/api/v1/bakery/jobs",
                files=[
                    (
                        "files",
                        (image_path.name, image_file, "image/jpeg"),
                    )
                ],
            )
        response.raise_for_status()
        accepted = response.json()
        assert accepted["status"] == "QUEUED"

        job_response = client.get(accepted["status_url"])
        job_response.raise_for_status()
        job = job_response.json()
        assert job["status"] == "COMPLETED", job
        assert job["total_quantity"] > 0

        excel_response = client.get(job["excel_url"])
        excel_response.raise_for_status()
        assert excel_response.content.startswith(b"PK")

        annotated_response = client.get(job["images"][0]["annotated_url"])
        annotated_response.raise_for_status()
        assert annotated_response.headers["content-type"].startswith("image/jpeg")

        with image_path.open("rb") as first_file, image_path.open("rb") as second_file:
            duplicate_response = client.post(
                "/api/v1/bakery/jobs",
                files=[
                    ("files", ("first.jpg", first_file, "image/jpeg")),
                    ("files", ("duplicate.jpg", second_file, "image/jpeg")),
                ],
            )
        assert duplicate_response.status_code == 400

        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "total_images": job["total_images"],
            "total_quantity": job["total_quantity"],
            "products": job["products"],
            "excel_bytes": len(excel_response.content),
            "annotated_bytes": len(annotated_response.content),
            "duplicate_upload_status": duplicate_response.status_code,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    arguments = parser.parse_args()
    summary = run(arguments.image.expanduser().resolve())
    print(json.dumps(summary, ensure_ascii=True))
