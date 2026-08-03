from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi import BackgroundTasks, HTTPException

from app.routes import local_jobs
from app.schemas.jobs import (
    CreateR2JobRequest,
    PresignUploadsRequest,
    R2JobFile,
    UploadFileRequest,
)


class FakeStorage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def object_info(self, object_key: str):
        payload = self.objects[object_key]
        return {"object_key": object_key, "size": len(payload)}

    def download_file(self, object_key: str, local_path: Path):
        local_path.write_bytes(self.objects[object_key])
        return local_path

    @staticmethod
    def job_key(job_id: str, category: str, filename: str) -> str:
        return f"purchase-intake/{job_id}/{category}/{filename}"

    def presign_upload(self, object_key: str, content_type: str, expires_in: int):
        return f"https://upload.example/{object_key}?expires={expires_in}"


class R2IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.previous_jobs_root = local_jobs.JOBS_ROOT
        local_jobs.JOBS_ROOT = Path(self.temporary.name).resolve()

    def tearDown(self):
        local_jobs.JOBS_ROOT = self.previous_jobs_root
        self.temporary.cleanup()

    def _write_manifest(self, job_id: str, entries: list[dict[str, object]]) -> None:
        local_jobs._write_json_atomic(
            local_jobs._manifest_path(job_id),
            {"job_id": job_id, "files": entries},
        )

    def test_validates_total_upload_size_before_presigning(self):
        files = [
            SimpleNamespace(
                filename="tray.jpg",
                content_type="image/jpeg",
                size_bytes=local_jobs.MAX_JOB_UPLOAD_SIZE_BYTES + 1,
            )
        ]
        with self.assertRaises(HTTPException) as context:
            local_jobs._validate_upload_metadata(files)
        self.assertEqual(context.exception.status_code, 413)

    def test_presign_retry_reuses_worker_supplied_job_id(self):
        job_id = "d" * 32
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(r2_storage_service=FakeStorage({}))
            )
        )
        payload = PresignUploadsRequest(
            job_id=job_id,
            files=[
                UploadFileRequest(
                    filename="tray.jpg",
                    content_type="image/jpeg",
                    size_bytes=123,
                )
            ],
        )

        first = local_jobs.presign_uploads(payload, request)
        second = local_jobs.presign_uploads(payload, request)

        self.assertEqual(first.job_id, job_id)
        self.assertEqual(second.job_id, job_id)
        self.assertEqual(first.uploads[0].object_key, second.uploads[0].object_key)

    def test_downloads_only_the_exact_manifest_objects(self):
        job_id = "a" * 32
        payload = b"not-a-real-image-but-valid-for-transport-test"
        object_key = f"purchase-intake/{job_id}/incoming/001_tray.jpg"
        self._write_manifest(
            job_id,
            [
                {
                    "filename": "tray.jpg",
                    "safe_name": "001_tray.jpg",
                    "size_bytes": len(payload),
                    "object_key": object_key,
                }
            ],
        )

        saved = local_jobs._download_r2_uploads(
            job_id, [object_key], FakeStorage({object_key: payload})
        )

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(Path(saved[0]["path"]).read_bytes(), payload)

    def test_rejects_object_keys_not_issued_for_the_session(self):
        job_id = "b" * 32
        expected_key = f"purchase-intake/{job_id}/incoming/001_tray.jpg"
        other_key = f"purchase-intake/{job_id}/incoming/999_other.jpg"
        self._write_manifest(
            job_id,
            [
                {
                    "filename": "tray.jpg",
                    "safe_name": "001_tray.jpg",
                    "size_bytes": 4,
                    "object_key": expected_key,
                }
            ],
        )

        with self.assertRaises(HTTPException) as context:
            local_jobs._download_r2_uploads(
                job_id, [other_key], FakeStorage({other_key: b"data"})
            )
        self.assertEqual(context.exception.status_code, 400)

    def test_resubmitting_existing_job_is_idempotent(self):
        job_id = "c" * 32
        job_directory = local_jobs._job_directory(job_id)
        job_directory.mkdir(parents=True)
        local_jobs._write_json_atomic(
            local_jobs._job_state_path(job_id),
            {
                "job_id": job_id,
                "status": "PROCESSING",
                "total_images": 2,
            },
        )
        app_state = SimpleNamespace(
            bakery_inference_service=object(),
            excel_service=object(),
            r2_storage_service=object(),
            kiotviet_service=object(),
            bakery_startup_error="",
        )
        request = SimpleNamespace(app=SimpleNamespace(state=app_state))
        tasks = BackgroundTasks()

        accepted = local_jobs.create_job_from_r2(
            CreateR2JobRequest(
                job_id=job_id,
                files=[R2JobFile(object_key="unused")],
            ),
            request,
            tasks,
        )

        self.assertEqual(accepted.status, "PROCESSING")
        self.assertEqual(accepted.total_images, 2)
        self.assertEqual(len(tasks.tasks), 0)


if __name__ == "__main__":
    unittest.main()
