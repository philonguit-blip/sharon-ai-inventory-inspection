"""Download/verify external Foundation weights and warm the DINOv2 cache."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from tempfile import NamedTemporaryFile


REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = REPO_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.config import FOUNDATION_REFERENCE_PATH, FOUNDATION_SAM_MODEL_PATH  # noqa: E402


MANIFEST_PATH = BACKEND_ROOT / "models" / "FOUNDATION_MANIFEST.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def download_atomic(url: str, target: Path, expected_sha: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="wb",
        suffix=".download",
        dir=target.parent,
        delete=False,
    ) as output:
        temporary = Path(output.name)
        print(f"Downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"  {downloaded * 100 / total:5.1f}%", end="\r")
    try:
        actual = sha256(temporary)
        if actual != expected_sha.upper():
            raise RuntimeError(
                f"Downloaded SAM checksum mismatch: expected {expected_sha}, got {actual}"
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_sam = str(manifest["sam_sha256"]).upper()
    if FOUNDATION_SAM_MODEL_PATH.is_file():
        actual = sha256(FOUNDATION_SAM_MODEL_PATH)
        if actual != expected_sam:
            raise RuntimeError(
                "Existing SAM file has the wrong checksum. Move it to _archive and rerun setup: "
                f"{FOUNDATION_SAM_MODEL_PATH}"
            )
        print("SAM2 checksum OK.")
    else:
        download_atomic(
            str(manifest["sam_source"]),
            FOUNDATION_SAM_MODEL_PATH,
            expected_sam,
        )
        print("SAM2 downloaded and verified.")

    if not FOUNDATION_REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            "Reference embeddings are missing. Restore backend/hybrid_data/reference_embeddings.npz "
            "or run scripts/hybrid_reference_manager.py build."
        )
    expected_reference = str(manifest["reference_sha256"]).upper()
    actual_reference = sha256(FOUNDATION_REFERENCE_PATH)
    if actual_reference != expected_reference:
        print(
            "WARNING: reference artifact differs from the shipped manifest. "
            "This is expected only after intentionally rebuilding references."
        )

    # Cache and validate DINOv2 now so the first production fallback does not
    # have to download weights while a user is waiting for a job.
    from app.services.foundation_inference_service import FoundationInferenceService

    service = FoundationInferenceService()
    service._ensure_loaded()
    print(f"Foundation ready with {service.health()['reference_count']} references.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
