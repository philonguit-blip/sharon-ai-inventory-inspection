from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app.services.developer_settings_service import (
    DeveloperSettingsError,
    DeveloperSettingsService,
)
from app.services.product_mapping_service import ProductMappingError, ProductMappingService


MAPPING_PATH = Path(__file__).resolve().parents[1] / "config" / "product_mapping.json"


def test_threshold_overrides_replace_only_requested_class() -> None:
    baseline = ProductMappingService(MAPPING_PATH)
    class_name = baseline.class_names()[0]
    original_other = baseline.class_names()[1]
    updated = ProductMappingService(MAPPING_PATH, {class_name: 0.33})

    assert updated.resolve(class_name)["confidence_threshold"] == 0.33
    assert (
        updated.resolve(original_other)["confidence_threshold"]
        == baseline.resolve(original_other)["confidence_threshold"]
    )


def test_threshold_override_rejects_unknown_class() -> None:
    with pytest.raises(ProductMappingError, match="unknown classes"):
        ProductMappingService(MAPPING_PATH, {"not-a-real-class": 0.5})


def test_runtime_settings_persist_and_reject_path_traversal() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        models = root / "models"
        models.mkdir()
        default_model = models / "best_default.pt"
        next_model = models / "best_next.pt"
        default_model.write_bytes(b"default")
        next_model.write_bytes(b"next")
        service = DeveloperSettingsService(
            models_root=models,
            settings_path=root / "runtime" / "developer_settings.json",
            default_model_path=default_model,
        )

        saved = service.save(model_path=next_model, thresholds={"class-a": 0.41})
        model_path, thresholds = service.startup_configuration()

        assert saved["active_model"] == "best_next.pt"
        assert model_path == next_model.resolve()
        assert thresholds == {"class-a": 0.41}
        with pytest.raises(DeveloperSettingsError, match="inside backend/models"):
            service.resolve_model("../outside.pt")
