"""Persistent, validated runtime settings for the developer control panel."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DeveloperSettingsError(RuntimeError):
    """Raised when a requested runtime setting is unsafe or invalid."""


class DeveloperSettingsService:
    """Keep model selection and class thresholds across backend restarts."""

    MODEL_SUFFIXES = {".pt", ".onnx"}

    def __init__(
        self,
        *,
        models_root: Path | str,
        settings_path: Path | str,
        default_model_path: Path | str,
    ) -> None:
        self.models_root = Path(models_root).expanduser().resolve()
        self.settings_path = Path(settings_path).expanduser().resolve()
        self.default_model_path = Path(default_model_path).expanduser().resolve()

    def _relative_model_name(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.models_root).as_posix()
        except ValueError as exc:
            raise DeveloperSettingsError(
                "Model must be stored inside backend/models."
            ) from exc

    def list_models(self) -> list[str]:
        if not self.models_root.is_dir():
            return []
        models = []
        for path in self.models_root.rglob("*"):
            if not path.is_file() or path.suffix.casefold() not in self.MODEL_SUFFIXES:
                continue
            # Foundation SAM weights are not valid YOLO detector choices.
            if "sam" in path.name.casefold():
                continue
            models.append(self._relative_model_name(path))
        return sorted(models, key=str.casefold)

    def resolve_model(self, model_name: str) -> Path:
        name = str(model_name or "").strip().replace("\\", "/")
        if not name:
            raise DeveloperSettingsError("Model name is required.")
        candidate = (self.models_root / name).resolve()
        try:
            candidate.relative_to(self.models_root)
        except ValueError as exc:
            raise DeveloperSettingsError(
                "Model path must stay inside backend/models."
            ) from exc
        if candidate.suffix.casefold() not in self.MODEL_SUFFIXES:
            raise DeveloperSettingsError("Only .pt and .onnx detector files are allowed.")
        if not candidate.is_file():
            raise DeveloperSettingsError(f"Model file not found: {name}")
        return candidate

    def load(self) -> dict[str, Any]:
        fallback_name = self._relative_model_name(self.default_model_path)
        fallback = {
            "active_model": fallback_name,
            "thresholds": {},
            "updated_at": None,
        }
        if not self.settings_path.is_file():
            return fallback
        try:
            payload = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return fallback
        if not isinstance(payload, dict):
            return fallback
        model_name = str(payload.get("active_model") or fallback_name).strip()
        raw_thresholds = payload.get("thresholds")
        thresholds: dict[str, float] = {}
        if isinstance(raw_thresholds, dict):
            for raw_class, raw_value in raw_thresholds.items():
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                if 0.0 < value <= 1.0:
                    thresholds[str(raw_class)] = value
        return {
            "active_model": model_name,
            "thresholds": thresholds,
            "updated_at": payload.get("updated_at"),
        }

    def startup_configuration(self) -> tuple[Path, dict[str, float]]:
        payload = self.load()
        try:
            model_path = self.resolve_model(str(payload["active_model"]))
        except DeveloperSettingsError:
            model_path = self.default_model_path
        return model_path, dict(payload["thresholds"])

    def save(self, *, model_path: Path, thresholds: dict[str, float]) -> dict[str, Any]:
        model_name = self._relative_model_name(model_path)
        payload = {
            "schema_version": 1,
            "active_model": model_name,
            "thresholds": {
                key: float(thresholds[key]) for key in sorted(thresholds, key=str.casefold)
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.settings_path.with_suffix(self.settings_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.settings_path)
        return payload
