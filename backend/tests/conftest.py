"""Make backend/app importable whether pytest starts at repo root or backend/."""

from __future__ import annotations

import os
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Match the Windows launchers: third-party model libraries must keep their
# mutable caches inside the project runtime folder, not AppData. This also
# makes direct service imports deterministic in restricted test environments.
ULTRALYTICS_RUNTIME = BACKEND_ROOT / "runtime" / "ultralytics-config"
MATPLOTLIB_RUNTIME = BACKEND_ROOT / "runtime" / "matplotlib-config"
ULTRALYTICS_RUNTIME.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ULTRALYTICS_CONFIG_DIR", str(ULTRALYTICS_RUNTIME))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_RUNTIME))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_RUNTIME))
