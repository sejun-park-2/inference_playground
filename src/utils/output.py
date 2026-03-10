from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def make_output_dir(base: str = "output", timestamp: str | None = None) -> Path:
    execute_time = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base) / execute_time
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
