from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_artifact_json(artifact_type: str, payload: dict[str, Any], path: Path) -> None:
    write_json(
        {
            "schema_version": 1,
            "artifact_type": artifact_type,
            "payload": payload,
        },
        path,
    )
