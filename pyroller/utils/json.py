from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = 1


class ArtifactLoadError(ValueError):
    def __init__(self, message: str, *, artifact_type: str | None = None, path: Path | None = None) -> None:
        super().__init__(message)
        self.artifact_type = artifact_type
        self.path = path
        self.code = "artifact_load_error"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_artifact_json(data: dict[str, Any], *, expected_type: str, path: Path) -> dict[str, Any]:
    schema_version = data.get("schema_version")
    if schema_version not in {None, ARTIFACT_SCHEMA_VERSION}:
        raise ArtifactLoadError(
            f"Unsupported artifact schema_version={schema_version!r} for {path}",
            artifact_type=expected_type,
            path=path,
        )
    artifact_type = data.get("artifact_type")
    if artifact_type != expected_type:
        raise ArtifactLoadError(
            f"Expected artifact_type={expected_type!r}, got {artifact_type!r} from {path}",
            artifact_type=expected_type,
            path=path,
        )
    payload = data.get("payload")
    if not isinstance(payload, dict):
        raise ArtifactLoadError(
            f"Expected artifact payload object for {expected_type!r} from {path}",
            artifact_type=expected_type,
            path=path,
        )
    return payload


def write_artifact_json(artifact_type: str, payload: dict[str, Any], path: Path) -> None:
    write_json(
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": artifact_type,
            "payload": payload,
        },
        path,
    )
