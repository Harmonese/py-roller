from __future__ import annotations

from pathlib import Path

import pytest

from pyroller.utils.json import json_default, read_json, write_artifact_json, write_json


def test_json_default_serializes_paths() -> None:
    assert json_default(Path("/tmp/example")) == "/tmp/example"


def test_json_default_rejects_unknown_objects() -> None:
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_default(object())


def test_write_and_read_json_round_trip_with_paths(tmp_path) -> None:
    path = tmp_path / "nested" / "data.json"

    write_json({"path": tmp_path / "file.txt", "value": 3}, path)

    assert read_json(path) == {"path": str(tmp_path / "file.txt"), "value": 3}


def test_write_artifact_json_wraps_payload(tmp_path) -> None:
    path = tmp_path / "artifact.json"

    write_artifact_json("timed_units", {"units": []}, path)

    assert read_json(path) == {
        "schema_version": 1,
        "artifact_type": "timed_units",
        "payload": {"units": []},
    }
