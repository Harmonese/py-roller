from __future__ import annotations

import pytest

from pyroller.domain import AlignmentResult

from .factories import make_alignment_result


def test_alignment_result_round_trips_through_artifact_json(tmp_path) -> None:
    path = tmp_path / "alignment.json"
    original = make_alignment_result()

    original.save(path)
    loaded = AlignmentResult.load(path)

    assert loaded.language == original.language
    assert loaded.unit_type == original.unit_type
    assert [line.raw_text for line in loaded.lines] == ["你好", "", "世界"]
    assert loaded.lines[0].aligned_units[0].normalized_symbol == "ni"


def test_alignment_result_rejects_wrong_artifact_type(tmp_path) -> None:
    path = tmp_path / "not-alignment.json"
    path.write_text('{"artifact_type": "parsed_lyrics", "payload": {}}', encoding="utf-8")

    with pytest.raises(ValueError, match="Expected artifact_type"):
        AlignmentResult.load(path)
