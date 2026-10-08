from __future__ import annotations

import pytest
import json

from pyroller.domain import AlignmentResult
from pyroller.utils.json import ArtifactLoadError

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


def test_alignment_result_load_error_exposes_protocol_details(tmp_path) -> None:
    path = tmp_path / "not-alignment.json"
    path.write_text('{"schema_version": 99, "artifact_type": "alignment_result", "payload": {}}', encoding="utf-8")

    with pytest.raises(ArtifactLoadError) as exc_info:
        AlignmentResult.load(path)

    assert exc_info.value.code == "artifact_load_error"
    assert exc_info.value.artifact_type == "alignment_result"
    assert exc_info.value.path == path


@pytest.mark.parametrize('assigned', [(99, 0), (2, 4)])
def test_alignment_rejects_conflicting_assigned_times_before_publication(tmp_path, assigned):
    from pyroller.writer.lrc import LRCWriter
    from pyroller.writer.ass_karaoke import ASSKaraokeWriter
    result = make_alignment_result()
    result.lines[0].assigned_time, result.lines[2].assigned_time = assigned
    path = tmp_path / 'alignment.json'
    path.write_text(json.dumps({'schema_version': 1, 'artifact_type': 'alignment_result', 'payload': result.to_dict()}))
    with pytest.raises(ValueError, match='assigned_time'):
        AlignmentResult.load(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='assigned_time'):
        result.save(path)
    assert path.read_bytes() == original
    for writer in (LRCWriter(), LRCWriter(compressed=True), ASSKaraokeWriter()):
        output = tmp_path / 'output.txt'
        output.write_text('keep existing output')
        with pytest.raises(ValueError, match='assigned_time'):
            writer.write(result, output)
        assert output.read_text() == 'keep existing output'
