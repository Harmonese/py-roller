from __future__ import annotations

from pathlib import Path

import pytest

from pyroller.batch import BatchBuilder, ManifestBatchBuilder
from pyroller.domain import PipelineRequest


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    return path


def test_batch_builder_pairs_audio_and_lyrics_by_stem(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    lyrics_dir = tmp_path / "lyrics"
    _touch(audio_dir / "song-a.mp3")
    _touch(audio_dir / "ignored.mp3")
    _touch(lyrics_dir / "song-a.txt")

    request = PipelineRequest(
        stages=["t", "p", "a", "w"],
        audio_path=audio_dir,
        lyrics_path=lyrics_dir,
        output_roller_path=tmp_path / "out",
        intermediate_dir=tmp_path / "work",
    )

    tasks = BatchBuilder().build_tasks(request)

    assert [task.stem for task in tasks] == ["song-a"]
    assert tasks[0].request.audio_path == audio_dir / "song-a.mp3"
    assert tasks[0].request.lyrics_path == lyrics_dir / "song-a.txt"
    assert tasks[0].request.output_roller_path == tmp_path / "out" / "song-a.lrc"


def test_batch_builder_uses_ass_suffix_for_ass_writer(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    _touch(audio_dir / "song.mp3")
    request = PipelineRequest(
        stages=["t"],
        audio_path=audio_dir,
        output_roller_path=tmp_path / "out",
        intermediate_dir=tmp_path / "work",
        backend_config={"writer": {"backend": "ass_karaoke"}},
    )

    task = BatchBuilder().build_tasks(request)[0]

    assert task.request.output_roller_path == tmp_path / "out" / "song.ass"


def test_batch_builder_rejects_duplicate_stems(tmp_path) -> None:
    audio_dir = tmp_path / "audio"
    _touch(audio_dir / "song.mp3")
    _touch(audio_dir / "song.wav")
    request = PipelineRequest(
        stages=["t"],
        audio_path=audio_dir,
        intermediate_dir=tmp_path / "work",
    )

    with pytest.raises(ValueError, match="Duplicate stems"):
        BatchBuilder(audio_glob="song.*").build_tasks(request)


def test_manifest_builder_resolves_relative_paths_and_rejects_output_conflicts(tmp_path) -> None:
    manifest = tmp_path / "jobs.yaml"
    manifest.write_text(
        """
tasks:
  - id: one
    alignment_result: artifacts/one.json
    output_roller: out/shared.lrc
  - id: two
    alignment_result: artifacts/two.json
    output_roller: out/shared.lrc
""",
        encoding="utf-8",
    )
    request = PipelineRequest(
        stages=["w"],
        intermediate_dir=tmp_path / "work",
    )

    with pytest.raises(ValueError, match="output path conflict"):
        ManifestBatchBuilder(manifest).build_tasks(request)
