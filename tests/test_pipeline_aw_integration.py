from __future__ import annotations

from pyroller.domain import PipelineRequest
from pyroller.pipeline import ComposablePipelineRunner

from .factories import make_parsed_lyrics, make_transcription


def test_artifact_to_lrc_pipeline_runs_without_audio_runtime(tmp_path) -> None:
    timed_units = tmp_path / "timed-units.json"
    parsed_lyrics = tmp_path / "parsed-lyrics.json"
    output = tmp_path / "song.lrc"
    make_transcription().save(timed_units)
    make_parsed_lyrics().save(parsed_lyrics)

    request = PipelineRequest(
        stages=["a", "w"],
        timed_units_path=timed_units,
        parsed_lyrics_path=parsed_lyrics,
        output_roller_path=output,
        intermediate_dir=tmp_path / "work",
        cleanup="on-success",
        backend_config={"writer": {"backend": "lrc_ms", "spacing": "drop"}},
    )
    runner = ComposablePipelineRunner()
    try:
        result = runner.run(request)
    finally:
        runner.close()

    assert result.executed_stages == ["aligner", "writer"]
    assert result.write_result is not None
    assert output.read_text(encoding="utf-8").endswith("[00:03.000] 世界\n")
    assert not request.intermediate_dir.exists()
