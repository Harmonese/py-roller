from __future__ import annotations

from dataclasses import replace

import pytest

from pyroller.domain import PipelineRequest
from pyroller.pipeline import ComposablePipelineRunner


def _validate(request: PipelineRequest) -> list[str]:
    runner = ComposablePipelineRunner()
    try:
        stages = runner._resolve_execution_plan(request)
        runner._validate_request(request, stages)
        return stages
    finally:
        runner.close()


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (["s", "f", "t", "p", "a", "w"], ["splitter", "filter", "transcriber", "parser", "aligner", "writer"]),
        (["t", "p", "a", "w"], ["transcriber", "parser", "aligner", "writer"]),
        (["a", "w"], ["aligner", "writer"]),
        (["w"], ["writer"]),
    ],
)
def test_valid_contiguous_stage_chains(base_request: PipelineRequest, requested: list[str], expected: list[str]) -> None:
    request = replace(
        base_request,
        stages=requested,
        audio_path=base_request.intermediate_dir / "song.wav" if requested[0] in {"s", "t"} else None,
        lyrics_path=base_request.intermediate_dir / "song.txt" if "p" in requested else None,
        timed_units_path=base_request.intermediate_dir / "timed.json" if requested[0] == "a" else None,
        parsed_lyrics_path=base_request.intermediate_dir / "parsed.json" if requested[0] == "a" else None,
        alignment_result_path=base_request.intermediate_dir / "alignment.json" if requested[0] == "w" else None,
    )

    assert _validate(request) == expected


def test_rejects_non_contiguous_stage_chain(base_request: PipelineRequest) -> None:
    request = replace(
        base_request,
        stages=["s", "t", "w"],
        audio_path=base_request.intermediate_dir / "song.wav",
    )

    with pytest.raises(ValueError, match="contiguous chain"):
        _validate(request)


def test_writer_stage_requires_output_path(base_request: PipelineRequest) -> None:
    request = replace(
        base_request,
        stages=["w"],
        alignment_result_path=base_request.intermediate_dir / "alignment.json",
        output_roller_path=None,
    )

    with pytest.raises(ValueError, match="output-roller"):
        _validate(request)


def test_aligner_chain_requires_both_artifacts(base_request: PipelineRequest) -> None:
    request = replace(
        base_request,
        stages=["a", "w"],
        timed_units_path=base_request.intermediate_dir / "timed.json",
        parsed_lyrics_path=None,
    )

    with pytest.raises(ValueError, match="parsed_lyrics"):
        _validate(request)


def test_alignment_result_is_only_valid_for_writer_start(base_request: PipelineRequest) -> None:
    request = replace(
        base_request,
        stages=["a", "w"],
        timed_units_path=base_request.intermediate_dir / "timed.json",
        parsed_lyrics_path=base_request.intermediate_dir / "parsed.json",
        alignment_result_path=base_request.intermediate_dir / "alignment.json",
    )

    with pytest.raises(ValueError, match="alignment-result"):
        _validate(request)


def test_ass_karaoke_tag_option_requires_ass_writer(base_request: PipelineRequest) -> None:
    request = replace(
        base_request,
        stages=["w"],
        alignment_result_path=base_request.intermediate_dir / "alignment.json",
        backend_config={"writer": {"backend": "lrc_ms", "tag_type": "kf"}},
    )

    with pytest.raises(ValueError, match="ass_karaoke"):
        _validate(request)
