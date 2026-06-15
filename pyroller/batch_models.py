from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pyroller.domain import PipelineRequest


@dataclass(slots=True)
class BatchTask:
    index: int
    stem: str
    request: PipelineRequest
    expected_outputs: list[Path]


@dataclass(slots=True)
class BatchTaskResult:
    index: int
    stem: str
    status: str
    message: str = ""
    outputs: list[Path] = field(default_factory=list)
    log_file: Optional[Path] = None
    cleaned: bool = False
    artifact_paths: dict[str, str] = field(default_factory=dict)
    error: dict[str, Any] | None = None


@dataclass(slots=True)
class BatchRunSummary:
    total: int
    completed: int
    failed: int
    skipped: int
    aborted: int
    results: list[BatchTaskResult]


def batch_task_log_file(intermediate_dir: Path) -> Path:
    return intermediate_dir / "logs" / "run.log"


def build_expected_outputs(request: PipelineRequest) -> list[Path]:
    outputs: list[Path] = []
    for path in (
        request.output_vocal_audio_path,
        request.output_filtered_audio_path,
        request.output_timed_units_path,
        request.output_parsed_lyrics_path,
        request.output_alignment_result_path,
        request.output_roller_path,
    ):
        if path is not None:
            outputs.append(path)
    return outputs


def artifact_paths_for_request(request: PipelineRequest) -> dict[str, str]:
    paths = {
        "vocal_audio": request.output_vocal_audio_path,
        "filtered_audio": request.output_filtered_audio_path,
        "timed_units": request.output_timed_units_path,
        "parsed_lyrics": request.output_parsed_lyrics_path,
        "alignment_result": request.output_alignment_result_path,
        "roller": request.output_roller_path,
    }
    return {key: str(path) for key, path in paths.items() if path is not None}
