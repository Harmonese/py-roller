from __future__ import annotations

import importlib.metadata
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyroller.batch import BatchRunSummary, BatchTaskResult
from pyroller.domain import PipelineRequest, RunPipelineResult
from pyroller.utils.json import read_json

PROTOCOL_VERSION = 1
ENGINE_NAME = "py-roller"
STAGE_ORDER = ["s", "f", "t", "p", "a", "w"]
LANGUAGES = ["zh", "en", "mul"]

_PATH_FIELDS = {
    "audio_path",
    "lyrics_path",
    "timed_units_path",
    "parsed_lyrics_path",
    "alignment_result_path",
    "intermediate_dir",
    "output_vocal_audio_path",
    "output_filtered_audio_path",
    "output_timed_units_path",
    "output_parsed_lyrics_path",
    "output_alignment_result_path",
    "output_roller_path",
}

_REQUEST_ALIASES = {
    "audio": "audio_path",
    "lyrics": "lyrics_path",
    "timed_units": "timed_units_path",
    "parsed_lyrics": "parsed_lyrics_path",
    "alignment_result": "alignment_result_path",
    "output_vocal_audio": "output_vocal_audio_path",
    "output_filtered_audio": "output_filtered_audio_path",
    "output_timed_units": "output_timed_units_path",
    "output_parsed_lyrics": "output_parsed_lyrics_path",
    "output_alignment_result": "output_alignment_result_path",
    "output_roller": "output_roller_path",
    "intermediate": "intermediate_dir",
}

_OPTION_METADATA: list[dict[str, Any]] = [
    {"name": "language", "type": "choice", "choices": LANGUAGES, "stages": STAGE_ORDER, "default": "mul"},
    {"name": "stages", "type": "stage_chain", "choices": STAGE_ORDER, "stages": STAGE_ORDER},
    {"name": "cleanup", "type": "choice", "choices": ["on-success", "never"], "default": "on-success"},
    {"name": "log_level", "type": "choice", "choices": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], "default": "INFO"},
    {"name": "splitter_backend", "type": "string", "stages": ["s"]},
    {"name": "filter_chain", "type": "string_list", "stages": ["f"]},
    {"name": "transcriber_backend", "type": "string", "stages": ["t"]},
    {"name": "transcriber_device", "type": "string", "stages": ["t"]},
    {"name": "transcriber_model_name", "type": "string", "stages": ["t"]},
    {"name": "transcriber_model_path", "type": "path", "stages": ["t"]},
    {"name": "transcriber_local_files_only", "type": "boolean", "stages": ["t"]},
    {"name": "parser_lyrics_encoding", "type": "choice", "stages": ["p"]},
    {"name": "aligner_backend", "type": "string", "stages": ["a"]},
    {"name": "writer_backend", "type": "string", "stages": ["w"]},
    {"name": "writer_spacing", "type": "choice", "choices": ["keep", "drop"], "stages": ["w"], "default": "keep"},
]


@dataclass(slots=True)
class ProtocolBatchOptions:
    continue_on_error: bool = False
    skip_existing: bool = False
    jobs: int = 1
    manifest: Path | None = None
    pair_by: str = "stem"
    audio_glob: str = "*.mp3"
    lyrics_glob: str = "*.txt"
    timed_units_glob: str = "*.json"
    parsed_lyrics_glob: str = "*.json"
    alignment_result_glob: str = "*.json"


@dataclass(slots=True)
class ProtocolBatchRequest:
    request: PipelineRequest
    options: ProtocolBatchOptions = field(default_factory=ProtocolBatchOptions)


@dataclass(slots=True)
class ProtocolErrorDetail:
    type: str
    message: str
    code: str = "engine_error"
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_exception(cls, exc: BaseException, *, code: str = "engine_error", detail: dict[str, Any] | None = None) -> "ProtocolErrorDetail":
        return cls(
            type=exc.__class__.__name__,
            message=str(exc),
            code=code,
            detail=detail or {},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "code": self.code,
            "message": self.message,
        }
        if self.detail:
            payload["detail"] = self.detail
        return payload


def protocol_envelope(report_type: str, *, status: str = "ok", artifact_paths: dict[str, str] | None = None, error: ProtocolErrorDetail | dict[str, Any] | None = None, **payload: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": PROTOCOL_VERSION,
        "engine": ENGINE_NAME,
        "engine_version": engine_version(),
        "protocol_version": PROTOCOL_VERSION,
        "type": report_type,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_paths": artifact_paths or {},
    }
    report.update(payload)
    if error is not None:
        report["error"] = error.to_dict() if isinstance(error, ProtocolErrorDetail) else error
    return report


def engine_version() -> str:
    source_version = _source_tree_version()
    if source_version:
        return source_version
    try:
        return importlib.metadata.version("py-roller")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _source_tree_version() -> str | None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        return None
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE)
    return match.group(1) if match else None


def capabilities() -> dict[str, Any]:
    payload = protocol_envelope(
        "capabilities",
        stage_order=STAGE_ORDER,
        languages=LANGUAGES,
        profiles=["auto", "cpu", "cu124"],
        commands={
            "capabilities": {"output_formats": ["json"]},
            "run": {"request": "json", "progress_formats": ["jsonl"], "output_formats": ["json"]},
            "batch": {"request": "json", "progress_formats": ["jsonl"], "output_formats": ["json"]},
            "doctor": {"output_formats": ["json"]},
            "install": {"progress_formats": ["jsonl"], "output_formats": ["json"]},
            "cache-model": {"progress_formats": ["jsonl"], "output_formats": ["json"]},
        },
        schemas={
            "request": 1,
            "event": 1,
            "result": 1,
        },
        options=list(_OPTION_METADATA),
    )
    payload.pop("artifact_paths", None)
    return payload


def _path_or_none(value: object) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _request_payload(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("protocol_version") not in {None, PROTOCOL_VERSION}:
        raise ValueError(f"Unsupported py-roller protocol version: {data.get('protocol_version')}")
    payload = data.get("request", data)
    if not isinstance(payload, dict):
        raise ValueError("Protocol request must be a JSON object.")
    return dict(payload)


def pipeline_request_from_dict(data: dict[str, Any]) -> PipelineRequest:
    payload = _request_payload(data)
    for source, target in _REQUEST_ALIASES.items():
        if source in payload and target not in payload:
            payload[target] = payload.pop(source)
    for field_name in _PATH_FIELDS:
        if field_name in payload:
            payload[field_name] = _path_or_none(payload[field_name])
    stages = payload.get("stages")
    if isinstance(stages, str):
        payload["stages"] = [item.strip() for item in stages.split(",") if item.strip()]
    backend_config = payload.get("backend_config")
    if backend_config is None:
        payload["backend_config"] = {}
    elif not isinstance(backend_config, dict):
        raise ValueError("backend_config must be a JSON object.")
    return PipelineRequest(**payload)


def pipeline_request_from_json(path: Path) -> PipelineRequest:
    return pipeline_request_from_dict(read_json(path))


def batch_request_from_json(path: Path) -> ProtocolBatchRequest:
    data = read_json(path)
    request = pipeline_request_from_dict(data)
    raw_options = data.get("batch") or data.get("options") or {}
    if not isinstance(raw_options, dict):
        raise ValueError("batch/options must be a JSON object.")
    options_data = dict(raw_options)
    if "manifest" in options_data:
        options_data["manifest"] = _path_or_none(options_data["manifest"])
    return ProtocolBatchRequest(request=request, options=ProtocolBatchOptions(**options_data))


def _artifact_paths_from_request(request: PipelineRequest) -> dict[str, str]:
    paths = {
        "vocal_audio": request.output_vocal_audio_path,
        "filtered_audio": request.output_filtered_audio_path,
        "timed_units": request.output_timed_units_path,
        "parsed_lyrics": request.output_parsed_lyrics_path,
        "alignment_result": request.output_alignment_result_path,
        "roller": request.output_roller_path,
    }
    return {key: str(path) for key, path in paths.items() if path is not None}


def run_result_report(result: RunPipelineResult, request: PipelineRequest) -> dict[str, Any]:
    return protocol_envelope(
        "run_result",
        artifact_paths=_artifact_paths_from_request(request),
        executed_stages=result.executed_stages,
        counts={
            "timed_units": len(result.transcription.units) if result.transcription is not None else None,
            "parsed_lyrics": len(result.parsed_lyrics.lines) if result.parsed_lyrics is not None else None,
            "aligned_lines": len(result.alignment.lines) if result.alignment is not None else None,
        },
        write_result=result.write_result.to_dict() if result.write_result is not None else None,
    )


def batch_task_result_report(item: BatchTaskResult) -> dict[str, Any]:
    artifact_paths = getattr(item, "artifact_paths", None)
    if not artifact_paths:
        artifact_paths = {path.stem: str(path) for path in item.outputs}
    error = None
    if item.status == "failed":
        error = {
            "type": "BatchTaskError",
            "code": "batch_task_failed",
            "message": item.message,
        }
    return {
        "index": item.index,
        "task_id": item.stem,
        "status": item.status,
        "message": item.message,
        "artifact_paths": artifact_paths,
        "outputs": [str(path) for path in item.outputs],
        "log_file": str(item.log_file) if item.log_file is not None else None,
        "cleaned": item.cleaned,
        "error": error,
    }


def batch_result_report(summary: BatchRunSummary) -> dict[str, Any]:
    return protocol_envelope(
        "batch_result",
        status="failed" if summary.failed else "ok",
        total=summary.total,
        completed=summary.completed,
        failed=summary.failed,
        skipped=summary.skipped,
        aborted=summary.aborted,
        results=[batch_task_result_report(item) for item in summary.results],
    )


def error_report(exc: BaseException) -> dict[str, Any]:
    return protocol_envelope(
        "error",
        status="failed",
        error=ProtocolErrorDetail.from_exception(exc),
    )


def as_jsonable(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    if hasattr(data, "to_dict"):
        return data.to_dict()
    if hasattr(data, "__dataclass_fields__"):
        return asdict(data)
    raise TypeError(f"Object of type {data.__class__.__name__} is not JSON serializable")
