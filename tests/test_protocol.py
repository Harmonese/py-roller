from __future__ import annotations

import json
from pathlib import Path

from pyroller.protocol import (
    PROTOCOL_VERSION,
    batch_result_report,
    batch_request_from_json,
    capabilities,
    error_report,
    pipeline_request_from_dict,
    run_result_report,
)
from pyroller.batch import BatchRunSummary, BatchTaskResult
from pyroller.domain import RunPipelineResult


def test_capabilities_exposes_protocol_v1_contract() -> None:
    payload = capabilities()

    assert payload["engine"] == "py-roller"
    assert payload["protocol_version"] == 1
    assert payload["stage_order"] == ["s", "f", "t", "p", "a", "w"]
    assert payload["commands"]["run"]["request"] == "json"
    assert payload["schemas"] == {"request": 1, "event": 1, "result": 1}
    assert payload["type"] == "capabilities"
    assert payload["status"] == "ok"


def test_pipeline_request_from_protocol_json_accepts_aliases(tmp_path: Path) -> None:
    request = pipeline_request_from_dict(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request": {
                "stages": "t,p,a,w",
                "audio": str(tmp_path / "audio.wav"),
                "lyrics": str(tmp_path / "plain.txt"),
                "output_roller": str(tmp_path / "out.lrc"),
                "intermediate": str(tmp_path / "intermediate"),
                "backend_config": {"writer": {"backend": "lrc_ms"}},
            },
        }
    )

    assert request.stages == ["t", "p", "a", "w"]
    assert request.audio_path == tmp_path / "audio.wav"
    assert request.lyrics_path == tmp_path / "plain.txt"
    assert request.output_roller_path == tmp_path / "out.lrc"
    assert request.backend_config["writer"]["backend"] == "lrc_ms"


def test_batch_request_from_json_loads_options(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "protocol_version": 1,
                "request": {
                    "stages": ["w"],
                    "alignment_result": str(tmp_path / "alignments"),
                    "output_roller": str(tmp_path / "out"),
                    "intermediate": str(tmp_path / "intermediate"),
                },
                "batch": {"jobs": 2, "skip_existing": True, "alignment_result_glob": "*.alignment.json"},
            }
        ),
        encoding="utf-8",
    )

    loaded = batch_request_from_json(request_path)

    assert loaded.request.stages == ["w"]
    assert loaded.request.alignment_result_path == tmp_path / "alignments"
    assert loaded.options.jobs == 2
    assert loaded.options.skip_existing is True
    assert loaded.options.alignment_result_glob == "*.alignment.json"


def test_run_result_report_contains_protocol_metadata(tmp_path: Path) -> None:
    request = pipeline_request_from_dict(
        {
            "stages": ["w"],
            "alignment_result": str(tmp_path / "alignment.json"),
            "output_roller": str(tmp_path / "out.lrc"),
            "intermediate": str(tmp_path / "intermediate"),
        }
    )
    report = run_result_report(RunPipelineResult(executed_stages=["writer"]), request)

    assert report["schema_version"] == 1
    assert report["type"] == "run_result"
    assert report["status"] == "ok"
    assert report["artifact_paths"]["roller"] == str(tmp_path / "out.lrc")
    assert "timestamp" in report


def test_batch_result_report_contains_task_artifact_paths_and_error(tmp_path: Path) -> None:
    output = tmp_path / "bad.lrc"
    summary = BatchRunSummary(
        total=1,
        completed=0,
        failed=1,
        skipped=0,
        aborted=0,
        results=[
            BatchTaskResult(
                index=1,
                stem="bad",
                status="failed",
                message="boom",
                outputs=[output],
                artifact_paths={"roller": str(output)},
                error={"type": "RuntimeError", "code": "batch_task_failed", "message": "boom"},
            )
        ],
    )

    report = batch_result_report(summary)

    assert report["status"] == "failed"
    assert report["artifact_paths"] == {}
    assert report["results"][0]["task_id"] == "bad"
    assert report["results"][0]["artifact_paths"] == {"roller": str(output)}
    assert report["results"][0]["error"]["message"] == "boom"


def test_protocol_result_envelope_keeps_stable_fields(tmp_path: Path) -> None:
    request = pipeline_request_from_dict(
        {
            "protocol_version": 1,
            "request": {
                "stages": ["w"],
                "alignment_result": str(tmp_path / "alignment.json"),
                "output_roller": str(tmp_path / "out.lrc"),
            },
        }
    )

    report = run_result_report(RunPipelineResult(executed_stages=["writer"]), request)

    assert report["schema_version"] == 1
    assert report["protocol_version"] == 1
    assert report["engine"] == "py-roller"
    assert report["type"] == "run_result"
    assert report["status"] == "ok"
    assert report["artifact_paths"]["roller"] == str(tmp_path / "out.lrc")


def test_error_report_uses_structured_error_detail() -> None:
    report = error_report(ValueError("bad request"))

    assert report["status"] == "failed"
    assert report["artifact_paths"] == {}
    assert report["error"]["type"] == "ValueError"
    assert report["error"]["code"] == "engine_error"
