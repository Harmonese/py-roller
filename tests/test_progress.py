from __future__ import annotations

import json
import time

from pyroller.progress import (
    _EVENT_PREFIX,
    LoggingStageProgress,
    MultiProgressReporter,
    NullProgressReporter,
    ProgressHeartbeat,
    build_cli_progress_reporter,
    normalize_progress_payload,
    progress_heartbeat,
)


def _jsonl_events(output: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in output.splitlines():
        if line.startswith(_EVENT_PREFIX):
            events.append(json.loads(line.removeprefix(_EVENT_PREFIX)))
    return events


def test_normalize_progress_payload_accepts_percent_alias_and_clamps() -> None:
    assert normalize_progress_payload({"percent": 40})["progress"] == 0.4
    assert normalize_progress_payload({"progress": 150})["progress"] == 1.0
    assert normalize_progress_payload({"progress": -0.25})["progress"] == 0.0
    bool_payload = normalize_progress_payload({"progress": True})
    assert bool_payload["progress"] is True
    assert "percent" not in bool_payload


def test_jsonl_stage_progress_emits_lifecycle_events(capsys) -> None:
    reporter = build_cli_progress_reporter("jsonl")
    stage = reporter.stage("aligner", total=2, unit="phase")
    stage.update(message="first")
    stage.event("custom", detail="value", percent=75)
    stage.close("done")

    events = _jsonl_events(capsys.readouterr().out)

    assert [event["type"] for event in events] == [
        "stage_started",
        "stage_progress",
        "custom",
        "stage_completed",
    ]
    assert events[1]["progress"] == 0.5
    assert events[1]["schema_version"] == 1
    assert "timestamp" in events[1]
    assert "message" in events[1]
    assert events[2]["progress"] == 0.75
    assert events[-1]["done"] is True


def test_jsonl_stage_progress_emits_failure_event(capsys) -> None:
    stage = build_cli_progress_reporter("jsonl").stage("writer", total=3)
    stage.update(advance=2, message="working")
    stage.fail("bad output")

    events = _jsonl_events(capsys.readouterr().out)

    assert events[-1]["type"] == "stage_failed"
    assert events[-1]["failed"] is True
    assert events[-1]["message"] == "bad output"
    assert events[-1]["progress"] == 2 / 3


def test_multi_progress_reporter_forwards_events_to_all_reporters(capsys) -> None:
    reporter = MultiProgressReporter([NullProgressReporter(), build_cli_progress_reporter("jsonl")])
    reporter.event("run_started", stage="run", progress=25)
    stage = reporter.stage("parser", total=1)
    stage.close()

    events = _jsonl_events(capsys.readouterr().out)

    assert events[0]["type"] == "run_started"
    assert events[0]["progress"] == 0.25
    assert events[-1]["type"] == "stage_completed"


def test_logging_stage_progress_clamps_completed_to_total() -> None:
    stage = LoggingStageProgress(name="stage", total=2)

    stage.update(advance=10)

    assert stage.completed == 2


def test_build_cli_progress_reporter_both_combines_human_and_jsonl() -> None:
    reporter = build_cli_progress_reporter("both")

    assert isinstance(reporter, MultiProgressReporter)


def test_progress_heartbeat_emits_payload_until_context_exits() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    class Collector:
        def event(self, event_type: str, **payload: object) -> None:
            events.append((event_type, dict(payload)))

    with progress_heartbeat(
        Collector(),
        event_stage="install",
        message="still running",
        interval_seconds=0.01,
        payload_factory=lambda: {"step": "pip"},
    ):
        time.sleep(0.035)

    assert events
    assert all(event_type == "heartbeat" for event_type, _payload in events)
    assert events[0][1]["stage"] == "install"
    assert events[0][1]["step"] == "pip"


def test_progress_heartbeat_handles_payload_factory_errors() -> None:
    events: list[dict[str, object]] = []

    class Collector:
        def event(self, _event_type: str, **payload: object) -> None:
            events.append(dict(payload))

    def failing_payload() -> dict[str, object]:
        raise RuntimeError("boom")

    heartbeat = ProgressHeartbeat(
        Collector(),
        event_stage="install",
        message="still running",
        interval_seconds=0.01,
        payload_factory=failing_payload,
    )
    with heartbeat:
        time.sleep(0.025)

    assert events
    assert "boom" in str(events[0]["heartbeat_error"])
