from __future__ import annotations

from pathlib import Path

import pyroller.batch_runner as batch_runner
from pyroller.batch import BatchRunner, BatchTask, BatchTaskResult
from pyroller.domain import PipelineRequest


class RecordingProgress:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def event(self, event_type: str, **payload: object) -> None:
        self.events.append({"type": event_type, **payload})

    def stage(self, name: str, *, total: int, unit: str = "step"):
        raise AssertionError("BatchRunner should emit batch-level events directly")


def _task(index: int, stem: str, tmp_path: Path, *, outputs: list[Path] | None = None) -> BatchTask:
    return BatchTask(
        index=index,
        stem=stem,
        request=PipelineRequest(stages=["w"], intermediate_dir=tmp_path / stem),
        expected_outputs=outputs or [],
    )


def test_batch_runner_skips_tasks_when_all_outputs_exist(tmp_path) -> None:
    output = tmp_path / "done.lrc"
    output.write_text("done", encoding="utf-8")
    task = _task(1, "done", tmp_path, outputs=[output])

    summary = BatchRunner().run([task], skip_existing=True)

    assert summary.total == 1
    assert summary.skipped == 1
    assert summary.completed == 0
    assert summary.results[0].status == "skipped"


def test_batch_runner_emits_protocol_task_events(monkeypatch, tmp_path) -> None:
    output = tmp_path / "out.lrc"
    task = _task(1, "one", tmp_path, outputs=[output])
    progress = RecordingProgress()

    def fake_run_single(task, execution_context=None):
        return BatchTaskResult(
            task.index,
            task.stem,
            "ok",
            "completed",
            task.expected_outputs,
            artifact_paths={"roller": str(output)},
        )

    monkeypatch.setattr(batch_runner, "_run_single_batch_task", fake_run_single)

    summary = BatchRunner().run([task], progress_reporter=progress)

    assert summary.completed == 1
    assert [event["type"] for event in progress.events] == [
        "batch_started",
        "batch_task_started",
        "batch_task_completed",
        "batch_completed",
    ]
    assert progress.events[2]["task_id"] == "one"
    assert progress.events[2]["artifact_paths"] == {"roller": str(output)}


def test_batch_runner_aborts_remaining_tasks_after_first_failure(monkeypatch, tmp_path) -> None:
    tasks = [_task(1, "one", tmp_path), _task(2, "two", tmp_path), _task(3, "three", tmp_path)]

    def fake_run_single(task, execution_context=None):
        if task.stem == "one":
            return BatchTaskResult(task.index, task.stem, "failed", "boom", task.expected_outputs)
        return BatchTaskResult(task.index, task.stem, "ok", "completed", task.expected_outputs)

    monkeypatch.setattr(batch_runner, "_run_single_batch_task", fake_run_single)

    summary = BatchRunner().run(tasks, continue_on_error=False)

    assert summary.failed == 1
    assert summary.aborted == 2
    assert [result.status for result in summary.results] == ["failed", "aborted", "aborted"]


def test_batch_runner_continues_after_failure_when_requested(monkeypatch, tmp_path) -> None:
    tasks = [_task(1, "one", tmp_path), _task(2, "two", tmp_path)]

    def fake_run_single(task, execution_context=None):
        status = "failed" if task.stem == "one" else "ok"
        return BatchTaskResult(task.index, task.stem, status, status, task.expected_outputs)

    monkeypatch.setattr(batch_runner, "_run_single_batch_task", fake_run_single)

    summary = BatchRunner().run(tasks, continue_on_error=True)

    assert summary.failed == 1
    assert summary.completed == 1
    assert summary.aborted == 0
    assert [result.status for result in summary.results] == ["failed", "ok"]


def test_batch_runner_sorts_results_by_original_index(monkeypatch, tmp_path) -> None:
    tasks = [_task(2, "two", tmp_path), _task(1, "one", tmp_path)]

    monkeypatch.setattr(
        batch_runner,
        "_run_single_batch_task",
        lambda task, execution_context=None: BatchTaskResult(task.index, task.stem, "ok", "completed", task.expected_outputs),
    )

    summary = BatchRunner().run(tasks)

    assert [result.index for result in summary.results] == [1, 2]


def test_run_single_batch_task_returns_failed_result_and_log_path(monkeypatch, tmp_path) -> None:
    task = _task(1, "bad", tmp_path)
    log_file = task.request.intermediate_dir / "logs" / "run.log"
    log_file.parent.mkdir(parents=True)
    log_file.write_text("log", encoding="utf-8")

    class FailingRunner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, _request):
            raise RuntimeError("pipeline failed")

        def close(self) -> None:
            pass

    monkeypatch.setattr(batch_runner, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "ComposablePipelineRunner", FailingRunner)

    result = batch_runner._run_single_batch_task(task)

    assert result.status == "failed"
    assert result.message == "pipeline failed"
    assert result.log_file == log_file


def test_run_single_batch_task_closes_owned_runner(monkeypatch, tmp_path) -> None:
    task = _task(1, "ok", tmp_path)
    closed: list[bool] = []

    class SuccessfulRunner:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def run(self, _request):
            return None

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(batch_runner, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(batch_runner, "ComposablePipelineRunner", SuccessfulRunner)

    result = batch_runner._run_single_batch_task(task, execution_context=None)

    assert result.status == "ok"
    assert result.cleaned is True
    assert closed == [True]
