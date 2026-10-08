from __future__ import annotations

import multiprocessing as mp
from queue import Empty
from pyroller.batch_cache import completed as has_completion, input_fingerprint, record_completion, completion_quality, receipt_path
from pyroller.progress import JsonlStageProgress

from pyroller.batch_models import (
    BatchRunSummary,
    BatchTask,
    BatchTaskResult,
    artifact_paths_for_request,
    batch_task_log_file,
)
from pyroller.i18n import _
from pyroller.logging_utils import configure_logging
from pyroller.pipeline import ComposablePipelineRunner
from pyroller.pipeline.execution_context import PipelineExecutionContext
from pyroller.process_control import install_worker_signal_handlers
from pyroller.progress import LoggingProgressReporter, ProgressReporter


class _ForwardStage(JsonlStageProgress):
    def __init__(self, reporter, name, total, unit):
        self.reporter = reporter
        super().__init__(name, total, unit)

    def _emit(self, event_type, **payload):
        payload.setdefault("stage", self.name)
        self.reporter.event(event_type, **payload)


class TaskProgress(ProgressReporter):
    def __init__(self, task_id, target=None, queue=None):
        self.task_id, self.target, self.queue = task_id, target, queue

    def event(self, event_type, **payload):
        payload["task_id"] = self.task_id
        if self.queue is not None:
            self.queue.put((event_type, payload))
        elif self.target is not None:
            self.target.event(event_type, **payload)

    def stage(self, name, *, total, unit="step"):
        return _ForwardStage(self, name, total, unit)


def _run_single_batch_task(task: BatchTask, execution_context: PipelineExecutionContext | None = None, progress_reporter=None) -> BatchTaskResult:
    log_file = batch_task_log_file(task.request.intermediate_dir)
    shared_context = execution_context is not None
    runner = ComposablePipelineRunner(
        progress_reporter=progress_reporter or LoggingProgressReporter(prefix=f"[{task.stem}] "),
        execution_context=execution_context or PipelineExecutionContext(),
    )
    try:
        fingerprint = input_fingerprint(task)
        run_result = runner.run(task.request)
        quality = run_result.alignment.report.get("quality") if run_result and run_result.alignment else None
        record_completion(task, fingerprint, quality)
        effective_request = getattr(runner, "last_request", task.request)
        log_file = batch_task_log_file(effective_request.intermediate_dir)
        cleaned = task.request.cleanup == "on-success" and not log_file.exists()
        return BatchTaskResult(
            index=task.index,
            stem=task.stem,
            status="ok",
            message=_("completed"),
            outputs=task.expected_outputs,
            log_file=None if cleaned else log_file,
            cleaned=cleaned,
            artifact_paths=artifact_paths_for_request(task.request),
            quality=quality,
        )
    except Exception as exc:
        log_file = batch_task_log_file(getattr(runner, "last_request", task.request).intermediate_dir)
        return BatchTaskResult(
            index=task.index,
            stem=task.stem,
            status="failed",
            message=str(exc),
            outputs=task.expected_outputs,
            log_file=log_file if log_file.exists() else None,
            cleaned=False,
            artifact_paths=artifact_paths_for_request(task.request),
            error={
                "type": exc.__class__.__name__,
                "code": getattr(exc, "code", "batch_task_failed"),
                "message": str(exc),
            },
        )
    finally:
        if not shared_context:
            runner.close()


def _worker_loop(task_queue, result_queue) -> None:
    install_worker_signal_handlers()
    shared_context = PipelineExecutionContext()
    try:
        while True:
            task = task_queue.get()
            if task is None:
                return
            reporter = TaskProgress(task.stem, queue=result_queue)
            reporter.event("batch_task_started", stage="batch", message=task.stem)
            result_queue.put(_run_single_batch_task(task, execution_context=shared_context, progress_reporter=reporter))
    finally:
        shared_context.close()


class BatchRunner:
    def run(
        self,
        tasks: list[BatchTask],
        *,
        continue_on_error: bool = False,
        skip_existing: bool = False,
        jobs: int = 1,
        progress_reporter: ProgressReporter | None = None,
    ) -> BatchRunSummary:
        if type(jobs) is not int or jobs < 1:
            raise ValueError("jobs must be a positive integer")
        if len({task.stem for task in tasks}) != len(tasks):
            raise ValueError("Batch task IDs must be unique")
        from dataclasses import fields
        input_paths = {getattr(task.request, field.name).resolve()
                       for task in tasks for field in fields(task.request)
                       if field.name.endswith("_path") and not field.name.startswith("output_")
                       and getattr(task.request, field.name) is not None}
        output_paths = [path.resolve() for task in tasks for path in task.expected_outputs]
        output_paths.extend(receipt_path(task).resolve() for task in tasks if task.expected_outputs)
        if len(set(output_paths)) != len(output_paths) or input_paths & set(output_paths):
            raise ValueError("Batch output paths must be unique and must not overwrite any task input")
        all_paths = list(input_paths) + output_paths
        if any(a in b.parents or b in a.parents for i, a in enumerate(all_paths) for b in all_paths[i + 1:]):
            raise ValueError("Batch input/output files cannot be ancestors of other input/output paths")
        results: list[BatchTaskResult] = []
        runnable: list[BatchTask] = []
        if progress_reporter is not None:
            progress_reporter.event("batch_started", stage="batch", total=len(tasks), completed=0, unit="task", message=_("batch started"))
        for task in tasks:
            if skip_existing and has_completion(task):
                result = BatchTaskResult(
                    index=task.index,
                    stem=task.stem,
                    status="skipped",
                    message="Verified matching input/configuration and output completion receipt",
                    quality=completion_quality(task),
                    outputs=task.expected_outputs,
                    artifact_paths=artifact_paths_for_request(task.request),
                )
                results.append(result)
                if progress_reporter is not None:
                    progress_reporter.event(
                        "batch_task_skipped",
                        stage="batch",
                        task_id=task.stem,
                        completed=len(results),
                        total=len(tasks),
                        unit="task",
                        message=result.message,
                        artifact_paths=result.artifact_paths,
                    )
            else:
                runnable.append(task)

        if jobs <= 1 or len(runnable) <= 1:
            shared_context = PipelineExecutionContext()
            try:
                for position, task in enumerate(runnable):
                    if progress_reporter is not None:
                        progress_reporter.event("batch_task_started", stage="batch", task_id=task.stem, completed=len(results), total=len(tasks), unit="task", message=task.stem)
                    result = _run_single_batch_task(task, execution_context=shared_context, progress_reporter=TaskProgress(task.stem, target=progress_reporter))
                    results.append(result)
                    if progress_reporter is not None:
                        progress_reporter.event(
                            "batch_task_completed" if result.status == "ok" else "batch_task_failed",
                            stage="batch",
                            task_id=task.stem,
                            completed=len(results),
                            total=len(tasks),
                            unit="task",
                            message=result.message,
                            artifact_paths=result.artifact_paths,
                            error=result.error,
                        )
                    if result.status == "failed" and not continue_on_error:
                        for remaining in runnable[position + 1 :]:
                            aborted = BatchTaskResult(
                                index=remaining.index,
                                stem=remaining.stem,
                                status="aborted",
                                message=_("batch stopped after earlier failure"),
                                outputs=remaining.expected_outputs,
                                artifact_paths=artifact_paths_for_request(remaining.request),
                            )
                            results.append(aborted)
                            if progress_reporter is not None:
                                progress_reporter.event(
                                    "batch_task_aborted",
                                    stage="batch",
                                    task_id=remaining.stem,
                                    completed=len(results),
                                    total=len(tasks),
                                    unit="task",
                                    message=aborted.message,
                                    artifact_paths=aborted.artifact_paths,
                                )
                        break
            finally:
                shared_context.close()
        else:
            ctx = mp.get_context("spawn")
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()
            workers = [ctx.Process(target=_worker_loop, args=(task_queue, result_queue), daemon=False) for _ in range(min(jobs, len(runnable)))]
            for worker in workers:
                worker.start()
            for task in runnable:
                task_queue.put(task)
            for _worker_sentinel in workers:
                task_queue.put(None)

            pending_stems = {task.stem for task in runnable}
            task_by_stem = {task.stem: task for task in runnable}
            aborted = False
            try:
                while pending_stems:
                    try:
                        result = result_queue.get(timeout=0.25)
                    except Empty:
                        dead = [worker for worker in workers if worker.exitcode not in (None, 0)]
                        if dead or all(worker.exitcode is not None for worker in workers):
                            for stem in sorted(pending_stems):
                                task = task_by_stem[stem]
                                error = {"type": "WorkerExitError", "code": "worker_exited", "message": "Worker exited before delivering a result"}
                                results.append(BatchTaskResult(task.index, stem, "failed", error["message"], task.expected_outputs, error=error,
                                                              artifact_paths=artifact_paths_for_request(task.request)))
                                if progress_reporter is not None:
                                    progress_reporter.event("batch_task_failed", stage="batch", task_id=stem, error=error, message=error["message"])
                            pending_stems.clear()
                            aborted = True
                            break
                        continue
                    if isinstance(result, tuple):
                        event_type, payload = result
                        if progress_reporter is not None:
                            progress_reporter.event(event_type, **payload)
                        continue
                    if result.stem not in pending_stems:
                        continue
                    pending_stems.remove(result.stem)
                    results.append(result)
                    if progress_reporter is not None:
                        progress_reporter.event(
                            "batch_task_completed" if result.status == "ok" else "batch_task_failed",
                            stage="batch",
                            task_id=result.stem,
                            completed=len(results),
                            total=len(tasks),
                            unit="task",
                            message=result.message,
                            artifact_paths=result.artifact_paths,
                            error=result.error,
                        )
                    if result.status == "failed" and not continue_on_error:
                        aborted = True
                        break
            finally:
                if aborted or pending_stems:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                for worker in workers:
                    worker.join(timeout=5)
                for worker in workers:
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=1)
                task_queue.cancel_join_thread()
                task_queue.close()
                result_queue.close()
            if aborted:
                for task in sorted((task_by_stem[stem] for stem in pending_stems), key=lambda item: item.index):
                    result = BatchTaskResult(
                        index=task.index,
                        stem=task.stem,
                        status="aborted",
                        message=_("batch stopped after earlier failure"),
                        outputs=task.expected_outputs,
                        artifact_paths=artifact_paths_for_request(task.request),
                    )
                    results.append(result)
                    if progress_reporter is not None:
                        progress_reporter.event(
                            "batch_task_aborted",
                            stage="batch",
                            task_id=task.stem,
                            completed=len(results),
                            total=len(tasks),
                            unit="task",
                            message=result.message,
                            artifact_paths=result.artifact_paths,
                        )

        completed = sum(1 for item in results if item.status == "ok")
        failed = sum(1 for item in results if item.status == "failed")
        skipped = sum(1 for item in results if item.status == "skipped")
        aborted_count = sum(1 for item in results if item.status == "aborted")
        summary = BatchRunSummary(
            total=len(tasks),
            completed=completed,
            failed=failed,
            skipped=skipped,
            aborted=aborted_count,
            results=sorted(results, key=lambda item: item.index),
        )
        if progress_reporter is not None:
            progress_reporter.event(
                "batch_completed" if failed == 0 else "batch_failed",
                stage="batch",
                completed=completed + skipped + aborted_count + failed,
                total=len(tasks),
                unit="task",
                progress=1.0,
                message=_("batch complete") if failed == 0 else _("batch finished with failures"),
                failed=failed > 0,
            )
        return summary
