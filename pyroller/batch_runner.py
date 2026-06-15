from __future__ import annotations

import multiprocessing as mp

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


def _run_single_batch_task(task: BatchTask, execution_context: PipelineExecutionContext | None = None) -> BatchTaskResult:
    log_file = batch_task_log_file(task.request.intermediate_dir)
    configure_logging(level=task.request.log_level, log_file=log_file)
    shared_context = execution_context is not None
    runner = ComposablePipelineRunner(
        progress_reporter=LoggingProgressReporter(prefix=f"[{task.stem}] "),
        execution_context=execution_context or PipelineExecutionContext(),
    )
    try:
        runner.run(task.request)
        cleaned = task.request.cleanup == "on-success"
        return BatchTaskResult(
            index=task.index,
            stem=task.stem,
            status="ok",
            message=_("completed"),
            outputs=task.expected_outputs,
            log_file=None if cleaned else log_file,
            cleaned=cleaned,
            artifact_paths=artifact_paths_for_request(task.request),
        )
    except Exception as exc:
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
                "code": "batch_task_failed",
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
            result_queue.put(_run_single_batch_task(task, execution_context=shared_context))
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
        results: list[BatchTaskResult] = []
        runnable: list[BatchTask] = []
        if progress_reporter is not None:
            progress_reporter.event("batch_started", stage="batch", total=len(tasks), completed=0, unit="task", message=_("batch started"))
        for task in tasks:
            if skip_existing and task.expected_outputs and all(path.exists() for path in task.expected_outputs):
                result = BatchTaskResult(
                    index=task.index,
                    stem=task.stem,
                    status="skipped",
                    message=_("all declared outputs already exist"),
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
                    result = _run_single_batch_task(task, execution_context=shared_context)
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
                    result: BatchTaskResult = result_queue.get()
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
                if aborted:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                for worker in workers:
                    worker.join(timeout=5)
                for worker in workers:
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=1)
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
