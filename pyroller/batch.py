from __future__ import annotations

import logging
import multiprocessing as mp
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from pyroller.i18n import _
from pyroller.domain import PipelineRequest
from pyroller.logging_utils import configure_logging
from pyroller.pipeline import ComposablePipelineRunner
from pyroller.pipeline.execution_context import PipelineExecutionContext
from pyroller.process_control import install_worker_signal_handlers
from pyroller.progress import LoggingProgressReporter, ProgressReporter

logger = logging.getLogger("pyroller.batch")

_MANIFEST_INPUT_KEYS = (
    "audio",
    "lyrics",
    "timed_units",
    "parsed_lyrics",
    "alignment_result",
)
_MANIFEST_OUTPUT_KEYS = (
    "output_vocal_audio",
    "output_filtered_audio",
    "output_timed_units",
    "output_parsed_lyrics",
    "output_alignment_result",
    "output_roller",
)
_MANIFEST_ALLOWED_KEYS = set(_MANIFEST_INPUT_KEYS + _MANIFEST_OUTPUT_KEYS + ("id",))


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


class BatchBuilder:
    def __init__(
        self,
        pair_by: str = "stem",
        audio_glob: str = "*.mp3",
        lyrics_glob: str = "*.txt",
        timed_units_glob: str = "*.json",
        parsed_lyrics_glob: str = "*.json",
        alignment_result_glob: str = "*.json",
    ) -> None:
        if pair_by != "stem":
            raise ValueError(_("Unsupported --pair-by: {}. Currently only 'stem' is supported.").format(pair_by))
        self.pair_by = pair_by
        self.audio_glob = audio_glob
        self.lyrics_glob = lyrics_glob
        self.timed_units_glob = timed_units_glob
        self.parsed_lyrics_glob = parsed_lyrics_glob
        self.alignment_result_glob = alignment_result_glob

    def build_tasks(self, request: PipelineRequest) -> list[BatchTask]:
        stages = ComposablePipelineRunner()._resolve_execution_plan(request)
        if not stages:
            raise ValueError(_("At least one stage is required for batch mode."))
        first_stage = stages[0]

        if first_stage in {"splitter", "filter", "transcriber"}:
            return self._build_from_audio_and_maybe_lyrics(request, stages)
        if first_stage == "parser":
            return self._build_from_lyrics_only(request)
        if first_stage == "aligner":
            return self._build_from_timed_and_parsed(request)
        if first_stage == "writer":
            return self._build_from_alignment_only(request)
        raise ValueError(_("Unsupported batch chain start: {}").format(first_stage))

    def _glob_dir(self, directory: Optional[Path], pattern: str, label: str) -> list[Path]:
        if directory is None:
            raise ValueError(_("Batch mode requires {} directory.").format(label))
        if not directory.exists() or not directory.is_dir():
            raise ValueError(_("{} must be an existing directory in batch mode: {}").format(label, directory))
        matches = sorted(p for p in directory.glob(pattern) if p.is_file())
        if not matches:
            raise ValueError(_("No files matched {!r} under {} for {}.").format(pattern, directory, label))
        return matches

    def _warn_unmatched(self, label: str, unmatched: Iterable[str]) -> None:
        items = sorted(set(unmatched))
        if items:
            logger.warning(_("Ignoring unmatched %s stems: %s"), label, ", ".join(items))

    def _map_by_stem(self, files: Iterable[Path]) -> dict[str, Path]:
        mapping: dict[str, Path] = {}
        duplicates: list[str] = []
        for path in files:
            stem = path.stem
            if stem in mapping:
                duplicates.append(stem)
            else:
                mapping[stem] = path
        if duplicates:
            dupes = ", ".join(sorted(set(duplicates)))
            raise ValueError(_("Duplicate stems are not allowed for --pair-by stem: {}").format(dupes))
        return mapping

    def _build_from_audio_and_maybe_lyrics(self, request: PipelineRequest, stages: list[str]) -> list[BatchTask]:
        audio_files = self._glob_dir(request.audio_path, self.audio_glob, "--audio")
        audio_by_stem = self._map_by_stem(audio_files)
        lyrics_by_stem: Optional[dict[str, Path]] = None
        if "parser" in stages:
            lyrics_files = self._glob_dir(request.lyrics_path, self.lyrics_glob, "--lyrics")
            lyrics_by_stem = self._map_by_stem(lyrics_files)
            matched = sorted(set(audio_by_stem) & set(lyrics_by_stem))
            self._warn_unmatched("audio", set(audio_by_stem) - set(matched))
            self._warn_unmatched("lyrics", set(lyrics_by_stem) - set(matched))
            if not matched:
                raise ValueError(_("No matched audio/lyrics pairs found by stem."))
            stems = matched
        else:
            stems = sorted(audio_by_stem)
        return [
            self._task_for_stem(
                index=index,
                stem=stem,
                request=request,
                audio_path=audio_by_stem[stem],
                lyrics_path=lyrics_by_stem[stem] if lyrics_by_stem is not None else None,
            )
            for index, stem in enumerate(stems, start=1)
        ]

    def _build_from_lyrics_only(self, request: PipelineRequest) -> list[BatchTask]:
        lyrics_files = self._glob_dir(request.lyrics_path, self.lyrics_glob, "--lyrics")
        lyrics_by_stem = self._map_by_stem(lyrics_files)
        return [
            self._task_for_stem(index=index, stem=stem, request=request, lyrics_path=path)
            for index, (stem, path) in enumerate(sorted(lyrics_by_stem.items()), start=1)
        ]

    def _build_from_timed_and_parsed(self, request: PipelineRequest) -> list[BatchTask]:
        timed_files = self._glob_dir(request.timed_units_path, self.timed_units_glob, "--timed-units")
        parsed_files = self._glob_dir(request.parsed_lyrics_path, self.parsed_lyrics_glob, "--parsed-lyrics")
        timed_by_stem = self._map_by_stem(timed_files)
        parsed_by_stem = self._map_by_stem(parsed_files)
        matched = sorted(set(timed_by_stem) & set(parsed_by_stem))
        self._warn_unmatched("timed-units", set(timed_by_stem) - set(matched))
        self._warn_unmatched("parsed-lyrics", set(parsed_by_stem) - set(matched))
        if not matched:
            raise ValueError(_("No matched timed_units/parsed_lyrics pairs found by stem."))
        return [
            self._task_for_stem(
                index=index,
                stem=stem,
                request=request,
                timed_units_path=timed_by_stem[stem],
                parsed_lyrics_path=parsed_by_stem[stem],
            )
            for index, stem in enumerate(matched, start=1)
        ]

    def _build_from_alignment_only(self, request: PipelineRequest) -> list[BatchTask]:
        alignment_files = self._glob_dir(request.alignment_result_path, self.alignment_result_glob, "--alignment-result")
        alignment_by_stem = self._map_by_stem(alignment_files)
        return [
            self._task_for_stem(index=index, stem=stem, request=request, alignment_result_path=path)
            for index, (stem, path) in enumerate(sorted(alignment_by_stem.items()), start=1)
        ]

    def _task_for_stem(
        self,
        index: int,
        stem: str,
        request: PipelineRequest,
        audio_path: Optional[Path] = None,
        lyrics_path: Optional[Path] = None,
        timed_units_path: Optional[Path] = None,
        parsed_lyrics_path: Optional[Path] = None,
        alignment_result_path: Optional[Path] = None,
    ) -> BatchTask:
        task_intermediate = request.intermediate_dir / stem
        task_request = PipelineRequest(
            stages=list(request.stages),
            audio_path=audio_path,
            lyrics_path=lyrics_path,
            timed_units_path=timed_units_path,
            parsed_lyrics_path=parsed_lyrics_path,
            alignment_result_path=alignment_result_path,
            language=request.language,
            intermediate_dir=task_intermediate,
            cleanup=request.cleanup,
            output_vocal_audio_path=_path_for_dir_output(request.output_vocal_audio_path, stem, suffix=".wav"),
            output_filtered_audio_path=_path_for_dir_output(request.output_filtered_audio_path, stem, suffix=".wav"),
            output_timed_units_path=_path_for_dir_output(request.output_timed_units_path, stem, suffix=".json"),
            output_parsed_lyrics_path=_path_for_dir_output(request.output_parsed_lyrics_path, stem, suffix=".json"),
            output_alignment_result_path=_path_for_dir_output(request.output_alignment_result_path, stem, suffix=".json"),
            output_roller_path=_path_for_dir_output(request.output_roller_path, stem, suffix=_roller_suffix(request.backend_config.get("writer", {}).get("backend"))),
            log_level=request.log_level,
            parser_lyrics_encoding=request.parser_lyrics_encoding,
            backend_config=request.backend_config,
        )
        expected_outputs = build_expected_outputs(task_request)
        return BatchTask(index=index, stem=stem, request=task_request, expected_outputs=expected_outputs)


class ManifestBatchBuilder:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def build_tasks(self, request: PipelineRequest) -> list[BatchTask]:
        runner = ComposablePipelineRunner()
        try:
            stages = runner._resolve_execution_plan(request)
            if not stages:
                raise ValueError(_("At least one stage is required for batch mode."))
            entries = self._load_entries()
            tasks: list[BatchTask] = []
            seen_stems: set[str] = set()
            output_owner: dict[Path, str] = {}
            for index, entry in enumerate(entries, start=1):
                if not isinstance(entry, dict):
                    raise ValueError(_("Manifest task #{} must be a mapping/object.").format(index))
                task = self._task_from_entry(index, entry, request)
                runner._validate_request(task.request, stages)
                if task.stem in seen_stems:
                    raise ValueError(_("Manifest task ids/stems must be unique. Duplicate: {}").format(task.stem))
                seen_stems.add(task.stem)
                for output_path in task.expected_outputs:
                    resolved = output_path.resolve()
                    owner = output_owner.get(resolved)
                    if owner is not None:
                        raise ValueError(
                            _("Manifest output path conflict: {} is declared by both '{}' and '{}'. "
                              "Each task must write to unique final output paths.").format(resolved, owner, task.stem)
                        )
                    output_owner[resolved] = task.stem
                tasks.append(task)
        finally:
            runner.close()
        if not tasks:
            raise ValueError(_("Manifest {} did not define any tasks.").format(self.manifest_path))
        return tasks

    def _load_entries(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists() or not self.manifest_path.is_file():
            raise ValueError(_("Manifest path must be an existing JSON/YAML file: {}").format(self.manifest_path))
        text = self.manifest_path.read_text(encoding="utf-8")
        if self.manifest_path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
        if data is None:
            raise ValueError(_("Manifest {} is empty.").format(self.manifest_path))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            tasks = data.get("tasks")
            if isinstance(tasks, list):
                return tasks
        raise ValueError(
            _("Manifest YAML must be either a top-level list of tasks or an object with a 'tasks' list. "
              "Example:\n"
              "tasks:\n"
              "  - id: song01\n"
              "    audio: ./audio/song01.mp3\n"
              "    lyrics: ./lyrics/song01.txt\n"
              "    output_roller: ./out/song01.lrc")
        )

    def _task_from_entry(self, index: int, entry: dict[str, Any], request: PipelineRequest) -> BatchTask:
        unknown = sorted(set(entry) - _MANIFEST_ALLOWED_KEYS)
        if unknown:
            raise ValueError(
                _("Manifest task #{} contains unsupported keys: {}. "
                  "Only input/output paths and optional 'id' are allowed. "
                  "Do not put stages, language, backend, jobs, or filter settings inside the manifest.").format(
                    index, ", ".join(unknown))
            )
        stem = self._derive_task_stem(index, entry)
        task_request = PipelineRequest(
            stages=list(request.stages),
            audio_path=self._path_value(entry, "audio"),
            lyrics_path=self._path_value(entry, "lyrics"),
            timed_units_path=self._path_value(entry, "timed_units"),
            parsed_lyrics_path=self._path_value(entry, "parsed_lyrics"),
            alignment_result_path=self._path_value(entry, "alignment_result"),
            language=request.language,
            intermediate_dir=request.intermediate_dir / stem,
            cleanup=request.cleanup,
            output_vocal_audio_path=self._path_value(entry, "output_vocal_audio"),
            output_filtered_audio_path=self._path_value(entry, "output_filtered_audio"),
            output_timed_units_path=self._path_value(entry, "output_timed_units"),
            output_parsed_lyrics_path=self._path_value(entry, "output_parsed_lyrics"),
            output_alignment_result_path=self._path_value(entry, "output_alignment_result"),
            output_roller_path=self._path_value(entry, "output_roller"),
            log_level=request.log_level,
            parser_lyrics_encoding=request.parser_lyrics_encoding,
            backend_config=request.backend_config,
        )
        expected_outputs = build_expected_outputs(task_request)
        return BatchTask(index=index, stem=stem, request=task_request, expected_outputs=expected_outputs)

    def _derive_task_stem(self, index: int, entry: dict[str, Any]) -> str:
        explicit = entry.get("id")
        if explicit is not None:
            stem = str(explicit).strip()
            if stem:
                return stem
            raise ValueError(_("Manifest task #{} has an empty 'id'.").format(index))
        for key in _MANIFEST_INPUT_KEYS + _MANIFEST_OUTPUT_KEYS:
            raw = entry.get(key)
            if raw is not None:
                return Path(str(raw)).stem
        raise ValueError(
            _("Manifest task #{} must declare at least one input or output path so a task id/stem can be derived.").format(index)
        )

    def _path_value(self, entry: dict[str, Any], key: str) -> Optional[Path]:
        raw = entry.get(key)
        if raw is None:
            return None
        value = str(raw).strip()
        if not value:
            raise ValueError(_("Manifest field '{}' cannot be an empty path.").format(key))
        path = Path(value)
        if not path.is_absolute():
            path = (self.manifest_path.parent / path).resolve()
        return path


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


def _path_for_dir_output(base: Optional[Path], stem: str, suffix: str) -> Optional[Path]:
    if base is None:
        return None
    return base / f"{stem}{suffix}"


def _roller_suffix(writer_backend: object) -> str:
    return ".ass" if str(writer_backend or "lrc_ms") == "ass_karaoke" else ".lrc"


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
