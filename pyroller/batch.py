from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from pyroller.domain import PipelineRequest
from pyroller.logging_utils import configure_logging
from pyroller.pipeline import ComposablePipelineRunner
from pyroller.process_control import install_worker_signal_handlers
from pyroller.progress import LoggingProgressReporter

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
            raise ValueError(f"Unsupported --pair-by: {pair_by}. Currently only 'stem' is supported.")
        self.pair_by = pair_by
        self.audio_glob = audio_glob
        self.lyrics_glob = lyrics_glob
        self.timed_units_glob = timed_units_glob
        self.parsed_lyrics_glob = parsed_lyrics_glob
        self.alignment_result_glob = alignment_result_glob

    def build_tasks(self, request: PipelineRequest) -> list[BatchTask]:
        stages = ComposablePipelineRunner()._resolve_execution_plan(request)
        if not stages:
            raise ValueError("At least one stage is required for batch mode.")
        first_stage = stages[0]

        if first_stage in {"splitter", "filter", "transcriber"}:
            return self._build_from_audio_and_maybe_lyrics(request, stages)
        if first_stage == "parser":
            return self._build_from_lyrics_only(request)
        if first_stage == "aligner":
            return self._build_from_timed_and_parsed(request)
        if first_stage == "writer":
            return self._build_from_alignment_only(request)
        raise ValueError(f"Unsupported batch chain start: {first_stage}")

    def _glob_dir(self, directory: Optional[Path], pattern: str, label: str) -> list[Path]:
        if directory is None:
            raise ValueError(f"Batch mode requires {label} directory.")
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"{label} must be an existing directory in batch mode: {directory}")
        matches = sorted(p for p in directory.glob(pattern) if p.is_file())
        if not matches:
            raise ValueError(f"No files matched {pattern!r} under {directory} for {label}.")
        return matches

    def _warn_unmatched(self, label: str, unmatched: Iterable[str]) -> None:
        items = sorted(set(unmatched))
        if items:
            logger.warning("Ignoring unmatched %s stems: %s", label, ", ".join(items))

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
            raise ValueError(f"Duplicate stems are not allowed for --pair-by stem: {dupes}")
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
                raise ValueError("No matched audio/lyrics pairs found by stem.")
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
            raise ValueError("No matched timed_units/parsed_lyrics pairs found by stem.")
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
        stages = runner._resolve_execution_plan(request)
        if not stages:
            raise ValueError("At least one stage is required for batch mode.")
        entries = self._load_entries()
        tasks: list[BatchTask] = []
        seen_stems: set[str] = set()
        output_owner: dict[Path, str] = {}
        for index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"Manifest task #{index} must be a mapping/object.")
            task = self._task_from_entry(index, entry, request)
            runner._validate_request(task.request, stages)
            if task.stem in seen_stems:
                raise ValueError(f"Manifest task ids/stems must be unique. Duplicate: {task.stem}")
            seen_stems.add(task.stem)
            for output_path in task.expected_outputs:
                resolved = output_path.resolve()
                owner = output_owner.get(resolved)
                if owner is not None:
                    raise ValueError(
                        f"Manifest output path conflict: {resolved} is declared by both '{owner}' and '{task.stem}'. "
                        "Each task must write to unique final output paths."
                    )
                output_owner[resolved] = task.stem
            tasks.append(task)
        if not tasks:
            raise ValueError(f"Manifest {self.manifest_path} did not define any tasks.")
        return tasks

    def _load_entries(self) -> list[dict[str, Any]]:
        if not self.manifest_path.exists() or not self.manifest_path.is_file():
            raise ValueError(f"Manifest path must be an existing YAML file: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            raise ValueError(f"Manifest {self.manifest_path} is empty.")
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            tasks = data.get("tasks")
            if isinstance(tasks, list):
                return tasks
        raise ValueError(
            "Manifest YAML must be either a top-level list of tasks or an object with a 'tasks' list. "
            "Example:\n"
            "tasks:\n"
            "  - id: song01\n"
            "    audio: ./audio/song01.mp3\n"
            "    lyrics: ./lyrics/song01.txt\n"
            "    output_roller: ./out/song01.lrc"
        )

    def _task_from_entry(self, index: int, entry: dict[str, Any], request: PipelineRequest) -> BatchTask:
        unknown = sorted(set(entry) - _MANIFEST_ALLOWED_KEYS)
        if unknown:
            raise ValueError(
                f"Manifest task #{index} contains unsupported keys: {', '.join(unknown)}. "
                "Only input/output paths and optional 'id' are allowed. "
                "Do not put stages, language, backend, jobs, or filter settings inside the manifest."
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
            raise ValueError(f"Manifest task #{index} has an empty 'id'.")
        for key in _MANIFEST_INPUT_KEYS + _MANIFEST_OUTPUT_KEYS:
            raw = entry.get(key)
            if raw is not None:
                return Path(str(raw)).stem
        raise ValueError(
            f"Manifest task #{index} must declare at least one input or output path so a task id/stem can be derived."
        )

    def _path_value(self, entry: dict[str, Any], key: str) -> Optional[Path]:
        raw = entry.get(key)
        if raw is None:
            return None
        value = str(raw).strip()
        if not value:
            raise ValueError(f"Manifest field '{key}' cannot be an empty path.")
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


def _path_for_dir_output(base: Optional[Path], stem: str, suffix: str) -> Optional[Path]:
    if base is None:
        return None
    return base / f"{stem}{suffix}"


def _roller_suffix(writer_backend: object) -> str:
    return ".ass" if str(writer_backend or "lrc_ms") == "ass_karaoke" else ".lrc"


def _run_single_batch_task(task: BatchTask) -> BatchTaskResult:
    log_file = batch_task_log_file(task.request.intermediate_dir)
    configure_logging(level=task.request.log_level, log_file=log_file)
    try:
        ComposablePipelineRunner(
            progress_reporter=LoggingProgressReporter(prefix=f"[{task.stem}] ")
        ).run(task.request)
        cleaned = task.request.cleanup == "on-success"
        return BatchTaskResult(
            index=task.index,
            stem=task.stem,
            status="ok",
            message="completed",
            outputs=task.expected_outputs,
            log_file=None if cleaned else log_file,
            cleaned=cleaned,
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
        )


def _worker_loop(task_queue, result_queue) -> None:
    install_worker_signal_handlers()
    while True:
        task = task_queue.get()
        if task is None:
            return
        result_queue.put(_run_single_batch_task(task))


class BatchRunner:
    def run(
        self,
        tasks: list[BatchTask],
        *,
        continue_on_error: bool = False,
        skip_existing: bool = False,
        jobs: int = 1,
    ) -> BatchRunSummary:
        results: list[BatchTaskResult] = []
        runnable: list[BatchTask] = []
        for task in tasks:
            if skip_existing and task.expected_outputs and all(path.exists() for path in task.expected_outputs):
                results.append(BatchTaskResult(index=task.index, stem=task.stem, status="skipped", message="all declared outputs already exist", outputs=task.expected_outputs))
            else:
                runnable.append(task)

        if jobs <= 1 or len(runnable) <= 1:
            for position, task in enumerate(runnable):
                result = _run_single_batch_task(task)
                results.append(result)
                if result.status == "failed" and not continue_on_error:
                    for remaining in runnable[position + 1 :]:
                        results.append(BatchTaskResult(index=remaining.index, stem=remaining.stem, status="aborted", message="batch stopped after earlier failure", outputs=remaining.expected_outputs))
                    break
        else:
            ctx = mp.get_context("spawn")
            task_queue = ctx.Queue()
            result_queue = ctx.Queue()
            workers = [ctx.Process(target=_worker_loop, args=(task_queue, result_queue), daemon=False) for _ in range(min(jobs, len(runnable)))]
            for worker in workers:
                worker.start()
            for task in runnable:
                task_queue.put(task)
            for _ in workers:
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
                    results.append(BatchTaskResult(index=task.index, stem=task.stem, status="aborted", message="batch stopped after earlier failure", outputs=task.expected_outputs))

        completed = sum(1 for item in results if item.status == "ok")
        failed = sum(1 for item in results if item.status == "failed")
        skipped = sum(1 for item in results if item.status == "skipped")
        aborted_count = sum(1 for item in results if item.status == "aborted")
        return BatchRunSummary(
            total=len(tasks),
            completed=completed,
            failed=failed,
            skipped=skipped,
            aborted=aborted_count,
            results=sorted(results, key=lambda item: item.index),
        )
