from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pyroller.batch import BatchBuilder, BatchRunner, ManifestBatchBuilder, batch_task_log_file
from pyroller.domain import PipelineRequest, RunPipelineResult
from pyroller.i18n import _
from pyroller.logging_utils import configure_logging
from pyroller.pipeline import ComposablePipelineRunner
from pyroller.progress import ProgressReporter, build_cli_progress_reporter
from pyroller.protocol import (
    ProtocolBatchOptions,
    batch_result_report,
    protocol_envelope,
    run_result_report,
)
from pyroller.transcriber.hf_download_config import HFDownloadConfig
from pyroller.transcriber.model_resolver import TranscriberModelResolver
from pyroller.transcriber.registry import resolve_transcriber_backend
from pyroller.utils.ids import make_id


@dataclass(slots=True)
class EngineRunResult:
    request: PipelineRequest
    result: RunPipelineResult
    report: dict[str, Any]


@dataclass(slots=True)
class EngineBatchResult:
    summary: Any
    report: dict[str, Any]


def prepare_single_run_request(request: PipelineRequest) -> PipelineRequest:
    return replace(request, intermediate_dir=request.intermediate_dir / make_id("run"))


def run_protocol_request(
    request: PipelineRequest,
    *,
    progress_reporter: ProgressReporter | None = None,
    progress_format: str = "jsonl",
) -> EngineRunResult:
    effective_request = prepare_single_run_request(request)
    log_file = batch_task_log_file(effective_request.intermediate_dir)
    configure_logging(level=effective_request.log_level, log_file=log_file)
    runner = ComposablePipelineRunner(progress_reporter=progress_reporter or build_cli_progress_reporter(progress_format))
    try:
        result = runner.run(effective_request)
    finally:
        runner.close()
    return EngineRunResult(
        request=effective_request,
        result=result,
        report=run_result_report(result, effective_request),
    )


def validate_batch_directory_outputs(request: PipelineRequest) -> None:
    for label, path in (
        ("--output-vocal-audio", request.output_vocal_audio_path),
        ("--output-filtered-audio", request.output_filtered_audio_path),
        ("--output-timed-units", request.output_timed_units_path),
        ("--output-parsed-lyrics", request.output_parsed_lyrics_path),
        ("--output-alignment-result", request.output_alignment_result_path),
        ("--output-roller", request.output_roller_path),
    ):
        if path is not None and path.exists() and not path.is_dir():
            raise ValueError(_("{} must be a directory in batch mode: {}").format(label, path))


def validate_manifest_batch_usage(request: PipelineRequest) -> None:
    for label, path in (
        ("--audio", request.audio_path),
        ("--lyrics", request.lyrics_path),
        ("--timed-units", request.timed_units_path),
        ("--parsed-lyrics", request.parsed_lyrics_path),
        ("--alignment-result", request.alignment_result_path),
        ("--output-vocal-audio", request.output_vocal_audio_path),
        ("--output-filtered-audio", request.output_filtered_audio_path),
        ("--output-timed-units", request.output_timed_units_path),
        ("--output-parsed-lyrics", request.output_parsed_lyrics_path),
        ("--output-alignment-result", request.output_alignment_result_path),
        ("--output-roller", request.output_roller_path),
    ):
        if path is not None:
            raise ValueError(
                _("{} cannot be used together with --manifest. Put per-task input/output paths inside the JSON/YAML manifest instead.").format(label)
            )


def run_batch_protocol_request(
    request: PipelineRequest,
    options: ProtocolBatchOptions,
    *,
    progress_reporter: ProgressReporter | None = None,
    progress_format: str = "jsonl",
) -> EngineBatchResult:
    if options.jobs < 1:
        raise ValueError(_("--jobs must be at least 1."))
    configure_logging(level=request.log_level, log_file=None)
    if options.jobs > 2:
        logging.getLogger("pyroller.engine").warning(
            _("Batch parallelism jobs=%d may be memory-heavy for audio pipelines. Consider jobs<=2 for stable CPU/GPU usage."),
            options.jobs,
        )

    if options.manifest is not None:
        validate_manifest_batch_usage(request)
        tasks = ManifestBatchBuilder(options.manifest).build_tasks(request)
    else:
        validate_batch_directory_outputs(request)
        runner = ComposablePipelineRunner()
        try:
            stages = runner._resolve_execution_plan(request)
            runner._validate_request(request, stages)
        finally:
            runner.close()
        tasks = BatchBuilder(
            pair_by=options.pair_by,
            audio_glob=options.audio_glob,
            lyrics_glob=options.lyrics_glob,
            timed_units_glob=options.timed_units_glob,
            parsed_lyrics_glob=options.parsed_lyrics_glob,
            alignment_result_glob=options.alignment_result_glob,
        ).build_tasks(request)

    if not tasks:
        raise ValueError(_("Batch mode found no runnable tasks."))
    summary = BatchRunner().run(
        tasks,
        continue_on_error=options.continue_on_error,
        skip_existing=options.skip_existing,
        jobs=options.jobs,
        progress_reporter=progress_reporter or build_cli_progress_reporter(progress_format),
    )
    return EngineBatchResult(summary=summary, report=batch_result_report(summary))


def cache_model_protocol_request(
    *,
    language: str,
    transcriber_backend: str | None,
    transcriber_model_name: str | None,
    transcriber_model_path: Path,
    transcriber_hf_xet: str | None = None,
    transcriber_hf_proxy: str | None = None,
    transcriber_hf_etag_timeout: int | None = None,
    transcriber_hf_download_timeout: int | None = None,
    transcriber_hf_max_workers: int | None = None,
    progress_reporter: ProgressReporter | None = None,
    progress_format: str = "jsonl",
) -> dict[str, Any]:
    configure_logging(level="INFO", log_file=None)
    progress = progress_reporter or build_cli_progress_reporter(progress_format)
    effective_language, backend = resolve_transcriber_backend(language, transcriber_backend)
    hf_config = HFDownloadConfig(
        xet=transcriber_hf_xet or "auto",
        proxy=transcriber_hf_proxy,
        etag_timeout=transcriber_hf_etag_timeout,
        download_timeout=transcriber_hf_download_timeout,
        max_workers=transcriber_hf_max_workers,
    )
    resolver = TranscriberModelResolver(
        backend=backend,
        language=effective_language,
        model_name=transcriber_model_name,
        model_path=transcriber_model_path,
        local_files_only=False,
        hf_download_config=hf_config,
    )
    stage = progress.stage("model_download", total=2, unit=_("phase"))
    stage.phase(_("resolving model"))
    plan = resolver.resolve(materialize=True, stage=stage)
    stage.phase(_("model cached"))
    stage.close(_("model download complete"))
    return protocol_envelope(
        "cache_model_result",
        artifact_paths={"model_dir": str(plan.resolved_model_dir)},
        backend=plan.backend,
        language=plan.language,
        effective_model_name=plan.effective_model_name,
        resolved_model_dir=plan.resolved_model_dir,
        model_store_root=plan.model_store_root,
    )


def doctor_protocol_request(report: Any | None = None) -> dict[str, Any]:
    from pyroller.cli.doctor import build_doctor_report

    report = report or build_doctor_report()
    return protocol_envelope(
        "doctor_result",
        status="ok" if report.ok else "failed",
        **report.to_dict(),
    )
