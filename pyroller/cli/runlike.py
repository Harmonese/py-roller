from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyroller.i18n import _
from pyroller.protocol import ProtocolBatchOptions


def split_stages(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def split_csv(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            piece = str(item).strip()
            if piece:
                out.append(piece)
        return out
    return [str(value).strip()] if str(value).strip() else []


def auto_detect_transcriber_device() -> tuple[str | None, str | None]:
    """Return (device, compute_type) if a CUDA GPU is available, otherwise (None, None)."""
    try:
        import torch
    except ImportError:
        return None, None
    try:
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return None, None


def build_backend_config(args: argparse.Namespace) -> dict[str, object]:
    splitter_cfg: dict[str, object] = {"two_stems": "vocals"}
    if args.splitter_backend is not None:
        splitter_cfg["backend"] = args.splitter_backend
    if args.splitter_demucs_model is not None:
        splitter_cfg["model"] = args.splitter_demucs_model
    if args.splitter_demucs_device is not None:
        splitter_cfg["device"] = args.splitter_demucs_device
    if args.splitter_demucs_jobs is not None:
        splitter_cfg["jobs"] = args.splitter_demucs_jobs
    if args.splitter_demucs_overlap is not None:
        splitter_cfg["overlap"] = args.splitter_demucs_overlap
    if args.splitter_demucs_segment is not None:
        splitter_cfg["segment"] = args.splitter_demucs_segment

    filter_cfg: dict[str, object] = {}
    if args.filter_chain is not None:
        filter_cfg["chain"] = split_csv(args.filter_chain)

    transcriber_cfg: dict[str, object] = {}
    if args.transcriber_backend is not None:
        transcriber_cfg["backend"] = args.transcriber_backend
    if args.transcriber_model_name is not None:
        transcriber_cfg["model_name"] = args.transcriber_model_name
    if args.transcriber_model_path is not None and Path(args.transcriber_model_path) != default_transcriber_model_path():
        transcriber_cfg["model_path"] = args.transcriber_model_path
    if args.transcriber_local_files_only is not None:
        transcriber_cfg["local_files_only"] = args.transcriber_local_files_only
    if args.transcriber_hf_xet is not None:
        transcriber_cfg["hf_xet"] = args.transcriber_hf_xet
    if args.transcriber_hf_proxy is not None:
        transcriber_cfg["hf_proxy"] = args.transcriber_hf_proxy
    if args.transcriber_hf_etag_timeout is not None:
        transcriber_cfg["hf_etag_timeout"] = args.transcriber_hf_etag_timeout
    if args.transcriber_hf_download_timeout is not None:
        transcriber_cfg["hf_download_timeout"] = args.transcriber_hf_download_timeout
    if args.transcriber_hf_max_workers is not None:
        transcriber_cfg["hf_max_workers"] = args.transcriber_hf_max_workers
    if args.transcriber_device is not None:
        transcriber_cfg["device"] = args.transcriber_device
    if args.transcriber_compute_type is not None:
        transcriber_cfg["compute_type"] = args.transcriber_compute_type
    if args.transcriber_batch_size is not None:
        transcriber_cfg["batch_size"] = args.transcriber_batch_size
    if args.transcriber_vad_filter is not None:
        transcriber_cfg["vad_filter"] = args.transcriber_vad_filter

    if transcriber_cfg.get("device") is None:
        auto_device, auto_compute = auto_detect_transcriber_device()
        if auto_device is not None:
            transcriber_cfg["device"] = auto_device
            if transcriber_cfg.get("compute_type") is None:
                transcriber_cfg["compute_type"] = auto_compute

    aligner_cfg: dict[str, object] = {}
    if args.aligner_backend is not None:
        aligner_cfg["backend"] = args.aligner_backend
    if args.aligner_min_gap is not None:
        aligner_cfg["min_gap"] = args.aligner_min_gap
    if args.aligner_repetition is not None:
        aligner_cfg["repetition"] = args.aligner_repetition

    writer_cfg: dict[str, object] = {"spacing": args.writer_spacing or "keep"}
    if args.writer_backend is not None:
        writer_cfg["backend"] = args.writer_backend
    if args.writer_by_tag is not None:
        writer_cfg["by_tag"] = args.writer_by_tag
    if args.writer_ass_karaoke_tag_type is not None:
        writer_cfg["tag_type"] = args.writer_ass_karaoke_tag_type

    return {
        "splitter": splitter_cfg,
        "filter": filter_cfg,
        "parser": {},
        "transcriber": transcriber_cfg,
        "aligner": aligner_cfg,
        "writer": writer_cfg,
    }


def build_request(args: argparse.Namespace):
    from pyroller.domain import PipelineRequest

    if not args.stages:
        raise ValueError(_("--stages is required unless --request is used."))
    return PipelineRequest(
        stages=split_stages(args.stages),
        audio_path=args.audio,
        lyrics_path=args.lyrics,
        timed_units_path=args.timed_units,
        parsed_lyrics_path=args.parsed_lyrics,
        alignment_result_path=args.alignment_result,
        language=args.language,
        intermediate_dir=args.intermediate,
        cleanup=args.cleanup,
        output_vocal_audio_path=args.output_vocal_audio,
        output_filtered_audio_path=args.output_filtered_audio,
        output_timed_units_path=args.output_timed_units,
        output_parsed_lyrics_path=args.output_parsed_lyrics,
        output_alignment_result_path=args.output_alignment_result,
        output_roller_path=args.output_roller,
        log_level=args.log_level,
        parser_lyrics_encoding=args.parser_lyrics_encoding,
        backend_config=build_backend_config(args),
    )


def print_run_summary(result, request) -> None:
    from pyroller.batch import batch_task_log_file

    print(_("[OK] pipeline complete"))
    print(_("  executed stages        : {}").format(", ".join(result.executed_stages)))
    if result.source_audio_artifact is not None:
        print(_("  input audio            : {} ({})").format(result.source_audio_artifact.path, result.source_audio_artifact.role))
    if result.current_audio_artifact is not None:
        print(_("  current audio          : {} ({})").format(result.current_audio_artifact.path, result.current_audio_artifact.role))
    if result.transcription is not None:
        print(_("  timed units            : {}").format(len(result.transcription.units)))
    if result.parsed_lyrics is not None:
        print(_("  parsed lyric lines     : {}").format(len(result.parsed_lyrics.lines)))
    if result.alignment is not None:
        print(_("  aligned lines          : {}").format(len(result.alignment.lines)))
    for label, path in (
        ("output vocal audio", request.output_vocal_audio_path),
        ("output filtered audio", request.output_filtered_audio_path),
        ("output timed units", request.output_timed_units_path),
        ("output parsed lyrics", request.output_parsed_lyrics_path),
        ("output alignment", request.output_alignment_result_path),
        ("output roller", result.write_result.output_path if result.write_result is not None else None),
    ):
        if path is not None:
            print(_("  {label:<22}: {path}").format(label=label, path=path))
    print(_("  cleanup policy         : {}").format(request.cleanup))
    if request.cleanup == "never":
        print(_("  intermediate dir       : {}").format(request.intermediate_dir))
        print(_("  log file               : {}").format(batch_task_log_file(request.intermediate_dir)))
    else:
        print(_("  intermediate dir       : cleaned after success"))
        print(_("  log file               : cleaned after success"))


def print_batch_summary(summary) -> None:
    print(_("[OK] batch complete"))
    print(_("  total tasks            : {}").format(summary.total))
    print(_("  completed              : {}").format(summary.completed))
    print(_("  failed                 : {}").format(summary.failed))
    print(_("  skipped                : {}").format(summary.skipped))
    print(_("  aborted                : {}").format(summary.aborted))
    if summary.failed:
        print(_("  outcome                : finished with failures"))
    elif summary.aborted:
        print(_("  outcome                : stopped after failure"))
    elif summary.skipped and summary.completed == 0:
        print(_("  outcome                : nothing new to do"))
    else:
        print(_("  outcome                : success"))
    for item in summary.results:
        tag = {"ok": _("OK"), "skipped": _("SKIP"), "failed": _("FAIL"), "aborted": _("ABORT")}.get(item.status, item.status.upper())
        print(_("  [{tag:<5}] #{index:03d} {stem} :: {msg}").format(tag=tag, index=item.index, stem=item.stem, msg=item.message))
        if item.outputs:
            print(_("           outputs       : {}").format(", ".join(str(path) for path in item.outputs)))
        if item.log_file is not None:
            print(_("           log           : {}").format(item.log_file))
        elif item.status == "ok" and item.cleaned:
            print(_("           log           : cleaned after success"))


def execute_run(request, *, progress_format: str = "human", output_format: str = "human") -> None:
    from pyroller.engine import run_protocol_request
    from pyroller.protocol import as_jsonable

    engine_result = run_protocol_request(request, progress_format=progress_format)
    if output_format == "json":
        print(json.dumps(engine_result.report, ensure_ascii=False, default=as_jsonable))
    else:
        print_run_summary(engine_result.result, engine_result.request)


def execute_batch(args: argparse.Namespace, request) -> int:
    from pyroller.engine import run_batch_protocol_request
    from pyroller.protocol import as_jsonable, batch_request_from_json

    if args.request is not None:
        protocol_request = batch_request_from_json(args.request)
        request = protocol_request.request
        options = protocol_request.options
        args.continue_on_error = options.continue_on_error
        args.skip_existing = options.skip_existing
        args.jobs = options.jobs
        args.manifest = options.manifest
        args.pair_by = options.pair_by
        args.audio_glob = options.audio_glob
        args.lyrics_glob = options.lyrics_glob
        args.timed_units_glob = options.timed_units_glob
        args.parsed_lyrics_glob = options.parsed_lyrics_glob
        args.alignment_result_glob = options.alignment_result_glob
    options = ProtocolBatchOptions(
        continue_on_error=args.continue_on_error,
        skip_existing=args.skip_existing,
        jobs=args.jobs,
        manifest=args.manifest,
        pair_by=args.pair_by,
        audio_glob=args.audio_glob,
        lyrics_glob=args.lyrics_glob,
        timed_units_glob=args.timed_units_glob,
        parsed_lyrics_glob=args.parsed_lyrics_glob,
        alignment_result_glob=args.alignment_result_glob,
    )
    engine_result = run_batch_protocol_request(request, options, progress_format=args.progress_format)
    if args.output_format == "json":
        print(json.dumps(engine_result.report, ensure_ascii=False, default=as_jsonable))
    else:
        print_batch_summary(engine_result.summary)
    return 1 if engine_result.summary.failed else 0


def execute_cache_model(args: argparse.Namespace) -> int:
    from pyroller.engine import cache_model_protocol_request
    from pyroller.protocol import as_jsonable

    report = cache_model_protocol_request(
        language=args.language,
        transcriber_backend=args.transcriber_backend,
        transcriber_model_name=args.transcriber_model_name,
        transcriber_model_path=args.transcriber_model_path,
        transcriber_hf_xet=args.transcriber_hf_xet,
        transcriber_hf_proxy=args.transcriber_hf_proxy,
        transcriber_hf_etag_timeout=args.transcriber_hf_etag_timeout,
        transcriber_hf_download_timeout=args.transcriber_hf_download_timeout,
        transcriber_hf_max_workers=args.transcriber_hf_max_workers,
        progress_format=args.progress_format,
    )
    if args.output_format == "json":
        print(json.dumps(report, ensure_ascii=False, default=as_jsonable))
    else:
        print(_("[OK] model cached: {}").format(report["effective_model_name"]))
        print(_("  backend      : {}").format(report["backend"]))
        print(_("  language     : {}").format(report["language"]))
        print(_("  model dir    : {}").format(report["resolved_model_dir"]))
        print(_("  store root   : {}").format(report["model_store_root"]))
    return 0


def default_transcriber_model_path() -> Path:
    return Path.home() / ".cache" / "py-roller" / "models" / "transcriber"
