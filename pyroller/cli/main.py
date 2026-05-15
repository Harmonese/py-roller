from __future__ import annotations

import argparse
import logging
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from pyroller.cli.config import apply_cli_config_defaults, load_cli_config, preparse_config_path

def _default_intermediate_dir() -> Path:
    return Path(tempfile.gettempdir()) / "py-roller-artifacts"

def _default_transcriber_model_path() -> Path:
    return Path.home() / ".cache" / "py-roller" / "models" / "transcriber"

def _positive_timeout_seconds_arg(value: str) -> int:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return max(1, int(math.ceil(seconds)))

def _build_subparser_description(*, batch_mode: bool) -> str:
    io_line = "All inputs/outputs are directories unless --manifest is used." if batch_mode else "All inputs/outputs are file paths."
    detail = (
        "Run the same contiguous stage chain across multiple tasks, either by stem-matching directories or by loading a YAML manifest."
        if batch_mode
        else "Run one contiguous pipeline stage chain. Inputs must match the first selected stage, and explicit artifacts are only allowed at legal chain starts."
    )
    language_hint = "For best transcription/parser defaults, pass --language zh or --language en when the song language is known."
    return f"{io_line}\n\n{detail}\n\n{language_hint}"

def _add_shared_runlike_arguments(parser: argparse.ArgumentParser, *, batch_mode: bool) -> None:
    stages_group = parser.add_argument_group("stages")
    stages_group.add_argument(
        "--stages",
        required=True,
        help=(
            "Comma-separated contiguous stage chain in canonical order s,f,t,p,a,w "
            "(splitter, filter, transcriber, parser, aligner, writer). "
            "Examples: s,f,t,p,a,w ; t,p,a,w ; a,w ; w. Do not skip intermediate stages."
        ),
    )

    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--audio", type=Path, default=None, help=("Input audio directory" if batch_mode else "Input audio file path"))
    inputs.add_argument("--lyrics", type=Path, default=None, help=("Input plain-text lyrics directory" if batch_mode else "Input plain-text lyrics file path"))
    inputs.add_argument("--timed-units", type=Path, default=None, help=("Input timed_units artifact directory" if batch_mode else "Input timed_units artifact JSON path"))
    inputs.add_argument("--parsed-lyrics", type=Path, default=None, help=("Input parsed_lyrics artifact directory" if batch_mode else "Input parsed_lyrics artifact JSON path"))
    inputs.add_argument("--alignment-result", type=Path, default=None, help=("Input alignment_result artifact directory" if batch_mode else "Input alignment_result artifact JSON path"))

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--output-vocal-audio", type=Path, default=None, help=("Output directory for final vocal audio artifacts" if batch_mode else "Output path for final vocal audio artifact"))
    outputs.add_argument("--output-filtered-audio", type=Path, default=None, help=("Output directory for final filtered audio artifacts" if batch_mode else "Output path for final filtered audio artifact"))
    outputs.add_argument("--output-timed-units", type=Path, default=None, help=("Output directory for final timed_units artifacts" if batch_mode else "Output path for final timed_units artifact"))
    outputs.add_argument("--output-parsed-lyrics", type=Path, default=None, help=("Output directory for final parsed_lyrics artifacts" if batch_mode else "Output path for final parsed_lyrics artifact"))
    outputs.add_argument("--output-alignment-result", type=Path, default=None, help=("Output directory for final alignment_result artifacts" if batch_mode else "Output path for final alignment_result artifact"))
    outputs.add_argument("--output-roller", type=Path, default=None, help=("Output directory for LRC/ASS writer results. Required when stages include w." if batch_mode else "Output path for the LRC/ASS writer result. Required when stages include w."))

    language = parser.add_argument_group("language and defaults")
    language.add_argument(
        "--language",
        choices=["zh", "en", "mul"],
        default="mul",
        help="Pipeline language. Use zh or en when known; mul is the multilingual fallback. Default: mul",
    )

    splitter = parser.add_argument_group("splitter options (stage s)")
    splitter.add_argument("--splitter-backend", default=None, help="Splitter backend override. Omit to use the language/backend default; use demucs for Demucs separation.")
    splitter.add_argument("--splitter-demucs-model", default=None, help="Demucs model name when --splitter-backend demucs, e.g. htdemucs.")
    splitter.add_argument("--splitter-demucs-device", default=None, help="Device passed to Demucs, e.g. cpu or cuda.")
    splitter.add_argument("--splitter-demucs-jobs", type=int, default=None, help="Demucs parallel jobs for separation.")
    splitter.add_argument("--splitter-demucs-overlap", type=float, default=None, help="Demucs chunk overlap ratio.")
    splitter.add_argument("--splitter-demucs-segment", type=float, default=None, help="Demucs chunk size in seconds.")

    filter_options = parser.add_argument_group("filter options (stage f)")
    filter_options.add_argument(
        "--filter-chain",
        default=None,
        help="Comma-separated filter steps for stage f, e.g. noise_gate,dereverb. YAML config may use a string or list.",
    )

    transcriber = parser.add_argument_group("transcriber options (stage t)")
    transcriber.add_argument("--transcriber-backend", default=None, help="Transcriber backend override. Defaults are language-aware; faster_whisper is the default for zh/en/mul.")
    transcriber.add_argument("--transcriber-device", default=None, help="Inference device passed to supported transcriber backends, e.g. cpu or cuda.")
    transcriber.add_argument("--transcriber-model-name", default=None, help="Model alias, Hugging Face repo id, or explicit local model path. faster-whisper aliases include large-v2, large-v3, and turbo.")
    transcriber.add_argument("--transcriber-model-path", type=Path, default=_default_transcriber_model_path(), help="Local transcriber model store. Missing remote models are materialized here unless --transcriber-local-files-only is set.")
    transcriber.add_argument("--transcriber-local-files-only", action=argparse.BooleanOptionalAction, default=None, help="Offline mode: do not access remote model sources; use only local files/cache.")
    transcriber.add_argument("--transcriber-compute-type", default=None, help="faster-whisper compute_type override, e.g. float16, int8, or int8_float16.")
    transcriber.add_argument("--transcriber-batch-size", type=int, default=None, help="faster-whisper inference batch size override.")
    transcriber.add_argument("--transcriber-vad-filter", action=argparse.BooleanOptionalAction, default=None, help="Enable faster-whisper VAD filtering to skip silence. Default: true")

    hf_download = parser.add_argument_group("Hugging Face model download options")
    hf_download.add_argument("--transcriber-hf-xet", choices=["auto", "on", "off"], default=None, help="XET/CAS download mode for Hugging Face models. Use off when XET hangs or fails on your network. Default: auto")
    hf_download.add_argument("--transcriber-hf-proxy", default=None, help="Proxy URL for Hugging Face model downloads, e.g. http://127.0.0.1:7890 or socks5://127.0.0.1:7890.")
    hf_download.add_argument("--transcriber-hf-etag-timeout", type=_positive_timeout_seconds_arg, default=None, help="Hugging Face metadata/etag timeout in seconds. Raise this on slow or high-latency networks.")
    hf_download.add_argument("--transcriber-hf-download-timeout", type=_positive_timeout_seconds_arg, default=None, help="Hugging Face file download timeout in seconds. Raise this when large model files time out.")
    hf_download.add_argument("--transcriber-hf-max-workers", type=int, default=None, help="Maximum parallel snapshot download workers. Lower values such as 1 or 2 can help fragile proxies.")

    parser_options = parser.add_argument_group("parser options (stage p)")
    parser_options.add_argument(
        "--parser-lyrics-encoding",
        default=None,
        choices=["auto", "utf-8", "utf-8-sig", "utf-16", "gbk", "gb18030", "shift-jis"],
        help="Lyrics text encoding. Default: auto",
    )

    aligner = parser.add_argument_group("aligner options (stage a)")
    aligner.add_argument("--aligner-backend", default=None, help="Aligner backend override. Omit to use the default global_dp_v1 backend.")
    aligner.add_argument("--aligner-min-gap", type=float, default=None, help="Minimum post-repair gap between aligned lyric lines, in seconds.")
    aligner.add_argument(
        "--aligner-repetition",
        choices=["none", "few", "full"],
        default=None,
        help=(
            "Repetition handling mode. "
            "none=standard global DP; "
            "few=repair sparse repeated/omitted regions between trusted anchors; "
            "full=anchorless candidate lattice for highly repetitive songs. "
            "Default: none"
        ),
    )

    writer = parser.add_argument_group("writer options (stage w)")
    writer.add_argument("--writer-backend", default=None, help="Writer backend override. Common values: lrc_ms, lrc_cs, lrc_compressed, ass_karaoke.")
    writer.add_argument("--writer-spacing", choices=["keep", "drop"], default=None, help="Whether to keep structural blank lyric lines in writer output. Default: keep")
    writer.add_argument("--writer-by-tag", default=None, help="Value for writer metadata BY tag in LRC/ASS outputs.")
    writer.add_argument("--writer-ass-karaoke-tag-type", choices=["k", "K", "kf", "ko"], default=None, help="ASS karaoke timing tag type when --writer-backend ass_karaoke. Default is writer-specific.")

    runtime = parser.add_argument_group("runtime control")
    runtime.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Load YAML defaults. Priority: built-in defaults < config file < explicit CLI flags.",
    )
    runtime.add_argument(
        "--intermediate",
        type=Path,
        default=_default_intermediate_dir(),
        help="Root directory for intermediate splitter/filter/log files. Use --cleanup never to keep these files after success.",
    )
    runtime.add_argument(
        "--cleanup",
        choices=["on-success", "never"],
        default="on-success",
        help="Intermediate cleanup policy: on-success removes successful task directories; never keeps them. Default: on-success",
    )
    runtime.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Console and file log verbosity. Use DEBUG for full tracebacks.")
    runtime.add_argument(
        "--progress-format",
        choices=["human", "jsonl", "both"],
        default="human",
        help=(
            "Progress output format. human keeps terminal-oriented progress; "
            "jsonl emits machine-readable PYROLLER_EVENT JSON lines for GUI frontends; "
            "both emits both. Default: human"
        ),
    )

    if batch_mode:
        batch = parser.add_argument_group("batch-only")
        batch.add_argument("--continue-on-error", action="store_true", help="Keep processing remaining tasks after failures.")
        batch.add_argument("--skip-existing", action="store_true", help="Skip tasks whose declared final outputs already exist.")
        batch.add_argument("--pair-by", choices=["stem"], default="stem", help="Directory pairing strategy. Current supported value: stem.")
        batch.add_argument("--jobs", type=int, default=1, help="Maximum number of parallel batch workers. For audio pipelines, start with 1. Default: 1")
        batch.add_argument("--audio-glob", default="*.mp3", help="Non-recursive glob for candidate audio files in batch mode. Default: *.mp3")
        batch.add_argument("--lyrics-glob", default="*.txt", help="Non-recursive glob for candidate lyric files in batch mode. Default: *.txt")
        batch.add_argument("--timed-units-glob", default="*.json", help="Non-recursive glob for candidate timed_units artifacts in batch mode. Default: *.json")
        batch.add_argument("--parsed-lyrics-glob", default="*.json", help="Non-recursive glob for candidate parsed_lyrics artifacts in batch mode. Default: *.json")
        batch.add_argument("--alignment-result-glob", default="*.json", help="Non-recursive glob for candidate alignment_result artifacts in batch mode. Default: *.json")
        batch.add_argument(
            "--manifest",
            type=Path,
            default=None,
            help="YAML manifest with per-task input/output file paths. When used, do not also pass batch input/output directories.",
        )

def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser]:
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
        prog="py-roller",
        description="Local lyric-audio alignment pipeline for LRC/ASS generation.",
        epilog=(
            "Common commands:\n"
            "  py-roller install\n"
            "  py-roller doctor\n"
            "  py-roller cache-model --language zh\n"
            "  py-roller run --help\n"
            "  py-roller batch --help"
        ),
        formatter_class=formatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run one contiguous pipeline stage chain for one song/task.",
        description=_build_subparser_description(batch_mode=False),
        epilog=(
            "Examples:\n"
            "  py-roller run --stages t,p,a,w --audio vocals.wav --lyrics song.txt --language zh --output-roller song.lrc\n"
            "  py-roller run --stages t,p,a,w --audio vocals.wav --lyrics song.txt --transcriber-hf-xet off --output-roller song.lrc"
        ),
        formatter_class=formatter,
    )
    _add_shared_runlike_arguments(run, batch_mode=False)

    batch = subparsers.add_parser(
        "batch",
        help="Run the same contiguous stage chain across many songs/tasks.",
        description=_build_subparser_description(batch_mode=True),
        epilog=(
            "Examples:\n"
            "  py-roller batch --stages t,p,a,w --audio audio_dir --lyrics lyrics_dir --language zh --output-roller out_dir\n"
            "  py-roller batch --stages t,p,a,w --manifest jobs.yaml --language zh --jobs 1"
        ),
        formatter_class=formatter,
    )
    _add_shared_runlike_arguments(batch, batch_mode=True)

    from pyroller.cli.install import build_install_parser as _build_install_parser
    install_spec = _build_install_parser()
    install = subparsers.add_parser(
        "install",
        help="Install/repair the official audio and transcriber runtime.",
        description=install_spec.description,
        formatter_class=formatter,
        parents=[install_spec],
        add_help=False,
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="Inspect the local audio, transcriber, and proxy environment.",
        description=(
            "Check whether torch, torchaudio, faster-whisper, CTranslate2, transformers, "
            "demucs, librosa, and SOCKS proxy support import successfully in the current environment."
        ),
        formatter_class=formatter,
    )
    doctor.add_argument(
        "--output-format",
        choices=["human", "json"],
        default="human",
        help="Doctor report output format. human prints the terminal checklist; json prints a machine-readable report. Default: human",
    )

    cache_model = subparsers.add_parser(
        "cache-model",
        help="Pre-download a transcriber model into the local model store.",
        description="Resolve and materialize a transcriber model into the local py-roller model store so later pipeline runs can use --transcriber-local-files-only.",
        formatter_class=formatter,
    )
    cache_model.add_argument("--language", choices=["zh", "en", "mul"], default="mul", help="Pipeline language for backend/model resolution. Default: mul")
    cache_model.add_argument("--transcriber-backend", default=None, help="Transcriber backend override. Defaults are language-aware.")
    cache_model.add_argument("--transcriber-model-name", default=None, help="Model alias, Hugging Face repo id, or explicit local model path. faster-whisper aliases include large-v2, large-v3, and turbo.")
    cache_model.add_argument("--transcriber-model-path", type=Path, default=_default_transcriber_model_path(), help="Local transcriber model store root.")
    cache_model.add_argument("--transcriber-hf-xet", choices=["auto", "on", "off"], default=None, help="XET/CAS download mode. Use off when XET hangs. Default: auto")
    cache_model.add_argument("--transcriber-hf-proxy", default=None, help="Proxy URL for Hugging Face model downloads.")
    cache_model.add_argument("--transcriber-hf-etag-timeout", type=_positive_timeout_seconds_arg, default=None, help="Hugging Face metadata/etag timeout in seconds.")
    cache_model.add_argument("--transcriber-hf-download-timeout", type=_positive_timeout_seconds_arg, default=None, help="Hugging Face file download timeout in seconds.")
    cache_model.add_argument("--transcriber-hf-max-workers", type=int, default=None, help="Maximum parallel snapshot download workers.")
    cache_model.add_argument("--progress-format", choices=["human", "jsonl", "both"], default="human", help="Progress output format. Default: human")

    return parser, run, batch

def _split_stages(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]

def _split_csv(value: object) -> list[str]:
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

def _auto_detect_transcriber_device() -> tuple[str | None, str | None]:
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


def _build_backend_config(args: argparse.Namespace) -> dict[str, object]:
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
        filter_cfg["chain"] = _split_csv(args.filter_chain)

    transcriber_cfg: dict[str, object] = {}
    if args.transcriber_backend is not None:
        transcriber_cfg["backend"] = args.transcriber_backend
    if args.transcriber_model_name is not None:
        transcriber_cfg["model_name"] = args.transcriber_model_name
    if args.transcriber_model_path is not None and Path(args.transcriber_model_path) != _default_transcriber_model_path():
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
        auto_device, auto_compute = _auto_detect_transcriber_device()
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

def _build_request(args: argparse.Namespace):
    from pyroller.domain import PipelineRequest

    return PipelineRequest(
        stages=_split_stages(args.stages),
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
        backend_config=_build_backend_config(args),
    )

def _print_run_summary(result, request) -> None:
    from pyroller.batch import batch_task_log_file

    print("[OK] pipeline complete")
    print(f"  executed stages        : {', '.join(result.executed_stages)}")
    if result.source_audio_artifact is not None:
        print(f"  input audio            : {result.source_audio_artifact.path} ({result.source_audio_artifact.role})")
    if result.current_audio_artifact is not None:
        print(f"  current audio          : {result.current_audio_artifact.path} ({result.current_audio_artifact.role})")
    if result.transcription is not None:
        print(f"  timed units            : {len(result.transcription.units)}")
    if result.parsed_lyrics is not None:
        print(f"  parsed lyric lines     : {len(result.parsed_lyrics.lines)}")
    if result.alignment is not None:
        print(f"  aligned lines          : {len(result.alignment.lines)}")
    for label, path in (
        ("output vocal audio", request.output_vocal_audio_path),
        ("output filtered audio", request.output_filtered_audio_path),
        ("output timed units", request.output_timed_units_path),
        ("output parsed lyrics", request.output_parsed_lyrics_path),
        ("output alignment", request.output_alignment_result_path),
        ("output roller", result.write_result.output_path if result.write_result is not None else None),
    ):
        if path is not None:
            print(f"  {label:<22}: {path}")
    print(f"  cleanup policy         : {request.cleanup}")
    if request.cleanup == "never":
        print(f"  intermediate dir       : {request.intermediate_dir}")
        print(f"  log file               : {batch_task_log_file(request.intermediate_dir)}")
    else:
        print("  intermediate dir       : cleaned after success")
        print("  log file               : cleaned after success")

def _print_batch_summary(summary) -> None:
    print("[OK] batch complete")
    print(f"  total tasks            : {summary.total}")
    print(f"  completed              : {summary.completed}")
    print(f"  failed                 : {summary.failed}")
    print(f"  skipped                : {summary.skipped}")
    print(f"  aborted                : {summary.aborted}")
    if summary.failed:
        print("  outcome                : finished with failures")
    elif summary.aborted:
        print("  outcome                : stopped after failure")
    elif summary.skipped and summary.completed == 0:
        print("  outcome                : nothing new to do")
    else:
        print("  outcome                : success")
    for item in summary.results:
        tag = {"ok": "OK", "skipped": "SKIP", "failed": "FAIL", "aborted": "ABORT"}.get(item.status, item.status.upper())
        print(f"  [{tag:<5}] #{item.index:03d} {item.stem} :: {item.message}")
        if item.outputs:
            print("           outputs       : " + ", ".join(str(path) for path in item.outputs))
        if item.log_file is not None:
            print(f"           log           : {item.log_file}")
        elif item.status == "ok" and item.cleaned:
            print("           log           : cleaned after success")

def _prepare_single_run_request(request):
    from pyroller.utils.ids import make_id

    run_id = make_id("run")
    return replace(request, intermediate_dir=request.intermediate_dir / run_id)

def _execute_run(request, *, progress_format: str = "human") -> None:
    from pyroller.batch import batch_task_log_file
    from pyroller.logging_utils import configure_logging
    from pyroller.pipeline import ComposablePipelineRunner
    from pyroller.progress import build_cli_progress_reporter

    effective_request = _prepare_single_run_request(request)
    log_file = batch_task_log_file(effective_request.intermediate_dir)
    configure_logging(level=effective_request.log_level, log_file=log_file)
    runner = ComposablePipelineRunner(progress_reporter=build_cli_progress_reporter(progress_format))
    try:
        result = runner.run(effective_request)
    finally:
        runner.close()
    _print_run_summary(result, effective_request)

def _validate_batch_directory_outputs(request) -> None:
    for label, path in (
        ("--output-vocal-audio", request.output_vocal_audio_path),
        ("--output-filtered-audio", request.output_filtered_audio_path),
        ("--output-timed-units", request.output_timed_units_path),
        ("--output-parsed-lyrics", request.output_parsed_lyrics_path),
        ("--output-alignment-result", request.output_alignment_result_path),
        ("--output-roller", request.output_roller_path),
    ):
        if path is not None and path.exists() and not path.is_dir():
            raise ValueError(f"{label} must be a directory in batch mode: {path}")

def _validate_manifest_batch_usage(request) -> None:
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
                f"{label} cannot be used together with --manifest. Put per-task input/output paths inside the YAML manifest instead."
            )

def _execute_batch(args: argparse.Namespace, request) -> int:
    from pyroller.batch import BatchBuilder, BatchRunner, ManifestBatchBuilder
    from pyroller.logging_utils import configure_logging
    from pyroller.pipeline import ComposablePipelineRunner

    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    configure_logging(level=request.log_level, log_file=None)
    if args.jobs > 2:
        logging.getLogger("pyroller.cli").warning(
            "Batch parallelism jobs=%d may be memory-heavy for audio pipelines. Consider jobs<=2 for stable CPU/GPU usage.",
            args.jobs,
        )

    if args.manifest is not None:
        _validate_manifest_batch_usage(request)
        tasks = ManifestBatchBuilder(args.manifest).build_tasks(request)
    else:
        _validate_batch_directory_outputs(request)
        runner = ComposablePipelineRunner()
        try:
            stages = runner._resolve_execution_plan(request)
            runner._validate_request(request, stages)
        finally:
            runner.close()
        tasks = BatchBuilder(
            pair_by=args.pair_by,
            audio_glob=args.audio_glob,
            lyrics_glob=args.lyrics_glob,
            timed_units_glob=args.timed_units_glob,
            parsed_lyrics_glob=args.parsed_lyrics_glob,
            alignment_result_glob=args.alignment_result_glob,
        ).build_tasks(request)

    if not tasks:
        raise ValueError("Batch mode found no runnable tasks.")
    summary = BatchRunner().run(
        tasks,
        continue_on_error=args.continue_on_error,
        skip_existing=args.skip_existing,
        jobs=args.jobs,
    )
    _print_batch_summary(summary)
    return 1 if summary.failed else 0

def _execute_cache_model(args: argparse.Namespace) -> int:
    from pyroller.logging_utils import configure_logging
    from pyroller.progress import build_cli_progress_reporter
    from pyroller.transcriber.hf_download_config import HFDownloadConfig
    from pyroller.transcriber.model_resolver import TranscriberModelResolver
    from pyroller.transcriber.registry import resolve_transcriber_backend

    configure_logging(level="INFO", log_file=None)
    progress = build_cli_progress_reporter(args.progress_format)
    language, backend = resolve_transcriber_backend(args.language, args.transcriber_backend)
    hf_config = HFDownloadConfig(
        xet=args.transcriber_hf_xet or "auto",
        proxy=args.transcriber_hf_proxy,
        etag_timeout=args.transcriber_hf_etag_timeout,
        download_timeout=args.transcriber_hf_download_timeout,
        max_workers=args.transcriber_hf_max_workers,
    )
    resolver = TranscriberModelResolver(
        backend=backend,
        language=language,
        model_name=args.transcriber_model_name,
        model_path=args.transcriber_model_path,
        local_files_only=False,
        hf_download_config=hf_config,
    )
    stage = progress.stage("model_download", total=2, unit="phase")
    stage.phase("resolving model")
    plan = resolver.resolve(materialize=True, stage=stage)
    stage.phase("model cached")
    stage.close("model download complete")
    print(f"[OK] model cached: {plan.effective_model_name}")
    print(f"  backend      : {plan.backend}")
    print(f"  language     : {plan.language}")
    print(f"  model dir    : {plan.resolved_model_dir}")
    print(f"  store root   : {plan.model_store_root}")
    return 0


def main() -> None:
    try:
        raw_argv = sys.argv[1:]
        config_path = preparse_config_path(raw_argv)
        parser, run_parser, batch_parser = build_parser()
        if config_path is not None:
            config = load_cli_config(config_path)
            apply_cli_config_defaults(run_parser=run_parser, batch_parser=batch_parser, config=config)
        args = parser.parse_args(raw_argv)
        if args.command == "doctor":
            from pyroller.cli.doctor import run_doctor
            raise SystemExit(run_doctor(output_format=args.output_format))
        if args.command == "install":
            from pyroller.cli.install import run_install_command
            raise SystemExit(run_install_command(args))
        if args.command == "cache-model":
            raise SystemExit(_execute_cache_model(args))

        request = _build_request(args)
        if args.command == "run":
            _execute_run(request, progress_format=args.progress_format)
            return
        if args.command == "batch":
            raise SystemExit(_execute_batch(args, request))
        raise ValueError(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        logging.getLogger("pyroller.cli").warning("Interrupted by user.")
        print("[ERROR] interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        cli_logger = logging.getLogger("pyroller.cli")
        if cli_logger.isEnabledFor(logging.DEBUG):
            cli_logger.exception("Pipeline command failed")
        else:
            cli_logger.error("Pipeline command failed: %s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
