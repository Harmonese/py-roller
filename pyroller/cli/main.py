from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from pyroller.batch import BatchBuilder, BatchRunner, ManifestBatchBuilder, batch_task_log_file
from pyroller.cli.config import apply_cli_config_defaults, load_cli_config, preparse_config_path
from pyroller.domain import PipelineRequest
from pyroller.logging_utils import configure_logging
from pyroller.pipeline import ComposablePipelineRunner
from pyroller.progress import build_cli_progress_reporter
from pyroller.utils.ids import make_id


def _default_intermediate_dir() -> Path:
    return Path(tempfile.gettempdir()) / "py-roller-artifacts"


def _add_shared_runlike_arguments(parser: argparse.ArgumentParser, *, batch_mode: bool) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config file for overriding CLI defaults. Priority is: built-in defaults < config file < explicit CLI flags.",
    )
    parser.add_argument(
        "--stages",
        required=True,
        help=(
            "Comma-separated contiguous stage chain. Allowed canonical order is s,f,t,p,a,w. "
            "Examples: s,f,t,p,a,w ; t,p,a,w ; a,w ; w."
        ),
    )

    parser.add_argument("--audio", type=Path, default=None, help=("Input audio directory" if batch_mode else "Input audio file path"))
    parser.add_argument("--lyrics", type=Path, default=None, help=("Input plain-text lyrics directory" if batch_mode else "Input plain-text lyrics file path"))
    parser.add_argument("--timed-units", type=Path, default=None, help=("Input timed_units artifact directory" if batch_mode else "Input timed_units artifact JSON path"))
    parser.add_argument("--parsed-lyrics", type=Path, default=None, help=("Input parsed_lyrics artifact directory" if batch_mode else "Input parsed_lyrics artifact JSON path"))
    parser.add_argument("--alignment-result", type=Path, default=None, help=("Input alignment_result artifact directory" if batch_mode else "Input alignment_result artifact JSON path"))

    parser.add_argument("--output-vocal-audio", type=Path, default=None, help=("Optional output directory for final vocal audio artifacts" if batch_mode else "Optional output path for final vocal audio artifact"))
    parser.add_argument("--output-filtered-audio", type=Path, default=None, help=("Optional output directory for final filtered audio artifacts" if batch_mode else "Optional output path for final filtered audio artifact"))
    parser.add_argument("--output-timed-units", type=Path, default=None, help=("Optional output directory for final timed_units artifacts" if batch_mode else "Optional output path for final timed_units artifact"))
    parser.add_argument("--output-parsed-lyrics", type=Path, default=None, help=("Optional output directory for final parsed_lyrics artifacts" if batch_mode else "Optional output path for final parsed_lyrics artifact"))
    parser.add_argument("--output-alignment-result", type=Path, default=None, help=("Optional output directory for final alignment_result artifacts" if batch_mode else "Optional output path for final alignment_result artifact"))
    parser.add_argument("--output-written", type=Path, default=None, help=("Required when stage chain includes writer. In batch mode this must be a directory." if batch_mode else "Required when stage chain includes writer."))

    parser.add_argument("--language", default="mul", help="Pipeline language. Supported values: zh, en, mul. Default: mul")
    parser.add_argument("--transcriber-backend", default=None, help="Optional transcriber backend override for the selected language")
    parser.add_argument("--aligner-backend", default=None, help="Optional aligner backend override. Internal default is used when omitted")
    parser.add_argument("--writer-backend", default=None, help="Optional writer backend override. Internal default is used when omitted")
    parser.add_argument(
        "--filter-chain",
        default=None,
        help="Filter chain for stage f. CLI accepts comma-separated steps, e.g. noise_gate,dereverb. YAML config may use either the same string or a YAML list.",
    )
    parser.add_argument(
        "--parser-lyrics-encoding",
        default=None,
        choices=["auto", "utf-8", "utf-8-sig", "utf-16", "gbk", "gb18030", "shift-jis"],
        help="Lyrics text encoding for stage p. Default: auto",
    )
    parser.add_argument("--reserve-spacing", dest="reserve_spacing", action="store_true", default=True, help="Preserve a single blank lyric line as a structural row. Default: enabled")
    parser.add_argument("--no-reserve-spacing", dest="reserve_spacing", action="store_false", help="Disable blank-line preservation")
    parser.add_argument(
        "--intermediate",
        type=Path,
        default=_default_intermediate_dir(),
        help="Root directory for intermediate splitter/filter/log files. The tool creates per-task splitter/, filter/, and logs/ subdirectories here.",
    )
    parser.add_argument(
        "--cleanup",
        choices=["on-success", "never"],
        default="on-success",
        help="Whether to remove intermediate directories after successful tasks. Default: on-success",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Console and file log verbosity")

    parser.add_argument("--transcriber-device", default=None, help="Inference device passed to supported transcriber backends")
    parser.add_argument("--transcriber-model-name", default=None, help="Optional transcriber model override")
    parser.add_argument("--transcriber-compute-type", default=None, help="Optional WhisperX compute type override")
    parser.add_argument("--transcriber-batch-size", type=int, default=None, help="Optional WhisperX inference batch size override")
    parser.add_argument("--transcriber-no-align-words", action="store_true", default=None, help="Disable WhisperX word alignment")
    parser.add_argument("--splitter-demucs-model", default=None, help="Optional Demucs model name override for stage s")

    parser.add_argument("--aligner-min-gap", type=float, default=None, help="Optional minimum post-repair gap between aligned lyric lines")
    parser.add_argument("--writer-by-tag", default=None, help="Optional writer BY tag used by LRC and ASS outputs")
    parser.add_argument("--writer-ass-karaoke-tag-type", choices=["k", "K", "kf", "ko"], default=None, help="ASS karaoke tag type when --writer-backend ass_karaoke")

    if batch_mode:
        parser.add_argument("--continue-on-error", action="store_true", help="Keep processing remaining tasks after a task failure")
        parser.add_argument("--skip-existing", action="store_true", help="Skip tasks whose declared final outputs already all exist")
        parser.add_argument("--pair-by", choices=["stem"], default="stem", help="Directory pairing strategy. Current supported value: stem")
        parser.add_argument("--jobs", type=int, default=1, help="Maximum number of parallel batch workers. Default: 1")
        parser.add_argument("--audio-glob", default="*.mp3", help="Non-recursive glob for candidate audio files in batch mode. Default: *.mp3")
        parser.add_argument("--lyrics-glob", default="*.txt", help="Non-recursive glob for candidate lyric files in batch mode. Default: *.txt")
        parser.add_argument(
            "--manifest",
            type=Path,
            default=None,
            help="Optional YAML manifest describing per-task input/output file paths. When used, do not also pass batch input/output directories.",
        )


def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="py-roller", description="Composable lyric-audio alignment pipeline with single-run and batch execution modes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run one contiguous pipeline stage chain for a single task.",
        description="Run one contiguous pipeline stage chain. Inputs must match the first selected stage, and explicit artifacts are only allowed at legal chain starts.",
    )
    _add_shared_runlike_arguments(run, batch_mode=False)

    batch = subparsers.add_parser(
        "batch",
        help="Run the same contiguous stage chain across multiple tasks.",
        description="Run the same contiguous stage chain across multiple tasks, either by stem-matching directories or by loading a YAML manifest.",
    )
    _add_shared_runlike_arguments(batch, batch_mode=True)
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


def _build_backend_config(args: argparse.Namespace) -> dict[str, object]:
    splitter_cfg: dict[str, object] = {"two_stems": "vocals"}
    if args.splitter_demucs_model is not None:
        splitter_cfg["model"] = args.splitter_demucs_model

    filter_cfg: dict[str, object] = {}
    if args.filter_chain is not None:
        filter_cfg["chain"] = _split_csv(args.filter_chain)

    transcriber_cfg: dict[str, object] = {}
    if args.transcriber_backend is not None:
        transcriber_cfg["backend"] = args.transcriber_backend
    if args.transcriber_model_name is not None:
        transcriber_cfg["model_name"] = args.transcriber_model_name
    if args.transcriber_device is not None:
        transcriber_cfg["device"] = args.transcriber_device
    if args.transcriber_compute_type is not None:
        transcriber_cfg["compute_type"] = args.transcriber_compute_type
    if args.transcriber_batch_size is not None:
        transcriber_cfg["batch_size"] = args.transcriber_batch_size
    if args.transcriber_no_align_words is not None:
        transcriber_cfg["align_words"] = not args.transcriber_no_align_words

    aligner_cfg: dict[str, object] = {}
    if args.aligner_backend is not None:
        aligner_cfg["backend"] = args.aligner_backend
    if args.aligner_min_gap is not None:
        aligner_cfg["min_gap"] = args.aligner_min_gap

    writer_cfg: dict[str, object] = {"reserve_spacing": args.reserve_spacing}
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


def _build_request(args: argparse.Namespace) -> PipelineRequest:
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
        output_written_path=args.output_written,
        log_level=args.log_level,
        reserve_spacing=args.reserve_spacing,
        parser_lyrics_encoding=args.parser_lyrics_encoding,
        backend_config=_build_backend_config(args),
    )


def _print_run_summary(result, request: PipelineRequest) -> None:
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
        ("written output", result.write_result.output_path if result.write_result is not None else None),
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


def _prepare_single_run_request(request: PipelineRequest) -> PipelineRequest:
    run_id = make_id("run")
    return replace(request, intermediate_dir=request.intermediate_dir / run_id)


def _execute_run(request: PipelineRequest) -> None:
    effective_request = _prepare_single_run_request(request)
    log_file = batch_task_log_file(effective_request.intermediate_dir)
    configure_logging(level=effective_request.log_level, log_file=log_file)
    result = ComposablePipelineRunner(progress_reporter=build_cli_progress_reporter()).run(effective_request)
    _print_run_summary(result, effective_request)


def _validate_batch_directory_outputs(request: PipelineRequest) -> None:
    for label, path in (
        ("--output-vocal-audio", request.output_vocal_audio_path),
        ("--output-filtered-audio", request.output_filtered_audio_path),
        ("--output-timed-units", request.output_timed_units_path),
        ("--output-parsed-lyrics", request.output_parsed_lyrics_path),
        ("--output-alignment-result", request.output_alignment_result_path),
        ("--output-written", request.output_written_path),
    ):
        if path is not None and path.exists() and not path.is_dir():
            raise ValueError(f"{label} must be a directory in batch mode: {path}")


def _validate_manifest_batch_usage(request: PipelineRequest) -> None:
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
        ("--output-written", request.output_written_path),
    ):
        if path is not None:
            raise ValueError(
                f"{label} cannot be used together with --manifest. Put per-task input/output paths inside the YAML manifest instead."
            )


def _execute_batch(args: argparse.Namespace, request: PipelineRequest) -> int:
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
        stages = runner._resolve_execution_plan(request)
        runner._validate_request(request, stages)
        tasks = BatchBuilder(pair_by=args.pair_by, audio_glob=args.audio_glob, lyrics_glob=args.lyrics_glob).build_tasks(request)

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


def main() -> None:
    try:
        raw_argv = sys.argv[1:]
        config_path = preparse_config_path(raw_argv)
        parser, run_parser, batch_parser = build_parser()
        if config_path is not None:
            config = load_cli_config(config_path)
            apply_cli_config_defaults(run_parser=run_parser, batch_parser=batch_parser, config=config)
        args = parser.parse_args(raw_argv)
        request = _build_request(args)
        if args.command == "run":
            _execute_run(request)
            return
        if args.command == "batch":
            raise SystemExit(_execute_batch(args, request))
        raise ValueError(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        logging.getLogger("pyroller.cli").warning("Interrupted by user.")
        print("[ERROR] interrupted by user", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        logging.getLogger("pyroller.cli").error("Pipeline command failed: %s", exc)
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
