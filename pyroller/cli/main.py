from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path

from pyroller.cli.config import apply_cli_config_defaults, load_cli_config, preparse_config_path
from pyroller.i18n import _, install_argparse_i18n
from pyroller.cli import runlike

def _default_intermediate_dir() -> Path:
    return Path(tempfile.gettempdir()) / "py-roller-artifacts"

def _default_transcriber_model_path() -> Path:
    return Path.home() / ".cache" / "py-roller" / "models" / "transcriber"

def _positive_timeout_seconds_arg(value: str) -> int:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(_("must be a number of seconds")) from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError(_("must be a finite number greater than 0"))
    return max(1, int(math.ceil(seconds)))

def _build_subparser_description(*, batch_mode: bool) -> str:
    io_line = _("All inputs/outputs are directories unless --manifest is used.") if batch_mode else _("All inputs/outputs are file paths.")
    detail = (
        _("Run the same contiguous stage chain across multiple tasks, either by stem-matching directories or by loading a JSON/YAML manifest.")
        if batch_mode
        else _("Run one contiguous pipeline stage chain. Inputs must match the first selected stage, and explicit artifacts are only allowed at legal chain starts.")
    )
    language_hint = _("For best transcription/parser defaults, pass --language zh or --language en when the song language is known. Display language auto-detects from LANG/LC_ALL; set PYROLLER_LANG=zh|ja|ko|pl|pt|sk to override.")
    return _("{}\n\n{}\n\n{}").format(io_line, detail, language_hint)

def _add_shared_runlike_arguments(parser: argparse.ArgumentParser, *, batch_mode: bool) -> None:
    stages_group = parser.add_argument_group(_("stages"))
    stages_group.add_argument(
        "--stages",
        required=False,
        help=_(
            "Comma-separated contiguous stage chain in canonical order s,f,t,p,a,w "
            "(splitter, filter, transcriber, parser, aligner, writer). "
            "Examples: s,f,t,p,a,w ; t,p,a,w ; a,w ; w. Do not skip intermediate stages."
        ),
    )

    inputs = parser.add_argument_group(_("inputs"))
    inputs.add_argument("--audio", type=Path, default=None, help=_("Input audio directory") if batch_mode else _("Input audio file path"))
    inputs.add_argument("--lyrics", type=Path, default=None, help=_("Input plain-text lyrics directory") if batch_mode else _("Input plain-text lyrics file path"))
    inputs.add_argument("--timed-units", type=Path, default=None, help=_("Input timed_units artifact directory") if batch_mode else _("Input timed_units artifact JSON path"))
    inputs.add_argument("--parsed-lyrics", type=Path, default=None, help=_("Input parsed_lyrics artifact directory") if batch_mode else _("Input parsed_lyrics artifact JSON path"))
    inputs.add_argument("--alignment-result", type=Path, default=None, help=_("Input alignment_result artifact directory") if batch_mode else _("Input alignment_result artifact JSON path"))

    outputs = parser.add_argument_group(_("outputs"))
    outputs.add_argument("--output-vocal-audio", type=Path, default=None, help=_("Output directory for final vocal audio artifacts") if batch_mode else _("Output path for final vocal audio artifact"))
    outputs.add_argument("--output-filtered-audio", type=Path, default=None, help=_("Output directory for final filtered audio artifacts") if batch_mode else _("Output path for final filtered audio artifact"))
    outputs.add_argument("--output-timed-units", type=Path, default=None, help=_("Output directory for final timed_units artifacts") if batch_mode else _("Output path for final timed_units artifact"))
    outputs.add_argument("--output-parsed-lyrics", type=Path, default=None, help=_("Output directory for final parsed_lyrics artifacts") if batch_mode else _("Output path for final parsed_lyrics artifact"))
    outputs.add_argument("--output-alignment-result", type=Path, default=None, help=_("Output directory for final alignment_result artifacts") if batch_mode else _("Output path for final alignment_result artifact"))
    outputs.add_argument("--output-roller", type=Path, default=None, help=_("Output directory for LRC/ASS writer results. Required when stages include w.") if batch_mode else _("Output path for the LRC/ASS writer result. Required when stages include w."))

    language = parser.add_argument_group(_("language and defaults"))
    language.add_argument(
        "--language",
        choices=["zh", "en", "mul"],
        default="mul",
        help=_("Pipeline language for transcription/parsing. Use zh or en when known; mul is the multilingual fallback. Default: mul. Display language (i18n) is controlled by LANG/LC_ALL/PYROLLER_LANG environment variables."),
    )

    splitter = parser.add_argument_group(_("splitter options (stage s)"))
    splitter.add_argument("--splitter-backend", default=None, help=_("Splitter backend override. Omit to use the language/backend default; use demucs for Demucs separation."))
    splitter.add_argument("--splitter-demucs-model", default=None, help=_("Demucs model name when --splitter-backend demucs, e.g. htdemucs."))
    splitter.add_argument("--splitter-demucs-device", default=None, help=_("Device passed to Demucs, e.g. cpu or cuda."))
    splitter.add_argument("--splitter-demucs-jobs", type=int, default=None, help=_("Demucs parallel jobs for separation."))
    splitter.add_argument("--splitter-demucs-overlap", type=float, default=None, help=_("Demucs chunk overlap ratio."))
    splitter.add_argument("--splitter-demucs-segment", type=float, default=None, help=_("Demucs chunk size in seconds."))

    filter_options = parser.add_argument_group(_("filter options (stage f)"))
    filter_options.add_argument(
        "--filter-chain",
        default=None,
        help=_("Comma-separated filter steps for stage f, e.g. noise_gate,dereverb. YAML config may use a string or list."),
    )

    transcriber = parser.add_argument_group(_("transcriber options (stage t)"))
    transcriber.add_argument("--transcriber-backend", default=None, help=_("Transcriber backend override. Defaults are language-aware; faster_whisper is the default for zh/en/mul."))
    transcriber.add_argument("--transcriber-device", default=None, help=_("Inference device passed to supported transcriber backends, e.g. cpu or cuda."))
    transcriber.add_argument("--transcriber-model-name", default=None, help=_("Model alias, Hugging Face repo id, or explicit local model path. faster-whisper aliases include large-v2, large-v3, and turbo."))
    transcriber.add_argument("--transcriber-model-path", type=Path, default=_default_transcriber_model_path(), help=_("Local transcriber model store. Missing remote models are materialized here unless --transcriber-local-files-only is set."))
    transcriber.add_argument("--transcriber-local-files-only", action=argparse.BooleanOptionalAction, default=None, help=_("Offline mode: do not access remote model sources; use only local files/cache."))
    transcriber.add_argument("--transcriber-compute-type", default=None, help=_("faster-whisper compute_type override, e.g. float16, int8, or int8_float16."))
    transcriber.add_argument("--transcriber-batch-size", type=int, default=None, help=_("faster-whisper inference batch size override."))
    transcriber.add_argument("--transcriber-vad-filter", action=argparse.BooleanOptionalAction, default=None, help=_("Enable faster-whisper VAD filtering to skip silence. Default: true"))

    hf_download = parser.add_argument_group(_("Hugging Face model download options"))
    hf_download.add_argument("--transcriber-hf-xet", choices=["auto", "on", "off"], default=None, help=_("XET/CAS download mode for Hugging Face models. Use off when XET hangs or fails on your network. Default: auto"))
    hf_download.add_argument("--transcriber-hf-proxy", default=None, help=_("Proxy URL for Hugging Face model downloads, e.g. http://127.0.0.1:7890 or socks5://127.0.0.1:7890."))
    hf_download.add_argument("--transcriber-hf-etag-timeout", type=_positive_timeout_seconds_arg, default=None, help=_("Hugging Face metadata/etag timeout in seconds. Raise this on slow or high-latency networks."))
    hf_download.add_argument("--transcriber-hf-download-timeout", type=_positive_timeout_seconds_arg, default=None, help=_("Hugging Face file download timeout in seconds. Raise this when large model files time out."))
    hf_download.add_argument("--transcriber-hf-max-workers", type=int, default=None, help=_("Maximum parallel snapshot download workers. Lower values such as 1 or 2 can help fragile proxies."))

    parser_options = parser.add_argument_group(_("parser options (stage p)"))
    parser_options.add_argument(
        "--parser-lyrics-encoding",
        default=None,
        choices=["auto", "utf-8", "utf-8-sig", "utf-16", "gbk", "gb18030", "shift-jis"],
        help=_("Lyrics text encoding. Default: auto"),
    )

    aligner = parser.add_argument_group(_("aligner options (stage a)"))
    aligner.add_argument("--aligner-backend", default=None, help=_("Aligner backend override. Omit to use the default global_dp_v1 backend."))
    aligner.add_argument("--aligner-min-gap", type=float, default=None, help=_("Minimum post-repair gap between aligned lyric lines, in seconds."))
    aligner.add_argument(
        "--aligner-repetition",
        choices=["none", "few", "full"],
        default=None,
        help=_(
            "Repetition handling mode. "
            "none=standard global DP; "
            "few=repair sparse repeated/omitted regions between trusted anchors; "
            "full=anchorless candidate lattice for highly repetitive songs. "
            "Default: none"
        ),
    )

    writer = parser.add_argument_group(_("writer options (stage w)"))
    writer.add_argument("--writer-backend", default=None, help=_("Writer backend override. Common values: lrc_ms, lrc_cs, lrc_compressed, ass_karaoke."))
    writer.add_argument("--writer-spacing", choices=["keep", "drop"], default=None, help=_("Whether to keep structural blank lyric lines in writer output. Default: keep"))
    writer.add_argument("--writer-by-tag", default=None, help=_("Value for writer metadata BY tag in LRC/ASS outputs."))
    writer.add_argument("--writer-ass-karaoke-tag-type", choices=["k", "K", "kf", "ko"], default=None, help=_("ASS karaoke timing tag type when --writer-backend ass_karaoke. Default is writer-specific."))

    runtime = parser.add_argument_group(_("runtime control"))
    runtime.add_argument(
        "--config",
        type=Path,
        default=None,
        help=_("Load YAML defaults. Priority: built-in defaults < config file < explicit CLI flags."),
    )
    runtime.add_argument(
        "--intermediate",
        type=Path,
        default=_default_intermediate_dir(),
        help=_("Root directory for intermediate splitter/filter/log files. Use --cleanup never to keep these files after success."),
    )
    runtime.add_argument(
        "--cleanup",
        choices=["on-success", "never"],
        default="on-success",
        help=_("Intermediate cleanup policy: on-success removes successful task directories; never keeps them. Default: on-success"),
    )
    runtime.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help=_("Console and file log verbosity. Use DEBUG for full tracebacks."))
    runtime.add_argument(
        "--progress-format",
        choices=["human", "jsonl", "both"],
        default="human",
        help=_(
            "Progress output format. human keeps terminal-oriented progress; "
            "jsonl emits machine-readable PYROLLER_EVENT JSON lines for GUI frontends; "
            "both emits both. Default: human"
        ),
    )
    runtime.add_argument(
        "--request",
        type=Path,
        default=None,
        help=_("Protocol v1 JSON request file for machine clients. When set, run/batch path and backend options are read from this file."),
    )
    runtime.add_argument(
        "--output-format",
        choices=["human", "json"],
        default="human",
        help=_("Final result output format. Use json for machine-readable protocol v1 reports. Default: human"),
    )

    if batch_mode:
        batch = parser.add_argument_group(_("batch-only"))
        batch.add_argument("--continue-on-error", action="store_true", help=_("Keep processing remaining tasks after failures."))
        batch.add_argument("--skip-existing", action="store_true", help=_("Skip tasks whose declared final outputs already exist."))
        batch.add_argument("--pair-by", choices=["stem"], default="stem", help=_("Directory pairing strategy. Current supported value: stem."))
        batch.add_argument("--jobs", type=int, default=1, help=_("Maximum number of parallel batch workers. For audio pipelines, start with 1. Default: 1"))
        batch.add_argument("--audio-glob", default="*.mp3", help=_("Non-recursive glob for candidate audio files in batch mode. Default: *.mp3"))
        batch.add_argument("--lyrics-glob", default="*.txt", help=_("Non-recursive glob for candidate lyric files in batch mode. Default: *.txt"))
        batch.add_argument("--timed-units-glob", default="*.json", help=_("Non-recursive glob for candidate timed_units artifacts in batch mode. Default: *.json"))
        batch.add_argument("--parsed-lyrics-glob", default="*.json", help=_("Non-recursive glob for candidate parsed_lyrics artifacts in batch mode. Default: *.json"))
        batch.add_argument("--alignment-result-glob", default="*.json", help=_("Non-recursive glob for candidate alignment_result artifacts in batch mode. Default: *.json"))
        batch.add_argument(
            "--manifest",
            type=Path,
            default=None,
            help=_("JSON/YAML manifest with per-task input/output file paths. When used, do not also pass batch input/output directories."),
        )

def build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser, argparse.ArgumentParser]:
    install_argparse_i18n()
    formatter = argparse.RawTextHelpFormatter
    parser = argparse.ArgumentParser(
        prog="py-roller",
        description=_("Local lyric-audio alignment pipeline for LRC/ASS generation."),
        epilog=_(
            "Common commands:\n"
            "  py-roller install\n"
            "  py-roller doctor\n"
            "  py-roller cache-model --language zh\n"
            "  PYROLLER_LANG=zh py-roller run --help\n"
            "  py-roller run --help\n"
            "  py-roller batch --help"
        ),
        formatter_class=formatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help=_("Run one contiguous pipeline stage chain for one song/task."),
        description=_build_subparser_description(batch_mode=False),
        epilog=_(
            "Examples:\n"
            "  py-roller run --stages t,p,a,w --audio vocals.wav --lyrics song.txt --language zh --output-roller song.lrc\n"
            "  py-roller run --stages t,p,a,w --audio vocals.wav --lyrics song.txt --transcriber-hf-xet off --output-roller song.lrc"
        ),
        formatter_class=formatter,
    )
    _add_shared_runlike_arguments(run, batch_mode=False)

    batch = subparsers.add_parser(
        "batch",
        help=_("Run the same contiguous stage chain across many songs/tasks."),
        description=_build_subparser_description(batch_mode=True),
        epilog=_(
            "Examples:\n"
            "  py-roller batch --stages t,p,a,w --audio audio_dir --lyrics lyrics_dir --language zh --output-roller out_dir\n"
            "  py-roller batch --stages t,p,a,w --manifest jobs.json --language zh --jobs 1"
        ),
        formatter_class=formatter,
    )
    _add_shared_runlike_arguments(batch, batch_mode=True)

    from pyroller.cli.install import build_install_parser as _build_install_parser
    install_spec = _build_install_parser()
    subparsers.add_parser(
        "install",
        help=_("Install/repair the official audio and transcriber runtime."),
        description=install_spec.description,
        formatter_class=formatter,
        parents=[install_spec],
        add_help=False,
    )

    doctor = subparsers.add_parser(
        "doctor",
        help=_("Inspect the local audio, transcriber, and proxy environment."),
        description=_(
            "Check whether torch, torchaudio, faster-whisper, CTranslate2, transformers, "
            "demucs, librosa, and SOCKS proxy support import successfully in the current environment."
        ),
        formatter_class=formatter,
    )
    doctor.add_argument(
        "--output-format",
        choices=["human", "json"],
        default="human",
        help=_("Doctor report output format. human prints the terminal checklist; json prints a machine-readable report. Default: human"),
    )

    cache_model = subparsers.add_parser(
        "cache-model",
        help=_("Pre-download a transcriber model into the local model store."),
        description=_("Resolve and materialize a transcriber model into the local py-roller model store so later pipeline runs can use --transcriber-local-files-only."),
        formatter_class=formatter,
    )
    cache_model.add_argument("--language", choices=["zh", "en", "mul"], default="mul", help=_("Pipeline language for backend/model resolution. Default: mul"))
    cache_model.add_argument("--transcriber-backend", default=None, help=_("Transcriber backend override. Defaults are language-aware."))
    cache_model.add_argument("--transcriber-model-name", default=None, help=_("Model alias, Hugging Face repo id, or explicit local model path. faster-whisper aliases include large-v2, large-v3, and turbo."))
    cache_model.add_argument("--transcriber-model-path", type=Path, default=_default_transcriber_model_path(), help=_("Local transcriber model store root."))
    cache_model.add_argument("--transcriber-hf-xet", choices=["auto", "on", "off"], default=None, help=_("XET/CAS download mode. Use off when XET hangs. Default: auto"))
    cache_model.add_argument("--transcriber-hf-proxy", default=None, help=_("Proxy URL for Hugging Face model downloads."))
    cache_model.add_argument("--transcriber-hf-etag-timeout", type=_positive_timeout_seconds_arg, default=None, help=_("Hugging Face metadata/etag timeout in seconds."))
    cache_model.add_argument("--transcriber-hf-download-timeout", type=_positive_timeout_seconds_arg, default=None, help=_("Hugging Face file download timeout in seconds."))
    cache_model.add_argument("--transcriber-hf-max-workers", type=int, default=None, help=_("Maximum parallel snapshot download workers."))
    cache_model.add_argument("--progress-format", choices=["human", "jsonl", "both"], default="human", help=_("Progress output format. Default: human"))
    cache_model.add_argument("--output-format", choices=["human", "json"], default="human", help=_("Final result output format. Default: human"))

    capabilities = subparsers.add_parser(
        "capabilities",
        help=_("Print py-roller protocol capabilities for machine clients."),
        formatter_class=formatter,
    )
    capabilities.add_argument("--output-format", choices=["json"], default="json", help=_("Capabilities output format. Default: json"))

    return parser, run, batch

# Compatibility aliases for tests and external callers that imported these helpers.
_split_stages = runlike.split_stages
_split_csv = runlike.split_csv
_auto_detect_transcriber_device = runlike.auto_detect_transcriber_device
_build_backend_config = runlike.build_backend_config
_build_request = runlike.build_request
_print_run_summary = runlike.print_run_summary
_print_batch_summary = runlike.print_batch_summary
_execute_run = runlike.execute_run
_execute_batch = runlike.execute_batch
_execute_cache_model = runlike.execute_cache_model

def main() -> None:
    try:
        raw_argv = sys.argv[1:]
        config_path = preparse_config_path(raw_argv)
        parser, run_parser, batch_parser = build_parser()
        if config_path is not None:
            config = load_cli_config(config_path)
            apply_cli_config_defaults(run_parser=run_parser, batch_parser=batch_parser, config=config)
        args = parser.parse_args(raw_argv)
        if args.command == "capabilities":
            from pyroller.protocol import as_jsonable, capabilities
            print(json.dumps(capabilities(), ensure_ascii=False, default=as_jsonable))
            return
        if args.command == "doctor":
            from pyroller.cli.doctor import run_doctor
            raise SystemExit(run_doctor(output_format=args.output_format))
        if args.command == "install":
            from pyroller.cli.install import run_install_command
            raise SystemExit(run_install_command(args))
        if args.command == "cache-model":
            raise SystemExit(runlike.execute_cache_model(args))

        if args.command == "run" and args.request is not None:
            from pyroller.protocol import pipeline_request_from_json
            request = pipeline_request_from_json(args.request)
        elif args.command == "batch" and args.request is not None:
            request = None
        else:
            request = runlike.build_request(args)
        if args.command == "run":
            runlike.execute_run(request, progress_format=args.progress_format, output_format=args.output_format)
            return
        if args.command == "batch":
            raise SystemExit(runlike.execute_batch(args, request))
        raise ValueError(_("Unknown command: {}").format(args.command))
    except KeyboardInterrupt:
        logging.getLogger("pyroller.cli").warning(_("Interrupted by user."))
        print(_("[ERROR] interrupted by user"), file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        cli_logger = logging.getLogger("pyroller.cli")
        if cli_logger.isEnabledFor(logging.DEBUG):
            cli_logger.exception(_("Pipeline command failed"))
        else:
            cli_logger.error(_("Pipeline command failed: %s"), exc)
        if "args" in locals() and getattr(args, "output_format", None) == "json":
            from pyroller.protocol import as_jsonable, error_report
            print(json.dumps(error_report(exc), ensure_ascii=False, default=as_jsonable))
        print(_("[ERROR] {}").format(exc), file=sys.stderr)
        raise SystemExit(1)

if __name__ == "__main__":
    main()
