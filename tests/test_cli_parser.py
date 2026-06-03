from __future__ import annotations

from pathlib import Path

import pytest

from pyroller.cli.main import _build_request, build_parser


def _parse(argv: list[str]):
    parser, _run_parser, _batch_parser = build_parser()
    return parser.parse_args(argv)


def test_run_help_exits_successfully(capsys) -> None:
    parser, _run_parser, _batch_parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run", "--help"])

    assert exc_info.value.code == 0
    assert "stage" in capsys.readouterr().out.lower()


def test_batch_help_exits_successfully(capsys) -> None:
    parser, _run_parser, _batch_parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["batch", "--help"])

    assert exc_info.value.code == 0
    assert "batch" in capsys.readouterr().out.lower()


def test_parse_run_command_builds_pipeline_request(tmp_path: Path) -> None:
    args = _parse(
        [
            "run",
            "--stages",
            "t,p,a,w",
            "--audio",
            str(tmp_path / "vocals.wav"),
            "--lyrics",
            str(tmp_path / "lyrics.txt"),
            "--language",
            "zh",
            "--filter-chain",
            "noise_gate,dereverb",
            "--writer-backend",
            "ass_karaoke",
            "--writer-ass-karaoke-tag-type",
            "K",
            "--writer-spacing",
            "drop",
            "--transcriber-hf-xet",
            "off",
            "--output-roller",
            str(tmp_path / "song.ass"),
        ]
    )

    request = _build_request(args)

    assert args.command == "run"
    assert request.stages == ["t", "p", "a", "w"]
    assert request.audio_path == tmp_path / "vocals.wav"
    assert request.lyrics_path == tmp_path / "lyrics.txt"
    assert request.language == "zh"
    assert request.output_roller_path == tmp_path / "song.ass"
    assert request.backend_config["filter"]["chain"] == ["noise_gate", "dereverb"]
    assert request.backend_config["writer"]["backend"] == "ass_karaoke"
    assert request.backend_config["writer"]["tag_type"] == "K"
    assert request.backend_config["writer"]["spacing"] == "drop"
    assert request.backend_config["transcriber"]["hf_xet"] == "off"


def test_parse_batch_command_preserves_batch_options(tmp_path: Path) -> None:
    args = _parse(
        [
            "batch",
            "--stages",
            "w",
            "--alignment-result",
            str(tmp_path / "alignments"),
            "--output-roller",
            str(tmp_path / "out"),
            "--jobs",
            "2",
            "--skip-existing",
            "--continue-on-error",
            "--alignment-result-glob",
            "*.alignment.json",
        ]
    )
    request = _build_request(args)

    assert args.command == "batch"
    assert args.jobs == 2
    assert args.skip_existing is True
    assert args.continue_on_error is True
    assert args.alignment_result_glob == "*.alignment.json"
    assert request.stages == ["w"]
    assert request.alignment_result_path == tmp_path / "alignments"
    assert request.output_roller_path == tmp_path / "out"


def test_parse_batch_protocol_request_does_not_require_stages(tmp_path: Path) -> None:
    args = _parse(["batch", "--request", str(tmp_path / "request.json"), "--progress-format", "jsonl", "--output-format", "json"])

    assert args.command == "batch"
    assert args.request == tmp_path / "request.json"
    assert args.stages is None


def test_cache_model_parser_accepts_download_options(tmp_path: Path) -> None:
    args = _parse(
        [
            "cache-model",
            "--language",
            "zh",
            "--transcriber-backend",
            "faster_whisper",
            "--transcriber-model-name",
            "turbo",
            "--transcriber-model-path",
            str(tmp_path / "models"),
            "--transcriber-hf-xet",
            "off",
            "--transcriber-hf-max-workers",
            "1",
            "--progress-format",
            "jsonl",
        ]
    )

    assert args.command == "cache-model"
    assert args.language == "zh"
    assert args.transcriber_backend == "faster_whisper"
    assert args.transcriber_model_name == "turbo"
    assert args.transcriber_model_path == tmp_path / "models"
    assert args.transcriber_hf_xet == "off"
    assert args.transcriber_hf_max_workers == 1
    assert args.progress_format == "jsonl"


def test_capabilities_parser_defaults_to_json() -> None:
    args = _parse(["capabilities"])

    assert args.command == "capabilities"
    assert args.output_format == "json"


def test_invalid_command_arguments_exit_with_error(capsys) -> None:
    parser, _run_parser, _batch_parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run", "--stages", "w", "--writer-spacing", "invalid"])

    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err.lower()
