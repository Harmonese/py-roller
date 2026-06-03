from __future__ import annotations

from pathlib import Path

import pytest

from pyroller.cli.config import (
    ConfigError,
    apply_cli_config_defaults,
    load_cli_config,
    preparse_config_path,
)
from pyroller.cli.main import _build_request, build_parser


def test_preparse_config_path_finds_config_after_command(tmp_path: Path) -> None:
    config = tmp_path / "pyroller.yaml"

    assert preparse_config_path(["run", "--config", str(config), "--stages", "w"]) == config


def test_load_flat_config_as_shared_defaults(tmp_path: Path) -> None:
    config = tmp_path / "pyroller.yaml"
    config.write_text(
        """
language: zh
filter_chain:
  - noise_gate
transcriber_hf_xet: off
intermediate: work
""",
        encoding="utf-8",
    )

    loaded = load_cli_config(config)

    assert loaded["shared"]["language"] == "zh"
    assert loaded["shared"]["filter_chain"] == ["noise_gate"]
    assert loaded["shared"]["transcriber_hf_xet"] is False
    assert loaded["shared"]["intermediate"] == "work"


def test_load_sectioned_config_merges_run_into_shared_and_keeps_batch(tmp_path: Path) -> None:
    config = tmp_path / "pyroller.yaml"
    config.write_text(
        """
shared:
  language: mul
run:
  writer_backend: ass_karaoke
batch:
  jobs: 2
  audio_glob: "*.wav"
""",
        encoding="utf-8",
    )

    loaded = load_cli_config(config)

    assert loaded["shared"]["language"] == "mul"
    assert loaded["shared"]["writer_backend"] == "ass_karaoke"
    assert loaded["batch"]["jobs"] == 2
    assert loaded["batch"]["audio_glob"] == "*.wav"


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config = tmp_path / "pyroller.yaml"
    config.write_text("shared:\n  not_a_real_option: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unsupported config keys"):
        load_cli_config(config)


def test_config_defaults_apply_to_run_and_cli_overrides_config(tmp_path: Path) -> None:
    config = tmp_path / "pyroller.yaml"
    config.write_text(
        """
shared:
  language: zh
  writer_backend: lrc_cs
  writer_spacing: drop
  intermediate: work
  transcriber_hf_xet: off
""",
        encoding="utf-8",
    )
    parser, run_parser, batch_parser = build_parser()
    apply_cli_config_defaults(
        run_parser=run_parser,
        batch_parser=batch_parser,
        config=load_cli_config(config),
    )

    args = parser.parse_args(
        [
            "run",
            "--config",
            str(config),
            "--stages",
            "w",
            "--alignment-result",
            str(tmp_path / "alignment.json"),
            "--writer-backend",
            "ass_karaoke",
            "--output-roller",
            str(tmp_path / "song.ass"),
        ]
    )
    request = _build_request(args)

    assert request.language == "zh"
    assert request.intermediate_dir == Path("work")
    assert request.backend_config["writer"]["backend"] == "ass_karaoke"
    assert request.backend_config["writer"]["spacing"] == "drop"
    assert request.backend_config["transcriber"]["hf_xet"] == "off"


def test_batch_config_defaults_apply_only_to_batch_options(tmp_path: Path) -> None:
    config = tmp_path / "pyroller.yaml"
    config.write_text(
        """
shared:
  language: en
batch:
  jobs: 3
  skip_existing: true
  lyrics_glob: "*.lyrics.txt"
""",
        encoding="utf-8",
    )
    parser, run_parser, batch_parser = build_parser()
    apply_cli_config_defaults(
        run_parser=run_parser,
        batch_parser=batch_parser,
        config=load_cli_config(config),
    )

    batch_args = parser.parse_args(
        [
            "batch",
            "--config",
            str(config),
            "--stages",
            "w",
            "--alignment-result",
            str(tmp_path / "alignments"),
            "--output-roller",
            str(tmp_path / "out"),
        ]
    )
    run_args = parser.parse_args(
        [
            "run",
            "--config",
            str(config),
            "--stages",
            "w",
            "--alignment-result",
            str(tmp_path / "alignment.json"),
            "--output-roller",
            str(tmp_path / "song.lrc"),
        ]
    )

    assert batch_args.language == "en"
    assert batch_args.jobs == 3
    assert batch_args.skip_existing is True
    assert batch_args.lyrics_glob == "*.lyrics.txt"
    assert run_args.language == "en"
    assert not hasattr(run_args, "jobs")
