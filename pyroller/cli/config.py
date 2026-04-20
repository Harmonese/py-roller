from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

_SHARED_ALLOWED_KEYS = {
    "language",
    "transcriber_backend",
    "aligner_backend",
    "writer_backend",
    "filter_chain",
    "parser_lyrics_encoding",
    "writer_spacing",
    "splitter_backend",
    "splitter_demucs_device",
    "splitter_demucs_jobs",
    "splitter_demucs_overlap",
    "splitter_demucs_segment",
    "intermediate",
    "log_level",
    "transcriber_device",
    "transcriber_model_name",
    "transcriber_model_path",
    "transcriber_local_files_only",
    "transcriber_compute_type",
    "transcriber_batch_size",
    "splitter_demucs_model",
    "aligner_min_gap",
    "writer_by_tag",
    "writer_ass_karaoke_tag_type",
    "cleanup",
}
_RUN_ALLOWED_KEYS: set[str] = set()
_BATCH_ALLOWED_KEYS = {
    "continue_on_error",
    "skip_existing",
    "pair_by",
    "jobs",
    "audio_glob",
    "lyrics_glob",
    "timed_units_glob",
    "parsed_lyrics_glob",
    "alignment_result_glob",
}
_ALLOWED_SECTIONS = {"shared", "run", "batch"}


class ConfigError(ValueError):
    pass


def preparse_config_path(argv: list[str]) -> Path | None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("command", nargs="?")
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    return known.config


def load_cli_config(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise ConfigError(f"Config path must be an existing YAML file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {"shared": {}, "run": {}, "batch": {}}
    if not isinstance(data, dict):
        raise ConfigError("Config YAML must be a mapping/object.")
    sections = _normalize_sections(data)
    _validate_section_keys(sections)
    return sections


def apply_cli_config_defaults(
    *,
    run_parser: argparse.ArgumentParser,
    batch_parser: argparse.ArgumentParser,
    config: dict[str, dict[str, Any]],
) -> None:
    shared_defaults = _coerce_values(config.get("shared", {}))
    run_defaults = _coerce_values(config.get("run", {}))
    batch_defaults = _coerce_values(config.get("batch", {}))
    if shared_defaults:
        run_parser.set_defaults(**shared_defaults)
        batch_parser.set_defaults(**shared_defaults)
    if run_defaults:
        run_parser.set_defaults(**run_defaults)
    if batch_defaults:
        batch_parser.set_defaults(**batch_defaults)


def _normalize_sections(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    explicit_sections = set(data) & _ALLOWED_SECTIONS
    other_keys = {k: v for k, v in data.items() if k not in _ALLOWED_SECTIONS}
    sections = {"shared": {}, "run": {}, "batch": {}}
    if explicit_sections:
        for section in _ALLOWED_SECTIONS:
            raw = data.get(section, {})
            if raw is None:
                raw = {}
            if not isinstance(raw, dict):
                raise ConfigError(f"Config section '{section}' must be a mapping/object.")
            sections[section] = dict(raw)
        if other_keys:
            raise ConfigError(
                "Top-level config keys must be under 'shared', 'run', or 'batch' when sectioned config is used. "
                f"Unexpected keys: {', '.join(sorted(other_keys))}"
            )
        return sections
    sections["shared"] = dict(data)
    return sections


def _validate_section_keys(sections: dict[str, dict[str, Any]]) -> None:
    shared_unknown = sorted(set(sections.get("shared", {})) - _SHARED_ALLOWED_KEYS)
    if shared_unknown:
        raise ConfigError(
            "Unsupported config keys under 'shared': "
            + ", ".join(shared_unknown)
            + ". Only default option overrides are allowed."
        )
    run_unknown = sorted(set(sections.get("run", {})) - _RUN_ALLOWED_KEYS)
    if run_unknown:
        raise ConfigError(
            "Unsupported config keys under 'run': "
            + ", ".join(run_unknown)
            + ". Run-specific config currently supports no extra keys beyond 'shared'."
        )
    batch_unknown = sorted(set(sections.get("batch", {})) - _BATCH_ALLOWED_KEYS)
    if batch_unknown:
        raise ConfigError(
            "Unsupported config keys under 'batch': "
            + ", ".join(batch_unknown)
            + ". Only batch default option overrides are allowed."
        )


def _coerce_values(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key in {"intermediate", "transcriber_model_path"} and value is not None:
            out[key] = Path(str(value))
        elif key == "filter_chain" and isinstance(value, list):
            out[key] = [str(item) for item in value]
        else:
            out[key] = value
    return out
