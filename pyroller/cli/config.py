from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pyroller.i18n import _
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
    "transcriber_hf_xet",
    "transcriber_hf_proxy",
    "transcriber_hf_etag_timeout",
    "transcriber_hf_download_timeout",
    "transcriber_hf_max_workers",
    "transcriber_compute_type",
    "transcriber_batch_size",
    "transcriber_vad_filter",
    "splitter_demucs_model",
    "aligner_min_gap",
    "aligner_repetition",
    "writer_by_tag",
    "writer_ass_karaoke_tag_type",
    "cleanup",
    "progress_format",
}
_RUN_ALLOWED_KEYS: set[str] = set()  # currently no extra keys beyond shared; if any are added, add them here
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
        raise ConfigError(_("Config path must be an existing YAML file: {}").format(path))
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {"shared": {}, "run": {}, "batch": {}}
    if not isinstance(data, dict):
        raise ConfigError(_("Config YAML must be a mapping/object."))
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
                raise ConfigError(_("Config section '{}' must be a mapping/object.").format(section))
            sections[section] = dict(raw)
        if sections["run"]:
            # The "run" section has no keys beyond shared; merge it into shared
            # so users aren't confused by rejected keys.
            sections["shared"].update(sections.pop("run"))
        if other_keys:
            raise ConfigError(
                _("Top-level config keys must be under 'shared', 'run', or 'batch' when sectioned config is used. "
                  "Unexpected keys: {}").format(', '.join(sorted(other_keys)))
            )
        return sections
    sections["shared"] = dict(data)
    return sections


def _validate_section_keys(sections: dict[str, dict[str, Any]]) -> None:
    shared_unknown = sorted(set(sections.get("shared", {})) - _SHARED_ALLOWED_KEYS)
    if shared_unknown:
        raise ConfigError(
            _("Unsupported config keys under 'shared': {}. Only default option overrides are allowed.").format(", ".join(shared_unknown))
        )
    run_unknown = sorted(set(sections.get("run", {})) - _RUN_ALLOWED_KEYS)
    run_unknown = sorted(set(sections.get("run", {})) - _RUN_ALLOWED_KEYS - _SHARED_ALLOWED_KEYS)
    if run_unknown:
        raise ConfigError(
            _("Unsupported config keys under 'run': {}. Run-specific config supports all shared keys.").format(", ".join(run_unknown))
        )
    batch_unknown = sorted(set(sections.get("batch", {})) - _BATCH_ALLOWED_KEYS)
    if batch_unknown:
        raise ConfigError(
            _("Unsupported config keys under 'batch': {}. Only batch default option overrides are allowed.").format(", ".join(batch_unknown))
        )


def _coerce_values(values: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if key in {"intermediate", "transcriber_model_path"} and value is not None:
            out[key] = Path(str(value))
        elif key == "filter_chain" and isinstance(value, list):
            out[key] = [str(item) for item in value]
        elif key == "transcriber_hf_xet" and isinstance(value, bool):
            # YAML 1.1 parsers commonly coerce unquoted on/off to booleans.
            out[key] = "on" if value else "off"
        else:
            out[key] = value
    return out
