from __future__ import annotations

import logging
from typing import Any

from pyroller.transcriber.base import Transcriber
from pyroller.transcriber.specs import TRANSCRIBER_SPECS, TranscriberSpec

logger = logging.getLogger("pyroller.transcriber")

_SUPPORTED_LANGUAGES = {language for language, _ in TRANSCRIBER_SPECS}
_DEFAULT_TRANSCRIBER_BY_LANGUAGE = {
    "zh": "mms_phonetic",
    "en": "whisperx",
    "mul": "wav2vec2_phoneme",
}
_FALLBACK_LANGUAGE = "mul"


def resolve_transcriber_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in _SUPPORTED_LANGUAGES:
        return normalized
    logger.error("Unsupported language=%s for transcriber. Falling back to language=%s.", language, _FALLBACK_LANGUAGE)
    return _FALLBACK_LANGUAGE


def _available_specs_for_language(language: str) -> dict[str, TranscriberSpec]:
    effective_language = resolve_transcriber_language(language)
    return {
        backend: spec
        for (spec_language, backend), spec in TRANSCRIBER_SPECS.items()
        if spec_language == effective_language
    }


def list_available_transcriber_backends(language: str) -> tuple[str, ...]:
    return tuple(_available_specs_for_language(language).keys())


def resolve_transcriber_backend(language: str, backend_name: str | None) -> tuple[str, str]:
    effective_language = resolve_transcriber_language(language)
    available_specs = _available_specs_for_language(effective_language)
    chosen_backend = backend_name or _DEFAULT_TRANSCRIBER_BY_LANGUAGE[effective_language]
    if chosen_backend not in available_specs:
        raise ValueError(
            f"Unsupported transcriber backend {chosen_backend!r} for language {effective_language!r}. "
            f"Available backends: {', '.join(available_specs)}"
        )
    return effective_language, chosen_backend


def _get_spec(language: str, backend_name: str | None) -> tuple[str, str, TranscriberSpec]:
    effective_language, chosen_backend = resolve_transcriber_backend(language, backend_name)
    return effective_language, chosen_backend, TRANSCRIBER_SPECS[(effective_language, chosen_backend)]


def get_transcriber_requirements(backend_name: str) -> tuple[str, ...]:
    for (_, spec_backend), spec in TRANSCRIBER_SPECS.items():
        if spec_backend == backend_name:
            return spec.requirements
    return ()


def sanitize_transcriber_config(backend_name: str, config: dict[str, Any] | None) -> dict[str, Any]:
    init_config = {key: value for key, value in dict(config or {}).items() if value is not None}
    init_config.pop("backend", None)
    accepted = next((spec.config_keys for (_, spec_backend), spec in TRANSCRIBER_SPECS.items() if spec_backend == backend_name), frozenset())
    filtered = {key: value for key, value in init_config.items() if key in accepted}
    ignored = sorted(set(init_config) - set(accepted))
    if ignored:
        logger.info(
            "Ignoring backend-incompatible transcriber option(s) for %s: %s",
            backend_name,
            ", ".join(ignored),
        )
    return filtered


def build_transcriber(language: str, backend_name: str | None, config: dict[str, Any]) -> Transcriber:
    effective_language, chosen_backend, spec = _get_spec(language, backend_name)
    init_config = sanitize_transcriber_config(chosen_backend, config)
    return spec.compose(init_config)
