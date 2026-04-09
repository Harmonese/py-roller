from __future__ import annotations

import logging
from typing import Any

from pyroller.transcriber.base import Transcriber
from pyroller.transcriber.en_whisperx import EnglishWhisperXTranscriber
from pyroller.transcriber.mms_phonetic import Wav2Vec2MMSPhoneticTranscriber
from pyroller.transcriber.mul_wav2vec2_phoneme import MultilingualWav2Vec2PhonemeTranscriber
from pyroller.transcriber.whisperx import WhisperXTranscriber

logger = logging.getLogger("pyroller.transcriber")

_SUPPORTED_LANGUAGES = {"zh", "en", "mul"}
_DEFAULT_TRANSCRIBER_BY_LANGUAGE = {
    "zh": "mms_phonetic",
    "en": "whisperx",
    "mul": "wav2vec2_phoneme",
}

_AVAILABLE_TRANSCRIBER_BACKENDS_BY_LANGUAGE = {
    "zh": ("mms_phonetic", "whisperx"),
    "en": ("whisperx",),
    "mul": ("wav2vec2_phoneme",),
}

_TRANSCRIBER_FACTORIES_BY_LANGUAGE = {
    "zh": {
        "mms_phonetic": Wav2Vec2MMSPhoneticTranscriber,
        "whisperx": WhisperXTranscriber,
    },
    "en": {
        "whisperx": EnglishWhisperXTranscriber,
    },
    "mul": {
        "wav2vec2_phoneme": MultilingualWav2Vec2PhonemeTranscriber,
    },
}

_TRANSCRIBER_REQUIREMENTS = {
    "whisperx": ("whisperx",),
    "mms_phonetic": ("librosa", "torch", "transformers"),
    "wav2vec2_phoneme": ("librosa", "torch", "transformers"),
}

_SUPPORTED_TRANSCRIBER_CONFIG_KEYS = {
    "whisperx": {"model_name", "device", "compute_type", "batch_size", "align_words"},
    "mms_phonetic": {"model_name", "device", "target_sample_rate", "trust_remote_code"},
    "wav2vec2_phoneme": {"model_name", "device", "target_sample_rate", "trust_remote_code"},
}

_FALLBACK_LANGUAGE = "mul"


def resolve_transcriber_language(language: str) -> str:
    normalized = (language or "").strip().lower()
    if normalized in _SUPPORTED_LANGUAGES:
        return normalized
    logger.error("Unsupported language=%s for transcriber. Falling back to language=%s.", language, _FALLBACK_LANGUAGE)
    return _FALLBACK_LANGUAGE


def list_available_transcriber_backends(language: str) -> tuple[str, ...]:
    effective_language = resolve_transcriber_language(language)
    return _AVAILABLE_TRANSCRIBER_BACKENDS_BY_LANGUAGE[effective_language]


def resolve_transcriber_backend(language: str, backend_name: str | None) -> tuple[str, str]:
    effective_language = resolve_transcriber_language(language)
    available = _AVAILABLE_TRANSCRIBER_BACKENDS_BY_LANGUAGE[effective_language]
    chosen_backend = backend_name or _DEFAULT_TRANSCRIBER_BY_LANGUAGE[effective_language]
    if chosen_backend not in available:
        raise ValueError(
            f"Unsupported transcriber backend {chosen_backend!r} for language {effective_language!r}. "
            f"Available backends: {', '.join(available)}"
        )
    return effective_language, chosen_backend


def get_transcriber_requirements(backend_name: str) -> tuple[str, ...]:
    return _TRANSCRIBER_REQUIREMENTS.get(backend_name, ())


def sanitize_transcriber_config(backend_name: str, config: dict[str, Any] | None) -> dict[str, Any]:
    init_config = {key: value for key, value in dict(config or {}).items() if value is not None}
    init_config.pop("backend", None)
    accepted = _SUPPORTED_TRANSCRIBER_CONFIG_KEYS.get(backend_name, set())
    filtered = {key: value for key, value in init_config.items() if key in accepted}
    ignored = sorted(set(init_config) - accepted)
    if ignored:
        logger.info(
            "Ignoring backend-incompatible transcriber option(s) for %s: %s",
            backend_name,
            ", ".join(ignored),
        )
    return filtered


def build_transcriber(language: str, backend_name: str | None, config: dict[str, Any]) -> Transcriber:
    effective_language, chosen_backend = resolve_transcriber_backend(language, backend_name)
    factory = _TRANSCRIBER_FACTORIES_BY_LANGUAGE[effective_language][chosen_backend]
    init_config = sanitize_transcriber_config(chosen_backend, config)
    return factory(**init_config)
