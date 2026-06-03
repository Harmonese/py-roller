from __future__ import annotations

import pytest

from pyroller.transcriber.composed import ComposedTranscriber
from pyroller.transcriber.registry import (
    build_transcriber,
    get_transcriber_config_keys,
    get_transcriber_requirements,
    list_available_transcriber_backends,
    resolve_transcriber_backend,
    resolve_transcriber_language,
    sanitize_transcriber_config,
)


@pytest.mark.parametrize(
    ("language", "available"),
    [
        ("zh", ("faster_whisper", "mms_phonetic")),
        ("en", ("faster_whisper",)),
        ("mul", ("faster_whisper", "wav2vec2_phoneme")),
    ],
)
def test_available_transcriber_backends_by_language(language: str, available: tuple[str, ...]) -> None:
    assert list_available_transcriber_backends(language) == available


@pytest.mark.parametrize("language", ["zh", "en", "mul"])
def test_default_transcriber_backend_is_faster_whisper(language: str) -> None:
    assert resolve_transcriber_backend(language, None) == (language, "faster_whisper")


def test_transcriber_language_falls_back_to_multilingual() -> None:
    assert resolve_transcriber_language("xx") == "mul"
    assert resolve_transcriber_backend("xx", None) == ("mul", "faster_whisper")


def test_resolve_transcriber_backend_rejects_language_incompatible_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported transcriber backend"):
        resolve_transcriber_backend("en", "mms_phonetic")


def test_transcriber_requirements_and_config_keys_are_backend_specific() -> None:
    assert get_transcriber_requirements("faster_whisper") == ("faster_whisper",)
    assert get_transcriber_requirements("mms_phonetic") == ("librosa", "torch", "transformers", "huggingface_hub")
    assert get_transcriber_requirements("unknown") == ()
    assert "compute_type" in get_transcriber_config_keys("faster_whisper")
    assert "trust_remote_code" not in get_transcriber_config_keys("faster_whisper")
    assert "trust_remote_code" in get_transcriber_config_keys("mms_phonetic")


def test_sanitize_transcriber_config_filters_unsupported_and_none_values() -> None:
    sanitized = sanitize_transcriber_config(
        "faster_whisper",
        {
            "backend": "faster_whisper",
            "model_name": "turbo",
            "compute_type": "int8",
            "target_sample_rate": 16000,
            "hf_proxy": None,
            "extra": "ignored",
        },
    )

    assert sanitized == {
        "model_name": "turbo",
        "compute_type": "int8",
    }


def test_build_transcriber_composes_engine_and_adapter_without_loading_model(tmp_path) -> None:
    transcriber = build_transcriber(
        "zh",
        "faster_whisper",
        {
            "model_path": tmp_path,
            "local_files_only": True,
            "hf_xet": "off",
        },
    )

    assert isinstance(transcriber, ComposedTranscriber)
    assert transcriber.backend_name == "faster_whisper"
    assert transcriber.adapter.backend == "faster_whisper"
