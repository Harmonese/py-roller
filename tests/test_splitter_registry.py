from __future__ import annotations

import pytest

from pyroller.splitter.demucs import DemucsSplitter
from pyroller.splitter.registry import (
    build_splitter,
    get_splitter_requirements,
    list_available_splitter_backends,
    resolve_splitter_backend,
    sanitize_splitter_config,
)


def test_splitter_registry_lists_default_backend() -> None:
    assert list_available_splitter_backends() == ("demucs",)
    assert resolve_splitter_backend(None) == "demucs"
    assert resolve_splitter_backend("demucs") == "demucs"


def test_splitter_registry_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported splitter backend"):
        resolve_splitter_backend("missing")


def test_splitter_requirements_are_backend_specific() -> None:
    assert get_splitter_requirements("demucs") == ("demucs",)
    assert get_splitter_requirements("missing") == ()


def test_sanitize_splitter_config_filters_unknown_and_none_values() -> None:
    config = sanitize_splitter_config(
        "demucs",
        {
            "backend": "demucs",
            "model": "htdemucs_ft",
            "device": "cpu",
            "jobs": None,
            "two_stems": "vocals",
            "unexpected": "ignored",
        },
    )

    assert config == {
        "model": "htdemucs_ft",
        "device": "cpu",
        "two_stems": "vocals",
    }


def test_build_splitter_passes_sanitized_config(tmp_path) -> None:
    splitter = build_splitter(
        "demucs",
        tmp_path / "split",
        {
            "model": "htdemucs_ft",
            "two_stems": "drums",
            "device": "cpu",
            "jobs": 1,
            "overlap": 0.25,
            "segment": 8.0,
            "ignored": "x",
        },
    )

    assert isinstance(splitter, DemucsSplitter)
    assert splitter.output_dir == tmp_path / "split"
    assert splitter.model == "htdemucs_ft"
    assert splitter.two_stems == "drums"
    assert splitter.device == "cpu"
    assert splitter.jobs == 1
    assert splitter.overlap == 0.25
    assert splitter.segment == 8.0
