from __future__ import annotations

import pytest

from pyroller.domain import AudioArtifact
from pyroller.filter.chain import FilterChain
from pyroller.filter.dereverb import DereverbFilter
from pyroller.filter.noise_gate import AdaptiveNoiseGateFilter
from pyroller.filter.registry import build_filter_chain, get_filter_requirements, list_available_filter_backends
from pyroller.progress import NullProgressReporter


def test_filter_registry_lists_available_filters() -> None:
    assert list_available_filter_backends() == ("dereverb", "noise_gate")


def test_filter_registry_exposes_requirements() -> None:
    assert get_filter_requirements("noise_gate") == ("numpy", "soundfile")
    assert get_filter_requirements("dereverb") == ("numpy", "soundfile", "scipy", "bottleneck", "nara_wpe")
    assert get_filter_requirements("unknown") == ()


def test_build_filter_chain_instantiates_requested_steps(tmp_path) -> None:
    chain = build_filter_chain(["noise_gate", "dereverb"], tmp_path / "out")

    assert isinstance(chain, FilterChain)
    assert isinstance(chain.filters[0], AdaptiveNoiseGateFilter)
    assert isinstance(chain.filters[1], DereverbFilter)
    assert chain.output_dir == tmp_path / "out"


def test_build_filter_chain_rejects_unknown_step(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported filter step"):
        build_filter_chain(["missing"], tmp_path / "out")


def test_empty_filter_chain_forwards_audio_artifact_without_processing(tmp_path) -> None:
    source = AudioArtifact(
        artifact_id="source",
        stage="input",
        kind="audio",
        path=tmp_path / "vocals.wav",
        sample_rate=48000,
        channels=2,
        duration=12.5,
        role="vocal_audio",
        metadata={"source": "test"},
    )

    result = build_filter_chain([], tmp_path / "out").process(source, progress=NullProgressReporter())

    assert result.path == source.path
    assert result.sample_rate == 48000
    assert result.channels == 2
    assert result.duration == 12.5
    assert result.role == "filtered_vocal_audio"
    assert result.metadata["source"] == "test"
    assert result.metadata["filter_chain"] == []
    assert result.metadata["source_artifact_id"] == "source"
