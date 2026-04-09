from __future__ import annotations

from pathlib import Path
from typing import Any

from pyroller.filter.base import AudioFilter
from pyroller.filter.chain import FilterChain
from pyroller.filter.dereverb import DereverbFilter
from pyroller.filter.noise_gate import AdaptiveNoiseGateFilter

_FILTER_FACTORIES: dict[str, type[AudioFilter]] = {
    "noise_gate": AdaptiveNoiseGateFilter,
    "dereverb": DereverbFilter,
}

_FILTER_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "noise_gate": ("numpy", "soundfile"),
    "dereverb": ("numpy", "soundfile", "scipy", "bottleneck", "nara_wpe"),
}


def list_available_filter_backends() -> tuple[str, ...]:
    return tuple(sorted(_FILTER_FACTORIES))


def get_filter_requirements(name: str) -> tuple[str, ...]:
    return _FILTER_REQUIREMENTS.get(name, ())


def build_filter_chain(
    chain_names: list[str] | tuple[str, ...] | None,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> FilterChain:
    del config
    filters: list[AudioFilter] = []
    for name in list(chain_names or []):
        try:
            factory = _FILTER_FACTORIES[name]
        except KeyError as exc:
            available = ", ".join(list_available_filter_backends()) or "<none registered yet>"
            raise ValueError(f"Unsupported filter step {name!r}. Available filter steps: {available}") from exc
        filters.append(factory())
    return FilterChain(filters=filters, output_dir=output_dir)
