from __future__ import annotations

from pathlib import Path
from typing import Any

from pyroller.filter.base import AudioFilter
from pyroller.filter.chain import FilterChain
from pyroller.filter.dereverb import DereverbFilter
from pyroller.filter.noise_gate import AdaptiveNoiseGateFilter
from pyroller.i18n import _

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
    config = dict(config or {})
    unknown = set(config) - {"chain", "steps"}
    if unknown:
        raise ValueError(f"Unknown filter configuration: {sorted(unknown)}")
    steps = config.get("steps", {})
    if not isinstance(steps, dict) or set(steps) - set(chain_names or []):
        raise ValueError("filter.steps must configure selected filter names")
    filters: list[AudioFilter] = []
    for name in list(chain_names or []):
        try:
            factory = _FILTER_FACTORIES[name]
        except KeyError as exc:
            available = ", ".join(list_available_filter_backends()) or "<none registered yet>"
            raise ValueError(_("Unsupported filter step {!r}. Available filter steps: {}").format(name, available)) from exc
        import inspect
        params = steps.get(name, {})
        if not isinstance(params, dict):
            raise ValueError(f"Filter configuration for {name} must be an object")
        accepted = {key for key, parameter in inspect.signature(factory).parameters.items()
                    if parameter.kind != inspect.Parameter.VAR_KEYWORD}
        if set(params) - accepted:
            raise ValueError(f"Unknown {name} options: {sorted(set(params) - accepted)}")
        filters.append(factory(**params))
    return FilterChain(filters=filters, output_dir=output_dir)
