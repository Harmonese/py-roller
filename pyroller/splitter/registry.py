from __future__ import annotations

import logging
from typing import Any

from pyroller.splitter.base import Splitter
from pyroller.splitter.demucs import DemucsSplitter

logger = logging.getLogger("pyroller.splitter")

_DEFAULT_SPLITTER_BACKEND = "demucs"
_AVAILABLE_SPLITTER_BACKENDS = ("demucs",)

_SPLITTER_FACTORIES = {
    "demucs": DemucsSplitter,
}

_SPLITTER_REQUIREMENTS = {
    "demucs": ("demucs",),
}

_SUPPORTED_SPLITTER_CONFIG_KEYS = {
    "demucs": {"model", "device", "jobs", "overlap", "segment", "two_stems"},
}


def list_available_splitter_backends() -> tuple[str, ...]:
    return _AVAILABLE_SPLITTER_BACKENDS


def resolve_splitter_backend(backend_name: str | None) -> str:
    chosen_backend = backend_name or _DEFAULT_SPLITTER_BACKEND
    if chosen_backend not in _AVAILABLE_SPLITTER_BACKENDS:
        raise ValueError(
            f"Unsupported splitter backend {chosen_backend!r}. "
            f"Available backends: {', '.join(_AVAILABLE_SPLITTER_BACKENDS)}"
        )
    return chosen_backend


def get_splitter_requirements(backend_name: str) -> tuple[str, ...]:
    return _SPLITTER_REQUIREMENTS.get(backend_name, ())


def sanitize_splitter_config(backend_name: str, config: dict[str, Any] | None) -> dict[str, Any]:
    init_config = {key: value for key, value in dict(config or {}).items() if value is not None}
    init_config.pop("backend", None)
    accepted = _SUPPORTED_SPLITTER_CONFIG_KEYS.get(backend_name, set())
    filtered = {key: value for key, value in init_config.items() if key in accepted}
    ignored = sorted(set(init_config) - accepted)
    if ignored:
        logger.info(
            "Ignoring backend-incompatible splitter option(s) for %s: %s",
            backend_name,
            ", ".join(ignored),
        )
    return filtered


def build_splitter(backend_name: str | None, output_dir, config: dict[str, Any]) -> Splitter:
    chosen_backend = resolve_splitter_backend(backend_name)
    factory = _SPLITTER_FACTORIES[chosen_backend]
    init_config = sanitize_splitter_config(chosen_backend, config)
    return factory(output_dir=output_dir, **init_config)
