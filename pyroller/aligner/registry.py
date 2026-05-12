from __future__ import annotations

import inspect
from typing import Any

from pyroller.aligner.base import Aligner
from pyroller.aligner.global_dp_v1 import GlobalDPAligner

_DEFAULT_ALIGNER_BACKEND = "global_dp_v1"

_ALIGNER_FACTORIES = {
    "global_dp_v1": GlobalDPAligner,
}


def list_available_aligner_backends() -> tuple[str, ...]:
    return tuple(_ALIGNER_FACTORIES.keys())


def build_aligner(backend_name: str | None, config: dict[str, Any]) -> Aligner:
    chosen_backend = backend_name or _DEFAULT_ALIGNER_BACKEND
    try:
        factory = _ALIGNER_FACTORIES[chosen_backend]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported aligner backend {chosen_backend!r}. Available backends: {', '.join(list_available_aligner_backends())}"
        ) from exc
    init_config = {key: value for key, value in dict(config).items() if value is not None}
    init_config.pop("backend", None)
    signature = inspect.signature(factory.__init__)
    accepted = {name for name in signature.parameters if name != "self"}
    filtered_config = {key: value for key, value in init_config.items() if key in accepted}
    return factory(**filtered_config)
