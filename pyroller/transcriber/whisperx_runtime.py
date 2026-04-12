from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

logger = logging.getLogger("pyroller.transcriber")

def collect_whisperx_runtime_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for pkg in ("whisperx", "torch", "torchaudio", "pyannote.audio", "lightning", "speechbrain"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


@contextmanager
def whisperx_load_model_context():
    key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(key)
    os.environ[key] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@contextmanager
def whisperx_runtime_context(plan):
    from pyroller.transcriber.model_resolver import transcriber_provider_environment

    with transcriber_provider_environment(plan):
        yield


def build_whisperx_helpful_error(exc: Exception, *, model_store_root: Path, local_files_only: bool) -> RuntimeError:
    message = str(exc)
    cause = f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__
    if (
        "Weights only load failed" in message
        or "Unsupported global" in message
        or "upgrade torch to at least v2.6" in message
        or "check_torch_load_is_safe" in message
    ):
        guidance = (
            "WhisperX failed while loading a checkpoint under the current Torch/Transformers safety rules. "
            "This usually means the installed WhisperX/Pyannote/Torch environment is not mutually compatible, or Torch is older than 2.6. "
            f"Model store: {model_store_root}. Underlying error: {cause}"
        )
        return RuntimeError(guidance)
    if local_files_only:
        guidance = (
            "WhisperX was forced into local-files-only mode, but a required local model or auxiliary asset was missing from the py-roller model store. "
            f"Populate {model_store_root} in advance, or retry without --transcriber-local-files-only. "
            f"Underlying error: {cause}"
        )
        return RuntimeError(guidance)
    guidance = (
        "WhisperX failed while resolving or loading models. This is usually either a restricted-network download problem or an incompatible local audio environment. "
        f"Model store: {model_store_root}. Underlying error: {cause}"
    )
    return RuntimeError(guidance)
