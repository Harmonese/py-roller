from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyroller.progress import StageProgress

logger = logging.getLogger("pyroller.transcriber")


def snapshot_download_with_logging(
    *,
    repo_id: str,
    cache_dir: str | Path,
    local_files_only: bool,
    log_label: str,
    stage: StageProgress | None = None,
    **kwargs: Any,
) -> str:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to materialize transcriber models from Hugging Face. Install with: pip install .[audio]"
        ) from exc

    cache_dir = str(Path(cache_dir))
    logger.info("Preparing model download: %s", log_label)
    logger.info("Model source repo: %s", repo_id)
    logger.info("Model cache destination: %s", cache_dir)
    if stage is not None:
        stage.update(0, message=f"model source: {repo_id}")
    if local_files_only:
        logger.info("Local-files-only mode is enabled; py-roller will only use existing local cache for %s", log_label)
        if stage is not None:
            stage.update(0, message="using local cache only")
    else:
        logger.info("Downloading model into local cache if missing: %s", log_label)
        if stage is not None:
            stage.update(0, message="checking/downloading model cache")

    resolved = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        **kwargs,
    )
    if stage is not None:
        stage.update(0, message="model cache ready")
    return resolved
