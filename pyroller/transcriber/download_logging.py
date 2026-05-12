from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pyroller.progress import StageProgress
from pyroller.transcriber.hf_download_config import HFDownloadConfig, hf_download_environment, huggingface_download_error_hints

logger = logging.getLogger("pyroller.transcriber")


def snapshot_download_with_logging(
    *,
    repo_id: str,
    cache_dir: str | Path,
    local_files_only: bool,
    log_label: str,
    stage: StageProgress | None = None,
    hf_download_config: HFDownloadConfig | None = None,
    **kwargs: Any,
) -> str:
    config = hf_download_config or HFDownloadConfig()
    cache_dir = str(Path(cache_dir))
    summary = config.summary()

    logger.info("Preparing model download: %s", log_label)
    logger.info("Model source repo: %s", repo_id)
    logger.info("Model cache destination: %s", cache_dir)
    logger.info(
        "HF download options: xet=%s proxy=%s etag_timeout=%s download_timeout=%s max_workers=%s",
        summary["xet"],
        summary["proxy"],
        summary["etag_timeout"],
        summary["download_timeout"],
        summary["max_workers"],
    )
    if stage is not None:
        stage.update(0, message=f"model source: {repo_id}")
    if local_files_only:
        logger.info("Local-files-only mode is enabled; py-roller will only use existing local cache for %s", log_label)
        if stage is not None:
            stage.update(0, message="using local cache only")
    else:
        logger.info("Downloading model into local cache if missing: %s", log_label)
        if stage is not None:
            stage.update(0, message=f"checking/downloading model cache (HF XET={summary['xet']})")

    with hf_download_environment(config, local_files_only=local_files_only):
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "huggingface_hub is required to materialize transcriber models from Hugging Face. Install with: pip install .[audio-core]"
            ) from exc

        try:
            resolved = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                **kwargs,
            )
        except Exception as exc:
            hints = huggingface_download_error_hints(exc)
            hint_text = f" Suggested fix: {'; '.join(hints)}." if hints else ""
            raise RuntimeError(f"Hugging Face model download failed for {repo_id!r}.{hint_text}") from exc
    if stage is not None:
        stage.update(0, message="model cache ready")
    return resolved
