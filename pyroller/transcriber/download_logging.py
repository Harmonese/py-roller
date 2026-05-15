from __future__ import annotations

import logging

from pyroller.i18n import _
import threading
import time
from pathlib import Path
from typing import Any

from pyroller.progress import StageProgress
from pyroller.transcriber.hf_download_config import HFDownloadConfig, hf_download_environment, huggingface_download_error_hints

logger = logging.getLogger("pyroller.transcriber")


def _repo_cache_dir(cache_dir: str | Path, repo_id: str) -> Path:
    safe = "models--" + repo_id.replace("/", "--")
    candidate = Path(cache_dir) / safe
    return candidate if candidate.exists() else Path(cache_dir)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(value, 0))
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"


def _repo_file_stats_from_hub(repo_id: str, *, local_files_only: bool, kwargs: dict[str, Any]) -> dict[str, Any]:
    if local_files_only:
        return {"bytes_total": None, "largest_file": None, "largest_file_size": None, "file_count": None}
    try:
        from huggingface_hub import HfApi  # type: ignore

        revision = kwargs.get("revision")
        api = HfApi()
        info = api.model_info(repo_id=repo_id, revision=revision, files_metadata=True)
        total = 0
        seen = False
        largest_file: str | None = None
        largest_size = 0
        file_count = 0
        for sibling in getattr(info, "siblings", []) or []:
            filename = getattr(sibling, "rfilename", None) or getattr(sibling, "filename", None)
            size = getattr(sibling, "size", None)
            file_count += 1
            if isinstance(size, int) and size > 0:
                total += size
                seen = True
                if size > largest_size:
                    largest_size = size
                    largest_file = str(filename) if filename else None
        return {
            "bytes_total": total if seen else None,
            "largest_file": largest_file,
            "largest_file_size": largest_size if largest_size > 0 else None,
            "file_count": file_count or None,
        }
    except Exception as exc:
        logger.debug(_("Unable to prefetch Hugging Face file sizes for %s: %s"), repo_id, exc)
        return {"bytes_total": None, "largest_file": None, "largest_file_size": None, "file_count": None}


class _DownloadProgressWatcher:
    def __init__(
        self,
        *,
        repo_id: str,
        cache_dir: str | Path,
        stage: StageProgress | None,
        bytes_total: int | None,
        summary: dict[str, Any],
        largest_file: str | None = None,
        file_count: int | None = None,
        interval_seconds: float = 1.0,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = Path(cache_dir)
        self.stage = stage
        self.bytes_total = bytes_total
        self.summary = summary
        self.largest_file = largest_file
        self.file_count = file_count
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_size: int | None = None
        self._last_time: float | None = None

    def __enter__(self):
        if self.stage is not None:
            self.stage.event(
                "download_started",
                stage="model_download",
                parent_stage="preflight",
                repo_id=self.repo_id,
                cache_dir=str(self.cache_dir),
                file=self.largest_file,
                file_count=self.file_count,
                bytes_total=self.bytes_total,
                message=_("Downloading model cache for {}").format(self.repo_id),
                hf_download=self.summary,
            )
        self._thread = threading.Thread(target=self._run, name="pyroller-hf-download-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.stage is None:
            return
        current = _directory_size(_repo_cache_dir(self.cache_dir, self.repo_id))
        event_type = "download_failed" if exc is not None else "download_completed"
        self.stage.event(
            event_type,
            stage="model_download",
            parent_stage="preflight",
            repo_id=self.repo_id,
            cache_dir=str(self.cache_dir),
            file=self.largest_file,
            file_count=self.file_count,
            bytes_downloaded=current,
            bytes_total=self.bytes_total,
            progress=(current / self.bytes_total if self.bytes_total else None),
            cached=exc is None and self.bytes_total is not None and current >= self.bytes_total,
            message=(_("Model cache download failed for {}").format(self.repo_id) if exc is not None else _("Model cache ready for {}").format(self.repo_id)),
            hf_download=self.summary,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._emit_progress()

    def _emit_progress(self) -> None:
        if self.stage is None:
            return
        root = _repo_cache_dir(self.cache_dir, self.repo_id)
        current = _directory_size(root)
        now = time.monotonic()
        speed: float | None = None
        if self._last_size is not None and self._last_time is not None:
            elapsed = max(now - self._last_time, 1e-6)
            speed = max(0.0, (current - self._last_size) / elapsed)
        self._last_size = current
        self._last_time = now
        total = self.bytes_total
        percent = current / total if total else None
        message = _("Downloading model cache: {}").format(_format_bytes(current))
        if total:
            message += _(" / {}").format(_format_bytes(total))
        if speed is not None:
            message += _(" at {}/s").format(_format_bytes(int(speed)))
        self.stage.event(
            "download_progress",
            stage="model_download",
            parent_stage="preflight",
            repo_id=self.repo_id,
            cache_dir=str(self.cache_dir),
            file=self.largest_file,
            file_count=self.file_count,
            bytes_downloaded=current,
            bytes_total=total,
            bytes_per_second=speed,
            progress=percent,
            message=message,
            hf_download=self.summary,
        )


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

    logger.info(_("Preparing model download: %s"), log_label)
    logger.info(_("Model source repo: %s"), repo_id)
    logger.info(_("Model cache destination: %s"), cache_dir)
    logger.info(
        _("HF download options: xet=%s proxy=%s etag_timeout=%s download_timeout=%s max_workers=%s"),
        summary["xet"],
        summary["proxy"],
        summary["etag_timeout"],
        summary["download_timeout"],
        summary["max_workers"],
    )
    if stage is not None:
        stage.update(0, message=_("model source: {}").format(repo_id))
    if local_files_only:
        logger.info(_("Local-files-only mode is enabled; py-roller will only use existing local cache for %s"), log_label)
        if stage is not None:
            stage.update(0, message=_("using local cache only"))
    else:
        logger.info(_("Downloading model into local cache if missing: %s"), log_label)
        if stage is not None:
            stage.update(0, message=_("checking/downloading model cache (HF XET={})").format(summary['xet']))

    with hf_download_environment(config, local_files_only=local_files_only):
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                _("huggingface_hub is required to materialize transcriber models from Hugging Face. Install with: pip install .[audio-core]")
            ) from exc

        file_stats = _repo_file_stats_from_hub(repo_id, local_files_only=local_files_only, kwargs=kwargs)
        bytes_total = file_stats.get("bytes_total")
        try:
            with _DownloadProgressWatcher(
                repo_id=repo_id,
                cache_dir=cache_dir,
                stage=stage,
                bytes_total=bytes_total,
                summary=summary,
                largest_file=file_stats.get("largest_file"),
                file_count=file_stats.get("file_count"),
            ):
                resolved = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    **kwargs,
                )
        except Exception as exc:
            hints = huggingface_download_error_hints(exc)
            hint_text = _(" Suggested fix: {}.").format("; ".join(hints)) if hints else ""
            raise RuntimeError(_("Hugging Face model download failed for {!r}.{}").format(repo_id, hint_text)) from exc
    if stage is not None:
        stage.update(0, message=_("model cache ready"))
    return resolved
