from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from typing import TextIO

logger = logging.getLogger("pyroller.progress")

_PROGRESS_MODES = {"auto", "plain", "json", "off"}


class StageProgress:
    def phase(self, message: str, advance: int = 1) -> None:
        raise NotImplementedError

    def update(self, advance: int = 1, message: str | None = None) -> None:
        raise NotImplementedError

    def close(self, message: str | None = None) -> None:
        raise NotImplementedError

    def fail(self, message: str | None = None) -> None:
        raise NotImplementedError


class ProgressReporter:
    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        raise NotImplementedError


class NullStageProgress(StageProgress):
    def phase(self, message: str, advance: int = 1) -> None:
        return None

    def update(self, advance: int = 1, message: str | None = None) -> None:
        return None

    def close(self, message: str | None = None) -> None:
        return None

    def fail(self, message: str | None = None) -> None:
        return None


class NullProgressReporter(ProgressReporter):
    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        return NullStageProgress()


@dataclass
class LoggingStageProgress(StageProgress):
    name: str
    total: int
    unit: str = "step"
    prefix: str = ""
    completed: int = 0

    def phase(self, message: str, advance: int = 1) -> None:
        self.update(advance=advance, message=message)

    def update(self, advance: int = 1, message: str | None = None) -> None:
        self.completed = min(self.total, self.completed + max(advance, 0))
        suffix = f" [{self.completed}/{self.total} {self.unit}]" if self.total > 0 else ""
        label = f"{self.prefix}{self.name}"
        if message:
            logger.info("%s%s - %s", label, suffix, message)
        else:
            logger.info("%s%s", label, suffix)

    def close(self, message: str | None = None) -> None:
        label = f"{self.prefix}{self.name}"
        if message:
            logger.info("%s complete - %s", label, message)
        else:
            logger.info("%s complete", label)

    def fail(self, message: str | None = None) -> None:
        label = f"{self.prefix}{self.name}"
        if message:
            logger.error("%s failed - %s", label, message)
        else:
            logger.error("%s failed", label)


class LoggingProgressReporter(ProgressReporter):
    def __init__(self, *, prefix: str = "") -> None:
        self.prefix = prefix

    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        return LoggingStageProgress(name=name, total=total, unit=unit, prefix=self.prefix)


class JsonStageProgress(StageProgress):
    def __init__(self, name: str, total: int, unit: str = "step", *, prefix: str = "", stream: TextIO | None = None) -> None:
        self.name = f"{prefix}{name}" if prefix else name
        self.total = max(int(total), 0)
        self.unit = unit
        self.completed = 0
        self.started_at = time.monotonic()
        self.stream = stream or sys.stderr
        self._emit("started", None)

    def phase(self, message: str, advance: int = 1) -> None:
        self.update(advance=advance, message=message)

    def update(self, advance: int = 1, message: str | None = None) -> None:
        self.completed = min(self.total, self.completed + max(int(advance), 0))
        self._emit("running", message)

    def close(self, message: str | None = None) -> None:
        if self.total > 0:
            self.completed = self.total
        self._emit("done", message)

    def fail(self, message: str | None = None) -> None:
        self._emit("failed", message)

    def _emit(self, status: str, message: str | None) -> None:
        payload = {
            "type": "progress",
            "stage": self.name,
            "status": status,
            "current": self.completed,
            "total": self.total,
            "unit": self.unit,
            "message": message or "",
            "elapsed": round(time.monotonic() - self.started_at, 3),
        }
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), file=self.stream, flush=True)


class JsonProgressReporter(ProgressReporter):
    def __init__(self, *, prefix: str = "", stream: TextIO | None = None) -> None:
        self.prefix = prefix
        self.stream = stream or sys.stderr

    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        return JsonStageProgress(name=name, total=total, unit=unit, prefix=self.prefix, stream=self.stream)


def normalize_progress_mode(mode: str | None) -> str:
    normalized = str(mode or "auto").strip().lower()
    if normalized not in _PROGRESS_MODES:
        raise ValueError(f"Unsupported progress mode {mode!r}. Expected one of: auto, plain, json, off.")
    return normalized


def build_cli_progress_reporter(mode: str | None = "auto", *, prefix: str = "") -> ProgressReporter:
    normalized = normalize_progress_mode(mode)
    if normalized == "off":
        return NullProgressReporter()
    if normalized == "json":
        return JsonProgressReporter(prefix=prefix)
    return LoggingProgressReporter(prefix=prefix)
