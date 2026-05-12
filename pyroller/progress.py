from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pyroller.utils.json import json_default

try:  # pragma: no cover - optional dependency in some environments
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

logger = logging.getLogger("pyroller.progress")
_EVENT_PREFIX = "PYROLLER_EVENT "


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _percent(completed: int | float, total: int | float) -> float | None:
    if total <= 0:
        return None
    return max(0.0, min(1.0, float(completed) / float(total)))


class StageProgress:
    def phase(self, message: str, advance: int = 1) -> None:
        raise NotImplementedError

    def update(self, advance: int = 1, message: str | None = None) -> None:
        raise NotImplementedError

    def close(self, message: str | None = None) -> None:
        raise NotImplementedError

    def fail(self, message: str | None = None) -> None:
        raise NotImplementedError

    def event(self, event_type: str, **payload: Any) -> None:
        return None


class ProgressReporter:
    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        raise NotImplementedError

    def event(self, event_type: str, **payload: Any) -> None:
        return None


class NullStageProgress(StageProgress):
    def phase(self, message: str, advance: int = 1) -> None:
        return None

    def update(self, advance: int = 1, message: str | None = None) -> None:
        return None

    def close(self, message: str | None = None) -> None:
        return None

    def fail(self, message: str | None = None) -> None:
        return None

    def event(self, event_type: str, **payload: Any) -> None:
        return None


class NullProgressReporter(ProgressReporter):
    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        return NullStageProgress()

    def event(self, event_type: str, **payload: Any) -> None:
        return None


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


class TqdmStageProgress(StageProgress):
    def __init__(self, name: str, total: int, unit: str = "step") -> None:
        self.name = name
        self.total = max(total, 1)
        self._bar = tqdm(total=self.total, desc=name, unit=unit, leave=False, file=sys.stderr)

    def phase(self, message: str, advance: int = 1) -> None:
        self.update(advance=advance, message=message)

    def update(self, advance: int = 1, message: str | None = None) -> None:
        if message:
            self._bar.set_postfix_str(message[:80], refresh=False)
        self._bar.update(max(advance, 0))

    def close(self, message: str | None = None) -> None:
        if message:
            self._bar.set_postfix_str(message[:80], refresh=False)
        remaining = self.total - self._bar.n
        if remaining > 0:
            self._bar.update(remaining)
        self._bar.close()

    def fail(self, message: str | None = None) -> None:
        if message:
            self._bar.set_postfix_str(("FAILED: " + message)[:80], refresh=False)
        self._bar.close()


class TqdmProgressReporter(ProgressReporter):
    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        if tqdm is None:
            return LoggingStageProgress(name=name, total=total, unit=unit)
        return TqdmStageProgress(name=name, total=total, unit=unit)


class JsonlStageProgress(StageProgress):
    def __init__(self, name: str, total: int, unit: str = "step", *, prefix: str = "") -> None:
        self.name = name
        self.total = max(total, 0)
        self.unit = unit
        self.prefix = prefix
        self.completed = 0
        self.event(
            "stage_started",
            stage=self.name,
            completed=self.completed,
            total=self.total,
            unit=self.unit,
            percent=_percent(self.completed, self.total),
            message=f"{self.name} started",
        )

    def _emit(self, event_type: str, **payload: Any) -> None:
        payload.setdefault("type", event_type)
        payload.setdefault("time", _utc_now())
        payload.setdefault("stage", self.name)
        if self.prefix:
            payload.setdefault("prefix", self.prefix)
        print(_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, default=json_default), flush=True)

    def phase(self, message: str, advance: int = 1) -> None:
        self.update(advance=advance, message=message)

    def update(self, advance: int = 1, message: str | None = None) -> None:
        self.completed = min(self.total, self.completed + max(advance, 0)) if self.total > 0 else self.completed
        self._emit(
            "stage_progress",
            completed=self.completed,
            total=self.total,
            unit=self.unit,
            percent=_percent(self.completed, self.total),
            message=message or "",
        )

    def close(self, message: str | None = None) -> None:
        if self.total > 0:
            self.completed = self.total
        self._emit(
            "stage_completed",
            completed=self.completed,
            total=self.total,
            unit=self.unit,
            percent=1.0 if self.total > 0 else None,
            message=message or f"{self.name} complete",
            done=True,
        )

    def fail(self, message: str | None = None) -> None:
        self._emit(
            "stage_failed",
            completed=self.completed,
            total=self.total,
            unit=self.unit,
            percent=_percent(self.completed, self.total),
            message=message or f"{self.name} failed",
            failed=True,
        )

    def event(self, event_type: str, **payload: Any) -> None:
        self._emit(event_type, **payload)


class JsonlProgressReporter(ProgressReporter):
    def __init__(self, *, prefix: str = "") -> None:
        self.prefix = prefix

    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        return JsonlStageProgress(name=name, total=total, unit=unit, prefix=self.prefix)

    def event(self, event_type: str, **payload: Any) -> None:
        payload.setdefault("type", event_type)
        payload.setdefault("time", _utc_now())
        if self.prefix:
            payload.setdefault("prefix", self.prefix)
        print(_EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, default=json_default), flush=True)


class MultiStageProgress(StageProgress):
    def __init__(self, stages: list[StageProgress]) -> None:
        self.stages = stages

    def phase(self, message: str, advance: int = 1) -> None:
        for stage in self.stages:
            stage.phase(message, advance=advance)

    def update(self, advance: int = 1, message: str | None = None) -> None:
        for stage in self.stages:
            stage.update(advance=advance, message=message)

    def close(self, message: str | None = None) -> None:
        for stage in self.stages:
            stage.close(message=message)

    def fail(self, message: str | None = None) -> None:
        for stage in self.stages:
            stage.fail(message=message)

    def event(self, event_type: str, **payload: Any) -> None:
        for stage in self.stages:
            stage.event(event_type, **payload)


class MultiProgressReporter(ProgressReporter):
    def __init__(self, reporters: list[ProgressReporter]) -> None:
        self.reporters = reporters

    def stage(self, name: str, *, total: int, unit: str = "step") -> StageProgress:
        return MultiStageProgress([reporter.stage(name, total=total, unit=unit) for reporter in self.reporters])

    def event(self, event_type: str, **payload: Any) -> None:
        for reporter in self.reporters:
            reporter.event(event_type, **payload)


def _human_progress_reporter() -> ProgressReporter:
    if sys.stderr.isatty():
        return TqdmProgressReporter()
    return LoggingProgressReporter()


def build_cli_progress_reporter(progress_format: str = "human") -> ProgressReporter:
    normalized = (progress_format or "human").strip().lower()
    if normalized == "jsonl":
        return JsonlProgressReporter()
    if normalized == "both":
        return MultiProgressReporter([_human_progress_reporter(), JsonlProgressReporter()])
    return _human_progress_reporter()
