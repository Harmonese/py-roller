from __future__ import annotations

from pyroller.i18n import _

import json
import logging
import sys
import threading
from collections.abc import Callable
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


def _ratio(completed: int | float, total: int | float) -> float | None:
    if total <= 0:
        return None
    return max(0.0, min(1.0, float(completed) / float(total)))


def _normalize_progress_value(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def normalize_progress_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize progress event payloads while keeping old keys compatible.

    ``progress`` is the canonical 0.0-1.0 value for frontends.  The older
    ``percent`` key is kept as a compatibility alias during the 0.5.x line.
    """

    progress = _normalize_progress_value(payload.get("progress"))
    if progress is None:
        progress = _normalize_progress_value(payload.get("percent"))
    if progress is not None:
        payload["progress"] = progress
        payload.setdefault("percent", progress)
    return payload


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
            logger.info(_("%s%s - %s"), label, suffix, message)
        else:
            logger.info(_("%s%s"), label, suffix)

    def close(self, message: str | None = None) -> None:
        label = f"{self.prefix}{self.name}"
        if message:
            logger.info(_("%s complete - %s"), label, message)
        else:
            logger.info(_("%s complete"), label)

    def fail(self, message: str | None = None) -> None:
        label = f"{self.prefix}{self.name}"
        if message:
            logger.error(_("%s failed - %s"), label, message)
        else:
            logger.error(_("%s failed"), label)


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
            self._bar.set_postfix_str((_("FAILED: ") + message)[:80], refresh=False)
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
            progress=_ratio(self.completed, self.total),
            message=_("{} started").format(self.name),
        )

    def _emit(self, event_type: str, **payload: Any) -> None:
        payload.setdefault("schema_version", 1)
        payload.setdefault("type", event_type)
        now = _utc_now()
        payload.setdefault("timestamp", now)
        payload.setdefault("time", now)  # compatibility alias used by early GUI builds
        payload.setdefault("stage", self.name)
        payload.setdefault("message", "")
        payload.setdefault("progress", None)
        if self.prefix:
            payload.setdefault("prefix", self.prefix)
        normalize_progress_payload(payload)
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
            progress=_ratio(self.completed, self.total),
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
            progress=1.0 if self.total > 0 else None,
            message=message or _("{} complete").format(self.name),
            done=True,
        )

    def fail(self, message: str | None = None) -> None:
        self._emit(
            "stage_failed",
            completed=self.completed,
            total=self.total,
            unit=self.unit,
            progress=_ratio(self.completed, self.total),
            message=message or _("{} failed").format(self.name),
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
        payload.setdefault("schema_version", 1)
        payload.setdefault("type", event_type)
        now = _utc_now()
        payload.setdefault("timestamp", now)
        payload.setdefault("time", now)
        payload.setdefault("stage", "")
        payload.setdefault("message", "")
        payload.setdefault("progress", None)
        if self.prefix:
            payload.setdefault("prefix", self.prefix)
        normalize_progress_payload(payload)
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


class ProgressHeartbeat:
    def __init__(
        self,
        stage: StageProgress | ProgressReporter | None,
        *,
        event_stage: str,
        message: str,
        interval_seconds: float = 10.0,
        payload_factory: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.stage = stage
        self.event_stage = event_stage
        self.message = message
        self.interval_seconds = interval_seconds
        self.payload_factory = payload_factory
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_emit = 0.0

    def __enter__(self) -> ProgressHeartbeat:
        if self.stage is not None:
            self._thread = threading.Thread(target=self._run, name=f"pyroller-{self.event_stage}-heartbeat", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            payload: dict[str, Any] = {}
            if self.payload_factory is not None:
                try:
                    payload = self.payload_factory() or {}
                except Exception as exc:  # pragma: no cover - diagnostic only
                    payload = {"heartbeat_error": str(exc)}
            payload.setdefault("stage", self.event_stage)
            payload.setdefault("message", self.message)
            payload.setdefault("seconds_since_last_progress", self.interval_seconds)
            self.stage.event("heartbeat", **payload)


def progress_heartbeat(
    stage: StageProgress | ProgressReporter | None,
    *,
    event_stage: str,
    message: str,
    interval_seconds: float = 10.0,
    payload_factory: Callable[[], dict[str, Any]] | None = None,
) -> ProgressHeartbeat:
    return ProgressHeartbeat(
        stage,
        event_stage=event_stage,
        message=message,
        interval_seconds=interval_seconds,
        payload_factory=payload_factory,
    )


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
