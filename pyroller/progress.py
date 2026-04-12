from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - optional dependency in some environments
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

logger = logging.getLogger("pyroller.progress")


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


def build_cli_progress_reporter() -> ProgressReporter:
    if sys.stderr.isatty():
        return TqdmProgressReporter()
    return LoggingProgressReporter()
