from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pyroller.domain import AlignmentResult, WriteResult


class Writer(ABC):
    @abstractmethod
    def write(self, alignment: AlignmentResult, output_path: Path) -> WriteResult:
        raise NotImplementedError
