from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pyroller.domain import AudioArtifact
from pyroller.progress import ProgressReporter


class Splitter(ABC):
    name = "splitter"
    accepts = ("mixed_audio",)
    produces = "vocal_audio"

    @abstractmethod
    def split(self, audio_path: Path, progress: ProgressReporter | None = None) -> AudioArtifact:
        raise NotImplementedError
