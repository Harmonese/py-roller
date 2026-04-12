from __future__ import annotations

from abc import ABC, abstractmethod

from pyroller.domain import AudioArtifact
from pyroller.progress import StageProgress
from pyroller.transcriber.engine_types import EngineOutput


class TranscriberEngine(ABC):
    name = "engine"

    def preflight(self, language: str, stage: StageProgress | None = None) -> dict[str, object]:
        return self.prepare(language, stage=stage)

    @abstractmethod
    def prepare(self, language: str, stage: StageProgress | None = None) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        language: str,
        stage: StageProgress | None = None,
    ) -> EngineOutput:
        raise NotImplementedError

    def close(self) -> None:
        return None

    @property
    def transcribe_phase_total(self) -> int:
        return 5
