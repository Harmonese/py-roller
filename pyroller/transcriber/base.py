from __future__ import annotations

from abc import ABC, abstractmethod

from pyroller.domain import AudioArtifact, TranscriptionResult
from pyroller.progress import ProgressReporter, StageProgress


class Transcriber(ABC):
    name = "transcriber"
    accepts = ("mixed_audio", "vocal_audio", "filtered_vocal_audio")
    produces = "timed_units"

    def preflight(self, language: str, stage: StageProgress | None = None) -> dict[str, object]:
        return {}

    def close(self) -> None:
        return None

    @abstractmethod
    def transcribe(self, audio_artifact: AudioArtifact, language: str, tone_mode: str, progress: ProgressReporter | None = None) -> TranscriptionResult:
        raise NotImplementedError
