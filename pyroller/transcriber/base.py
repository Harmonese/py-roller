from __future__ import annotations

from abc import ABC, abstractmethod

from pyroller.domain import AudioArtifact, TranscriptionResult
from pyroller.progress import ProgressReporter, StageProgress


class Transcriber(ABC):
    name = "transcriber"
    accepts = ("filtered_vocal_audio",)
    produces = "timed_units"

    def preflight(self, language: str, stage: StageProgress | None = None) -> dict[str, object]:
        return {}

    def preflight_phase_total(self, language: str) -> int:
        return 0

    def transcribe_phase_total(self, language: str) -> int:
        return 0

    def close(self) -> None:
        return None

    @abstractmethod
    def transcribe(self, audio_artifact: AudioArtifact, language: str, tone_mode: str, progress: ProgressReporter | None = None) -> TranscriptionResult:
        raise NotImplementedError
