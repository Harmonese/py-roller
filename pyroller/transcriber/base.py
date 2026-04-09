from __future__ import annotations

from abc import ABC, abstractmethod

from pyroller.domain import AudioArtifact, TranscriptionResult
from pyroller.progress import ProgressReporter


class Transcriber(ABC):
    name = "transcriber"
    accepts = ("mixed_audio", "vocal_audio", "filtered_vocal_audio")
    produces = "timed_units"

    @abstractmethod
    def transcribe(self, audio_artifact: AudioArtifact, language: str, tone_mode: str, progress: ProgressReporter | None = None) -> TranscriptionResult:
        raise NotImplementedError
