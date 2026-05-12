from __future__ import annotations

from abc import ABC, abstractmethod

from pyroller.domain import AlignmentResult, ParsedLyrics, TranscriptionResult
from pyroller.progress import ProgressReporter


class Aligner(ABC):
    name = "aligner"
    accepts = ("timed_units", "parsed_lyrics")
    produces = "alignment_result"

    @abstractmethod
    def align(self, transcription: TranscriptionResult, parsed_lyrics: ParsedLyrics, progress: ProgressReporter | None = None) -> AlignmentResult:
        raise NotImplementedError
