from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pyroller.domain import AudioArtifact


class AudioFilter(ABC):
    name = "filter"
    accepts = ("vocal_audio",)
    produces = "filtered_vocal_audio"

    @abstractmethod
    def process(self, audio_artifact: AudioArtifact, output_dir: Path) -> AudioArtifact:
        raise NotImplementedError
