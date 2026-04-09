from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pyroller.domain import AudioArtifact


class AudioFilter(ABC):
    name = "filter"
    accepts = ("mixed_audio", "vocal_audio", "filtered_vocal_audio")
    produces = "filtered_vocal_audio"

    @abstractmethod
    def process(self, audio_artifact: AudioArtifact, output_dir: Path) -> AudioArtifact:
        raise NotImplementedError
