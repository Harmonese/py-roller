from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from pyroller.domain import AudioArtifact
from pyroller.filter.base import AudioFilter
from pyroller.progress import ProgressReporter
from pyroller.utils.ids import make_id

logger = logging.getLogger("pyroller.filter")


class FilterChain:
    def __init__(self, filters: Iterable[AudioFilter], output_dir: Path) -> None:
        self.filters = list(filters)
        self.output_dir = output_dir

    def process(self, audio_artifact: AudioArtifact, progress: ProgressReporter | None = None) -> AudioArtifact:
        total_phases = max(1, len(self.filters) + 1)
        stage = progress.stage("filter", total=total_phases, unit="phase") if progress is not None else None
        if not self.filters:
            logger.info("Filter chain empty: forwarding %s unchanged", audio_artifact.path)
            metadata = dict(audio_artifact.metadata)
            metadata["filter_chain"] = []
            metadata["source_artifact_id"] = audio_artifact.artifact_id
            if stage is not None:
                stage.phase("no filter steps configured; forwarding audio unchanged")
                stage.close("filter stage skipped")
            return AudioArtifact(
                artifact_id=make_id("artifact"),
                stage="filter",
                kind="audio",
                path=audio_artifact.path,
                sample_rate=audio_artifact.sample_rate,
                channels=audio_artifact.channels,
                duration=audio_artifact.duration,
                role="filtered_vocal_audio",
                metadata=metadata,
            )

        current = audio_artifact
        applied: list[str] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if stage is not None:
            stage.phase("preparing filter chain")
        for audio_filter in self.filters:
            filter_name = getattr(audio_filter, "name", audio_filter.__class__.__name__)
            logger.info("Applying filter step: %s", filter_name)
            current = audio_filter.process(current, self.output_dir)
            applied.append(filter_name)
            if stage is not None:
                stage.phase(f"applied {filter_name}")

        metadata = dict(current.metadata)
        metadata["filter_chain"] = applied
        if "source_artifact_id" not in metadata:
            metadata["source_artifact_id"] = audio_artifact.artifact_id
        current.metadata = metadata
        current.role = "filtered_vocal_audio"
        if stage is not None:
            stage.close("filter stage complete")
        return current
