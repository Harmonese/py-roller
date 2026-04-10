from __future__ import annotations

import logging
import sys
from pathlib import Path

from pyroller.domain import AudioArtifact
from pyroller.process_control import run_subprocess
from pyroller.progress import ProgressReporter
from pyroller.splitter.base import Splitter
from pyroller.utils.ids import make_id

logger = logging.getLogger("pyroller.splitter")


class DemucsSplitter(Splitter):
    def __init__(
        self,
        output_dir: Path,
        model: str = "htdemucs",
        two_stems: str = "vocals",
        device: str | None = None,
        jobs: int | None = None,
        overlap: float | None = None,
        segment: float | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.model = model
        self.two_stems = two_stems
        self.device = device
        self.jobs = jobs
        self.overlap = overlap
        self.segment = segment

    def split(self, audio_path: Path, progress: ProgressReporter | None = None) -> AudioArtifact:
        stage = progress.stage("splitter", total=2, unit="phase") if progress is not None else None
        if stage is not None:
            stage.phase("starting Demucs (native progress below)")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            self.model,
            "--two-stems",
            self.two_stems,
        ]
        if self.device:
            cmd.extend(["-d", self.device])
        if self.jobs is not None:
            cmd.extend(["-j", str(self.jobs)])
        if self.overlap is not None:
            cmd.extend(["--overlap", str(self.overlap)])
        if self.segment is not None:
            cmd.extend(["--segment", str(self.segment)])
        cmd.extend([
            "-o",
            str(self.output_dir),
            str(audio_path),
        ])
        logger.info("Running Demucs: %s", " ".join(cmd))
        run_subprocess(cmd)

        stem_path = self.output_dir / self.model / audio_path.stem / f"{self.two_stems}.wav"
        if not stem_path.exists():
            raise FileNotFoundError(f"Demucs output not found: {stem_path}")
        if stage is not None:
            stage.phase("collecting vocal stem")
            stage.close("splitter output ready")

        return AudioArtifact(
            artifact_id=make_id("artifact"),
            stage="splitter",
            kind="audio",
            path=stem_path,
            role="vocal_audio",
            metadata={
                "backend": "demucs",
                "model": self.model,
                "two_stems": self.two_stems,
                "device": self.device,
                "jobs": self.jobs,
                "overlap": self.overlap,
                "segment": self.segment,
                "source_audio": str(audio_path),
            },
        )
