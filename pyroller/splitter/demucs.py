from __future__ import annotations

import logging

from pyroller.i18n import _
import re
import sys
from pathlib import Path

from pyroller.domain import AudioArtifact
from pyroller.process_control import run_subprocess
from pyroller.progress import ProgressReporter
from pyroller.splitter.base import Splitter
from pyroller.utils.ids import make_id

logger = logging.getLogger("pyroller.splitter")
_DEMUCS_TQDM_RE = re.compile(r"(?P<percent>\d{1,3}(?:\.\d+)?)%\|.*?\|\s*(?P<current>\d+(?:\.\d+)?)/(?P<total>\d+(?:\.\d+)?)(?:\s+\[(?P<bracket>[^\]]+)\])?")


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
            stage.phase(_("starting Demucs (native progress below)"))
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
        logger.info(_("Running Demucs: %s"), " ".join(cmd))

        def _handle_demucs_output(record: str, _separator: str) -> None:
            match = _DEMUCS_TQDM_RE.search(record)
            if stage is None or match is None:
                return
            current = float(match.group("current"))
            total = float(match.group("total"))
            progress_value = max(0.0, min(1.0, current / total)) if total > 0 else None
            stage.event(
                "stage_progress",
                stage="splitter",
                current=current,
                total=total,
                unit="seconds",
                progress=progress_value,
                message=_("Separating vocals: {:.1f} / {:.1f} seconds").format(current, total),
            )

        run_subprocess(cmd, output_callback=_handle_demucs_output if stage is not None else None)

        expected_dir = self.output_dir / self.model
        stem_path = expected_dir / audio_path.stem / f"{self.two_stems}.wav"
        if not stem_path.exists():
            # Demucs version or path canonicalization differences may produce a
            # different directory layout.  Search for any WAV file matching the
            # expected stem name as a graceful fallback.
            candidates = list(expected_dir.glob(f"**/{audio_path.stem}/**/{self.two_stems}.wav"))
            if not candidates:
                candidates = list(expected_dir.glob(f"**/{self.two_stems}.wav"))
            if not candidates:
                raise FileNotFoundError(
                    _("Demucs output not found at expected path {}, and no fallback match found under {}").format(stem_path, expected_dir)
                )
            stem_path = candidates[0]
            logger.info(_("Demucs output resolved via fallback glob: %s"), stem_path)
        if stage is not None:
            stage.phase(_("collecting vocal stem"))
            stage.close(_("splitter output ready"))

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
