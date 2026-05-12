from __future__ import annotations

from pyroller.domain import AudioArtifact, TranscriptionResult
from pyroller.progress import ProgressReporter
from pyroller.transcriber.base import Transcriber
from pyroller.transcriber.engines.base import TranscriberEngine
from pyroller.transcriber.unitizers.base import TranscriptionAdapter


class ComposedTranscriber(Transcriber):
    def __init__(self, *, engine: TranscriberEngine, adapter: TranscriptionAdapter, backend_name: str) -> None:
        self.engine = engine
        self.adapter = adapter
        self.backend_name = backend_name

    def preflight(self, language: str, stage=None) -> dict[str, object]:
        return self.engine.preflight(language, stage=stage)

    def preflight_phase_total(self, language: str) -> int:
        return self.engine.preflight_phase_total(language)

    def transcribe_phase_total(self, language: str) -> int:
        return self.engine.transcribe_phase_total(language)

    def close(self) -> None:
        self.engine.close()

    def transcribe(
        self,
        audio_artifact: AudioArtifact,
        language: str,
        tone_mode: str,
        progress: ProgressReporter | None = None,
    ) -> TranscriptionResult:
        stage = progress.stage("transcriber", total=self.engine.transcribe_phase_total(language), unit="phase") if progress is not None else None
        stage_failed = False
        stage_failure_message = "transcriber failed"
        try:
            engine_output = self.engine.transcribe(audio_artifact, language, stage=stage)
            result = self.adapter.adapt(engine_output, language=language, tone_mode=tone_mode)
            return result
        except Exception as exc:
            stage_failed = True
            stage_failure_message = f"transcriber failed: {exc.__class__.__name__}: {exc}"
            raise
        finally:
            if stage is not None:
                if stage_failed:
                    stage.fail(stage_failure_message)
                else:
                    stage.close("transcriber complete")
