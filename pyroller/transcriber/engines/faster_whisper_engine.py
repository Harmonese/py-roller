from __future__ import annotations

import gc
import importlib.metadata
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyroller.domain import AudioArtifact
from pyroller.transcriber.engine_types import ENGINE_OUTPUT_SCHEMA_VERSION, EngineOutput, EngineSpan
from pyroller.transcriber.engines.base import TranscriberEngine
from pyroller.progress import progress_heartbeat
from pyroller.transcriber.hf_download_config import HFDownloadConfig, hf_download_environment

logger = logging.getLogger("pyroller.transcriber")


@dataclass
class _PreparedFasterWhisperBundle:
    language: str
    plan: Any
    model: Any
    batched_pipeline: Any | None
    runtime_report: dict[str, object]


class FasterWhisperEngine(TranscriberEngine):
    name = "faster_whisper"

    def __init__(
        self,
        *,
        model_name: str = "large-v2",
        model_path: str | Path | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 8,
        vad_filter: bool = True,
        hf_xet: str = "auto",
        hf_proxy: str | None = None,
        hf_etag_timeout: float | None = None,
        hf_download_timeout: float | None = None,
        hf_max_workers: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.model_path = str(model_path) if model_path is not None else None
        self.local_files_only = local_files_only
        self.device = device
        self.compute_type = compute_type
        self.batch_size = max(int(batch_size), 1)
        self.vad_filter = bool(vad_filter)
        self.hf_download_config = HFDownloadConfig(
            xet=hf_xet,
            proxy=hf_proxy,
            etag_timeout=hf_etag_timeout,
            download_timeout=hf_download_timeout,
            max_workers=hf_max_workers,
        )
        self._prepared: _PreparedFasterWhisperBundle | None = None

    def _prepare_phase_count(self, language: str) -> int:
        if self._is_prepared_for(language):
            return 1
        return 2

    def preflight_phase_total(self, language: str) -> int:
        return self._prepare_phase_count(language)

    def transcribe_phase_total(self, language: str) -> int:
        return 1 + self._prepare_phase_count(language) + 1

    def _build_resolution_plan(self, language: str, *, materialize: bool, stage=None):
        from pyroller.transcriber.model_resolver import TranscriberModelResolver

        resolver = TranscriberModelResolver(
            backend="faster_whisper",
            language=language,
            model_name=self.model_name,
            model_path=self.model_path,
            local_files_only=self.local_files_only,
            hf_download_config=self.hf_download_config,
        )
        return resolver.resolve(materialize=materialize, stage=stage)

    def _is_prepared_for(self, language: str) -> bool:
        return self._prepared is not None and self._prepared.language == language

    def _clear_device_cache(self) -> None:
        gc.collect()
        try:
            import torch  # type: ignore
        except ImportError:
            return
        if str(self.device).startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _runtime_versions(self) -> dict[str, str | None]:
        versions: dict[str, str | None] = {}
        for dist_name in ("faster-whisper", "ctranslate2"):
            try:
                versions[dist_name] = importlib.metadata.version(dist_name)
            except importlib.metadata.PackageNotFoundError:
                versions[dist_name] = None
        return versions

    def _prepare_bundle(self, language: str, stage=None) -> _PreparedFasterWhisperBundle:
        if self._is_prepared_for(language):
            if stage is not None:
                stage.phase("reusing prepared faster-whisper model")
            return self._prepared  # type: ignore[return-value]

        with hf_download_environment(self.hf_download_config, local_files_only=self.local_files_only):
            try:
                from faster_whisper import WhisperModel  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("faster-whisper is not installed. Install with: pip install faster-whisper") from exc

            try:
                from faster_whisper import BatchedInferencePipeline  # type: ignore
            except Exception:  # pragma: no cover
                BatchedInferencePipeline = None

            self.close()
            if stage is not None:
                stage.phase("resolving transcriber model")
            plan = self._build_resolution_plan(language, materialize=True, stage=stage)
            model = None
            batched_pipeline = None
            try:
                model_ref = str(plan.resolved_model_dir) if plan.resolved_model_dir is not None else plan.effective_model_name
                logger.info("Preparing faster-whisper model=%s from=%s device=%s", self.model_name, model_ref, self.device)
                if stage is not None:
                    stage.phase("loading faster-whisper model")
                model = WhisperModel(
                    model_ref,
                    device=self.device,
                    compute_type=self.compute_type,
                    download_root=str(plan.download_root) if plan.download_root is not None else None,
                    local_files_only=True,
                )
                if self.batch_size > 1 and BatchedInferencePipeline is not None:
                    batched_pipeline = BatchedInferencePipeline(model=model)
                elif self.batch_size > 1 and BatchedInferencePipeline is None:
                    logger.warning("BatchedInferencePipeline is unavailable in the installed faster-whisper package; falling back to non-batched inference")
                runtime = plan.runtime_record()
                runtime["compatibility_bundle"] = self._runtime_versions()
                runtime["batching_enabled"] = batched_pipeline is not None
                runtime["native_word_timestamps"] = True
                bundle = _PreparedFasterWhisperBundle(
                    language=language,
                    plan=plan,
                    model=model,
                    batched_pipeline=batched_pipeline,
                    runtime_report=runtime,
                )
                self._prepared = bundle
                return bundle
            except Exception:
                if batched_pipeline is not None:
                    del batched_pipeline
                if model is not None:
                    del model
                self._clear_device_cache()
                raise

    def close(self) -> None:
        bundle = self._prepared
        self._prepared = None
        if bundle is None:
            logger.debug("FasterWhisperEngine close() called with no prepared bundle to release")
            return
        logger.info("Closing prepared faster-whisper model=%s language=%s device=%s", self.model_name, bundle.language, self.device)
        try:
            bundle.batched_pipeline = None
            bundle.model = None
        finally:
            self._clear_device_cache()
            logger.info("Closed prepared faster-whisper model=%s language=%s device=%s", self.model_name, bundle.language, self.device)

    def prepare(self, language: str, stage=None) -> dict[str, object]:
        bundle = self._prepare_bundle(language, stage=stage)
        return dict(bundle.runtime_report)

    def transcribe(self, audio_artifact: AudioArtifact, language: str, stage=None) -> EngineOutput:
        if stage is not None:
            stage.phase("loading faster-whisper backend")
        try:
            import faster_whisper  # noqa: F401  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("faster-whisper is not installed. Install with: pip install faster-whisper") from exc

        audio_path = Path(audio_artifact.path) if audio_artifact.path is not None else None
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for faster-whisper backend: {audio_artifact.path}")

        bundle = self._prepare_bundle(language, stage=stage)
        logger.info(
            "Using prepared faster-whisper model=%s from=%s device=%s batch_size=%s",
            self.model_name,
            bundle.plan.resolved_model_dir,
            self.device,
            self.batch_size,
        )

        if stage is not None:
            # Do not advance the coarse phase counter to 3/3 before inference has
            # actually completed. GUI frontends otherwise look stuck at 100% while
            # faster-whisper is still transcribing. Detailed percent updates are
            # emitted below as segments are yielded.
            stage.update(advance=0, message="running transcription inference")

        transcribe_kwargs = {
            "word_timestamps": True,
        }
        if language != "mul":
            transcribe_kwargs["language"] = language
        if bundle.batched_pipeline is not None:
            segments_iter, info = bundle.batched_pipeline.transcribe(
                str(audio_path),
                batch_size=self.batch_size,
                **transcribe_kwargs,
            )
        else:
            segments_iter, info = bundle.model.transcribe(
                str(audio_path),
                vad_filter=self.vad_filter,
                **transcribe_kwargs,
            )

        audio_duration_hint = self._float_or_none(self._field(info, "duration"))
        duration_after_vad = self._float_or_none(self._field(info, "duration_after_vad"))
        if stage is not None:
            stage.event(
                "stage_progress",
                stage="transcriber",
                message="Transcribing audio",
                audio_duration=audio_duration_hint,
                duration_after_vad=duration_after_vad,
                progress=None,
            )

        segments = []
        last_percent_reported = -1
        last_segment_end: float | None = None

        def _heartbeat_payload() -> dict[str, object]:
            payload: dict[str, object] = {
                "message": "Still running transcription inference",
                "segments": len(segments),
                "audio_duration": audio_duration_hint,
                "duration_after_vad": duration_after_vad,
            }
            if last_segment_end is not None:
                payload["audio_time_processed"] = last_segment_end
            if audio_duration_hint and last_segment_end is not None:
                payload["progress"] = max(0.0, min(1.0, last_segment_end / audio_duration_hint))
            return payload

        with progress_heartbeat(
            stage,
            event_stage="transcriber",
            message="Still running transcription inference",
            interval_seconds=10.0,
            payload_factory=_heartbeat_payload,
        ):
            for segment in segments_iter:
                segments.append(segment)
                segment_end = self._float_or_none(self._field(segment, "end"))
                if segment_end is not None:
                    last_segment_end = segment_end
                progress_value = None
                if audio_duration_hint and segment_end is not None:
                    progress_value = max(0.0, min(1.0, segment_end / audio_duration_hint))
                percent_bucket = int((progress_value or 0.0) * 100)
                if stage is not None and (progress_value is None or percent_bucket >= last_percent_reported + 2 or percent_bucket >= 100):
                    last_percent_reported = percent_bucket
                    stage.event(
                        "stage_progress",
                        stage="transcriber",
                        message="Transcribing audio",
                        completed=percent_bucket if progress_value is not None else len(segments),
                        total=100 if progress_value is not None else 0,
                        unit="%" if progress_value is not None else "segment",
                        progress=progress_value,
                        audio_time_processed=segment_end,
                        audio_duration=audio_duration_hint,
                        duration_after_vad=duration_after_vad,
                        segments=len(segments),
                        text=self._normalized_text(self._field(segment, "text")),
                    )

        if stage is not None:
            stage.update(advance=1, message="transcription inference complete")
        spans = self._segments_to_spans(segments)
        raw_text = " ".join(text for text in (self._normalized_text(self._field(segment, "text")) for segment in segments) if text) or None
        audio_duration = self._infer_audio_duration(info, segments)
        metadata = {
            "engine_output_schema": ENGINE_OUTPUT_SCHEMA_VERSION,
            "engine": self.name,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "native_word_timestamps": True,
            "batched_inference": bundle.batched_pipeline is not None,
            "audio_duration": audio_duration,
            "detected_language": self._field(info, "language"),
            "detected_language_probability": self._float_or_none(self._field(info, "language_probability")),
            "duration_after_vad": self._float_or_none(self._field(info, "duration_after_vad")),
            "source_audio": str(audio_artifact.path) if audio_artifact.path else None,
            "source_role": audio_artifact.role,
            "runtime": dict(bundle.runtime_report),
        }
        return EngineOutput(
            language=language,
            engine=self.name,
            raw_text=raw_text,
            spans=spans,
            metadata=metadata,
        )

    def _segments_to_spans(self, segments: list[Any]) -> list[EngineSpan]:
        spans: list[EngineSpan] = []
        for seg_index, segment in enumerate(segments):
            segment_span_id = f"segment:{seg_index}"
            segment_start = self._float_or_default(self._field(segment, "start"), 0.0)
            segment_end = self._float_or_default(self._field(segment, "end"), segment_start)
            words = list(self._field(segment, "words") or [])
            spans.append(
                EngineSpan(
                    span_id=segment_span_id,
                    level="segment",
                    start_time=segment_start,
                    end_time=segment_end,
                    text=self._field(segment, "text"),
                    confidence=self._float_or_none(self._field(segment, "avg_logprob")),
                    segment_index=seg_index,
                    metadata={"word_count": len(words)},
                )
            )
            for word_index, word in enumerate(words):
                word_start = self._float_or_default(self._field(word, "start"), segment_start)
                word_end = self._float_or_default(self._field(word, "end"), segment_end)
                spans.append(
                    EngineSpan(
                        span_id=f"{segment_span_id}:word:{word_index}",
                        level="word",
                        start_time=word_start,
                        end_time=word_end,
                        text=self._field(word, "word") or "",
                        confidence=self._float_or_none(self._field(word, "probability")),
                        segment_index=seg_index,
                        word_index=word_index,
                        parent_span_id=segment_span_id,
                        metadata={},
                    )
                )
        return spans

    def _infer_audio_duration(self, info: Any, segments: list[Any]) -> float:
        duration = self._float_or_none(self._field(info, "duration"))
        if duration is not None:
            return duration
        if not segments:
            return 0.0
        return max(self._float_or_default(self._field(segment, "end"), 0.0) for segment in segments)

    @staticmethod
    def _field(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    @staticmethod
    def _float_or_default(value: Any, default: float) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        return float(default)

    @staticmethod
    def _normalized_text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
