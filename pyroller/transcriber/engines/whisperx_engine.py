from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from typing import Any

from pyroller.domain import AudioArtifact
from pyroller.transcriber.engine_types import ENGINE_OUTPUT_SCHEMA_VERSION, EngineOutput, EngineSpan
from pyroller.transcriber.engines.base import TranscriberEngine

logger = logging.getLogger("pyroller.transcriber")


@dataclass
class _PreparedWhisperXBundle:
    language: str
    plan: Any
    model: Any
    align_model: Any | None
    align_metadata: Any | None
    runtime_report: dict[str, object]


class WhisperXEngine(TranscriberEngine):
    name = "whisperx"

    def __init__(
        self,
        *,
        model_name: str = "large-v2",
        model_path: str | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 8,
        align_words: bool = True,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.local_files_only = local_files_only
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.align_words = align_words
        self._prepared: _PreparedWhisperXBundle | None = None

    @property
    def transcribe_phase_total(self) -> int:
        return 6 if self.align_words else 5

    def _build_resolution_plan(self, language: str, *, materialize: bool, stage=None):
        from pyroller.transcriber.model_resolver import TranscriberModelResolver

        resolver = TranscriberModelResolver(
            backend="whisperx",
            language=language,
            model_name=self.model_name,
            model_path=self.model_path,
            local_files_only=self.local_files_only,
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

    def _prepare_bundle(self, language: str, stage=None) -> _PreparedWhisperXBundle:
        if self._is_prepared_for(language):
            if stage is not None:
                stage.phase("reusing prepared WhisperX model")
            return self._prepared  # type: ignore[return-value]

        try:
            import whisperx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("whisperx is not installed. Install with: pip install whisperx") from exc

        from pyroller.transcriber.whisperx_runtime import (
            build_whisperx_helpful_error,
            collect_whisperx_runtime_versions,
            whisperx_load_model_context,
            whisperx_runtime_context,
        )

        self.close()
        if stage is not None:
            stage.phase("resolving transcriber model")
        plan = self._build_resolution_plan(language, materialize=True, stage=stage)
        model = None
        align_model = None
        align_metadata = None
        try:
            logger.info("Preparing WhisperX model=%s device=%s store=%s", self.model_name, self.device, plan.model_store_root)
            with whisperx_runtime_context(plan):
                model_ref = str(plan.resolved_model_dir) if plan.resolved_model_dir is not None else plan.effective_model_name
                logger.info("Loading WhisperX ASR model from local path: %s", model_ref)
                if stage is not None:
                    stage.phase("loading WhisperX model")
                with whisperx_load_model_context():
                    model = whisperx.load_model(
                        model_ref,
                        self.device,
                        compute_type=self.compute_type,
                        language=language,
                        download_root=str(plan.download_root) if plan.download_root is not None else None,
                    )
                if self.align_words:
                    logger.info("Preparing WhisperX alignment model for language=%s (may download auxiliary assets)", language)
                    if stage is not None:
                        stage.phase("preparing WhisperX alignment model")
                    align_model, align_metadata = whisperx.load_align_model(language_code=language, device=self.device)
            runtime = plan.runtime_record()
            runtime["compatibility_bundle"] = collect_whisperx_runtime_versions()
            runtime["preflight_probe"] = "model-load-only"
            bundle = _PreparedWhisperXBundle(
                language=language,
                plan=plan,
                model=model,
                align_model=align_model,
                align_metadata=align_metadata,
                runtime_report=runtime,
            )
            self._prepared = bundle
            return bundle
        except Exception as exc:
            if align_model is not None:
                del align_model
            if align_metadata is not None:
                del align_metadata
            if model is not None:
                del model
            self._clear_device_cache()
            if logger.isEnabledFor(logging.DEBUG):
                logger.exception("WhisperX model preparation failed")
            else:
                logger.error("WhisperX model preparation failed: %s: %s", exc.__class__.__name__, exc)
            raise build_whisperx_helpful_error(exc, model_store_root=plan.model_store_root, local_files_only=self.local_files_only) from exc

    def close(self) -> None:
        bundle = self._prepared
        self._prepared = None
        if bundle is None:
            logger.debug("WhisperXEngine close() called with no prepared bundle to release")
            return
        logger.info("Closing prepared WhisperX model=%s language=%s device=%s", self.model_name, bundle.language, self.device)
        try:
            bundle.align_model = None
            bundle.align_metadata = None
            bundle.model = None
        finally:
            self._clear_device_cache()
            logger.info("Closed prepared WhisperX model=%s language=%s device=%s", self.model_name, bundle.language, self.device)

    def prepare(self, language: str, stage=None) -> dict[str, object]:
        bundle = self._prepare_bundle(language, stage=stage)
        return dict(bundle.runtime_report)

    def transcribe(self, audio_artifact: AudioArtifact, language: str, stage=None) -> EngineOutput:
        if stage is not None:
            stage.phase("loading WhisperX backend")
        try:
            import whisperx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("whisperx is not installed. Install with: pip install whisperx") from exc
        from pyroller.transcriber.whisperx_runtime import build_whisperx_helpful_error, whisperx_runtime_context

        bundle = self._prepare_bundle(language, stage=stage)
        resolution = bundle.plan
        logger.info("Using prepared WhisperX model=%s device=%s store=%s", self.model_name, self.device, resolution.model_store_root)
        try:
            with whisperx_runtime_context(resolution):
                logger.info("Loading audio for transcription: %s", audio_artifact.path)
                if stage is not None:
                    stage.phase("loading audio")
                audio = whisperx.load_audio(str(audio_artifact.path))
                audio_duration = float(len(audio) / 16000.0) if len(audio) else 0.0
                if stage is not None:
                    stage.phase("running transcription inference")
                result = bundle.model.transcribe(audio, batch_size=self.batch_size, language=language)
                segments = result.get("segments", [])

                aligned = False
                if self.align_words and bundle.align_model is not None and bundle.align_metadata is not None:
                    try:
                        logger.info("Running WhisperX word alignment")
                        if stage is not None:
                            stage.phase("running word alignment")
                        result = whisperx.align(
                            result["segments"],
                            bundle.align_model,
                            bundle.align_metadata,
                            audio,
                            self.device,
                            return_char_alignments=False,
                        )
                        segments = result.get("segments", segments)
                        aligned = True
                    except Exception as exc:
                        logger.warning("WhisperX word alignment failed; falling back to segment timestamps: %s", exc)
                        aligned = False
        except Exception as exc:
            if logger.isEnabledFor(logging.DEBUG):
                logger.exception("WhisperX transcription runtime failed")
            else:
                logger.error("WhisperX transcription runtime failed: %s: %s", exc.__class__.__name__, exc)
            raise build_whisperx_helpful_error(exc, model_store_root=resolution.model_store_root, local_files_only=self.local_files_only) from exc

        spans = self._segments_to_spans(segments)
        raw_text = " ".join(str(seg.get("text", "")).strip() for seg in segments if str(seg.get("text", "")).strip()) or None
        metadata = {
            "engine_output_schema": ENGINE_OUTPUT_SCHEMA_VERSION,
            "engine": self.name,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "compute_type": self.compute_type,
            "batch_size": self.batch_size,
            "align_words": self.align_words,
            "alignment_used": aligned,
            "audio_duration": audio_duration,
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

    def _segments_to_spans(self, segments: list[dict[str, Any]]) -> list[EngineSpan]:
        spans: list[EngineSpan] = []
        for seg_index, segment in enumerate(segments):
            segment_span_id = f"segment:{seg_index}"
            spans.append(
                EngineSpan(
                    span_id=segment_span_id,
                    level="segment",
                    start_time=float(segment.get("start", 0.0)),
                    end_time=float(segment.get("end", 0.0)),
                    text=segment.get("text"),
                    confidence=float(segment.get("avg_logprob")) if isinstance(segment.get("avg_logprob"), (int, float)) else None,
                    segment_index=seg_index,
                    metadata={"word_count": len(segment.get("words") or [])},
                )
            )
            for word_index, word in enumerate(segment.get("words") or []):
                spans.append(
                    EngineSpan(
                        span_id=f"{segment_span_id}:word:{word_index}",
                        level="word",
                        start_time=float(word.get("start", segment.get("start", 0.0))),
                        end_time=float(word.get("end", segment.get("end", 0.0))),
                        text=word.get("word", ""),
                        confidence=float(word.get("score")) if isinstance(word.get("score"), (int, float)) else None,
                        segment_index=seg_index,
                        word_index=word_index,
                        parent_span_id=segment_span_id,
                        metadata={},
                    )
                )
        return spans
