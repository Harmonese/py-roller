from __future__ import annotations

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyroller.domain import AudioArtifact
from pyroller.transcriber.ctc_utils import ctc_token_segments
from pyroller.transcriber.engine_types import ENGINE_OUTPUT_SCHEMA_VERSION, EngineOutput, EngineSpan
from pyroller.transcriber.engines.base import TranscriberEngine

logger = logging.getLogger("pyroller.transcriber")


@dataclass
class _PreparedWav2Vec2Bundle:
    language: str
    plan: Any
    processor: Any
    model: Any
    runtime_report: dict[str, object]


class Wav2Vec2CTCEngine(TranscriberEngine):
    name = "wav2vec2_ctc"

    def __init__(
        self,
        *,
        backend_name: str,
        model_name: str,
        model_path: str | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        target_sample_rate: int = 16000,
        trust_remote_code: bool = False,
    ) -> None:
        self.backend_name = backend_name
        self.model_name = model_name
        self.model_path = model_path
        self.local_files_only = local_files_only
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.trust_remote_code = trust_remote_code
        self._prepared: _PreparedWav2Vec2Bundle | None = None

    @property
    def transcribe_phase_total(self) -> int:
        return 6

    def _build_resolution_plan(self, language: str, *, materialize: bool, stage=None):
        from pyroller.transcriber.model_resolver import TranscriberModelResolver

        resolver = TranscriberModelResolver(
            backend=self.backend_name,
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

    def _prepare_bundle(self, language: str, stage=None) -> _PreparedWav2Vec2Bundle:
        if self._is_prepared_for(language):
            if stage is not None:
                stage.phase("reusing prepared wav2vec2 model")
            return self._prepared  # type: ignore[return-value]

        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"{self.backend_name} dependencies are not installed. Install with: pip install librosa transformers torch"
            ) from exc

        self.close()
        if stage is not None:
            stage.phase("resolving transcriber model")
        plan = self._build_resolution_plan(language, materialize=True, stage=stage)
        if plan.resolved_model_dir is None:
            raise RuntimeError(f"Unable to resolve local {self.backend_name} model directory.")
        processor = None
        model = None
        try:
            logger.info("Preparing %s model=%s from=%s device=%s", self.backend_name, self.model_name, plan.resolved_model_dir, self.device)
            if stage is not None:
                stage.phase("loading wav2vec2 model")
            processor = Wav2Vec2Processor.from_pretrained(
                str(plan.resolved_model_dir),
                local_files_only=True,
                trust_remote_code=self.trust_remote_code,
            )
            model = Wav2Vec2ForCTC.from_pretrained(
                str(plan.resolved_model_dir),
                local_files_only=True,
                trust_remote_code=self.trust_remote_code,
            ).to(self.device)
            model.eval()
            bundle = _PreparedWav2Vec2Bundle(
                language=language,
                plan=plan,
                processor=processor,
                model=model,
                runtime_report=plan.runtime_record(),
            )
            self._prepared = bundle
            return bundle
        except Exception:
            if model is not None:
                del model
            if processor is not None:
                del processor
            self._clear_device_cache()
            raise

    def close(self) -> None:
        bundle = self._prepared
        self._prepared = None
        if bundle is None:
            logger.debug("%s close() called with no prepared bundle to release", self.__class__.__name__)
            return
        logger.info("Closing prepared %s model=%s language=%s device=%s", self.backend_name, self.model_name, bundle.language, self.device)
        try:
            bundle.model = None
            bundle.processor = None
        finally:
            self._clear_device_cache()
            logger.info("Closed prepared %s model=%s language=%s device=%s", self.backend_name, self.model_name, bundle.language, self.device)

    def prepare(self, language: str, stage=None) -> dict[str, object]:
        bundle = self._prepare_bundle(language, stage=stage)
        return dict(bundle.runtime_report)

    def transcribe(self, audio_artifact: AudioArtifact, language: str, stage=None) -> EngineOutput:
        if stage is not None:
            stage.phase("loading wav2vec2 backend")
        try:
            import librosa  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                f"{self.backend_name} dependencies are not installed. Install with: pip install librosa transformers torch"
            ) from exc

        audio_path = Path(audio_artifact.path) if audio_artifact.path is not None else None
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for {self.backend_name} backend: {audio_artifact.path}")

        bundle = self._prepare_bundle(language, stage=stage)
        logger.info("Using prepared %s model=%s from=%s device=%s", self.backend_name, self.model_name, bundle.plan.resolved_model_dir, self.device)

        if stage is not None:
            stage.phase("loading audio")
        speech, sampling_rate = librosa.load(str(audio_path), sr=self.target_sample_rate, mono=True)
        audio_duration = float(len(speech) / sampling_rate) if len(speech) else 0.0

        if stage is not None:
            stage.phase("running phonetic inference")
        inputs = bundle.processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        with torch.no_grad():
            if attention_mask is not None:
                logits = bundle.model(input_values=input_values, attention_mask=attention_mask).logits
            else:
                logits = bundle.model(input_values=input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)[0]
        raw_transcription = bundle.processor.batch_decode(predicted_ids.unsqueeze(0), skip_special_tokens=True)[0]
        time_offset = bundle.model.config.inputs_to_logits_ratio / sampling_rate
        blank_id = bundle.processor.tokenizer.pad_token_id
        if blank_id is None:
            blank_id = bundle.model.config.pad_token_id
        if blank_id is None:
            raise RuntimeError(f"Unable to determine CTC blank token id for {self.backend_name} backend")

        token_segments = ctc_token_segments(predicted_ids, bundle.processor.tokenizer, blank_id=blank_id, time_offset=time_offset)
        spans = self._token_segments_to_spans(token_segments)
        metadata = {
            "engine_output_schema": ENGINE_OUTPUT_SCHEMA_VERSION,
            "engine": self.name,
            "source_backend_key": self.backend_name,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "target_sample_rate": self.target_sample_rate,
            "trust_remote_code": self.trust_remote_code,
            "audio_duration": audio_duration,
            "source_audio": str(audio_artifact.path) if audio_artifact.path else None,
            "source_role": audio_artifact.role,
            "runtime": dict(bundle.runtime_report),
        }
        return EngineOutput(language=language, engine=self.name, raw_text=raw_transcription, spans=spans, metadata=metadata)

    def _token_segments_to_spans(self, token_segments: list[dict[str, Any]]) -> list[EngineSpan]:
        spans: list[EngineSpan] = []
        for index, segment in enumerate(token_segments):
            token = str(segment.get("token", ""))
            spans.append(
                EngineSpan(
                    span_id=f"token:{index}",
                    level="token",
                    start_time=float(segment["start_time"]),
                    end_time=float(segment["end_time"]),
                    text=token,
                    token=token,
                    token_index=index,
                    segment_index=index,
                    metadata={},
                )
            )
        return spans
