from __future__ import annotations

import gc

from pyroller.i18n import _
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyroller.domain import AudioArtifact
from pyroller.transcriber.ctc_utils import ctc_token_segments
from pyroller.transcriber.engine_types import ENGINE_OUTPUT_SCHEMA_VERSION, EngineOutput, EngineSpan
from pyroller.transcriber.engines.base import TranscriberEngine
from pyroller.transcriber.hf_download_config import HFDownloadConfig, hf_download_environment

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
        model_path: str | Path | None = None,
        local_files_only: bool = False,
        device: str = "cpu",
        target_sample_rate: int = 16000,
        trust_remote_code: bool = False,
        hf_xet: str = "auto",
        hf_proxy: str | None = None,
        hf_etag_timeout: float | None = None,
        hf_download_timeout: float | None = None,
        hf_max_workers: int | None = None,
    ) -> None:
        self.backend_name = backend_name
        self.model_name = model_name
        self.model_path = str(model_path) if model_path is not None else None
        self.local_files_only = local_files_only
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.trust_remote_code = trust_remote_code
        self.hf_download_config = HFDownloadConfig(
            xet=hf_xet,
            proxy=hf_proxy,
            etag_timeout=hf_etag_timeout,
            download_timeout=hf_download_timeout,
            max_workers=hf_max_workers,
        )
        self._prepared: _PreparedWav2Vec2Bundle | None = None

    def _prepare_phase_count(self, language: str) -> int:
        if self._is_prepared_for(language):
            return 1
        return 2

    def preflight_phase_total(self, language: str) -> int:
        return self._prepare_phase_count(language)

    def transcribe_phase_total(self, language: str) -> int:
        return 1 + self._prepare_phase_count(language) + 2

    def _build_resolution_plan(self, language: str, *, materialize: bool, stage=None):
        from pyroller.transcriber.model_resolver import TranscriberModelResolver

        resolver = TranscriberModelResolver(
            backend=self.backend_name,
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

    def _prepare_bundle(self, language: str, stage=None) -> _PreparedWav2Vec2Bundle:
        if self._is_prepared_for(language):
            if stage is not None:
                stage.phase(_("reusing prepared wav2vec2 model"))
            return self._prepared  # type: ignore[return-value]

        with hf_download_environment(self.hf_download_config, local_files_only=self.local_files_only):
            try:
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    _("{} dependencies are not installed. Install with: pip install librosa transformers torch").format(self.backend_name)
                ) from exc

            self.close()
            if stage is not None:
                stage.phase(_("resolving transcriber model"))
            plan = self._build_resolution_plan(language, materialize=True, stage=stage)
            if plan.resolved_model_dir is None:
                raise RuntimeError(_("Unable to resolve local {} model directory.").format(self.backend_name))
            processor = None
            model = None
            try:
                logger.info(_("Preparing %s model=%s from=%s device=%s"), self.backend_name, self.model_name, plan.resolved_model_dir, self.device)
                if stage is not None:
                    stage.phase(_("loading wav2vec2 model"))
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
            logger.debug(_("%s close() called with no prepared bundle to release"), self.__class__.__name__)
            return
        logger.info(_("Closing prepared %s model=%s language=%s device=%s"), self.backend_name, self.model_name, bundle.language, self.device)
        try:
            bundle.model = None
            bundle.processor = None
        finally:
            self._clear_device_cache()
            logger.info(_("Closed prepared %s model=%s language=%s device=%s"), self.backend_name, self.model_name, bundle.language, self.device)

    def prepare(self, language: str, stage=None) -> dict[str, object]:
        bundle = self._prepare_bundle(language, stage=stage)
        return dict(bundle.runtime_report)

    def transcribe(self, audio_artifact: AudioArtifact, language: str, stage=None) -> EngineOutput:
        if stage is not None:
            stage.phase(_("loading wav2vec2 backend"))
        try:
            import librosa  # type: ignore
            import torch  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                _("{} dependencies are not installed. Install with: pip install librosa transformers torch").format(self.backend_name)
            ) from exc

        audio_path = Path(audio_artifact.path) if audio_artifact.path is not None else None
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(_("Audio file not found for {} backend: {}").format(self.backend_name, audio_artifact.path))

        bundle = self._prepare_bundle(language, stage=stage)
        logger.info(_("Using prepared %s model=%s from=%s device=%s"), self.backend_name, self.model_name, bundle.plan.resolved_model_dir, self.device)

        if stage is not None:
            stage.phase(_("loading audio"))
        speech, sampling_rate = librosa.load(str(audio_path), sr=self.target_sample_rate, mono=True)
        audio_duration = float(len(speech) / sampling_rate) if len(speech) else 0.0

        if stage is not None:
            stage.phase(_("running phonetic inference"))
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
            raise RuntimeError(_("Unable to determine CTC blank token id for {} backend").format(self.backend_name))

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
