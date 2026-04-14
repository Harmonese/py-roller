from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pyroller.transcriber.composed import ComposedTranscriber
from pyroller.transcriber.engines import FasterWhisperEngine, WhisperXEngine, Wav2Vec2CTCEngine
from pyroller.transcriber.engines.base import TranscriberEngine
from pyroller.transcriber.unitizers import (
    EnArpabetUnitizer,
    MulIpaUnitizer,
    TranscriptionAdapter,
    ZhPinyinFromCTCUnitizer,
    ZhPinyinFromTextUnitizer,
)

EngineFactory = Callable[[dict[str, Any]], TranscriberEngine]
AdapterFactory = Callable[[], TranscriptionAdapter]


@dataclass(frozen=True)
class TranscriberSpec:
    language: str
    backend: str
    requirements: tuple[str, ...]
    config_keys: frozenset[str]
    engine_factory: EngineFactory
    adapter_factory: AdapterFactory

    def compose(self, config: dict[str, Any]) -> ComposedTranscriber:
        return ComposedTranscriber(
            engine=self.engine_factory(config),
            adapter=self.adapter_factory(),
            backend_name=self.backend,
        )


def _build_faster_whisper_engine(config: dict[str, Any]) -> FasterWhisperEngine:
    return FasterWhisperEngine(**config)


def _build_whisperx_engine(config: dict[str, Any]) -> WhisperXEngine:
    return WhisperXEngine(**config)


def _build_mms_ctc_engine(config: dict[str, Any]) -> Wav2Vec2CTCEngine:
    resolved = {"model_name": "Chuatury/wav2vec2-mms-1b-cmn-phonetic", **config}
    return Wav2Vec2CTCEngine(backend_name="mms_phonetic", **resolved)


def _build_mul_ctc_engine(config: dict[str, Any]) -> Wav2Vec2CTCEngine:
    resolved = {"model_name": "facebook/wav2vec2-lv-60-espeak-cv-ft", **config}
    return Wav2Vec2CTCEngine(backend_name="wav2vec2_phoneme", **resolved)


TRANSCRIBER_SPECS: dict[tuple[str, str], TranscriberSpec] = {
    ("zh", "whisperx"): TranscriberSpec(
        language="zh",
        backend="whisperx",
        requirements=("whisperx",),
        config_keys=frozenset({"model_name", "model_path", "local_files_only", "device", "compute_type", "batch_size", "align_words"}),
        engine_factory=_build_whisperx_engine,
        adapter_factory=lambda: ZhPinyinFromTextUnitizer(backend="whisperx"),
    ),
    ("en", "whisperx"): TranscriberSpec(
        language="en",
        backend="whisperx",
        requirements=("whisperx",),
        config_keys=frozenset({"model_name", "model_path", "local_files_only", "device", "compute_type", "batch_size", "align_words"}),
        engine_factory=_build_whisperx_engine,
        adapter_factory=lambda: EnArpabetUnitizer(backend="whisperx"),
    ),
    ("en", "faster_whisper"): TranscriberSpec(
        language="en",
        backend="faster_whisper",
        requirements=("faster_whisper",),
        config_keys=frozenset({"model_name", "model_path", "local_files_only", "device", "compute_type", "batch_size"}),
        engine_factory=_build_faster_whisper_engine,
        adapter_factory=lambda: EnArpabetUnitizer(backend="faster_whisper"),
    ),
    ("zh", "mms_phonetic"): TranscriberSpec(
        language="zh",
        backend="mms_phonetic",
        requirements=("librosa", "torch", "transformers", "huggingface_hub"),
        config_keys=frozenset({"model_name", "model_path", "local_files_only", "device", "target_sample_rate", "trust_remote_code"}),
        engine_factory=_build_mms_ctc_engine,
        adapter_factory=lambda: ZhPinyinFromCTCUnitizer(backend="mms_phonetic"),
    ),
    ("mul", "wav2vec2_phoneme"): TranscriberSpec(
        language="mul",
        backend="wav2vec2_phoneme",
        requirements=("librosa", "torch", "transformers", "huggingface_hub"),
        config_keys=frozenset({"model_name", "model_path", "local_files_only", "device", "target_sample_rate", "trust_remote_code"}),
        engine_factory=_build_mul_ctc_engine,
        adapter_factory=lambda: MulIpaUnitizer(backend="wav2vec2_phoneme"),
    ),
}
