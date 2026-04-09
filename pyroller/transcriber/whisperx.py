from __future__ import annotations

import gc
import logging
from typing import Any

from pyroller.domain import AudioArtifact, TimedUnit, TranscriptionResult
from pyroller.progress import ProgressReporter
from pyroller.transcriber.base import Transcriber
from pyroller.utils.ids import make_id
from pyroller.utils.text import chinese_text_to_pinyin_syllables, normalize_chinese_text

logger = logging.getLogger("pyroller.transcriber")


class WhisperXTranscriber(Transcriber):
    def __init__(
        self,
        model_name: str = "large-v2",
        device: str = "cpu",
        compute_type: str = "int8",
        batch_size: int = 8,
        align_words: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.align_words = align_words

    def transcribe(self, audio_artifact: AudioArtifact, language: str, tone_mode: str, progress: ProgressReporter | None = None) -> TranscriptionResult:
        stage = progress.stage("transcriber", total=5 if self.align_words else 4, unit="phase") if progress is not None else None
        if stage is not None:
            stage.phase("loading WhisperX backend")
        try:
            import whisperx  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("whisperx is not installed. Install with: pip install whisperx") from exc

        logger.info("Loading WhisperX model=%s device=%s", self.model_name, self.device)
        if stage is not None:
            stage.phase("loading WhisperX model")
        model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            language=language,
        )
        logger.info("Loading audio for transcription: %s", audio_artifact.path)
        if stage is not None:
            stage.phase("loading audio")
        audio = whisperx.load_audio(str(audio_artifact.path))
        audio_duration = float(len(audio) / 16000.0) if len(audio) else 0.0
        if stage is not None:
            stage.phase("running transcription inference")
        result = model.transcribe(audio, batch_size=self.batch_size, language=language)
        segments = result.get("segments", [])

        aligned = False
        if self.align_words:
            try:
                logger.info("Running WhisperX word alignment")
                if stage is not None:
                    stage.phase("running word alignment")
                align_model, align_metadata = whisperx.load_align_model(language_code=language, device=self.device)
                result = whisperx.align(
                    result["segments"],
                    align_model,
                    align_metadata,
                    audio,
                    self.device,
                    return_char_alignments=False,
                )
                segments = result.get("segments", segments)
                aligned = True
                del align_model
                del align_metadata
            except Exception as exc:
                logger.warning("WhisperX word alignment failed; falling back to segment timestamps: %s", exc)
                aligned = False

        if stage is not None and not self.align_words:
            stage.phase("post-processing transcription")
        units = self._segments_to_pinyin_units(segments, language=language, tone_mode=tone_mode)
        raw_text = " ".join(seg.get("text", "") for seg in segments).strip() or None

        del model
        gc.collect()

        logger.info("Transcription complete: %d segments, %d normalized timed units", len(segments), len(units))
        if stage is not None:
            stage.close("transcriber complete")
        return TranscriptionResult(
            language=language,
            backend="whisperx",
            units=units,
            raw_text=raw_text,
            raw_segments=segments,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "batch_size": self.batch_size,
                "align_words": self.align_words,
                "aligned": aligned,
                "audio_duration": audio_duration,
                "source_audio": str(audio_artifact.path) if audio_artifact.path else None,
                "source_role": audio_artifact.role,
            },
        )

    def _segments_to_pinyin_units(
        self,
        segments: list[dict[str, Any]],
        language: str,
        tone_mode: str,
    ) -> list[TimedUnit]:
        units: list[TimedUnit] = []
        for seg_index, segment in enumerate(segments):
            words = segment.get("words") or []
            if words:
                for word_index, word in enumerate(words):
                    units.extend(
                        self._text_chunk_to_units(
                            text=word.get("word", ""),
                            start=float(word.get("start", segment.get("start", 0.0))),
                            end=float(word.get("end", segment.get("end", 0.0))),
                            confidence=word.get("score"),
                            language=language,
                            tone_mode=tone_mode,
                            chunk_prefix=f"seg{seg_index}_word{word_index}",
                            source_segment_index=seg_index,
                            source_word_index=word_index,
                        )
                    )
            else:
                units.extend(
                    self._text_chunk_to_units(
                        text=segment.get("text", ""),
                        start=float(segment.get("start", 0.0)),
                        end=float(segment.get("end", 0.0)),
                        confidence=None,
                        language=language,
                        tone_mode=tone_mode,
                        chunk_prefix=f"seg{seg_index}",
                        source_segment_index=seg_index,
                        source_word_index=None,
                    )
                )
        return units

    def _text_chunk_to_units(
        self,
        text: str,
        start: float,
        end: float,
        confidence: float | None,
        language: str,
        tone_mode: str,
        chunk_prefix: str,
        source_segment_index: int,
        source_word_index: int | None,
    ) -> list[TimedUnit]:
        normalized = normalize_chinese_text(text)
        if not normalized:
            return []

        pinyin_units = chinese_text_to_pinyin_syllables(normalized, tone_mode=tone_mode)
        if not pinyin_units:
            return []

        count = len(pinyin_units)
        duration = max(end - start, 0.0)
        step = duration / count if count > 0 else 0.0

        units: list[TimedUnit] = []
        for idx, syllable in enumerate(pinyin_units):
            unit_start = start + (idx * step)
            unit_end = end if idx == count - 1 else start + ((idx + 1) * step)
            units.append(
                TimedUnit(
                    unit_id=make_id("timed_unit"),
                    symbol=syllable["symbol"] or syllable["normalized_symbol"],
                    normalized_symbol=syllable["normalized_symbol"] or syllable["symbol"],
                    unit_type="pinyin_syllable",
                    language=language,
                    tone=syllable["tone"],
                    start_time=unit_start,
                    end_time=unit_end,
                    confidence=float(confidence) if confidence is not None else None,
                    source_backend="whisperx",
                    raw_tokens=[normalized[idx]] if idx < len(normalized) else [text],
                    metadata={
                        "source_text": text,
                        "normalized_text": normalized,
                        "chunk_prefix": chunk_prefix,
                        "source_segment_index": source_segment_index,
                        "source_word_index": source_word_index,
                    },
                )
            )
        return units
