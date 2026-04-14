from __future__ import annotations

from pyroller.domain import TimedUnit
from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
from pyroller.transcriber.protocol import build_unit_trace_metadata
from pyroller.transcriber.unitizers.base import TranscriptionAdapter
from pyroller.transcriber.unitizers.common import preferred_text_spans
from pyroller.utils.ids import make_id
from pyroller.utils.text import english_text_to_arpabet_units, normalize_english_text


class EnArpabetUnitizer(TranscriptionAdapter):
    name = "en_arpabet"
    unit_timing_semantics = "interpolated_non_acoustic"

    def __init__(self, *, backend: str = "whisperx") -> None:
        self.backend = backend

    def _unitize(self, engine_output: EngineOutput, *, language: str, tone_mode: str) -> list[TimedUnit]:
        units: list[TimedUnit] = []
        for span in preferred_text_spans(engine_output):
            units.extend(self._text_span_to_units(span, language=language))
        return units

    def _text_span_to_units(self, span: EngineSpan, *, language: str) -> list[TimedUnit]:
        text = span.text or ""
        normalized = normalize_english_text(text)
        phones = english_text_to_arpabet_units(text)
        if not phones:
            return []

        start = float(span.start_time)
        end = float(span.end_time)
        count = len(phones)
        duration = max(end - start, 0.0)
        step = duration / count if count > 0 else 0.0
        units: list[TimedUnit] = []
        for idx, phone in enumerate(phones):
            unit_start = start + (idx * step)
            unit_end = end if idx == count - 1 else start + ((idx + 1) * step)
            units.append(
                TimedUnit(
                    unit_id=make_id("timed_unit"),
                    symbol=phone["symbol"] or phone["normalized_symbol"],
                    normalized_symbol=phone["normalized_symbol"] or phone["symbol"],
                    unit_type="arpabet_phone",
                    language=language,
                    tone=phone.get("stress"),
                    start_time=unit_start,
                    end_time=unit_end,
                    confidence=float(span.confidence) if span.confidence is not None else None,
                    source_backend=self.backend,
                    raw_tokens=[phone["symbol"]],
                    metadata=build_unit_trace_metadata(
                        backend=self.backend,
                        source_segment_index=span.segment_index if span.segment_index is not None else 0,
                        source_segment_level=span.level,
                        source_word_index=span.word_index,
                        source_start_time=start,
                        source_end_time=end,
                        source_text=text,
                        normalized_text=normalized,
                        timing_mode="interpolated_from_word" if span.level == "word" else "interpolated_from_segment",
                        extra={
                            "engine": self.backend,
                            "engine_span_id": span.span_id,
                            "chunk_prefix": span.span_id.replace(":", "_"),
                            "source_word": phone.get("source_word"),
                            "timing_is_interpolated": True,
                            "timing_is_acoustic": False,
                            "timing_basis": f"{self.backend}_word_span" if span.level == "word" else f"{self.backend}_segment_span",
                        },
                    ),
                )
            )
        return units
