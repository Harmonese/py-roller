from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from pyroller.domain import TimedUnit, TranscriptionResult
from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
from pyroller.transcriber.unitizers.common import (
    base_result_metadata,
    preferred_raw_segment_spans,
    raw_segments_from_spans,
)


class TranscriptionAdapter(ABC):
    name = "adapter"
    backend = ""
    unit_timing_semantics = "unknown"

    def adapt(self, engine_output: EngineOutput, *, language: str, tone_mode: str) -> TranscriptionResult:
        units = self._unitize(engine_output, language=language, tone_mode=tone_mode)
        raw_segment_spans = list(self._raw_segment_spans(engine_output))
        metadata = base_result_metadata(
            engine_output,
            unitizer_name=self.name,
            raw_segment_level=raw_segment_spans[0].level if raw_segment_spans else "segment",
        )
        metadata["unit_timing_semantics"] = self.unit_timing_semantics
        if self.name in {"zh_pinyin_from_text", "mul_ipa_from_text"}:
            from pyroller.language_diagnostics import route_warnings
            from pyroller.utils.text import summarize_multilingual_routes, summarize_zh_router_routes
            text = engine_output.raw_text or ""
            summary = (summarize_zh_router_routes(text) if language == "zh" else
                       summarize_multilingual_routes(text, getattr(self, "latin_language", None)))
            metadata["language_warnings"] = route_warnings(text, summary)
        if self.name == "en_arpabet":
            from pyroller.language_diagnostics import english_warnings
            from pyroller.utils.text import english_text_to_arpabet_units
            text = engine_output.raw_text or ""
            metadata["language_warnings"] = english_warnings(text, english_text_to_arpabet_units(text))
        metadata.update(self._extra_result_metadata(engine_output))
        from pyroller.transcriber.unitizers.common import preferred_text_spans
        span_warnings = [span.metadata['text_span_warning'] for span in preferred_text_spans(engine_output)
                         if 'text_span_warning' in span.metadata]
        metadata['text_span_warnings'] = span_warnings
        metadata.setdefault('language_warnings', []).extend(span_warnings)
        return TranscriptionResult(
            language=language,
            backend=self.backend,
            units=units,
            raw_text=engine_output.raw_text,
            raw_segments=raw_segments_from_spans(raw_segment_spans),
            metadata=metadata,
        )

    def _raw_segment_spans(self, engine_output: EngineOutput) -> Sequence[EngineSpan]:
        return preferred_raw_segment_spans(engine_output)

    def _extra_result_metadata(self, engine_output: EngineOutput) -> dict[str, object]:
        return {}

    @abstractmethod
    def _unitize(self, engine_output: EngineOutput, *, language: str, tone_mode: str) -> list[TimedUnit]:
        raise NotImplementedError
