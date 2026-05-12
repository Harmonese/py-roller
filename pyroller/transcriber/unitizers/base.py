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
        metadata.update(self._extra_result_metadata(engine_output))
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
