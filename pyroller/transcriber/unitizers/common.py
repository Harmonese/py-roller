from __future__ import annotations

from typing import Iterable

from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
from pyroller.transcriber.protocol import RAW_SEGMENTS_SCHEMA_VERSION, UNIT_TRACE_SCHEMA_VERSION, build_raw_segment


def engine_spans_by_level(engine_output: EngineOutput, level: str) -> list[EngineSpan]:
    return [span for span in engine_output.spans if span.level == level]


def preferred_text_spans(engine_output: EngineOutput) -> list[EngineSpan]:
    words = engine_spans_by_level(engine_output, "word")
    if words:
        return words
    return engine_spans_by_level(engine_output, "segment")


def preferred_raw_segment_spans(engine_output: EngineOutput) -> list[EngineSpan]:
    segments = engine_spans_by_level(engine_output, "segment")
    if segments:
        return segments
    return list(engine_output.spans)


def raw_segments_from_spans(spans: Iterable[EngineSpan]) -> list[dict]:
    raw_segments: list[dict] = []
    for index, span in enumerate(spans):
        raw_segments.append(
            build_raw_segment(
                segment_index=span.segment_index if span.segment_index is not None else index,
                segment_level=span.level,
                start=float(span.start_time),
                end=float(span.end_time),
                text=span.text,
                normalized_text=span.normalized_text,
                token=span.token,
                confidence=span.confidence,
                metadata=dict(span.metadata or {}),
            )
        )
    return raw_segments


def base_result_metadata(engine_output: EngineOutput, *, unitizer_name: str, raw_segment_level: str) -> dict:
    metadata = dict(engine_output.metadata or {})
    metadata.setdefault("engine", engine_output.engine)
    metadata["unitizer"] = unitizer_name
    metadata["engine_output_schema"] = metadata.get("engine_output_schema", "pyroller.transcriber.engine_output.v1")
    metadata["raw_segments_schema"] = RAW_SEGMENTS_SCHEMA_VERSION
    metadata["unit_trace_schema"] = UNIT_TRACE_SCHEMA_VERSION
    metadata["raw_segment_level"] = raw_segment_level
    return metadata
