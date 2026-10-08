from __future__ import annotations

from typing import Iterable
from dataclasses import replace
import unicodedata

from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
from pyroller.transcriber.protocol import RAW_SEGMENTS_SCHEMA_VERSION, UNIT_TRACE_SCHEMA_VERSION, build_raw_segment


def engine_spans_by_level(engine_output: EngineOutput, level: str) -> list[EngineSpan]:
    return [span for span in engine_output.spans if span.level == level]


def _text_key(text: str | None) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKC', text or '').casefold() if c.isalnum())


def preferred_text_spans(engine_output: EngineOutput) -> list[EngineSpan]:
    words = engine_spans_by_level(engine_output, "word")
    segments = engine_spans_by_level(engine_output, "segment")
    selected = []
    used = set()
    for segment in segments:
        children = [word for word in words if id(word) not in used and (word.text or "").strip() and
                    (word.parent_span_id == segment.span_id or
                     (word.parent_span_id is None and segment.segment_index is not None and word.segment_index == segment.segment_index) or
                     (word.parent_span_id is None and word.segment_index is None and
                      segment.start_time <= word.start_time <= word.end_time <= segment.end_time))]
        children.sort(key=lambda word: (word.start_time, word.end_time))
        complete = ((not _text_key(segment.text) or _text_key(segment.text) == ''.join(_text_key(word.text) for word in children))
                    and all(segment.start_time <= word.start_time <= word.end_time <= segment.end_time for word in children))
        if children and not complete:
            selected.append(replace(segment, metadata={**segment.metadata, 'text_span_warning': {
                'code': 'incomplete_word_coverage', 'text': segment.text,
                'span_id': segment.span_id, 'fallback': 'segment_timing',
            }}))
        else:
            selected.extend(children or [segment])
        used.update(id(word) for word in children)
    selected.extend(word for word in words if id(word) not in used)
    return sorted(selected, key=lambda span: (span.start_time, span.end_time))


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


def span_confidence(span: EngineSpan) -> float | None:
    import math
    if span.confidence is None:
        return None
    value = float(span.confidence)
    if not math.isfinite(value):
        raise ValueError("Non-finite transcription confidence")
    # faster-whisper segments provide average log probability; words provide
    # probability. Keep this distinction when falling back to segment timing.
    if span.level == "segment":
        return min(1.0, math.exp(min(0.0, value)))
    if not 0 <= value <= 1:
        raise ValueError("Word confidence must be between zero and one")
    return value
