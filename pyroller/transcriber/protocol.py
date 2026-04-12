from __future__ import annotations

from typing import Any

RAW_SEGMENTS_SCHEMA_VERSION = "pyroller.transcriber.raw_segments.v1"
UNIT_TRACE_SCHEMA_VERSION = "pyroller.transcriber.unit_trace.v1"


def _clean_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def build_raw_segment(
    *,
    segment_index: int,
    segment_level: str,
    start: float,
    end: float,
    text: str | None = None,
    normalized_text: str | None = None,
    token: str | None = None,
    confidence: float | None = None,
    parent_segment_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": RAW_SEGMENTS_SCHEMA_VERSION,
        "segment_index": int(segment_index),
        "segment_level": segment_level,
        "start": float(start),
        "end": float(end),
        "text": text,
        "normalized_text": normalized_text,
        "token": token,
        "confidence": float(confidence) if confidence is not None else None,
        "parent_segment_index": int(parent_segment_index) if parent_segment_index is not None else None,
        "metadata": dict(metadata or {}),
    }
    return _clean_none_values(payload)


def build_unit_trace_metadata(
    *,
    backend: str,
    source_segment_index: int,
    source_segment_level: str,
    source_start_time: float,
    source_end_time: float,
    source_word_index: int | None = None,
    source_token_index: int | None = None,
    source_text: str | None = None,
    normalized_text: str | None = None,
    source_token: str | None = None,
    timing_mode: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "trace_version": UNIT_TRACE_SCHEMA_VERSION,
        "backend": backend,
        "source_segment_index": int(source_segment_index),
        "source_segment_level": source_segment_level,
        "source_word_index": int(source_word_index) if source_word_index is not None else None,
        "source_token_index": int(source_token_index) if source_token_index is not None else None,
        "source_start_time": float(source_start_time),
        "source_end_time": float(source_end_time),
        "source_text": source_text,
        "normalized_text": normalized_text,
        "source_token": source_token,
        "timing_mode": timing_mode,
    }
    payload.update(dict(extra or {}))
    return _clean_none_values(payload)
