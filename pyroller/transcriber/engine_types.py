from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ENGINE_OUTPUT_SCHEMA_VERSION = "pyroller.transcriber.engine_output.v1"


@dataclass(slots=True)
class EngineSpan:
    span_id: str
    level: str
    start_time: float
    end_time: float
    text: str | None = None
    normalized_text: str | None = None
    token: str | None = None
    confidence: float | None = None
    segment_index: int | None = None
    word_index: int | None = None
    token_index: int | None = None
    parent_span_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EngineOutput:
    language: str
    engine: str
    raw_text: str | None
    spans: list[EngineSpan]
    metadata: dict[str, Any] = field(default_factory=dict)
