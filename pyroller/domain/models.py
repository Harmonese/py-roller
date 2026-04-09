from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from pyroller.utils.json import read_json, write_artifact_json


@dataclass(slots=True)
class Artifact:
    artifact_id: str
    stage: str
    kind: str
    path: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.path is not None:
            data["path"] = str(self.path)
        return data


@dataclass(slots=True)
class AudioArtifact(Artifact):
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration: Optional[float] = None
    role: str = "audio"


@dataclass(slots=True)
class AlignmentUnit:
    unit_id: str
    symbol: str
    normalized_symbol: str
    unit_type: str
    language: str
    tone: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentUnit":
        return cls(**data)


@dataclass(slots=True)
class LyricUnit(AlignmentUnit):
    line_index: int = 0
    unit_index_in_line: int = 0
    source_text_span: Optional[tuple[int, int]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LyricUnit":
        return cls(**data)


@dataclass(slots=True)
class TimedUnit(AlignmentUnit):
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: Optional[float] = None
    source_backend: str = ""
    raw_tokens: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TimedUnit":
        return cls(**data)


@dataclass(slots=True)
class AlignedUnit:
    unit_id: str
    unit_index_in_line: int
    text: str
    normalized_symbol: str
    unit_type: str
    language: str
    start_time: float
    end_time: float
    confidence: float = 0.0
    source_audio_unit_index: Optional[int] = None
    source_audio_unit_range: Optional[tuple[int, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignedUnit":
        return cls(**data)


@dataclass(slots=True)
class LyricLine:
    line_index: int
    raw_text: str
    normalized_text: Optional[str] = None
    units: list[LyricUnit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_index": self.line_index,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "units": [u.to_dict() for u in self.units],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LyricLine":
        return cls(
            line_index=data["line_index"],
            raw_text=data["raw_text"],
            normalized_text=data.get("normalized_text"),
            units=[LyricUnit.from_dict(item) for item in data.get("units", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class LyricsDocument:
    source_path: Path
    raw_text: str
    encoding: str
    lines: list[LyricLine]
    language: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "raw_text": self.raw_text,
            "encoding": self.encoding,
            "lines": [line.to_dict() for line in self.lines],
            "language": self.language,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class TranscriptionResult:
    language: str
    backend: str
    units: list[TimedUnit]
    raw_text: Optional[str] = None
    raw_segments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    artifact_type = "timed_units"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "backend": self.backend,
            "units": [u.to_dict() for u in self.units],
            "raw_text": self.raw_text,
            "raw_segments": self.raw_segments,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionResult":
        return cls(
            language=data["language"],
            backend=data["backend"],
            units=[TimedUnit.from_dict(item) for item in data.get("units", [])],
            raw_text=data.get("raw_text"),
            raw_segments=data.get("raw_segments", []),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        write_artifact_json(self.artifact_type, self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "TranscriptionResult":
        data = read_json(path)
        artifact_type = data.get("artifact_type")
        if artifact_type != cls.artifact_type:
            raise ValueError(f"Expected artifact_type={cls.artifact_type!r}, got {artifact_type!r} from {path}")
        return cls.from_dict(data["payload"])


@dataclass(slots=True)
class ParsedLyrics:
    language: str
    backend: str
    lines: list[LyricLine]
    unit_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    artifact_type = "parsed_lyrics"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "backend": self.backend,
            "unit_type": self.unit_type,
            "lines": [line.to_dict() for line in self.lines],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParsedLyrics":
        return cls(
            language=data["language"],
            backend=data["backend"],
            lines=[LyricLine.from_dict(item) for item in data.get("lines", [])],
            unit_type=data["unit_type"],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        write_artifact_json(self.artifact_type, self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "ParsedLyrics":
        data = read_json(path)
        artifact_type = data.get("artifact_type")
        if artifact_type != cls.artifact_type:
            raise ValueError(f"Expected artifact_type={cls.artifact_type!r}, got {artifact_type!r} from {path}")
        return cls.from_dict(data["payload"])


@dataclass(slots=True)
class AlignmentLine:
    line_index: int
    raw_text: str
    assigned_time: float
    start_time: float
    end_time: Optional[float] = None
    confidence: float = 0.0
    method: str = "unknown"
    matched_audio_unit_range: Optional[tuple[int, int]] = None
    lyric_unit_range: Optional[tuple[int, int]] = None
    aligned_units: list[AlignedUnit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["aligned_units"] = [unit.to_dict() for unit in self.aligned_units]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentLine":
        return cls(
            line_index=data["line_index"],
            raw_text=data["raw_text"],
            assigned_time=data["assigned_time"],
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            confidence=data.get("confidence", 0.0),
            method=data.get("method", "unknown"),
            matched_audio_unit_range=data.get("matched_audio_unit_range"),
            lyric_unit_range=data.get("lyric_unit_range"),
            aligned_units=[AlignedUnit.from_dict(item) for item in data.get("aligned_units", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass(slots=True)
class AlignmentResult:
    language: str
    unit_type: str
    lines: list[AlignmentLine]
    overall_confidence: Optional[float] = None
    report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    artifact_type = "alignment_result"

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "unit_type": self.unit_type,
            "lines": [line.to_dict() for line in self.lines],
            "overall_confidence": self.overall_confidence,
            "report": self.report,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentResult":
        return cls(
            language=data["language"],
            unit_type=data["unit_type"],
            lines=[AlignmentLine.from_dict(item) for item in data.get("lines", [])],
            overall_confidence=data.get("overall_confidence"),
            report=data.get("report", {}),
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        write_artifact_json(self.artifact_type, self.to_dict(), path)

    @classmethod
    def load(cls, path: Path) -> "AlignmentResult":
        data = read_json(path)
        artifact_type = data.get("artifact_type")
        if artifact_type != cls.artifact_type:
            raise ValueError(f"Expected artifact_type={cls.artifact_type!r}, got {artifact_type!r} from {path}")
        return cls.from_dict(data["payload"])


@dataclass(slots=True)
class WriteResult:
    output_path: Path
    writer_backend: str
    artifacts_dir: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "writer_backend": self.writer_backend,
            "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class PipelineRequest:
    stages: list[str]
    audio_path: Optional[Path] = None
    lyrics_path: Optional[Path] = None
    timed_units_path: Optional[Path] = None
    parsed_lyrics_path: Optional[Path] = None
    alignment_result_path: Optional[Path] = None
    language: str = "mul"
    intermediate_dir: Path = Path(".")
    cleanup: str = "on-success"
    output_vocal_audio_path: Optional[Path] = None
    output_filtered_audio_path: Optional[Path] = None
    output_timed_units_path: Optional[Path] = None
    output_parsed_lyrics_path: Optional[Path] = None
    output_alignment_result_path: Optional[Path] = None
    output_written_path: Optional[Path] = None
    log_level: str = "INFO"
    reserve_spacing: bool = True
    parser_lyrics_encoding: Optional[str] = None
    backend_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunPipelineResult:
    executed_stages: list[str]
    source_audio_artifact: Optional[AudioArtifact] = None
    current_audio_artifact: Optional[AudioArtifact] = None
    lyrics_document: Optional[LyricsDocument] = None
    transcription: Optional[TranscriptionResult] = None
    parsed_lyrics: Optional[ParsedLyrics] = None
    alignment: Optional[AlignmentResult] = None
    write_result: Optional[WriteResult] = None
    artifacts: list[Artifact] = field(default_factory=list)
