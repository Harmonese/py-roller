from __future__ import annotations

from typing import Any

from pyroller.writer.ass_karaoke import ASSKaraokeWriter
from pyroller.writer.base import Writer
from pyroller.writer.lrc import LRCWriter

_DEFAULT_WRITER = "lrc_ms"
_AVAILABLE = ("lrc_ms", "lrc_cs", "lrc_compressed", "ass_karaoke")


def list_available_writer_backends() -> tuple[str, ...]:
    return _AVAILABLE


def build_writer(backend_name: str | None, config: dict[str, Any] | None = None) -> Writer:
    backend = backend_name or _DEFAULT_WRITER
    config = dict(config or {})
    by_tag = str(config.get("by_tag") or "py-roller")
    tag_type = str(config.get("tag_type") or "kf")
    reserve_spacing = bool(config.get("reserve_spacing", False))
    skip_structural_lines = bool(config.get("skip_structural_lines", True))
    unmatched_line_duration = float(config.get("unmatched_line_duration", 0.6))

    if backend == "lrc_ms":
        return LRCWriter(decimals=3, compressed=False, by_tag=by_tag, reserve_spacing=reserve_spacing)
    if backend == "lrc_cs":
        return LRCWriter(decimals=2, compressed=False, by_tag=by_tag, reserve_spacing=reserve_spacing)
    if backend == "lrc_compressed":
        return LRCWriter(decimals=2, compressed=True, by_tag=by_tag, reserve_spacing=False)
    if backend == "ass_karaoke":
        return ASSKaraokeWriter(
            by_tag=by_tag,
            tag_type=tag_type,
            skip_structural_lines=skip_structural_lines,
            unmatched_line_duration=unmatched_line_duration,
        )

    available = ", ".join(_AVAILABLE)
    raise ValueError(f"Unsupported writer backend {backend!r}. Available: {available}")
