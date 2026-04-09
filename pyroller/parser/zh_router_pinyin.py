from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from pyroller.domain import LyricLine, LyricUnit, LyricsDocument, ParsedLyrics
from pyroller.parser.base import LyricsParser
from pyroller.utils.ids import make_id
from pyroller.utils.text import segmented_zh_text_to_pinyin_units

logger = logging.getLogger("pyroller.parser")


class ZhRouterPinyinParser(LyricsParser):
    def __init__(self, **_: Any) -> None:
        pass

    def parse(self, lyrics_document: LyricsDocument, language: str, tone_mode: str) -> ParsedLyrics:
        parsed_lines: list[LyricLine] = []
        total_routes: Counter[str] = Counter()
        total_segment_types: Counter[str] = Counter()
        total_segments = 0
        total_foreign_segments = 0

        logger.info("=" * 58)
        logger.info("ZH ROUTER PINYIN PARSER")
        logger.info("Strategy: han->pinyin, digit->digit_by_digit, latin->borrowed pinyin proxy")
        logger.info("=" * 58)

        for line in lyrics_document.lines:
            if bool(line.metadata.get("is_spacing")) or not line.raw_text.strip():
                parsed_lines.append(
                    LyricLine(
                        line_index=line.line_index,
                        raw_text=line.raw_text,
                        normalized_text="",
                        units=[],
                        metadata={**dict(line.metadata), "is_structural": True, "route_summary": {"route_counts": {"structural": 1}, "segment_type_counts": {}, "segments": [], "foreign_segment_count": 0}},
                    )
                )
                total_routes["structural"] += 1
                continue

            routed_units, route_summary = segmented_zh_text_to_pinyin_units(line.raw_text, tone_mode=tone_mode)
            total_routes.update(route_summary.get("route_counts", {}))
            total_segment_types.update(route_summary.get("segment_type_counts", {}))
            total_segments += len(route_summary.get("segments", []))
            total_foreign_segments += int(route_summary.get("foreign_segment_count", 0))

            normalized = " ".join(str(unit.get("normalized_symbol") or unit.get("symbol") or "") for unit in routed_units if str(unit.get("normalized_symbol") or unit.get("symbol") or ""))
            units: list[LyricUnit] = []
            for idx, syllable in enumerate(routed_units):
                units.append(
                    LyricUnit(
                        unit_id=make_id("lyric_unit"),
                        symbol=str(syllable.get("symbol") or syllable.get("normalized_symbol") or ""),
                        normalized_symbol=str(syllable.get("normalized_symbol") or syllable.get("symbol") or ""),
                        unit_type="pinyin_syllable",
                        language=language,
                        tone=syllable.get("tone"),
                        line_index=line.line_index,
                        unit_index_in_line=idx,
                        metadata={
                            "raw_text": line.raw_text,
                            "normalized_text": normalized,
                            "source_text": syllable.get("source_text"),
                            "segment_type": syllable.get("segment_type"),
                            "segment_text": syllable.get("segment_text"),
                            "segment_index": syllable.get("segment_index"),
                            "route_backend": syllable.get("backend"),
                        },
                    )
                )

            parsed_lines.append(
                LyricLine(
                    line_index=line.line_index,
                    raw_text=line.raw_text,
                    normalized_text=normalized,
                    units=units,
                    metadata={**dict(line.metadata), "route_summary": route_summary},
                )
            )
            logger.debug(
                "L%-3d routes=%s segment_types=%s segments=%s",
                line.line_index + 1,
                route_summary.get("route_counts"),
                route_summary.get("segment_type_counts"),
                [seg.get("route") for seg in route_summary.get("segments", [])],
            )

        logger.info(
            "Parsed %d lyric lines with zh router (segments=%d routes=%s segment_types=%s foreign_segments=%d)",
            len(parsed_lines),
            total_segments,
            dict(total_routes),
            dict(total_segment_types),
            total_foreign_segments,
        )
        return ParsedLyrics(
            language=language,
            backend="zh_router_pinyin",
            lines=parsed_lines,
            unit_type="pinyin_syllable",
            metadata={
                "tone_mode": tone_mode,
                "line_count": len(parsed_lines),
                "route_counts": dict(total_routes),
                "segment_type_counts": dict(total_segment_types),
                "segment_count": total_segments,
                "foreign_segment_count": total_foreign_segments,
            },
        )
