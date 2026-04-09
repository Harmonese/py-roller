from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from pyroller.domain import LyricLine, LyricUnit, LyricsDocument, ParsedLyrics
from pyroller.parser.base import LyricsParser
from pyroller.utils.ids import make_id
from pyroller.utils.text import multilingual_text_to_ipa_units, summarize_multilingual_routes

logger = logging.getLogger("pyroller.parser")


class MultilingualIPAParser(LyricsParser):
    def __init__(self, **_: Any) -> None:
        pass

    def parse(self, lyrics_document: LyricsDocument, language: str, tone_mode: str) -> ParsedLyrics:
        del tone_mode

        parsed_lines: list[LyricLine] = []
        total_routes: Counter[str] = Counter()
        total_languages: Counter[str] = Counter()
        total_segments = 0

        logger.info("=" * 58)
        logger.info("MULTILINGUAL IPA PARSER")
        logger.info("Strategy priority: A=gruut, B=dedicated, C=phonemizer/espeak, D=empty")
        logger.info("=" * 58)

        for line in lyrics_document.lines:
            phones = multilingual_text_to_ipa_units(line.raw_text)
            route_summary = summarize_multilingual_routes(line.raw_text)
            total_routes.update(route_summary["route_counts"])
            total_languages.update(route_summary["language_counts"])
            total_segments += len(route_summary["segments"])

            normalized = " ".join(phone["normalized_symbol"] for phone in phones)
            units: list[LyricUnit] = []
            for idx, phone in enumerate(phones):
                units.append(
                    LyricUnit(
                        unit_id=make_id("lyric_unit"),
                        symbol=phone["symbol"] or phone["normalized_symbol"],
                        normalized_symbol=phone["normalized_symbol"] or phone["symbol"],
                        unit_type="ipa_phone",
                        language=str(phone.get("segment_language") or language),
                        tone=phone.get("stress"),
                        line_index=line.line_index,
                        unit_index_in_line=idx,
                        metadata={
                            "raw_text": line.raw_text,
                            "normalized_text": normalized,
                            "source_word": phone.get("source_word"),
                            "source_language": phone.get("source_language"),
                            "segment_language": phone.get("segment_language"),
                            "segment_text": phone.get("segment_text"),
                            "segment_index": phone.get("segment_index"),
                            "route_backend": phone.get("backend"),
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
                "L%-3d routes=%s languages=%s segments=%s",
                line.line_index + 1,
                route_summary["route_counts"],
                route_summary["language_counts"],
                [seg["route"] for seg in route_summary["segments"]],
            )

        logger.info(
            "Parsed %d lyric lines into IPA phones (segments=%d routes=%s languages=%s)",
            len(parsed_lines),
            total_segments,
            dict(total_routes),
            dict(total_languages),
        )
        return ParsedLyrics(
            language=language,
            backend="mul_ipa",
            lines=parsed_lines,
            unit_type="ipa_phone",
            metadata={
                "line_count": len(parsed_lines),
                "route_counts": dict(total_routes),
                "language_counts": dict(total_languages),
                "segment_count": total_segments,
            },
        )
