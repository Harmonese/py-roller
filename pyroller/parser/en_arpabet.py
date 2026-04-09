from __future__ import annotations

import logging
from typing import Any

from pyroller.domain import LyricLine, LyricUnit, LyricsDocument, ParsedLyrics
from pyroller.parser.base import LyricsParser
from pyroller.utils.ids import make_id
from pyroller.utils.text import english_text_to_arpabet_units, normalize_english_text

logger = logging.getLogger("pyroller.parser")


class EnglishARPAbetParser(LyricsParser):
    def __init__(self, **_: Any) -> None:
        pass

    def parse(self, lyrics_document: LyricsDocument, language: str, tone_mode: str) -> ParsedLyrics:
        parsed_lines: list[LyricLine] = []

        for line in lyrics_document.lines:
            normalized = normalize_english_text(line.raw_text)
            phones = english_text_to_arpabet_units(line.raw_text)
            units: list[LyricUnit] = []
            for idx, phone in enumerate(phones):
                units.append(
                    LyricUnit(
                        unit_id=make_id("lyric_unit"),
                        symbol=phone["symbol"] or phone["normalized_symbol"],
                        normalized_symbol=phone["normalized_symbol"] or phone["symbol"],
                        unit_type="arpabet_phone",
                        language=language,
                        tone=phone.get("stress"),
                        line_index=line.line_index,
                        unit_index_in_line=idx,
                        metadata={
                            "raw_text": line.raw_text,
                            "normalized_text": normalized,
                            "source_word": phone.get("source_word"),
                        },
                    )
                )
            parsed_lines.append(
                LyricLine(
                    line_index=line.line_index,
                    raw_text=line.raw_text,
                    normalized_text=normalized,
                    units=units,
                    metadata=dict(line.metadata),
                )
            )

        logger.info("Parsed %d lyric lines into ARPAbet phones", len(parsed_lines))
        return ParsedLyrics(
            language=language,
            backend="en_arpabet",
            lines=parsed_lines,
            unit_type="arpabet_phone",
            metadata={
                "line_count": len(parsed_lines),
                "stress_normalized": True,
            },
        )
