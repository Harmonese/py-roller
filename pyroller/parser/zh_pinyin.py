from __future__ import annotations

import logging

from pyroller.domain import LyricLine, LyricUnit, LyricsDocument, ParsedLyrics
from pyroller.parser.base import LyricsParser
from pyroller.utils.ids import make_id
from pyroller.utils.text import chinese_text_to_pinyin_syllables, normalize_chinese_text

logger = logging.getLogger("pyroller.parser")


class ChinesePinyinParser(LyricsParser):
    def parse(self, lyrics_document: LyricsDocument, language: str, tone_mode: str) -> ParsedLyrics:
        parsed_lines: list[LyricLine] = []

        for line in lyrics_document.lines:
            normalized = normalize_chinese_text(line.raw_text)
            syllables = chinese_text_to_pinyin_syllables(normalized, tone_mode=tone_mode)
            units: list[LyricUnit] = []
            for idx, syllable in enumerate(syllables):
                units.append(
                    LyricUnit(
                        unit_id=make_id("lyric_unit"),
                        symbol=syllable["symbol"] or syllable["normalized_symbol"],
                        normalized_symbol=syllable["normalized_symbol"] or syllable["symbol"],
                        unit_type="pinyin_syllable",
                        language=language,
                        tone=syllable["tone"],
                        line_index=line.line_index,
                        unit_index_in_line=idx,
                        metadata={
                            "raw_text": line.raw_text,
                            "normalized_text": normalized,
                            "source_char": syllable.get("source_char"),
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

        logger.info("Parsed %d lyric lines into normalized pinyin syllables", len(parsed_lines))
        return ParsedLyrics(
            language=language,
            backend="zh_pinyin",
            lines=parsed_lines,
            unit_type="pinyin_syllable",
            metadata={
                "tone_mode": tone_mode,
                "line_count": len(parsed_lines),
            },
        )
