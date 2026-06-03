from __future__ import annotations

from pathlib import Path

from pyroller.domain import LyricLine, LyricsDocument
from pyroller.parser.registry import get_lyrics_parser
from pyroller.parser.zh_pinyin import ChinesePinyinParser


def _lyrics_document(language: str, lines: list[LyricLine]) -> LyricsDocument:
    return LyricsDocument(
        source_path=Path("song.txt"),
        raw_text="\n".join(line.raw_text for line in lines),
        encoding="utf-8",
        lines=lines,
        language=language,
    )


def test_zh_router_parser_preserves_structural_lines_and_emits_pinyin_units() -> None:
    document = _lyrics_document(
        "zh",
        [
            LyricLine(line_index=0, raw_text="你好，世界"),
            LyricLine(line_index=1, raw_text="", metadata={"is_spacing": True}),
        ],
    )

    parsed = get_lyrics_parser("zh").parse(document, language="zh", tone_mode="ignore")

    assert parsed.backend == "zh_router_pinyin"
    assert parsed.unit_type == "pinyin_syllable"
    assert parsed.lines[0].normalized_text == "ni hao shi jie"
    assert [unit.normalized_symbol for unit in parsed.lines[0].units] == ["ni", "hao", "shi", "jie"]
    assert all(unit.unit_type == "pinyin_syllable" for unit in parsed.lines[0].units)
    assert parsed.lines[1].units == []
    assert parsed.lines[1].metadata["is_spacing"] is True
    assert parsed.lines[1].metadata["is_structural"] is True
    assert parsed.metadata["route_counts"]["structural"] == 1


def test_zh_router_parser_tracks_latin_and_digit_routes() -> None:
    document = _lyrics_document("zh", [LyricLine(line_index=0, raw_text="AI 2026")])

    parsed = get_lyrics_parser("zh").parse(document, language="zh", tone_mode="ignore")
    line = parsed.lines[0]

    assert line.units
    assert line.metadata["route_summary"]["foreign_segment_count"] >= 1
    assert any(unit.metadata["segment_type"] == "latin" for unit in line.units)
    assert any(unit.metadata["segment_type"] == "digit" for unit in line.units)


def test_plain_zh_pinyin_parser_keeps_raw_line_metadata() -> None:
    document = _lyrics_document(
        "zh",
        [LyricLine(line_index=0, raw_text="你好", metadata={"section": "verse"})],
    )

    parsed = ChinesePinyinParser().parse(document, language="zh", tone_mode="ignore")

    assert parsed.backend == "zh_pinyin"
    assert parsed.lines[0].normalized_text == "你好"
    assert [unit.normalized_symbol for unit in parsed.lines[0].units] == ["ni", "hao"]
    assert parsed.lines[0].metadata == {"section": "verse"}


def test_english_parser_normalizes_text_and_stress_for_alignment() -> None:
    document = _lyrics_document("en", [LyricLine(line_index=0, raw_text="Hello, world!")])

    parsed = get_lyrics_parser("en").parse(document, language="en", tone_mode="ignore")
    line = parsed.lines[0]

    assert parsed.backend == "en_arpabet"
    assert parsed.unit_type == "arpabet_phone"
    assert line.normalized_text == "hello world"
    assert [unit.normalized_symbol for unit in line.units[:4]] == ["HH", "AH", "L", "OW"]
    assert [unit.symbol for unit in line.units[:4]] == ["HH", "AH0", "L", "OW1"]
    assert {unit.metadata["source_word"] for unit in line.units} == {"Hello", "world"}
    assert parsed.metadata["stress_normalized"] is True


def test_multilingual_parser_emits_ipa_units_with_route_summary() -> None:
    document = _lyrics_document("mul", [LyricLine(line_index=0, raw_text="Hello world")])

    parsed = get_lyrics_parser("mul").parse(document, language="mul", tone_mode="ignore")
    line = parsed.lines[0]

    assert parsed.backend == "mul_ipa"
    assert parsed.unit_type == "ipa_phone"
    assert line.units
    assert all(unit.unit_type == "ipa_phone" for unit in line.units)
    assert all(unit.normalized_symbol for unit in line.units)
    assert line.metadata["route_summary"]["segments"]
    assert parsed.metadata["segment_count"] >= 1
    assert parsed.metadata["route_counts"]
