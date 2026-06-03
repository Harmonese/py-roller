from __future__ import annotations

from pyroller.domain import (
    AlignedUnit,
    AlignmentLine,
    AlignmentResult,
    LyricLine,
    LyricUnit,
    ParsedLyrics,
    TimedUnit,
    TranscriptionResult,
)


def make_alignment_result() -> AlignmentResult:
    return AlignmentResult(
        language="zh",
        unit_type="pinyin",
        overall_confidence=0.95,
        lines=[
            AlignmentLine(
                line_index=0,
                raw_text="你好",
                assigned_time=1.0,
                start_time=1.0,
                end_time=2.0,
                confidence=0.98,
                aligned_units=[
                    AlignedUnit(
                        unit_id="aligned-1",
                        unit_index_in_line=0,
                        text="你",
                        normalized_symbol="ni",
                        unit_type="pinyin",
                        language="zh",
                        start_time=1.0,
                        end_time=1.45,
                    ),
                    AlignedUnit(
                        unit_id="aligned-2",
                        unit_index_in_line=1,
                        text="好",
                        normalized_symbol="hao",
                        unit_type="pinyin",
                        language="zh",
                        start_time=1.45,
                        end_time=2.0,
                    ),
                ],
            ),
            AlignmentLine(
                line_index=1,
                raw_text="",
                assigned_time=2.25,
                start_time=2.25,
                end_time=2.5,
                metadata={"is_spacing": True},
            ),
            AlignmentLine(
                line_index=2,
                raw_text="世界",
                assigned_time=3.0,
                start_time=3.0,
                end_time=4.0,
                confidence=0.92,
                aligned_units=[
                    AlignedUnit(
                        unit_id="aligned-3",
                        unit_index_in_line=0,
                        text="世",
                        normalized_symbol="shi",
                        unit_type="pinyin",
                        language="zh",
                        start_time=3.0,
                        end_time=3.5,
                    ),
                    AlignedUnit(
                        unit_id="aligned-4",
                        unit_index_in_line=1,
                        text="界",
                        normalized_symbol="jie",
                        unit_type="pinyin",
                        language="zh",
                        start_time=3.5,
                        end_time=4.0,
                    ),
                ],
            ),
        ],
    )


def make_repeated_alignment_result() -> AlignmentResult:
    result = make_alignment_result()
    result.lines.append(
        AlignmentLine(
            line_index=3,
            raw_text="你好",
            assigned_time=5.0,
            start_time=5.0,
            end_time=6.0,
            confidence=0.9,
        )
    )
    return result


def make_parsed_lyrics() -> ParsedLyrics:
    return ParsedLyrics(
        language="zh",
        backend="test",
        unit_type="pinyin",
        lines=[
            LyricLine(
                line_index=0,
                raw_text="你好",
                units=[
                    LyricUnit("lyric-1", "你", "ni", "pinyin", "zh", line_index=0, unit_index_in_line=0),
                    LyricUnit("lyric-2", "好", "hao", "pinyin", "zh", line_index=0, unit_index_in_line=1),
                ],
            ),
            LyricLine(
                line_index=1,
                raw_text="世界",
                units=[
                    LyricUnit("lyric-3", "世", "shi", "pinyin", "zh", line_index=1, unit_index_in_line=0),
                    LyricUnit("lyric-4", "界", "jie", "pinyin", "zh", line_index=1, unit_index_in_line=1),
                ],
            ),
        ],
    )


def make_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        language="zh",
        backend="test",
        units=[
            TimedUnit("audio-1", "你", "ni", "pinyin", "zh", start_time=1.0, end_time=1.4, source_backend="test"),
            TimedUnit("audio-2", "好", "hao", "pinyin", "zh", start_time=1.4, end_time=2.0, source_backend="test"),
            TimedUnit("audio-3", "世", "shi", "pinyin", "zh", start_time=3.0, end_time=3.5, source_backend="test"),
            TimedUnit("audio-4", "界", "jie", "pinyin", "zh", start_time=3.5, end_time=4.0, source_backend="test"),
        ],
    )
