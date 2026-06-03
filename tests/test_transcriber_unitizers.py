from __future__ import annotations

from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
from pyroller.transcriber.unitizers.common import (
    base_result_metadata,
    engine_spans_by_level,
    preferred_raw_segment_spans,
    preferred_text_spans,
    raw_segments_from_spans,
)
from pyroller.transcriber.unitizers.en_arpabet import EnArpabetUnitizer
from pyroller.transcriber.unitizers.mul_ipa_from_text import MulIpaFromTextUnitizer
from pyroller.transcriber.unitizers.zh_pinyin_from_text import ZhPinyinFromTextUnitizer


def _engine_output() -> EngineOutput:
    return EngineOutput(
        language="en",
        engine="test_engine",
        raw_text="Hello world",
        spans=[
            EngineSpan(
                span_id="seg:0",
                level="segment",
                start_time=0.0,
                end_time=2.0,
                text="Hello world",
                normalized_text="hello world",
                confidence=0.8,
                segment_index=3,
                metadata={"kind": "segment"},
            ),
            EngineSpan(
                span_id="word:0",
                level="word",
                start_time=0.0,
                end_time=1.0,
                text="Hello",
                confidence=0.9,
                segment_index=3,
                word_index=0,
            ),
        ],
        metadata={"custom": "value"},
    )


def test_unitizer_common_prefers_word_text_spans_and_segment_raw_spans() -> None:
    output = _engine_output()

    assert engine_spans_by_level(output, "word")[0].span_id == "word:0"
    assert preferred_text_spans(output)[0].level == "word"
    assert preferred_raw_segment_spans(output)[0].level == "segment"


def test_unitizer_common_falls_back_to_segments_when_no_words() -> None:
    output = EngineOutput(
        language="en",
        engine="test",
        raw_text="Hello",
        spans=[EngineSpan("seg:0", "segment", 0.0, 1.0, text="Hello")],
    )

    assert preferred_text_spans(output)[0].level == "segment"
    assert preferred_raw_segment_spans(output)[0].span_id == "seg:0"


def test_raw_segments_from_spans_and_metadata_contract() -> None:
    output = _engine_output()
    raw_segments = raw_segments_from_spans(preferred_raw_segment_spans(output))
    metadata = base_result_metadata(output, unitizer_name="test_unitizer", raw_segment_level="segment")

    assert raw_segments[0]["segment_index"] == 3
    assert raw_segments[0]["metadata"] == {"kind": "segment"}
    assert metadata["custom"] == "value"
    assert metadata["engine"] == "test_engine"
    assert metadata["unitizer"] == "test_unitizer"
    assert metadata["raw_segment_level"] == "segment"


def test_english_arpabet_unitizer_adapts_word_spans_to_timed_units() -> None:
    result = EnArpabetUnitizer(backend="test_backend").adapt(_engine_output(), language="en", tone_mode="ignore")

    assert result.backend == "test_backend"
    assert result.raw_text == "Hello world"
    assert result.raw_segments[0]["segment_level"] == "segment"
    assert [unit.normalized_symbol for unit in result.units[:4]] == ["HH", "AH", "L", "OW"]
    assert result.units[0].start_time == 0.0
    assert result.units[-1].end_time == 1.0
    assert result.units[0].metadata["timing_mode"] == "interpolated_from_word"
    assert result.metadata["unitizer"] == "en_arpabet"
    assert result.metadata["unit_timing_semantics"] == "interpolated_non_acoustic"


def test_zh_pinyin_from_text_unitizer_interpolates_segment_timing() -> None:
    output = EngineOutput(
        language="zh",
        engine="test",
        raw_text="你好",
        spans=[EngineSpan("seg:0", "segment", 2.0, 4.0, text="你好", confidence=0.7, segment_index=0)],
    )

    result = ZhPinyinFromTextUnitizer(backend="test_backend").adapt(output, language="zh", tone_mode="ignore")

    assert [unit.normalized_symbol for unit in result.units] == ["ni", "hao"]
    assert result.units[0].start_time == 2.0
    assert result.units[0].end_time == 3.0
    assert result.units[1].end_time == 4.0
    assert result.units[0].metadata["timing_mode"] == "interpolated_from_segment"


def test_mul_ipa_from_text_unitizer_emits_ipa_units() -> None:
    output = EngineOutput(
        language="mul",
        engine="test",
        raw_text="Hello",
        spans=[EngineSpan("word:0", "word", 0.0, 1.0, text="Hello", word_index=0)],
    )

    result = MulIpaFromTextUnitizer(backend="test_backend").adapt(output, language="mul", tone_mode="ignore")

    assert result.units
    assert all(unit.unit_type == "ipa_phone" for unit in result.units)
    assert result.units[0].metadata["timing_mode"] == "interpolated_from_word"
