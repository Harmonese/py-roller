from __future__ import annotations

from pyroller.aligner.global_dp_v1 import GlobalDPAligner
from pyroller.domain import ParsedLyrics, TranscriptionResult

from .factories import make_parsed_lyrics, make_transcription


def test_global_dp_aligner_matches_small_symbol_sequence() -> None:
    alignment = GlobalDPAligner().align(make_transcription(), make_parsed_lyrics())

    assert [line.raw_text for line in alignment.lines] == ["你好", "世界"]
    assert alignment.lines[0].start_time <= 1.05
    assert alignment.lines[1].start_time >= alignment.lines[0].start_time
    assert all(line.end_time is None or line.end_time >= line.start_time for line in alignment.lines)
    assert alignment.metadata["backend"] == "global_dp_v1"


def test_global_dp_aligner_interpolates_when_no_audio_units() -> None:
    transcription = TranscriptionResult(language="zh", backend="test", units=[])

    alignment = GlobalDPAligner().align(transcription, make_parsed_lyrics())

    assert len(alignment.lines) == 2
    assert alignment.metadata["fallback"] == "full_interpolation_no_units"
    assert alignment.lines[0].start_time <= alignment.lines[1].start_time


def test_global_dp_few_repetition_mode_smoke() -> None:
    alignment = GlobalDPAligner(repetition="few").align(make_transcription(), make_parsed_lyrics())

    assert len(alignment.lines) == 2
    assert alignment.metadata["repetition"] == "few"
    assert "repetition_stats" in alignment.report


def test_global_dp_full_repetition_mode_smoke() -> None:
    alignment = GlobalDPAligner(repetition="full").align(make_transcription(), make_parsed_lyrics())

    assert len(alignment.lines) == 2
    assert alignment.metadata["repetition"] == "full"
    assert alignment.report["dp_stats"]["skipped"] is True


def test_global_dp_interpolates_when_lyrics_have_no_units() -> None:
    parsed = ParsedLyrics(language="zh", backend="test", unit_type="pinyin", lines=[], metadata={})

    alignment = GlobalDPAligner().align(make_transcription(), parsed)

    assert alignment.lines == []
    assert alignment.metadata["fallback"] == "full_interpolation_no_lyric_units"
    assert alignment.overall_confidence == 0.0


def test_global_dp_rejects_invalid_repetition_mode() -> None:
    try:
        GlobalDPAligner(repetition="sometimes")
    except ValueError as exc:
        assert "repetition must be one of" in str(exc)
    else:
        raise AssertionError("expected invalid repetition mode to raise")
