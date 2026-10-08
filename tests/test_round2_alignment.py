from pathlib import Path
import json
import re

import pytest

from pyroller.aligner.global_dp_v1 import GlobalDPAligner
from pyroller.domain import AlignmentResult, LyricsDocument, LyricLine, LyricUnit, ParsedLyrics, TimedUnit, TranscriptionResult
from pyroller.parser import get_lyrics_parser
from pyroller.quality import AlignmentQualityError, enforce_quality
from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
from pyroller.transcriber.unitizers.common import preferred_text_spans
from pyroller.transcriber.unitizers.zh_pinyin_from_text import ZhPinyinFromTextUnitizer
from pyroller.writer.ass_karaoke import ASSKaraokeWriter
from pyroller.writer.lrc import LRCWriter
from .factories import make_alignment_result


def parsed(text, language='zh'):
    document = LyricsDocument(Path('unused'), text, 'utf-8', [LyricLine(0, text)], language)
    return get_lyrics_parser(language).parse(document, language=language, tone_mode='ignore')


@pytest.mark.parametrize('mode', ['none', 'few', 'full'])
@pytest.mark.parametrize('missing', [0, 2, 4])
def test_missing_units_never_move_observed_anchors(mode, missing):
    symbols = ['ni', 'hao', 'shi', 'jie']
    lyrics = symbols[:missing] + ['xxxxxx'] + symbols[missing:]
    line = LyricLine(0, 'ABCDE', units=[LyricUnit(str(i), s, s, 'pinyin', 'zh',
                        line_index=0, unit_index_in_line=i, source_text_span=(i, i+1)) for i, s in enumerate(lyrics)])
    transcription = TranscriptionResult('zh', 'test', [TimedUnit(str(i), s, s, 'pinyin', 'zh',
                        start_time=1+i*.3, end_time=1+i*.3+.2, confidence=1,
                        metadata={'timing_is_acoustic': True}) for i, s in enumerate(symbols)])
    result = GlobalDPAligner(repetition=mode).align(transcription, ParsedLyrics('zh', 'test', [line], 'pinyin'))
    units = result.lines[0].aligned_units
    for unit in units:
        if unit.source_audio_unit_index is not None:
            original = transcription.units[unit.source_audio_unit_index]
            assert (unit.start_time, unit.end_time) == (original.start_time, original.end_time)
    inferred = units[missing]
    assert inferred.confidence == 0 and inferred.source_audio_unit_range is None
    assert inferred.metadata['match_status'] == 'unmatched'
    if missing in (0, 4):
        assert inferred.start_time == inferred.end_time
        assert inferred.metadata['timing_source'] == 'unresolved'
    else:
        assert units[missing-1].end_time <= inferred.start_time < inferred.end_time <= units[missing+1].start_time
    with pytest.raises(AlignmentQualityError):
        enforce_quality(result, {'mode': 'strict'})


@pytest.mark.parametrize('word_text', ['你', '你好', '世界', '别的'])
def test_partial_word_coverage_retains_segment_and_reports_fallback(word_text):
    output = EngineOutput('zh', 'faster_whisper', '你好世界', [
        EngineSpan('s', 'segment', 1, 5, text='你好世界'),
        EngineSpan('w', 'word', 1, 2, text=word_text, parent_span_id='s'),
    ])
    result = ZhPinyinFromTextUnitizer().adapt(output, language='zh', tone_mode='ignore')
    assert [u.normalized_symbol for u in result.units] == ['ni', 'hao', 'shi', 'jie']
    assert result.metadata['text_span_warnings'][0]['code'] == 'incomplete_word_coverage'
    assert result.metadata['language_warnings']
    assert all(u.metadata['timing_mode'] == 'interpolated_from_segment' for u in result.units)


def test_complete_word_coverage_keeps_native_spans():
    words = [EngineSpan('w1', 'word', 1, 2, text='你好', parent_span_id='s'),
             EngineSpan('w2', 'word', 3, 4, text='世界', parent_span_id='s')]
    output = EngineOutput('zh', 'test', '你好，世界！', [EngineSpan('s', 'segment', 0, 5, text='你好，世界！'), *words])
    assert preferred_text_spans(output) == words


def test_empty_segment_text_does_not_discard_words():
    word = EngineSpan('w', 'word', 1, 2, text='hello', parent_span_id='s')
    output = EngineOutput('en', 'test', 'hello', [EngineSpan('s', 'segment', 0, 3), word])
    assert preferred_text_spans(output) == [word]


def test_estimated_timing_survives_roundtrip_and_strict_policy(tmp_path):
    output = EngineOutput('zh', 'faster_whisper', '你好世界', [EngineSpan('s', 'segment', 1, 9, text='你好世界', confidence=0)])
    transcription = ZhPinyinFromTextUnitizer().adapt(output, language='zh', tone_mode='ignore')
    result = GlobalDPAligner().align(transcription, parsed('你好世界'))
    path = tmp_path / 'alignment.json'
    result.save(path)
    result = AlignmentResult.load(path)
    quality = result.report['quality']
    assert quality['coverage'] == 1 and quality['interpolated_line_ratio'] == 0
    assert quality['interpolated_unit_ratio'] == 1 and quality['acoustic_unit_ratio'] == 0
    assert len(quality['unit_timing_diagnostics']) == 4
    assert result.lines[0].aligned_units[0].metadata['timing_provenance']['timing_mode'] == 'interpolated_from_segment'
    with pytest.raises(AlignmentQualityError):
        enforce_quality(result, {'mode': 'strict'})
    assert enforce_quality(result, {'mode': 'strict', 'timing_policy': 'line'})['timing_needs_review']


def test_acoustic_timing_passes_and_unknown_timing_requires_review():
    lyrics = parsed('你好')
    transcription = TranscriptionResult('zh', 'test', [TimedUnit(str(i), u.symbol, u.normalized_symbol,
        lyrics.unit_type, 'zh', start_time=1+i, end_time=2+i, confidence=1,
        metadata={'timing_is_acoustic': True}) for i, u in enumerate(lyrics.lines[0].units)])
    result = GlobalDPAligner().align(transcription, lyrics)
    assert enforce_quality(result, {'mode': 'strict'})['acoustic_unit_ratio'] == 1
    for unit in transcription.units:
        unit.metadata.clear()
    result = GlobalDPAligner().align(transcription, lyrics)
    assert result.report['quality']['unknown_timing_unit_ratio'] == 1
    with pytest.raises(AlignmentQualityError):
        enforce_quality(result, {'mode': 'strict'})


@pytest.mark.parametrize('text', ['你好世界', '你好，世界！', '傳統漢字'])
def test_multilingual_chinese_has_per_character_karaoke(text, tmp_path):
    lyrics = parsed(text, 'mul')
    transcription = TranscriptionResult('mul', 'test', [TimedUnit(str(i), u.symbol, u.normalized_symbol,
        lyrics.unit_type, 'mul', start_time=i*.2, end_time=(i+1)*.2) for i, u in enumerate(lyrics.lines[0].units)])
    result = GlobalDPAligner().align(transcription, lyrics)
    spans = {tuple(unit.source_text_span) for unit in lyrics.lines[0].units}
    assert len(spans) == 4
    output = tmp_path / 'song.ass'
    ASSKaraokeWriter().write(result, output)
    dialogue = next(line for line in output.read_text().splitlines() if line.startswith('Dialogue:'))
    assert len(re.findall(r'\{\\kf\d+\}', dialogue)) == 4
    assert re.sub(r'\{[^}]*\}', '', dialogue).endswith(text)


@pytest.mark.parametrize('defect', ['outside', 'unordered', 'before_line'])
def test_invalid_alignment_rejected_on_load_save_and_export(tmp_path, defect):
    result = make_alignment_result()
    units = result.lines[0].aligned_units
    if defect == 'outside':
        units[0].start_time, units[0].end_time = 100, 101
    elif defect == 'before_line':
        units[0].start_time = .5
    else:
        units[0].start_time, units[0].end_time = 1.6, 1.8
    output = tmp_path / 'existing'
    output.write_text('keep')
    for publish in (result.save, lambda p: ASSKaraokeWriter().write(result, p), lambda p: LRCWriter().write(result, p)):
        with pytest.raises(ValueError):
            publish(output)
        assert output.read_text() == 'keep'
    artifact = tmp_path / 'bad.json'
    artifact.write_text(json.dumps({'artifact_type': 'alignment_result', 'schema_version': 1, 'payload': result.to_dict()}))
    with pytest.raises(ValueError):
        AlignmentResult.load(artifact)


def test_overlapping_observed_units_are_preserved_and_valid(tmp_path):
    lyrics = parsed('你好')
    transcription = TranscriptionResult('zh', 'test', [TimedUnit(str(i), u.symbol, u.normalized_symbol,
        lyrics.unit_type, 'zh', start_time=1+i*.2, end_time=1.5+i*.2,
        metadata={'timing_is_acoustic': True}) for i, u in enumerate(lyrics.lines[0].units)])
    result = GlobalDPAligner().align(transcription, lyrics)
    assert [(u.start_time,u.end_time) for u in result.lines[0].aligned_units] == [(1,1.5),(1.2,1.7)]
    result.save(tmp_path / 'valid.json')
    assert result.report['quality']['timing_anomaly_unit_count'] == 1
    with pytest.raises(AlignmentQualityError):
        enforce_quality(result, {'mode': 'strict'})


def test_missing_unit_timing_is_not_masked_by_other_acoustic_lines():
    result = make_alignment_result()
    for line in result.lines:
        for unit in line.aligned_units:
            unit.metadata['timing_source'] = 'acoustic'
    result.lines[2].aligned_units = []
    with pytest.raises(AlignmentQualityError):
        enforce_quality(result, {'mode': 'strict'})
    assert result.report['quality']['missing_unit_timing_lines'] == [2]


def test_no_audio_units_does_not_invent_unit_durations():
    transcription = TranscriptionResult('zh', 'test', [], metadata={'audio_duration': 2})
    result = GlobalDPAligner().align(transcription, parsed('你好世界'))
    assert all(unit.start_time == unit.end_time for unit in result.lines[0].aligned_units)
    assert result.report['quality']['unresolved_unit_ratio'] == 1


def test_cli_timing_policy_and_strict_writer_preserve_existing_output(tmp_path):
    from pyroller.cli.main import build_parser
    from pyroller.cli.runlike import build_request
    from pyroller.pipeline import ComposablePipelineRunner
    artifact = tmp_path / 'alignment.json'
    make_alignment_result().save(artifact)
    output = tmp_path / 'song.ass'
    output.write_text('keep')
    parser, _, _ = build_parser()
    arguments = ['run', '--stages', 'w', '--alignment-result', str(artifact),
                 '--output-roller', str(output), '--writer-backend', 'ass_karaoke',
                 '--intermediate', str(tmp_path / 'scratch'), '--quality-mode', 'strict']
    runner = ComposablePipelineRunner()
    try:
        with pytest.raises(AlignmentQualityError):
            runner.run(build_request(parser.parse_args(arguments)))
        assert output.read_text() == 'keep'
        request = build_request(parser.parse_args([*arguments, '--quality-timing-policy', 'line']))
        runner.run(request)
        assert 'Dialogue:' in output.read_text()
    finally:
        runner.close()
