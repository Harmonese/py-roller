from pathlib import Path
import json
import re

import pytest

from pyroller.domain import PipelineRequest, LyricsDocument, LyricLine, TimedUnit, TranscriptionResult
from pyroller.pipeline import ComposablePipelineRunner
from pyroller.batch_builder import ManifestBatchBuilder
from pyroller.parser import get_lyrics_parser
from pyroller.aligner.global_dp_v1 import GlobalDPAligner
from pyroller.writer.ass_karaoke import ASSKaraokeWriter
from .factories import make_alignment_result, make_parsed_lyrics, make_transcription


def test_existing_workspace_files_and_outputs_are_preserved(tmp_path):
    artifact = tmp_path / 'alignment.json'
    make_alignment_result().save(artifact)
    old_marker = tmp_path / '.py-roller-owner.json'
    old_marker.write_text('{"owned_by":"py-roller"}')
    sentinel = tmp_path / 'unrelated.txt'
    sentinel.write_text('keep')
    request = PipelineRequest(stages=['w'], alignment_result_path=artifact,
                              intermediate_dir=tmp_path, output_roller_path=tmp_path / 'out.lrc')
    runner = ComposablePipelineRunner()
    runner.run(request)
    first = runner.last_request.intermediate_dir
    runner.run(request)
    assert runner.last_request.intermediate_dir != first
    assert not first.exists()
    assert sentinel.read_text() == 'keep'
    assert old_marker.read_text() == '{"owned_by":"py-roller"}'
    assert artifact.exists() and request.output_roller_path.exists()


@pytest.mark.parametrize('task_id', ['..', '.', '../escape', '/absolute', r'a\b', 'C:drive', 'x/y', 'x\x00'])
def test_manifest_rejects_path_ids(tmp_path, task_id):
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps([{'id': task_id, 'alignment_result': 'in.json', 'output_roller': 'out.ass'}]))
    with pytest.raises(ValueError, match='safe single'):
        ManifestBatchBuilder(manifest).build_tasks(PipelineRequest(stages=['w'], intermediate_dir=tmp_path))


@pytest.mark.parametrize('language,text', [('zh','你好，世界！'), ('zh','我愛你 AI 2024！'),
                                          ('en',"Hello, hello world!"), ('mul','Hello, world!')])
def test_real_parser_alignment_ass_preserves_original(tmp_path, language, text):
    parsed = get_lyrics_parser(language).parse(LyricsDocument(Path('in.txt'), text, 'utf-8',
                                              [LyricLine(0, text)], language), language, 'ignore')
    assert parsed.lines[0].units
    assert all(u.source_text_span is not None for u in parsed.lines[0].units)
    transcription = TranscriptionResult(language, 'test', [
        TimedUnit(str(i), u.symbol, u.normalized_symbol, u.unit_type, u.language,
                  start_time=1+i*.1, end_time=1+(i+1)*.1)
        for i,u in enumerate(parsed.lines[0].units)])
    parsed_path = tmp_path / 'parsed.json'
    parsed.save(parsed_path)
    parsed = type(parsed).load(parsed_path)
    alignment = GlobalDPAligner().align(transcription, parsed)
    output = tmp_path / 'out.ass'
    ASSKaraokeWriter().write(alignment, output)
    dialogue = next(line for line in output.read_text().splitlines() if line.startswith('Dialogue:'))
    rendered = re.sub(r'\{[^}]*\}', '', dialogue.split(',',9)[9])
    assert rendered == text


def test_instrumental_gap_does_not_stretch_units():
    transcription = make_transcription()
    for u in transcription.units[2:]:
        u.start_time += 27
        u.end_time += 27
    alignment = GlobalDPAligner().align(transcription, make_parsed_lyrics())
    assert alignment.lines[0].end_time == 2
    assert alignment.lines[0].aligned_units[-1].end_time == 2


def test_ass_preserves_pause_between_units():
    line = make_alignment_result().lines[0]
    line.aligned_units[0].end_time = 1.2
    line.aligned_units[1].start_time = 1.7
    assert ASSKaraokeWriter()._line_to_ass_text(line) == r'{\kf20}你{\k50}{\kf30}好'


def test_plain_chinese_parser_maps_traditional_text_and_numbers():
    from pyroller.parser.zh_pinyin import ChinesePinyinParser
    text = '我愛你，2024！'
    parsed = ChinesePinyinParser().parse(LyricsDocument(Path('x.txt'), text, 'utf-8', [LyricLine(0,text)], 'zh'), 'zh', 'ignore')
    alignment = GlobalDPAligner().align(TranscriptionResult('zh','test',[], metadata={'audio_duration': 10}), parsed)
    ass = ASSKaraokeWriter()._line_to_ass_text(alignment.lines[0])
    assert re.sub(r'\{[^}]*\}', '', ass) == text


def test_nfkc_combining_characters_and_fullwidth_display():
    text = '你 e\u0301 ＡＩ ２０２４！'
    parsed = get_lyrics_parser('zh').parse(LyricsDocument(Path('x.txt'),text,'utf-8',[LyricLine(0,text)],'zh'),'zh','ignore')
    alignment = GlobalDPAligner().align(TranscriptionResult('zh','test',[],metadata={'audio_duration':10}), parsed)
    assert re.sub(r'\{[^}]*\}', '', ASSKaraokeWriter()._line_to_ass_text(alignment.lines[0])) == text


def test_min_gap_does_not_move_fast_matched_lyrics():
    transcription = make_transcription()
    for index, unit in enumerate(transcription.units):
        unit.start_time = 1 + index * .1
        unit.end_time = 1 + (index + 1) * .1
    alignment = GlobalDPAligner(min_gap=.5).align(transcription, make_parsed_lyrics())
    assert alignment.lines[1].start_time == pytest.approx(1.2)
    assert alignment.lines[1].aligned_units[0].start_time == pytest.approx(1.2)
