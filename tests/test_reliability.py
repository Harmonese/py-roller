import json
import os
import time
from pathlib import Path

import pytest

from pyroller.batch import BatchRunner, BatchTask
from pyroller.domain import PipelineRequest, TranscriptionResult
from pyroller.aligner.global_dp_v1 import GlobalDPAligner
from pyroller.pipeline import ComposablePipelineRunner
from pyroller.quality import AlignmentQualityError
from .factories import make_alignment_result, make_parsed_lyrics, make_transcription


def writer_task(tmp_path, index=0):
    source = tmp_path / f'{index}.json'
    make_alignment_result().save(source)
    request = PipelineRequest(stages=['w'], alignment_result_path=source,
                              intermediate_dir=tmp_path / 'work', output_roller_path=tmp_path / f'{index}.lrc')
    return BatchTask(index, str(index), request, [request.output_roller_path])


class Events:
    def __init__(self):
        self.events = []
    def event(self, kind, **payload):
        self.events.append((kind, payload))


def dead_worker(*args):
    os._exit(7)


def test_parallel_worker_death_returns_terminal_failures(monkeypatch, tmp_path):
    import pyroller.batch_runner as module
    monkeypatch.setattr(module, '_worker_loop', dead_worker)
    start = time.monotonic()
    result = BatchRunner().run([writer_task(tmp_path, i) for i in range(2)], jobs=2)
    assert time.monotonic() - start < 15
    assert result.failed == 2
    assert all(item.error['code'] == 'worker_exited' for item in result.results)


@pytest.mark.parametrize('jobs', [1, 2])
def test_batch_progress_and_completion_receipts(tmp_path, jobs):
    tasks = [writer_task(tmp_path, i) for i in range(2)]
    progress = Events()
    assert BatchRunner().run(tasks, jobs=jobs, progress_reporter=progress).completed == 2
    assert any(kind == 'stage_started' and payload['task_id'] == '0' and payload['stage'] == 'writer'
               for kind, payload in progress.events)
    assert BatchRunner().run(tasks, jobs=jobs, skip_existing=True).skipped == 2
    tasks[0].expected_outputs[0].write_text('')
    rerun = BatchRunner().run(tasks, jobs=jobs, skip_existing=True)
    assert (rerun.completed, rerun.skipped) == (1, 1)
    tasks[0].request.backend_config = {'writer': {'by_tag': 'changed'}}
    assert BatchRunner().run(tasks, skip_existing=True).completed == 1
    source = tasks[0].request.alignment_result_path
    data = json.loads(source.read_text()); data['payload']['lines'][0]['raw_text'] = 'changed'
    source.write_text(json.dumps(data))
    assert BatchRunner().run(tasks, skip_existing=True).completed == 1


def test_unsafe_output_alias_rejected_without_truncation(tmp_path):
    task = writer_task(tmp_path)
    source = task.request.alignment_result_path
    original = source.read_bytes()
    task.request.output_roller_path = source
    with pytest.raises(ValueError, match='collide'):
        ComposablePipelineRunner().run(task.request)
    assert source.read_bytes() == original


@pytest.mark.parametrize('jobs', [1, 2])
@pytest.mark.parametrize('collision', ['input', 'output'])
def test_batch_reserves_completion_receipt_paths(tmp_path, jobs, collision):
    from pyroller.batch_cache import receipt_path
    tasks = [writer_task(tmp_path, i) for i in range(2)]
    reserved = receipt_path(tasks[0])
    if collision == 'input':
        make_alignment_result().save(reserved)
        tasks[1].request.alignment_result_path = reserved
    else:
        reserved.write_text('existing output')
        tasks[1].request.output_roller_path = reserved
        tasks[1].expected_outputs = [reserved]
    original = reserved.read_bytes()
    with pytest.raises(ValueError, match='must be unique'):
        BatchRunner().run(tasks, jobs=jobs)
    assert reserved.read_bytes() == original
    assert not tasks[0].expected_outputs[0].exists()


def test_quality_accounts_for_unmatched_lines_and_strict_mode(tmp_path):
    parsed = make_parsed_lyrics()
    for unit in parsed.lines[1].units:
        unit.normalized_symbol = 'xxxxxx'
    alignment = GlobalDPAligner().align(make_transcription(), parsed)
    assert alignment.overall_confidence == .5
    assert alignment.report['quality']['interpolated_line_ratio'] == .5
    source = tmp_path / 'in.json'; alignment.save(source)
    output = tmp_path / 'out.ass'; output.write_text('old output')
    with pytest.raises(AlignmentQualityError):
        ComposablePipelineRunner().run(PipelineRequest(stages=['w'], alignment_result_path=source,
            output_roller_path=output, intermediate_dir=tmp_path / 'work',
            backend_config={'quality': {'mode': 'strict'}}))
    assert output.read_text() == 'old output'


def test_low_asr_confidence_reduces_matching_confidence():
    transcription = make_transcription()
    for unit in transcription.units:
        unit.confidence = .1
    result = GlobalDPAligner().align(transcription, make_parsed_lyrics())
    assert result.overall_confidence < .5
    assert result.report['quality']['needs_review']


@pytest.mark.parametrize('change', ['nan', 'reverse', 'index', 'type', 'language'])
def test_alignment_inputs_validate_semantics(change):
    transcription, parsed = make_transcription(), make_parsed_lyrics()
    if change == 'nan': transcription.units[0].start_time = float('nan')
    elif change == 'reverse': transcription.units[0].end_time = 0
    elif change == 'index': parsed.lines[0].line_index = 7
    elif change == 'type': transcription.units[0].unit_type = 'ipa_phone'
    elif change == 'language': transcription.language = 'en'
    with pytest.raises(ValueError):
        GlobalDPAligner().align(transcription, parsed)


def test_atomic_writer_failure_preserves_old_output(monkeypatch, tmp_path):
    from pyroller.writer.ass_karaoke import ASSKaraokeWriter
    output = tmp_path / 'out.ass'; output.write_text('old')
    writer = ASSKaraokeWriter()
    monkeypatch.setattr(writer, '_line_to_ass_text', lambda line: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError):
        writer.write(make_alignment_result(), output)
    assert output.read_text() == 'old'
    assert list(tmp_path.iterdir()) == [output]


def test_noise_gate_preserves_nonzero_tail_and_antiphase(tmp_path):
    import numpy as np
    import soundfile as sf
    from pyroller.domain import AudioArtifact
    from pyroller.filter.noise_gate import AdaptiveNoiseGateFilter
    source = tmp_path / 'input.wav'
    audio = np.stack([np.ones(5000)*.2, np.ones(5000)*-.2], axis=1)
    sf.write(source, audio, 16000, subtype='FLOAT')
    result = AdaptiveNoiseGateFilter().process(AudioArtifact('test','input','audio',path=source), tmp_path / 'out')
    actual, rate = sf.read(result.path)
    assert actual.shape == audio.shape
    assert np.allclose(actual, audio)


def test_filter_parameters_are_applied(tmp_path):
    from pyroller.filter.registry import build_filter_chain
    chain = build_filter_chain(['noise_gate'], tmp_path, {'steps': {'noise_gate': {'ramp_ms': 8, 'threshold_percentile': 30}}})
    assert chain.filters[0].ramp_ms == 8
    assert chain.filters[0].threshold_percentile == 30
    with pytest.raises(ValueError, match='Unknown'):
        build_filter_chain(['noise_gate'], tmp_path, {'steps': {'noise_gate': {'typo': 1}}})


def test_mixed_word_segment_timestamps_retain_both_segments():
    from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
    from pyroller.transcriber.unitizers.common import preferred_text_spans
    out = EngineOutput('zh','test','你好',[
        EngineSpan('s0','segment',0,1,text='你'), EngineSpan('w0','word',0,1,text='你',parent_span_id='s0'),
        EngineSpan('s1','segment',2,3,text='好')])
    assert [span.text for span in preferred_text_spans(out)] == ['你','好']


def test_chinese_normalization_shared_for_mixed_text():
    from pyroller.transcriber.unitizers.zh_pinyin_from_text import ZhPinyinFromTextUnitizer
    from pyroller.transcriber.engine_types import EngineSpan
    from pyroller.utils.text import segmented_zh_text_to_pinyin_units
    text = '我愛你 AI 2024'
    expected, _ = segmented_zh_text_to_pinyin_units(text)
    units = ZhPinyinFromTextUnitizer()._text_span_to_units(EngineSpan('word','word',1,2,text=text), language='zh', tone_mode='ignore')
    assert [unit.normalized_symbol for unit in units] == [item['normalized_symbol'] for item in expected]


def test_unsupported_and_assumed_languages_are_reported():
    from pyroller.utils.text import summarize_multilingual_routes, split_multilingual_text_segments
    from pyroller.language_diagnostics import route_warnings
    text = 'こんにちは 사랑해 bonjour'
    warnings = route_warnings(text, summarize_multilingual_routes(text))
    assert any(w['code'] == 'unsupported_text' and w['language'] == 'ja' for w in warnings)
    assert any(w['code'] == 'unsupported_text' and w['language'] == 'ko' for w in warnings)
    assert any(w['code'] == 'assumed_english' for w in warnings)
    assert split_multilingual_text_segments('bonjour', 'fr')[0]['language'] == 'fr'


def test_artifact_load_rejects_invalid_timestamps(tmp_path):
    from pyroller.utils.json import ArtifactLoadError
    source = tmp_path / 'timed.json'
    make_transcription().save(source)
    data = json.loads(source.read_text())
    data['payload']['units'][0]['start_time'] = -1
    source.write_text(json.dumps(data))
    with pytest.raises(ArtifactLoadError):
        TranscriptionResult.load(source)


def test_cuda_does_not_inject_transcriber_flags_into_writer(monkeypatch, tmp_path):
    from pyroller.cli.main import build_parser
    from pyroller.cli.runlike import build_request
    from pyroller.pipeline.validation import validate_pipeline_request
    from pyroller.pipeline.stages import resolve_execution_plan
    import pyroller.transcriber.device as device
    monkeypatch.setattr(device, 'auto_detect_transcriber_device', lambda: ('cuda','float16'))
    parser, _, _ = build_parser()
    args = parser.parse_args(['run','--stages','w','--alignment-result',str(tmp_path/'in.json'),
                              '--output-roller',str(tmp_path/'out.lrc')])
    request = build_request(args)
    validate_pipeline_request(request, resolve_execution_plan(request))
    assert not request.backend_config['transcriber']


def test_capabilities_matches_cli_and_backend_definitions():
    from pyroller.protocol import capabilities
    from pyroller.cli.main import build_parser
    from pyroller.config_contracts import QUALITY_OPTIONS
    result = capabilities()
    _, parser, _ = build_parser()
    assert {action.dest for action in parser._actions if action.dest != 'help'} == {item['name'] for item in result['options']}
    assert result['backend_schemas']['quality'] == QUALITY_OPTIONS
    assert 'latin_language' in result['backend_schemas']['parser']['mul_ipa']


def model_writer(store, model):
    from pyroller.transcriber.model_resolver import TranscriberModelResolver
    TranscriberModelResolver(backend='faster_whisper', language='zh', model_name=str(model), model_path=store).resolve(materialize=False)


def test_concurrent_model_index_updates_do_not_lose_entries(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    store = tmp_path / 'store'
    models = [tmp_path / f'model-{i}' for i in range(4)]
    for model in models: model.mkdir()
    workers = [ctx.Process(target=model_writer, args=(store, model)) for model in models]
    for worker in workers: worker.start()
    for worker in workers:
        worker.join(timeout=15)
        if worker.is_alive():
            worker.kill(); worker.join()
            pytest.fail('Model index worker timed out')
        assert worker.exitcode == 0
    data = json.loads((store / 'manifests' / 'transcriber-index.json').read_text())
    assert len(data['models']) == 4


def test_workspace_replaced_marker_cannot_authorize_cleanup(tmp_path):
    from pyroller.pipeline.workspace import RunWorkspace
    workspace = RunWorkspace(PipelineRequest(stages=['w'], intermediate_dir=tmp_path))
    keep = workspace.path / 'keep'; keep.write_text('data')
    workspace.marker.write_text('{}')
    assert not workspace.cleanup()
    assert keep.read_text() == 'data'


def test_batch_cannot_overwrite_another_tasks_input(tmp_path):
    first, second = writer_task(tmp_path, 0), writer_task(tmp_path, 1)
    first.expected_outputs = [second.request.alignment_result_path]
    first.request.output_roller_path = second.request.alignment_result_path
    with pytest.raises(ValueError, match='overwrite'):
        BatchRunner().run([first, second], jobs=2)


def test_cli_parser_to_aligner_to_ass(tmp_path):
    import subprocess
    import sys
    lyrics = tmp_path / 'lyrics.txt'; lyrics.write_text('你好，世界！')
    parsed = tmp_path / 'parsed.json'
    common = [sys.executable, '-m', 'pyroller.cli', 'run', '--language', 'zh',
              '--intermediate', str(tmp_path / 'work'), '--output-format', 'json', '--progress-format', 'jsonl']
    process = subprocess.run(common + ['--stages','p','--lyrics',str(lyrics),'--output-parsed-lyrics',str(parsed)],
                             text=True, capture_output=True, timeout=20)
    assert process.returncode == 0, process.stderr
    timed = tmp_path / 'timed.json'; make_transcription().save(timed)
    # Match the production parser's public unit type, not the old synthetic fixture alias.
    transcription = make_transcription()
    for unit in transcription.units: unit.unit_type = 'pinyin_syllable'
    transcription.save(timed)
    output = tmp_path / 'out.ass'
    process = subprocess.run(common + ['--stages','a,w','--timed-units',str(timed),'--parsed-lyrics',str(parsed),
                             '--output-roller',str(output),'--writer-backend','ass_karaoke'],
                             text=True, capture_output=True, timeout=20)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout.splitlines()[-1])
    assert report['status'] == 'ok' and report['quality']['coverage'] == 1
    assert '你' in output.read_text() and '世界' not in output.read_text()  # tags split characters
    assert 'ni' not in output.read_text().split('[Events]')[-1]


def test_partial_segment_confidence_is_log_probability():
    from pyroller.transcriber.unitizers.en_arpabet import EnArpabetUnitizer
    from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
    output = EngineOutput('en','faster_whisper','Hello',[EngineSpan('s','segment',0,1,text='Hello',confidence=-1)])
    result = EnArpabetUnitizer().adapt(output, language='en', tone_mode='ignore')
    assert 0 < result.units[0].confidence < 1


def test_noise_gate_gain_ramps_without_sample_jump(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from pyroller.filter.noise_gate import AdaptiveNoiseGateFilter
    from pyroller.domain import AudioArtifact
    gate = AdaptiveNoiseGateFilter(hangover_frames=0)
    source = tmp_path / 'constant.wav'
    sf.write(source, np.ones(10000) * .2, 16000, subtype='FLOAT')
    def selected_frames(np, db, threshold):
        keep = np.zeros(len(db), dtype=bool)
        keep[len(db)//2:] = True
        return keep
    monkeypatch.setattr(gate, '_build_keep_mask', selected_frames)
    result = gate.process(AudioArtifact('a', 'input', 'audio', path=source), tmp_path/'out')
    audio, _ = sf.read(result.path)
    gain = audio / .2
    assert gain.min() == 0 and gain.max() > .99
    assert np.max(np.abs(np.diff(gain))) < .1
    assert len(audio) == 10000 and gain[-1] > .99
    with pytest.raises(ValueError):
        AdaptiveNoiseGateFilter(hop_length=4096)


def test_retained_runs_use_unique_diagnostic_directories(tmp_path):
    from pyroller.engine import run_protocol_request
    task = writer_task(tmp_path)
    task.request.cleanup = 'never'
    first = run_protocol_request(task.request)
    second = run_protocol_request(task.request)
    assert first.request.intermediate_dir != second.request.intermediate_dir
    assert (first.request.intermediate_dir/'logs'/'run.log').exists()
    assert (second.request.intermediate_dir/'logs'/'run.log').exists()


def test_strict_quality_rejects_unmatched_long_region():
    from pyroller.quality import evaluate_quality
    result = GlobalDPAligner().align(TranscriptionResult('zh','test',[],metadata={'audio_duration':100}), make_parsed_lyrics())
    quality = evaluate_quality(result)
    assert quality['longest_unmatched_seconds'] > 10
    assert 'long_unmatched_region' in quality['reasons']


def test_dropped_digits_and_grapheme_fallback_are_reported():
    from pyroller.language_diagnostics import route_warnings, english_warnings
    from pyroller.utils.text import summarize_multilingual_routes, english_text_to_arpabet_units
    assert any(w['text'] == '2024' and w['code'] == 'unsupported_text'
               for w in route_warnings('2024', summarize_multilingual_routes('2024')))
    unknown = 'qzxxqzxx'
    assert any(w['code'] == 'approximate_pronunciation'
               for w in english_warnings(unknown, english_text_to_arpabet_units(unknown)))
