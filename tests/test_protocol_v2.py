import json
from pathlib import Path
import subprocess
import sys

import pytest

from pyroller import cli
from pyroller.protocol import EventStream, load_request
from test_runtime import fake_runtime


def request_file(tmp_path, **updates):
    value=dict(protocol_version=2, request_id='job-42', audio='audio.wav', lyrics='lyrics.txt',
               output='result', options=dict(offline=True))
    value.update(updates)
    path=tmp_path/'request.json'
    path.write_text(json.dumps(value),encoding='utf-8')
    return path


def test_relative_paths_and_types(tmp_path):
    request=load_request(request_file(tmp_path,options={'cache_dir':'cache','previews':False}))
    assert request.audio==tmp_path/'audio.wav'
    assert request.config.cache_dir==tmp_path/'cache'
    assert request.config.previews is False


@pytest.mark.parametrize('updates',[
    {'protocol_version':1}, {'protocol_version':True}, {'request_id':''},
    {'audio':None}, {'options':{'cpu_threads':True}}, {'options':{'cpu_threads':0}},
    {'options':{'offline':'false'}}, {'options':{'timing_mode':'joint'}}, {'unexpected':0},
    {'options':{'whisper_model':None}}, {'options':{'previews':1}},
])
def test_bad_requests(tmp_path,updates):
    with pytest.raises((ValueError,TypeError)):load_request(request_file(tmp_path,**updates))


def test_duplicate_keys(tmp_path):
    path=tmp_path/'request.json';path.write_text('{"protocol_version":2,"protocol_version":1}')
    with pytest.raises(ValueError,match='Duplicate'):load_request(path)


def test_stream_single_terminal():
    events=[];stream=EventStream('job',events.append)
    stream.emit(dict(type='started'));stream.emit(dict(type='failed',code='invalid_input',message='bad'))
    with pytest.raises(RuntimeError):stream.emit(dict(type='completed'))
    assert [e['sequence'] for e in events]==[1,2]


def test_event_log_failure_can_still_report_to_consumer():
    class FullDisk:
        def write(self,line):raise OSError('disk full')
        def close(self):pass
    events=[];stream=EventStream('job',events.append)
    stream.emit(dict(type='started'));stream.file=FullDisk()
    with pytest.raises(OSError):stream.emit(dict(type='progress',stage='ctc'))
    stream.emit(dict(type='failed',code='execution_failed',message='disk full'))
    assert [e['type'] for e in events]==['started','failed']
    assert [e['sequence'] for e in events]==[1,2]


def test_worker_stdout_contract(fake_runtime,tmp_path,capsys):
    path=request_file(tmp_path)
    assert cli.main(['run','--request',str(path)])==0
    captured=capsys.readouterr()
    events=[json.loads(line) for line in captured.out.splitlines()]
    assert events[0]['type']=='started' and events[-1]['type']=='completed'
    assert all(e['request_id']=='job-42' for e in events)
    assert sum(e['type'] in ('completed','failed','cancelled') for e in events)==1
    assert json.loads((tmp_path/'result/alignment.json').read_text())['lines']==[
        e for e in events if e['type']=='snapshot'][-1]['lines']
    disk=[json.loads(line) for line in (tmp_path/'result/events.jsonl').read_text().splitlines()]
    assert disk==events


def test_failure_before_output_is_jsonl(tmp_path,capsys):
    path=request_file(tmp_path)
    assert cli.main(['run','--request',str(path)])==1
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e['type'] for e in events]==['started','failed']
    assert events[-1]['request_id']=='job-42'
    assert events[-1]['code']=='file_not_found'
    assert not (tmp_path/'result').exists()


def test_invalid_request_has_terminal_event(tmp_path,capsys):
    path=request_file(tmp_path,protocol_version=1)
    assert cli.main(['run','--request',str(path)])==1
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [e['type'] for e in events]==['started','failed']
    assert events[-1]['code']=='invalid_input'


def test_cancel_worker(fake_runtime,tmp_path,capsys,monkeypatch):
    from pyroller import runtime
    class Interrupted:
        def __init__(self,*args):pass
        def __iter__(self):raise KeyboardInterrupt()
        def close(self):pass
    monkeypatch.setattr(runtime,'CTCStream',Interrupted)
    assert cli.main(['run','--request',str(request_file(tmp_path))])==130
    events=[json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]['type']=='cancelled'
    assert sum(e['type']=='cancelled' for e in events)==1


def test_cli_help_does_not_import_neural_runtime():
    code="from pyroller.cli import main; import sys; main(['models','list']); assert 'torch' not in sys.modules; assert 'faster_whisper' not in sys.modules"
    completed=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr


def test_default_model_is_pinned_v3():
    from pyroller.backends.catalog import MODELS
    assert MODELS['whisper'][:2]==('Systran/faster-whisper-large-v3','edaa852ec7e145841d8ffdb056a99866b5f0a478')


def test_separation_lock_is_exclusive_and_released(tmp_path):
    from pyroller.backends.separation import cache_lock
    with pytest.raises(KeyboardInterrupt):
        with cache_lock(tmp_path):
            with pytest.raises(RuntimeError,match='busy'):
                with cache_lock(tmp_path):pass
            raise KeyboardInterrupt()
    assert not (tmp_path/'working.lock').exists()
    with cache_lock(tmp_path):pass


def test_incomplete_auto_model_cache_is_not_accepted(tmp_path,monkeypatch):
    from pyroller.config import EngineConfig
    from pyroller.backends.catalog import local_model, MODELS
    monkeypatch.setattr(Path,'home',classmethod(lambda cls: tmp_path/'home'))
    config=EngineConfig(cache_dir=tmp_path)
    repo,revision,_=MODELS['whisper']
    folder=tmp_path/'models'/('models--'+repo.replace('/','--'))/'snapshots'/revision
    folder.mkdir(parents=True)
    (folder/'config.json').write_text('{}')
    assert local_model('whisper',config) is None
    for name in ('model.bin','tokenizer.json','vocabulary.json'):(folder/name).write_bytes(b'x')
    assert local_model('whisper',config)==folder


def test_existing_output_not_modified(fake_runtime,tmp_path):
    from pyroller.engine import align
    output=tmp_path/'existing';output.mkdir();marker=output/'keep';marker.write_text('unchanged')
    events=[]
    with pytest.raises(FileExistsError):align(*fake_runtime,output,on_event=events.append)
    assert events[-1]['code']=='output_exists'
    assert list(output.iterdir())==[marker] and marker.read_text()=='unchanged'


def test_schemas_match_request_events_and_result(fake_runtime,tmp_path):
    import jsonschema
    from referencing import Registry, Resource
    from pyroller.engine import align
    root=Path(__file__).resolve().parents[1]/'docs/protocol-v2'
    schemas={p.name:json.loads(p.read_text()) for p in root.glob('*.schema.json')}
    registry=Registry().with_resources((name,Resource.from_contents(value)) for name,value in schemas.items())
    for schema in schemas.values():jsonschema.Draft202012Validator.check_schema(schema)
    request=request_file(tmp_path)
    jsonschema.Draft202012Validator(schemas['request.schema.json']).validate(json.loads(request.read_text()))
    events=[]
    result=align(*fake_runtime,tmp_path/'out',on_event=events.append)
    jsonschema.Draft202012Validator(schemas['alignment.schema.json'],registry=registry).validate(result)
    validator=jsonschema.Draft202012Validator(schemas['event.schema.json'],registry=registry)
    for event in events:validator.validate(event)
    assert all(0<=r['start']<=r['end']<=result['duration'] for r in result['lines'])
