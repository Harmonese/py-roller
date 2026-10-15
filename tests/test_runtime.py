import json
from pathlib import Path

import numpy as np
import pytest

from pyroller.config import EngineConfig
from pyroller import runtime
from pyroller.hybrid.core import align_evidence
from pyroller.hybrid.io import read_json, save_json
from pyroller.backends.catalog import evidence_directory


def test_right_context_clock():
    blocks = [dict(end_frame=200,audio_available_until=6),dict(end_frame=400,audio_available_until=10)]
    assert runtime.visible_frames(blocks,5.99)==0
    assert runtime.visible_frames(blocks,6)==200
    assert runtime.visible_frames(blocks,9)==200


def test_cache_identity_includes_audio_model_and_options(tmp_path):
    base=dict(audio='a',model='b',beam=5)
    directories={evidence_directory(tmp_path,base)}
    for key in base:
        directories.add(evidence_directory(tmp_path,dict(base,**{key:'different'})))
    assert len(directories)==4
    assert evidence_directory(tmp_path,base) in directories


@pytest.fixture
def fake_runtime(tmp_path,monkeypatch):
    audio=tmp_path/'audio.wav';audio.write_bytes(b'test audio')
    lyrics=tmp_path/'lyrics.txt';lyrics.write_text('a')
    logp=np.log(np.array([[.99,.01]]*300,dtype=np.float32))
    logp[20:24]=np.log([.01,.99])
    class CTC:
        inference_seconds=load_seconds=0.;cache_hits=0
        origin=0.;step=.1;vocabulary={'<pad>':0,'A':1};cache=tmp_path/'ctc'
        def __init__(self,*args):pass
        def __iter__(self):
            yield dict(start_frame=0,end_frame=200,logp=logp[:200],cached=False,audio_available_until=22)
            yield dict(start_frame=200,end_frame=300,logp=logp[200:],cached=False,audio_available_until=30)
        def close(self):pass
    class Whisper:
        inference_seconds=load_seconds=0.;cache_hits=0;cache=tmp_path/'whisper'
        def __init__(self,*args):pass
        def __iter__(self):
            yield dict(index=0,start=0,end=28,core_start=0,core_end=26,
                segments=[dict(words=[dict(word='a',start=1.8,end=2.6,probability=.9)])])
            yield dict(index=1,start=24,end=30,core_start=26,core_end=30,segments=[])
        def close(self):pass
    monkeypatch.setattr(runtime,'decode_audio',lambda path:np.zeros(30*16000,dtype=np.float32))
    monkeypatch.setattr(runtime,'resolve_model',lambda *args:tmp_path)
    monkeypatch.setattr(runtime,'CTCStream',CTC)
    monkeypatch.setattr(runtime,'WhisperStream',Whisper)
    return audio,lyrics


def test_runtime_snapshots_artifacts_and_no_future_ctc(fake_runtime,tmp_path,monkeypatch):
    audio,lyrics=fake_runtime
    seen=[];real=runtime.align_evidence
    def observe(*args,**kwargs):
        seen.append(len(args[2]))
        return real(*args,**kwargs)
    monkeypatch.setattr(runtime,'align_evidence',observe)
    out=tmp_path/'out'
    result=runtime.align(audio,lyrics,out)
    assert seen==[200,300]
    events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
    snapshots=[e for e in events if e['type']=='snapshot']
    assert [e['final'] for e in snapshots]==[False,True]
    assert snapshots[-1]['lines']==result['lines']
    assert events[-1]['type']=='completed'
    assert [e['sequence'] for e in events]==list(range(1,len(events)+1))
    assert all(e['protocol_version']==2 for e in events)
    assert read_json(out/'run.json')['status']=='complete'
    assert '[00:02.00] a' in (out/'result.lrc').read_text()
    with pytest.raises(FileExistsError):runtime.align(audio,lyrics,out)


def test_no_previews_same_final(fake_runtime,tmp_path):
    first=runtime.align(*fake_runtime,tmp_path/'a')
    second=runtime.align(*fake_runtime,tmp_path/'b',EngineConfig(previews=False))
    assert first['lines']==second['lines']
    events=[json.loads(x) for x in (tmp_path/'b/events.jsonl').read_text().splitlines()]
    assert sum(e['type']=='snapshot' for e in events)==1


def test_public_result_separate_from_internal_diagnostics(fake_runtime,tmp_path):
    result=runtime.align(*fake_runtime,tmp_path/'out',request_id='test-job')
    diagnostic=read_json(tmp_path/'out/result.json')
    assert diagnostic['settings']['matching_mode']=='robust'
    assert diagnostic['settings']['timing_mode']=='legacy'
    assert diagnostic['settings']['stitching_mode']=='evidence'
    assert diagnostic['settings']['scoring_mode']=='weighted'
    assert 'candidates' not in result
    assert all('character_path' not in r for r in result['lines'])
    assert result['request_id']=='test-job'
    json.dumps(result,allow_nan=False)
    assert read_json(tmp_path/'out/alignment.json')==result


def test_cli_defaults_and_no_experiment_flags(monkeypatch,tmp_path):
    from pyroller import cli
    seen=[]
    monkeypatch.setattr(cli,'align',lambda audio,lyrics,output,config,progress,**kwargs:seen.append(config))
    args=['align','song.wav','lyrics.txt','-o',str(tmp_path/'output')]
    assert cli.main(args)==0
    assert seen[-1].audio_mode=='mix'
    assert cli.main(args+['--audio-mode','vocals','--offline'])==0
    assert seen[-1].audio_mode=='vocals' and seen[-1].offline
    with pytest.raises(SystemExit): cli.main(args+['--timing-mode','joint'])


def test_cancellation_is_not_final_success(fake_runtime,tmp_path,monkeypatch):
    class Interrupted:
        def __init__(self,*args):pass
        def __iter__(self):raise KeyboardInterrupt()
        def close(self):pass
    monkeypatch.setattr(runtime,'CTCStream',Interrupted)
    with pytest.raises(KeyboardInterrupt):runtime.align(*fake_runtime,tmp_path/'out')
    assert read_json(tmp_path/'out/run.json')['status']=='cancelled'
    assert not (tmp_path/'out/result.json').exists()
    assert not any(json.loads(line).get('final') for line in (tmp_path/'out/events.jsonl').read_text().splitlines())
    events=[json.loads(line) for line in (tmp_path/'out/events.jsonl').read_text().splitlines()]
    assert events[-1]['type']=='cancelled'


def test_invalid_ctc_metadata():
    with pytest.raises(ValueError):
        align_evidence('a',[],np.zeros((10,2)),{'<pad>':0,'A':1},0.,0.,10.)


def test_fresh_numpy_asr_times_match_cached_json_times():
    window=dict(index=0,start=0.,end=4.,core_start=0.,core_end=4.,segments=[dict(words=[
        dict(word='a',start=np.float64(1.),end=np.float64(2.),probability=.9)])])
    logp=np.log(np.array([[.99,.01]]*40,dtype=np.float32));logp[12:16]=np.log([.01,.99])
    args=(logp,{'<pad>':0,'A':1},0.,.1,4.)
    fresh=align_evidence('a',[window],*args)
    cached=align_evidence('a',[json.loads(json.dumps(window))],*args)
    assert fresh['occurrences']==cached['occurrences']
    json.dumps(fresh,allow_nan=False)


@pytest.mark.parametrize('config',[dict(audio_mode='auto'),dict(ctc_device='cuda'),dict(cpu_threads=0)])
def test_configuration_validation(config):
    with pytest.raises(ValueError):EngineConfig(**config)
