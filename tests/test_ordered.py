import json

import numpy as np
import pytest

from pyroller.config import EngineConfig
from pyroller.hybrid.config import AlignmentSettings
from pyroller.hybrid.core import align_evidence
from pyroller.hybrid.ordered import ordered_lines, align_words
from pyroller.protocol import alignment_document
from test_runtime import fake_runtime


def test_repeats_and_physical_line_identity():
    rows=ordered_lines('[Verse]\nHello world\n\nHello world')
    assert [r['input_line_index'] for r in rows]==[1,3]
    assert rows[0]['line_id']==rows[1]['line_id']
    assert rows[0]['occurrence_id']!=rows[1]['occurrence_id']


def test_global_mapping_missing_phrase_and_contraction():
    pairs,diag=align_words("we can't go stay here go home".split(),
                           'we can not go go home'.split())
    assert (5,4,0.) in pairs and (6,5,0.) in pairs
    assert not any(i in (3,4) for i,j,c in pairs)


def test_ambiguous_repeat_not_used_as_anchor():
    pairs,diag=align_words(['hello','hello'],['hello'])
    assert pairs==[] and diag['ambiguous_pairs']==2


def test_preview_does_not_force_future_text():
    pairs,diag=align_words(['hello','world','future'],['hello','world'],False)
    assert diag['lyric_prefix_words']==2
    assert len(pairs)==2


def test_env_and_explicit_override(monkeypatch):
    monkeypatch.delenv('PYROLLER_ALIGNMENT_MODE',raising=False)
    assert EngineConfig().alignment_mode=='ordered'
    assert AlignmentSettings().alignment_mode=='ordered'
    monkeypatch.setenv('PYROLLER_ALIGNMENT_MODE','ordered')
    assert EngineConfig().alignment_mode=='ordered'
    assert EngineConfig(alignment_mode='unordered').alignment_mode=='unordered'
    monkeypatch.setenv('PYROLLER_ALIGNMENT_MODE','typo')
    with pytest.raises(ValueError):EngineConfig()


def test_cli_override(monkeypatch,tmp_path):
    from pyroller import cli
    seen=[]
    monkeypatch.setenv('PYROLLER_ALIGNMENT_MODE','ordered')
    monkeypatch.setattr(cli,'align',lambda *args,**kwargs:seen.append(args[3]))
    args=['align','audio.wav','lyrics.txt','-o',str(tmp_path/'out')]
    assert cli.main(args)==0 and seen[-1].alignment_mode=='ordered'
    assert cli.main(args+['--alignment-mode','unordered'])==0
    assert seen[-1].alignment_mode=='unordered'


def test_ctc_recovers_asr_missing_line_and_preserves_unsupported():
    vocabulary={'<pad>':0,'A':1,'B':2,'C':3,'D':4}
    logp=np.full((100,5),.001);logp[:,0]=.996
    for index,label in [(10,1),(30,2),(50,3)]:
        logp[index:index+3]=.001;logp[index:index+3,label]=.996
    windows=[dict(index=0,start=0,end=10,core_start=0,core_end=10,segments=[dict(words=[
        dict(word='a',start=1,end=1.3),dict(word='c',start=5,end=5.3)])])]
    result=align_evidence('a\nb\nc\nd',windows,np.log(logp),vocabulary,0,.1,10,
                          settings=AlignmentSettings(alignment_mode='ordered'))
    assert [r['text'] for r in result['occurrences']]==['a','b','c']
    assert result['occurrences'][1]['start']==pytest.approx(3)
    assert [r['text'] for r in result['unaligned_lines']]==['d']
    doc=alignment_document(result,'test',10)
    assert doc['unaligned_lines'][0]['input_line_index']==3
    assert all(r['start'] is not None for r in doc['lines'])


def test_ordered_runtime_events_and_schema(fake_runtime,tmp_path):
    from pyroller.runtime import align
    from jsonschema import Draft202012Validator
    from pathlib import Path
    audio,lyrics=fake_runtime
    lyrics.write_text('a\nz')
    doc=align(audio,lyrics,tmp_path/'result',EngineConfig(alignment_mode='ordered'))
    schema=json.loads((Path(__file__).resolve().parents[1]/'docs/protocol-v2/alignment.schema.json').read_text())
    Draft202012Validator(schema).validate(doc)
    assert len(doc['lines'])+len(doc['unaligned_lines'])==2
    events=[json.loads(line) for line in (tmp_path/'result/events.jsonl').read_text().splitlines()]
    last=[e for e in events if e['type']=='snapshot'][-1]
    assert last['unaligned_lines']==doc['unaligned_lines']


def test_consecutive_missing_lines_share_one_ctc_path():
    vocabulary={'<pad>':0,'A':1,'B':2,'C':3}
    logp=np.full((100,4),.001);logp[:,0]=.997
    for index,label in [(10,1),(30,2),(50,2),(70,3)]:
        logp[index:index+3]=.001;logp[index:index+3,label]=.997
    windows=[dict(index=0,start=0,end=10,core_start=0,core_end=10,segments=[dict(words=[
        dict(word='a',start=1,end=1.3),dict(word='c',start=7,end=7.3)])])]
    result=align_evidence('a\nb\nb\nc',windows,np.log(logp),vocabulary,0,.1,10,
                          settings=AlignmentSettings(alignment_mode='ordered'))
    assert [r['start'] for r in result['occurrences']]==pytest.approx([1,3,5,7])
    assert not result['unaligned_lines']


def test_request_overrides_environment(monkeypatch,tmp_path):
    from pyroller.protocol import load_request
    monkeypatch.setenv('PYROLLER_ALIGNMENT_MODE','ordered')
    file=tmp_path/'request.json'
    request=dict(protocol_version=2,request_id='test',audio='a',lyrics='b',output='c')
    file.write_text(json.dumps(request))
    assert load_request(file).config.alignment_mode=='ordered'
    request['options']={'alignment_mode':'unordered'}
    file.write_text(json.dumps(request))
    assert load_request(file).config.alignment_mode=='unordered'


def test_word_lattice_budget():
    with pytest.raises(ValueError,match='million'):
        align_words(['a']*2000,['a']*2000)
