import json
from types import SimpleNamespace
import numpy as np
import pytest
from pyroller.hybrid._alignment import forced_path
from pyroller.hybrid.config import AlignmentSettings,EngineConfig
from pyroller.hybrid.context import retrieve_context
from pyroller.hybrid.matching import lyric_inventory
from pyroller.hybrid.joint import select_paths,refine_joint
from test_p0_p2 import enumerate_paths


def test_entry_priors_match_exhaustive_complete_paths():
    rng=np.random.default_rng(45)
    x=rng.uniform(.1,1,(7,3));x=np.log(x/x.sum(-1,keepdims=True)).astype(np.float32)
    costs={0:[0.,1.,2.,0.,3.,2.,1.],1:[2.,1.,0.,1.,0.,1.,2.]}
    expected=max(score-sum(costs[i][t] for i,t in starts.items()) for score,starts in enumerate_paths(x,[1,2]))
    actual=forced_path(x,[1,2],0,6.,{},costs)
    assert actual['score']==pytest.approx(expected)
    with pytest.raises(ValueError):forced_path(x,[1],0,6.,{},{0:[0.]})


def test_interval_competition_and_unknown_retention():
    a=dict(start=1.,end=3.,gain=8.)
    b=dict(start=2.,end=4.,gain=9.)
    c=dict(start=4.,end=5.,gain=-1.)
    score,_,chosen=select_paths([[a],[b],[c]])
    assert score==9. and chosen==[None,b,None]
    score,_,chosen=select_paths([[a],[dict(start=3.,end=4.,gain=3.)]])
    assert score==11. and all(chosen)
    # An uncertain middle instance must not revert unrelated good timing.
    score,_,chosen=select_paths([[dict(start=1.,end=2.,gain=3.)],[],
        [dict(start=5.,end=6.,gain=4.)]],[1.2,3.,5.2])
    assert score==7. and chosen[0] and chosen[1] is None and chosen[2]


def test_add_far_instance_does_not_repartition_local_evidence():
    x=np.tile(np.log([.99,.005,.005]),(200,1)).astype(np.float32)
    x[40:44]=np.log([.005,.99,.005]);x[150:154]=np.log([.005,.005,.99])
    rows=[dict(text='a',line_id='a',start=4.,end=4.5,token_start=0,token_end=1,
               matched_words=[dict(lyric_word=0,asr_word=0,cost=0.)]),
          dict(text='b',line_id='b',start=15.,end=15.5,token_start=1,token_end=2,
               matched_words=[dict(lyric_word=0,asr_word=1,cost=0.)])]
    words=[dict(start=4.,end=4.5),dict(start=15.,end=15.5)]
    args=(words,x,{'<pad>':0,'A':1,'B':2},0.,.1,20.,AlignmentSettings(timing_mode='joint'))
    alone,ra=refine_joint(rows[:1],*args)
    both,rb=refine_joint(rows,*args)
    assert ra[0]==rb[0]
    assert alone[0]['start']==both[0]['start']==4.
    assert both[1]['start']==15.


def test_joint_budget_exit_preserves_text_and_coarse_time():
    row=dict(text='a',line_id='a',start=3.,end=34.,token_start=0,token_end=1,matched_words=[])
    x=np.tile(np.log([.99,.01]),(400,1)).astype(np.float32)
    output,reports=refine_joint([row],[],x,{'<pad>':0,'A':1},0.,.1,40.,AlignmentSettings(timing_mode='joint'))
    assert reports[0]['status']=='budget_or_path_failure'
    assert output[0]['text']=='a' and output[0]['start']==3.
    assert output[0]['timing_status']=='coarse_unconfirmed'


def test_background_replacement_is_independent_of_outer_padding():
    x=np.tile(np.log([.99,.005,.005]),(80,1)).astype(np.float32)
    x[30:33]=np.log([.005,.99,.005]);x[36:39]=np.log([.005,.005,.99])
    def gain(a,b):
        block=x[a:b]
        path=forced_path(block,[1,2])
        background=np.maximum(block[:,0],block[:,1:].max(axis=1)-6.)
        return path['score']-background.astype(np.float64).sum(),path['starts'][0]+a
    small,small_start=gain(20,50);large,large_start=gain(0,80)
    assert small==pytest.approx(large) and small_start==large_start==30


def test_multiple_word_priors_allow_shared_large_offset():
    x=np.tile(np.log([.99,.005,.005]),(100,1)).astype(np.float32)
    x[50:53]=np.log([.005,.99,.005]);x[60:63]=np.log([.005,.005,.99])
    row=dict(text='a b',line_id='ab',start=2.,end=3.4,token_start=0,token_end=2,
             matched_words=[dict(lyric_word=i,asr_word=i,cost=0.) for i in range(2)])
    words=[dict(start=2.,end=2.4),dict(start=3.,end=3.4)]
    output,_=refine_joint([row],words,x,{'<pad>':0,'A':1,'B':2},0.,.1,10.,AlignmentSettings(timing_mode='joint'))
    assert output[0]['start']==5.
    assert output[0]['timing_method']=='joint_ctc'


class Tokenizer:
    def encode(self,text):return SimpleNamespace(ids=text.split())


def window(text):
    return dict(index=0,start=0.,end=30.,segments=[dict(words=[dict(word=w,start=i*.3,end=(i+1)*.3)
        for i,w in enumerate(text.split())])])


def test_prompt_uses_plain_inventory_not_inventory_order_or_timestamps():
    a=lyric_inventory('I got spirit I got wings\nTake me high\nCome on')
    b=lyric_inventory('Come on\nTake me high\nI got spirit I got wings')
    first=retrieve_context(window("I've got spirit I've got wings"),a,Tokenizer())
    assert first==retrieve_context(window("I've got spirit I've got wings"),b,Tokenizer())
    assert first['selected'][0]['text']=='I got spirit I got wings'
    assert 'Take me high' not in first['prompt']
    assert first['reference_used'] is False
    assert retrieve_context(window(''),a,Tokenizer())['prompt']==''
    assert retrieve_context(window('take me high'),a,Tokenizer(),token_budget=2)['prompt']==''
    with pytest.raises(ValueError):lyric_inventory('[00:12.00] Take me high')


def test_experimental_configs_validate():
    assert EngineConfig(timing_mode='joint',whisper_context='lyrics').matching_mode=='robust'
    with pytest.raises(ValueError):EngineConfig(whisper_context='previous_reference')
    with pytest.raises(ValueError):AlignmentSettings(joint_stage='tune')


def test_context_single_pass_cache_and_no_prompt_fallback(tmp_path):
    from pyroller.backends.whisper import LyricContextPass
    from dataclasses import dataclass
    @dataclass
    class Segment:
        text:str
        words:list
    calls=[]
    class Model:
        def transcribe(self,audio,**options):
            calls.append(options)
            return iter([Segment('Take me high',[])]),SimpleNamespace(language='en')
    context=LyricContextPass.__new__(LyricContextPass)
    context.samples=np.zeros(30*16000,dtype=np.float32)
    context.inventory=lyric_inventory('Take me high')
    context.tokenizer=Tokenizer();context.cache=tmp_path;context.model=Model()
    context.cache_hits=0;context.inference_seconds=0.
    original=window('take me high')
    result=context.process(original)
    assert calls[0]['initial_prompt']=='Take me high'
    assert calls[0]['condition_on_previous_text'] is False
    assert context.process(original)==result and len(calls)==1 and context.cache_hits==1
    no_context=context.process(window('unrelated sounds'))
    assert no_context['lyric_context']['status']=='no_context_reuse_unprompted'
    assert len(calls)==1
    json.dumps(result,allow_nan=False)
