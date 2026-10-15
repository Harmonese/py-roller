from dataclasses import replace
import itertools

import numpy as np
import pytest

from pyroller.hybrid._alignment import forced_path, onset_profiles
from pyroller.hybrid.config import AlignmentSettings
from pyroller.hybrid.discovery import discover
from pyroller.hybrid.matching import lyric_inventory, select_instances, discover_instances
from pyroller.hybrid.text_distance import substitution, phrase_cost
from pyroller.hybrid.timing import profile_decision, refine_profiles


def word_stream(text, start=0.):
    return [dict(token=w,start=start+i*.4,end=start+(i+1)*.4) for i,w in enumerate(text.split())]


def selected(text, lyrics):
    words=word_stream(text)
    settings=AlignmentSettings(matching_mode='robust')
    candidates,diag=discover(words,lyric_inventory(lyrics),settings)
    path=select_instances(words,candidates,settings)[0]
    return [candidates[i] for i in path['candidates']],diag


def test_contraction_not_equivalence():
    rows,_=selected("i've got spirit i've got wings",'I got spirit, I got wings')
    assert len(rows)==1 and rows[0]['text_cost']==.5
    assert [op['kind'] for op in rows[0]['operations']].count('auxiliary_contraction')==2
    assert substitution("don't",'do')[0]>.25


@pytest.mark.parametrize('audio,lyrics',[("i have found you","I've found you"),
    ("i've found you","I have found you"),('in to the sky','Into the sky')])
def test_phrase_correspondence(audio,lyrics):
    rows,_=selected(audio,lyrics)
    assert len(rows)==1 and rows[0]['text']==lyrics
    assert any(op['kind'] in ('contraction_expansion','word_segmentation') for op in rows[0]['operations'])


def test_short_line_can_examine_extra_word():
    rows,_=selected("we'll all stay together","We'll stay together")
    assert len(rows)==1 and rows[0]['token_end']==4
    assert rows[0]['similarity']==.75
    assert any(op['kind']=='extra_asr_word' for op in rows[0]['operations'])


def test_dictionary_distance_does_not_rewrite_identity():
    assert substitution('sea','see')==(.25,'dictionary_homophone')
    rows,_=selected('see the light','Sea the light')
    assert rows[0]['text']=='Sea the light'
    assert substitution('xyzabc','pqrst')[0]==1


def test_diagnostics_keep_rejections_and_unknown_audio():
    rows,diag=selected('thank you for watching please subscribe','Take me high')
    assert rows==[]
    assert diag['counts']['similarity']>0
    assert any(c['rejection']=='similarity' for c in diag['rejected'])


def test_legacy_diagnostics_do_not_change_candidates():
    words=word_stream('thank you take me high goodbye');inventory=lyric_inventory('Take me high')
    settings=AlignmentSettings();diagnostics={}
    original=discover_instances(words,inventory,settings)
    traced=discover_instances(words,inventory,settings,diagnostics=diagnostics)
    assert original==traced
    assert diagnostics['counts']['similarity']>0


def test_repeated_instances_survive_candidate_caps():
    rows,_=selected('take me high take me high take me high','Take me high')
    assert len(rows)==3
    assert all(a['token_end']<=b['token_start'] for a,b in zip(rows,rows[1:]))


def enumerate_paths(logp,labels,blank=0,cost=6.):
    states=[blank]
    for token in labels:states.extend([token,blank])
    complete=[]
    def visit(t,s,score,starts):
        if t==len(logp):
            if s>=len(states)-2:complete.append((score,starts))
            return
        for nxt in range(s,min(s+3,len(states))):
            if nxt==s+2 and (states[nxt]==blank or states[nxt]==states[s]):continue
            new=dict(starts)
            if nxt%2 and nxt!=s:new[nxt//2]=t
            value=(max(float(logp[t,blank]),float(logp[t,1:].max())-cost)
                   if nxt in (0,len(states)-1) else float(logp[t,states[nxt]]))
            visit(t+1,nxt,score+value,new)
    visit(0,0,0.,{})
    return complete


@pytest.mark.parametrize('labels',[[1,2],[1,1],[2,1,2]])
def test_profiles_and_constrained_paths_match_exhaustive_enumeration(labels):
    rng=np.random.default_rng(346)
    x=rng.uniform(.01,1,(7,3));x=np.log(x/x.sum(-1,keepdims=True)).astype(np.float32)
    complete=enumerate_paths(x,labels)
    profiles=onset_profiles(x,labels,list(range(len(labels))))
    for token,t in itertools.product(range(len(labels)),range(len(x))):
        expected=max((score for score,starts in complete if starts[token]==t),default=-np.inf)
        assert profiles[token,t]==pytest.approx(expected)
        if np.isfinite(expected):
            path=forced_path(x,labels,0,6.,{token:[t,t+1]})
            assert path['score']==pytest.approx(expected)
            assert path['starts'][token]==t
        else:
            with pytest.raises(RuntimeError):forced_path(x,labels,0,6.,{token:[t,t+1]})


def test_equal_peak_abstains_without_claiming_probability():
    scores=np.full(100,-20.);scores[20]=scores[60]=0
    result=profile_decision(scores,2.,np.arange(100)*.1,AlignmentSettings())
    assert result['status']=='ambiguous_onset'
    assert result['separated_peak_margin']==0


def test_profile_allows_large_clear_shift_without_first_letter_argmax():
    scores=np.full(100,-20.);scores[60]=0
    result=profile_decision(scores,2.,np.arange(100)*.1,AlignmentSettings())
    assert result['status']=='ctc_preferred' and result['proposed_start']==6.
    assert result['score_gain_over_coarse']==20.


def test_complete_path_onset_beats_misleading_early_character():
    x=np.full((100,3),.005,dtype=np.float32);x[:,0]=.99
    # Early A has weak support; later A/B complete the actual phrase.
    x[15]=[.45,.50,.05]
    x[60]=[.005,.99,.005];x[65]=[.005,.005,.99]
    words=word_stream('ab',2.)
    rows=[dict(text='ab',line_id='ab',start=2.,end=6.6,token_start=0,token_end=1,
               matched_words=[dict(lyric_word=0,asr_word=0,cost=0.)])]
    output,report=refine_profiles(rows,words,np.log(x),{'<pad>':0,'A':1,'B':2},0.,.1,10.,
                                  AlignmentSettings(timing_mode='profile'))
    assert output[0]['start']==6.
    assert report[0]['status']=='aligned'


def test_profile_validation_and_budget():
    with pytest.raises(ValueError):onset_profiles(np.zeros((10,2),dtype=np.float32),[1],[1])
    with pytest.raises(ValueError):onset_profiles(np.zeros((10,2),dtype=np.float32),[1],[0],0,float('nan'))
    with pytest.raises(ValueError):forced_path(np.zeros((10,2),dtype=np.float32),[1],0,6.,{0:[5,2]})
    with pytest.raises(ValueError,match='budget'):
        onset_profiles(np.zeros((4001,2),dtype=np.float32),[1]*1000,[0])


@pytest.mark.parametrize('failure',['budget','connection'])
def test_profile_failures_preserve_coarse_result(monkeypatch,failure):
    from pyroller.hybrid import timing
    original=timing.forced_path
    def forced(*args):
        if len(args)>4:raise RuntimeError('no legal joint connection')
        return original(*args)
    def over_budget(*args):raise ValueError('over-budget onset profiles')
    if failure=='budget':monkeypatch.setattr(timing,'onset_profiles',over_budget)
    else:monkeypatch.setattr(timing,'forced_path',forced)
    x=np.log(np.array([[.9,.1]]*100,dtype=np.float32));x[20:25]=np.log([.01,.99])
    rows=[dict(text='a',line_id='a',start=2.,end=2.6,token_start=0,token_end=1)]
    output,reports=refine_profiles(rows,word_stream('a',2.),x,{'<pad>':0,'A':1},0.,.1,10.,AlignmentSettings())
    assert output[0]['start']==2. and output[0]['end']==2.6
    assert reports[0]['status']==('coarse_fallback' if failure=='budget' else 'connection_failed')
