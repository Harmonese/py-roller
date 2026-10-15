import itertools
import numpy as np
import pytest
from pyroller.hybrid._alignment import onset_posteriors
from pyroller.hybrid.config import AlignmentSettings, EngineConfig
from pyroller.hybrid.posterior import support_decision, refine_posteriors
from test_p0_p2 import enumerate_paths


@pytest.mark.parametrize('labels', [[1], [1,2], [1,1], [2,1,2]])
def test_posterior_matches_all_complete_paths(labels):
    rng=np.random.default_rng(918)
    x=rng.uniform(.01,1,(7,3));x=np.log(x/x.sum(-1,keepdims=True)).astype(np.float32)
    paths=enumerate_paths(x,labels)
    weights=np.exp(np.array([s for s,_ in paths])-max(s for s,_ in paths))
    weights/=weights.sum()
    actual=onset_posteriors(x,labels,list(range(len(labels))))
    assert actual.sum(axis=1)==pytest.approx(np.ones(len(labels)))
    for token,t in itertools.product(range(len(labels)),range(len(x))):
        expected=sum(w for w,(_,starts) in zip(weights,paths) if starts[token]==t)
        assert actual[token,t]==pytest.approx(expected,abs=1e-10)


def test_two_distant_peaks_abstain_and_compact_support_accepts():
    p=np.zeros(100);p[20]=.5;p[60]=.5;times=np.arange(100)*.1
    assert support_decision(p,times,AlignmentSettings())['status']=='diffuse_onset'
    p[:]=0;p[20:24]=.25
    d=support_decision(p,times,AlignmentSettings())
    assert d['status']=='concentrated' and d['interval_width']==pytest.approx(.3)


def test_support_on_window_edge_abstains():
    p=np.zeros(100);p[0]=1.
    assert support_decision(p,np.arange(100)*.1,AlignmentSettings())['status']=='boundary_limited'


@pytest.mark.parametrize('failure',['diffuse','budget','connection'])
def test_uncertain_or_failed_timing_keeps_whisper(monkeypatch,failure):
    from pyroller.hybrid import posterior
    def profile(x,*args):
        if failure=='budget':raise ValueError('budget')
        p=np.zeros((1,len(x)))
        if failure=='diffuse':p[0,5]=.5;p[0,-6]=.5
        else:p[0,20]=1.
        return p
    monkeypatch.setattr(posterior,'onset_posteriors',profile)
    if failure=='connection':
        def fail(*args):raise RuntimeError('connection')
        monkeypatch.setattr(posterior,'forced_path',fail)
    row=dict(text='a',start=2.,end=2.6,token_start=0,token_end=1)
    out,_=refine_posteriors([row],[],np.log(np.full((100,2),.5,dtype=np.float32)),
        {'<pad>':0,'A':1},0.,.1,10.,AlignmentSettings(timing_mode='posterior'))
    assert (out[0]['start'],out[0]['end'])==(2.,2.6)
    assert out[0]['timing_method']=='whisper_matched_words'


def test_impossible_repeated_labels_and_validation():
    with pytest.raises(RuntimeError):onset_posteriors(np.zeros((2,2),np.float32),[1,1],[0])
    with pytest.raises(ValueError):onset_posteriors(np.zeros((3,2),np.float32),[1],[1])
    with pytest.raises(ValueError):onset_posteriors(np.full((3,2),np.nan,np.float32),[1],[0])
    with pytest.raises(ValueError):onset_posteriors(np.zeros((4001,2),np.float32),[1]*1000,[0])
    with pytest.raises(ValueError):AlignmentSettings(posterior_mass=1.)
    assert EngineConfig(timing_mode='posterior').timing_mode=='posterior'
