import numpy as np
import pytest
from pyroller.hybrid._alignment import forced_path
from pyroller.hybrid.conditional import complete_chain,refine_conditional
from pyroller.hybrid.config import AlignmentSettings


def test_negative_scores_do_not_reject_selected_text():
    first=dict(start=1.,end=2.,gain=-40.)
    second=dict(start=3.,end=4.,gain=-50.)
    assert complete_chain([[first],[second]])[2]==[first,second]


def test_secondary_path_can_release_neighbor():
    long=dict(start=1.,end=5.,gain=-1.)
    short=dict(start=1.,end=2.,gain=-3.)
    next_line=dict(start=3.,end=4.,gain=-2.)
    assert complete_chain([[long,short],[next_line]])[2]==[short,next_line]
    assert complete_chain([[long],[next_line]]) is None


def test_dense_retains_positions_and_far_instance_does_not_change_proposals():
    x=np.tile(np.log([.99,.005,.005]),(200,1)).astype(np.float32)
    x[40:44]=np.log([.005,.99,.005]);x[150:154]=np.log([.005,.005,.99])
    rows=[dict(text='a',line_id='a',start=4.,end=4.5,token_start=0,token_end=1,matched_words=[]),
          dict(text='b',line_id='b',start=15.,end=15.5,token_start=1,token_end=2,matched_words=[])]
    args=([],x,{'<pad>':0,'A':1,'B':2},0.,.1,20.,AlignmentSettings(timing_mode='conditional'))
    one,reports=refine_conditional(rows[:1],*args)
    two,others=refine_conditional(rows,*args)
    assert one[0]['start']==two[0]['start']==4.
    assert reports[0]['candidates']==others[0]['candidates']
    assert len(reports[0]['candidates'])>2


def test_no_complete_chain_preserves_coarse_times():
    x=np.tile(np.log([.99,.01]),(400,1)).astype(np.float32)
    row=dict(text='a',line_id='a',start=2.,end=35.,token_start=0,token_end=1,matched_words=[])
    result,_=refine_conditional([row],[],x,{'<pad>':0,'A':1},0.,.1,40.,AlignmentSettings(timing_mode='conditional'))
    assert result[0]['start']==2. and result[0]['timing_status']=='coarse_no_complete_conditional_chain'


@pytest.mark.parametrize('labels',[[1],[1,2],[1,1]])
def test_final_end_bounds_against_exhaustive_paths(labels):
    rng=np.random.default_rng(91)
    x=rng.uniform(.1,1,(6,3));x=np.log(x/x.sum(-1,keepdims=True)).astype(np.float32)
    emissions=[0]
    for token in labels:emissions.extend([token,0])
    paths=[]
    def visit(t,state,score,last):
        if t==len(x):
            if state>=len(emissions)-2:paths.append((score,last))
            return
        for nxt in range(state,min(state+3,len(emissions))):
            if nxt==state+2 and (emissions[nxt]==0 or emissions[nxt]==emissions[state]):continue
            value=max(float(x[t,0]),float(x[t,1:].max())-6.) if nxt in (0,len(emissions)-1) else float(x[t,emissions[nxt]])
            visit(t+1,nxt,score+value,t+1 if nxt==len(emissions)-2 else last)
    visit(0,0,0.,0)
    for end in range(1,7):
        expected=max((score for score,last in paths if last==end),default=None)
        if expected is None:
            with pytest.raises(RuntimeError):forced_path(x,labels,0,6.,{},{},[end,end])
        else:
            result=forced_path(x,labels,0,6.,{},{},[end,end])
            assert result['score']==pytest.approx(expected)
            assert result['ends'][-1]==end
