import pytest

from pyroller.hybrid.config import AlignmentSettings
from pyroller.hybrid.discovery import discover
from pyroller.hybrid.matching import lyric_inventory, select_instances
from pyroller.hybrid.text_distance import word_weight


def candidates(text, lyrics, mode='weighted'):
    words = [dict(token=w,start=i*.5,end=(i+1)*.5) for i,w in enumerate(text.split())]
    settings = AlignmentSettings(scoring_mode=mode)
    rows, diagnostics = discover(words,lyric_inventory(lyrics),settings)
    selected = [rows[i] for i in select_instances(words,rows,settings)[0]['candidates']]
    return rows, selected, diagnostics


@pytest.mark.parametrize('mode',['consistent','weighted'])
def test_admission_and_competition_share_mass(mode):
    rows, _, _ = candidates('see the light in the sky','Sea the light\nLight in the sky',mode)
    assert rows
    for r in rows:
        assert r['similarity'] == pytest.approx(1-r['text_cost']/r['scoring_mass'])
        assert r['score'] == pytest.approx(r['scoring_mass']*(2*r['similarity']-1)-.5)


def test_extra_word_keeps_original_admission_normalization():
    _, rows, _ = candidates("we'll all stay together", "We'll stay together", 'consistent')
    assert len(rows)==1 and rows[0]['similarity']==.75
    assert rows[0]['score']==1.5


def test_discounting_errors_does_not_discount_correct_word_support():
    for mode in ('consistent','weighted'):
        rows, _, _ = candidates('narrow visions of astronomy','Narrow visions of autonomy',mode)
        full = next(r for r in rows if r['token_start']==0 and r['token_end']==4)
        assert full['similarity'] >= .75
        assert full['scoring_mass']==4


@pytest.mark.parametrize('text,lyrics',[
    ("what i've never seen before", "A world I've never seen before"),
    ('your silhouette in the sun','A silhouette into the sun'),
    ('the candle on the table','A candle at the table')])
def test_function_errors_allow_full_span_without_fake_anchors(text,lyrics):
    rows, selected, _ = candidates(text, lyrics)
    full = [r for r in rows if r['token_start']==0 and r['token_end']==len(text.split())]
    assert full and selected
    for r in full:
        for op in r['operations']:
            if op['kind']=='substitution': assert op['cost']==1
        assert any(op['cost']==1 and op['penalty']<1 for op in r['operations'])


def test_no_discount_for_negation_quantifiers_or_content():
    for w in ('not','never','no','all','world','sun','you'): assert word_weight(w)==1
    assert word_weight('the') < 1
    assert not candidates('the of at by','A silhouette into the sun')[1]
    assert not candidates('i can fly','I cannot fly')[1]


def test_edge_insertions_do_not_consume_neighbours():
    rows, _, _ = candidates('garbage take me high garbage','Take me high')
    assert rows and all(r['token_start']>=1 and r['token_end']<=4 for r in rows)


def test_repetitions_are_separate_and_unknown_words_stay_unused():
    _, rows, _ = candidates('take me high nonsense take me high','Take me high')
    assert len(rows)==2
    assert rows[0]['token_end']==3 and rows[1]['token_start']==4


def test_function_only_substitutions_do_not_pass():
    assert not candidates('of by','In the')[1]
