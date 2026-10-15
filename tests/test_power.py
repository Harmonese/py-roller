from pyroller.hybrid.power_distance import power_phrase_cost
from pyroller.hybrid.text_distance import phrase_cost
from pyroller.hybrid.config import AlignmentSettings
from pyroller.hybrid.discovery import discover
from pyroller.hybrid.matching import lyric_inventory, select_instances


def test_exact_and_unknown():
    assert power_phrase_cost(('hello',), ('hello',)) == (0., 'exact')
    a, b = ('zzunknown',), ('hello',)
    assert power_phrase_cost(a, b) == phrase_cost(a, b)


def test_cross_word_homophone():
    cost, reason = power_phrase_cost(('ice', 'cream'), ('i', 'scream'))
    assert reason == 'power_phonetic_phrase'
    assert 0 <= cost < 1


def test_pipeline_candidates_repeat():
    words = [dict(token=w, start=i, end=i+.5) for i,w in enumerate('ice cream ice cream'.split())]
    candidates, diagnostics = discover(words, lyric_inventory('ice cream'), AlignmentSettings(scoring_mode='power'))
    best = select_instances(words, candidates, AlignmentSettings())[0]
    assert len(best['candidates']) == 2
    assert diagnostics['scoring_mode'] == 'power'
    assert AlignmentSettings().scoring_mode == 'weighted'
