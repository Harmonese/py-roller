from copy import deepcopy

import pytest

from pyroller.aligner.global_dp_v1 import GlobalDPAligner
from pyroller.aligner.repetition import LineCandidate, select_best_candidate_path
from tests.factories import make_parsed_lyrics
from tests.factories import make_transcription


def candidate(line, position, start, end, score=.9):
    return LineCandidate(line, position, position, start, end, score, 1., 1., [])


@pytest.mark.parametrize('position, accepted', [(5, True), (20, False)])
def test_sparse_repair_respects_retained_neighbour(monkeypatch, position, accepted):
    """Selective application must not splice a later occurrence before a kept line."""
    import pyroller.aligner.global_dp_v1 as module
    aligner = GlobalDPAligner(repetition='few')
    lyrics = make_parsed_lyrics().lines
    assignments = [dict(lyric_idx=i, pos=p, end_pos=p, time=float(p), confidence=.1,
                        metadata={'coverage': .1}, method='global_dp', text=line.raw_text)
                   for i, (p, line) in enumerate(zip([0, 10], lyrics))]
    original = deepcopy(assignments)
    proposal = candidate(0, position, float(position), float(position + 1))
    monkeypatch.setattr(module, 'select_anchor_chain', lambda **kw: [])
    monkeypatch.setattr(aligner, '_find_candidates_for_lines', lambda **kw: ([[proposal], []], 1))
    monkeypatch.setattr(module, 'select_best_candidate_path', lambda **kw: [proposal, None])
    units = [{'start_time': float(i), 'end_time': float(i + 1)} for i in range(30)]
    aligner._repair_few_repetition_regions(assignments, lyrics, units, 0, 30)
    assert assignments[1] == original[1]
    assert assignments[0]['pos'] == (position if accepted else 0)
    aligner._ensure_monotonic(assignments, 0, 30, .5)


def test_skipped_line_does_not_reset_beam_time_boundary():
    first = candidate(0, 0, 0, 5)
    # Prefer this candidate on score unless the timing constraint survives.
    overlapping = candidate(2, 2, 3, 4, score=1.0)
    later = candidate(2, 3, 6, 7)
    path = select_best_candidate_path(candidates_by_line=[[first], [], [overlapping, later]],
                                      line_count=3, min_time=0, max_time=10)
    assert path == [first, None, later]


@pytest.mark.parametrize('mode', ['none', 'few', 'full'])
def test_similarity_cache_preserves_alignment(monkeypatch, mode):
    from pyroller.aligner.common import SequenceAlignmentSupport

    def without_ids(value):
        if isinstance(value, dict):
            return {key: without_ids(item) for key, item in value.items() if key != 'unit_id'}
        if isinstance(value, list):
            return [without_ids(item) for item in value]
        return value

    transcription, lyrics = make_transcription(), make_parsed_lyrics()
    cached = GlobalDPAligner(repetition=mode).align(transcription, lyrics)
    function = SequenceAlignmentSupport._symbol_similarity.__wrapped__
    monkeypatch.setattr(SequenceAlignmentSupport, '_symbol_similarity', staticmethod(function))
    uncached = GlobalDPAligner(repetition=mode).align(transcription, lyrics)
    assert without_ids(cached.to_dict()) == without_ids(uncached.to_dict())
