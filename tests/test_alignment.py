import unittest
import numpy as np

from pyroller.hybrid.config import AlignmentSettings as HybridSettings
from pyroller.hybrid.matching import (lyric_inventory, stitch_windows, discover_instances, select_instances)
from pyroller.hybrid.alignment import refine, select_onsets, choose_variants
from pyroller.hybrid._alignment import forced_path


def words(text, start=0., step=.4):
    return [dict(token=t, start=start+i*step, end=start+(i+1)*step, probability=.9,
                 window=0, segment=0, word=i, available_until=30.) for i, t in enumerate(text.split())]


def window(index, start, end, core, tokens):
    return dict(index=index, start=start, end=end, core_start=core,
        segments=[dict(words=[dict(word=w['token'], start=w['start']-start,
                                  end=w['end']-start, probability=.9) for w in tokens])])


class HybridTests(unittest.TestCase):
    def setUp(self): self.settings = HybridSettings()

    def selected(self, text, lyrics):
        stream = words(text)
        candidates = discover_instances(stream, lyric_inventory(lyrics), self.settings)
        paths = select_instances(stream, candidates, self.settings)
        return [candidates[i] for i in paths[0]['candidates']], paths, candidates

    def test_unordered_and_repeated_inventory(self):
        a = lyric_inventory('take me high\ncome on\ntake me high')
        b = lyric_inventory('come on\ntake me high')
        self.assertEqual(a, b)
        rows, _, _ = self.selected('take me high come on take me high', 'come on\ntake me high')
        self.assertEqual([r['text'] for r in rows], ['take me high','come on','take me high'])

    def test_reject_timed_lyrics(self):
        with self.assertRaises(ValueError): lyric_inventory('[00:01] take me high')

    def test_missing_word_matches_original(self):
        rows, _, _ = self.selected('i need to know', 'I need to know now')
        self.assertEqual([r['text'] for r in rows], ['I need to know now'])
        self.assertGreater(rows[0]['text_cost'], 0)

    def test_unknown_audio_remains_unknown(self):
        rows, _, _ = self.selected('thank you for watching please subscribe', 'take me high')
        self.assertEqual(rows, [])

    def test_grouping_alternatives_do_not_reuse_words(self):
        rows, paths, candidates = self.selected('come on take me high', 'come on\ntake me high\ncome on take me high')
        self.assertEqual(len(rows), 1)
        self.assertIn(2, [len(p['candidates']) for p in paths])
        for path in paths:
            used = []
            for i in path['candidates']:
                c = candidates[i]; used.extend(range(c['token_start'], c['token_end']))
            self.assertEqual(len(used), len(set(used)))

    def test_shared_seam_preserves_word_with_crossed_midpoints(self):
        a = [dict(token='god',start=154.8,end=155.4),dict(token='cannot',start=155.84,end=156.48),dict(token='take',start=156.5,end=157.)]
        b = [dict(token='cannot',start=155.30,end=156.48),dict(token='take',start=156.5,end=157.),dict(token='us',start=157.,end=157.8)]
        stream, seams = stitch_windows([window(0,128,158,130,a),window(1,154,184,156,b)], self.settings)
        self.assertEqual([w['token'] for w in stream], ['god','cannot','take','us'])
        self.assertEqual(seams[0]['method'], 'time_gated_word_anchor')

    def test_different_time_repetition_is_not_deduplicated(self):
        a = words('take me high', 1.)+words('take me high', 6.)
        b = words('take me high', 6.)+words('take me high', 11.)
        stream, _ = stitch_windows([window(0,0,10,0,a),window(1,5,15,8,b)], self.settings)
        self.assertEqual([w['token'] for w in stream], 'take me high take me high take me high'.split())

    def test_ctc_repeated_characters_need_blank(self):
        logp = np.log(np.array([[.99,.01],[.01,.99],[.99,.01],[.01,.99],[.99,.01]],dtype=np.float32))
        result = forced_path(logp,[1,1],0,6.)
        self.assertEqual(result['starts'], [1,3])
        self.assertEqual(result['ends'], [2,4])
        with self.assertRaises(RuntimeError): forced_path(logp[:2],[1,1],0,6.)

    def test_unknown_ctc_character_falls_back_without_deleting_identity(self):
        logp = np.log(np.array([[.99,.01]]*100,dtype=np.float32))
        row = dict(line_id='number',text='42',start=5.,end=5.4,token_start=0,token_end=1,matched_words=[])
        rows, reports = refine([row], [], logp, {'<pad>':0,'A':1},0.,.1,10.,self.settings)
        self.assertEqual(rows[0]['start'],5.)
        self.assertEqual(reports[0]['status'],'coarse_fallback')

    def test_onset_decisions_do_not_use_neighboring_end(self):
        choices = [
            [dict(start=1.,rank=0),dict(start=3.,rank=1)],
            [dict(start=2.,rank=0),dict(start=4.,rank=1)],
        ]
        self.assertEqual([c['start'] for c in select_onsets(choices)], [3.,4.])

    def test_conflicting_onsets_choose_legal_chain(self):
        choices = [[dict(start=1.,rank=0),dict(start=4.,rank=1)],
                   [dict(start=3.,rank=0),dict(start=2.,rank=2)]]
        self.assertEqual([c['start'] for c in select_onsets(choices)],[1.,2.])

    def test_missing_suffix_does_not_silently_rewrite_identity(self):
        stream=words('a b c d',1.)
        inventory=lyric_inventory('a b c\na b c d')
        candidates=discover_instances(stream,inventory,self.settings)
        selected=[candidates[i] for i in select_instances(stream,candidates,self.settings)[0]['candidates']]
        probabilities=np.full((60,5),.0025)
        probabilities[:,0]=.99
        for frame,label in [(10,1),(15,2),(20,3)]:
            probabilities[frame]=.0025;probabilities[frame,label]=.99
        output,reports=choose_variants(selected,inventory,stream,np.log(probabilities).astype(np.float32),
            {'<pad>':0,'A':1,'B':2,'C':3,'D':4},0.,.1,6.,self.settings)
        self.assertEqual(output[0]['text'],'a b c d')
        self.assertEqual(reports[0]['status'],'alternative_pending_review')

    def test_nonfinite_evidence_is_rejected(self):
        with self.assertRaises(ValueError): forced_path(np.array([[np.nan,0]],dtype=np.float32),[1],0,6.)

    def test_bad_unknown_cost_is_rejected(self):
        for cost in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):
                forced_path(np.zeros((10,2),dtype=np.float32),[1],0,cost)

    def test_end_displacement_does_not_veto_onset(self):
        probabilities=np.array([[.99,.01]]*100,dtype=np.float32)
        probabilities[20:60]=[.01,.99]
        observed=[dict(token='a',start=1.,end=4.)]
        row=dict(line_id='a',text='a',start=1.,end=4.,token_start=0,token_end=1,
                 matched_words=[dict(lyric_word=0,asr_word=0,cost=0.)])
        rows,_=refine([row],observed,np.log(probabilities),{'<pad>':0,'A':1},0.,.1,10.,self.settings)
        self.assertEqual(rows[0]['start'],2.)
        self.assertEqual(rows[0]['end'],6.)

    def test_large_anchored_shift_is_allowed(self):
        probabilities=np.array([[.99,.01]]*100,dtype=np.float32)
        probabilities[40:60]=[.01,.99]
        observed=[dict(token='a',start=1.,end=5.)]
        row=dict(line_id='a',text='a',start=1.,end=5.,token_start=0,token_end=1,
                 matched_words=[dict(lyric_word=0,asr_word=0,cost=0.)])
        rows,_=refine([row],observed,np.log(probabilities),{'<pad>':0,'A':1},0.,.1,10.,self.settings)
        self.assertEqual(rows[0]['start'],4.)
        self.assertTrue(rows[0]['onset_decision']['anchored_shift'])

    def test_cpp_path_matches_exhaustive_small_ctc(self):
        rng = np.random.default_rng(724)
        for labels in ([1,2], [1,1], [2,1,2]):
            raw = rng.uniform(.01,1.,(7,3)); logp = np.log(raw/raw.sum(-1,keepdims=True)).astype(np.float32)
            states = [0]
            for label in labels: states.extend([label,0])
            complete = []
            def visit(frame, state, score):
                if frame == len(logp):
                    if state >= len(states)-2: complete.append(score)
                    return
                for nxt in range(state,min(state+3,len(states))):
                    if nxt==state+2 and (states[nxt]==0 or states[nxt]==states[state]): continue
                    value = (max(float(logp[frame,0]),float(logp[frame,1:].max())-6.)
                             if nxt in (0,len(states)-1) else float(logp[frame,states[nxt]]))
                    visit(frame+1,nxt,score+value)
            visit(0,0,0.)
            result = forced_path(logp,labels,0,6.)
            self.assertAlmostEqual(max(complete),result['score'],places=5)
            self.assertTrue(all(a<b for a,b in zip(result['starts'],result['ends'])))


if __name__ == '__main__': unittest.main()
