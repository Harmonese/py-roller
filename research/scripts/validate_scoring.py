"""Fixed scoring ablation on cached evidence; references enter evaluation only."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pyroller.hybrid.config import AlignmentSettings
from pyroller.hybrid.core import align_evidence
from pyroller.hybrid.io import read_json, save_json, atomic_write, sha256
from pyroller.hybrid.replay import load_evidence, write_replay
from pyroller.hybrid.text_distance import lexicon, phrase_cost, substitution, phones
from report_coverage_timing import pair_identity
from evaluation import read_lrc, normalize

SONGS = ('E01', 'E06', 'E08', 'E02', 'E05', 'E14', 'E07')
MODES = ('original', 'consistent', 'weighted')


def views(sid):
    # These three mixes have Whisper transcripts but no matching CTC cache.
    return ('vocals',) if sid in ('E02', 'E05', 'E07') else ('mix', 'vocals')


def evidence(sid, view):
    artifacts = ROOT/'src-reconstruct/artifacts'
    if sid == 'E01': old = 'blossom-two-branches-final/joint'
    elif sid in ('E02', 'E05', 'E14'): old = f'joint-three-song-validation/{sid}/joint'
    else: old = f'extended-vocal-validation/{sid}/joint'
    run = read_json(artifacts/old/'run.json')
    whisper = artifacts/('whisper-probe' if sid in ('E07', 'E08') else 'whisper-extended')/view/sid
    source = read_json(whisper/'result.json')
    metadata = dict(run['model_metadata'], audio_sha256=source['audio_sha256'])
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return load_evidence(whisper, run['lyrics'], artifacts/'evidence-cache'/key, run['model'])


def evaluate(rows, refs):
    pg, rg = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows): pg[normalize(r['text'])].append(i)
    for j, r in enumerate(refs): rg[normalize(r['text'])].append(j)
    pairs, missing, extra = [], [], []
    for key in sorted(pg.keys() | rg.keys()):
        p, m, e = pair_identity(rows, refs, pg[key], rg[key])
        pairs += p; missing += m; extra += e
    errors = [abs(rows[i]['start']-refs[j]['start']) for i, j in pairs]
    return dict(missing=[dict(index=j, **refs[j]) for j in missing],
        extra=[rows[i] for i in extra], over_two=sum(e > 2 for e in errors),
        within_half=sum(e <= .5 for e in errors),
        pairs=[dict(index=j, reference=refs[j], output=rows[i], error=rows[i]['start']-refs[j]['start']) for i,j in pairs])


def main():
    out = Path(sys.argv[1]); lexicon()
    save_json(out/'experiment.json', dict(reference_used=False, additional_model_seconds=0,
        parameter_scan=False, modes=MODES, songs=SONGS,
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'pyroller/hybrid').glob('*.py')}))
    for sid in SONGS:
        for view in views(sid):
            inputs, provenance = evidence(sid, view)
            for mode in MODES:
                folder = out/mode/view/sid
                if (folder/'result.json').exists(): continue
                for memo in (phrase_cost, substitution, phones): memo.cache_clear()
                result = align_evidence(*inputs, settings=AlignmentSettings(scoring_mode=mode, alignment_mode='unordered'))
                write_replay(result, provenance, folder)
                print(sid, view, mode, result['metrics'], flush=True)
    summaries = []; md = ['# Fixed scoring validation', '',
        'Unprompted cached ASR and CTC; evidence stitching and legacy timing in every arm. '
        'Scoring also affects stitching. References only used below; no parameter scan. '
        'Exact identity pairing has no time gate: a hallucinated repeat can mask a missing instance.', '',
        '| Song/view | Scoring | Missing | >2s | <=0.5s | Extra | Seconds |',
        '|---|---|---:|---:|---:|---:|---:|']
    for sid in SONGS:
        ref = ROOT/'src-reconstruct/artifacts'/(
            f'extended-vocal-validation/{sid}/reference.lrc' if sid in ('E07','E08') else f'whisper-extended/references/{sid}.lrc')
        refs = read_lrc(ref)
        for view in views(sid):
            for mode in MODES:
                result = read_json(out/mode/view/sid/'result.json'); rows = result['occurrences']
                assert all(0 <= r['start'] <= r['end'] for r in rows)
                assert all(a['start'] <= b['start'] and a['token_end'] <= b['token_start'] for a,b in zip(rows,rows[1:]))
                summary = dict(song=sid,view=view,mode=mode,metrics=result['metrics'],**evaluate(rows,refs))
                summaries.append(summary)
                md.append(f"| {sid}/{view} | {mode} | {len(summary['missing'])} | {summary['over_two']} | {summary['within_half']} | {len(summary['extra'])} | {result['metrics']['processing_seconds']:.3f} |")
    save_json(out/'summary.json', summaries)
    for s in summaries:
        if s['mode'] == 'original': continue
        base = next(b for b in summaries if (b['song'],b['view'],b['mode']) == (s['song'],s['view'],'original'))
        bm = {r['index']:r for r in base['missing']}; nm = {r['index']:r for r in s['missing']}
        md += ['',f"## {s['song']}/{s['view']} {s['mode']}"]
        for label, indices, lookup in [('Recovered identity',bm.keys()-nm.keys(),bm),('Lost identity',nm.keys()-bm.keys(),nm)]:
            for j in sorted(indices): md.append(f"- {label}: {lookup[j]['start']:.2f} {lookup[j]['text']}")
        for r in s['extra']: md.append(f"- Extra: {r['start']:.2f} {r['text']}")
    atomic_write(out/'summary.md','\n'.join(md)+'\n')


if __name__ == '__main__': main()
