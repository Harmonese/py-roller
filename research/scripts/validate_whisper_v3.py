"""Compare v2/v3 on fixed audio and CTC evidence; never prompt with references."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pyroller.hybrid.config import EngineConfig, AlignmentSettings
from pyroller.hybrid.core import align_evidence
from pyroller.backends.whisper import WhisperStream, WHISPER_OPTIONS
from pyroller.hybrid.io import read_json, save_json, atomic_write, decode_audio, sha256
from pyroller.hybrid.replay import write_replay
from validate_scoring import evidence, evaluate, views
from report_coverage_timing import read_lrc


def load(sid, view):
    if sid != 'E18':
        inputs, provenance = evidence(sid, view)
        reference = ROOT/'src-reconstruct/artifacts'/(
            f'extended-vocal-validation/{sid}/reference.lrc' if sid in ('E07', 'E08')
            else f'whisper-extended/references/{sid}.lrc')
        return inputs, provenance, reference
    folder = ROOT/'py-roller-artifacts'/('expanded-default-11' if view == 'mix' else 'expanded-vocals-11')
    run = read_json(folder/sid/'run.json')
    cache = Path(run['ctc_cache'])
    blocks = [np.load(p, allow_pickle=False) for p in sorted(cache.glob('*.npy'))]
    windows = [read_json(p) for p in sorted(Path(run['whisper_cache']).glob('window-*.json'))]
    inputs = (Path(run['lyrics']).read_text(), windows, np.concatenate(blocks),
        read_json(Path(run['ctc_model'])/'vocab.json'), run['frame_origin'], run['frame_step'], run['duration'])
    entry = next(e for e in read_json(folder/'experiment.json')['songs'] if e['id'] == sid)
    return inputs, dict(audio=run['selected_audio'], audio_sha256=run['selected_audio_sha256'],
        source_run=str(folder/sid/'run.json'), reference_used=False), Path(entry['reference'])


def summarize(out, rows):
    md = ['# Whisper large-v2 / large-v3', '',
        'Same cached CTC, default weighted/evidence/legacy pipeline. No prompts or parameter scan.',
        'Missing is identity-count deficit; <=2s gains/losses below guard against distant repeat matches.', '',
        '| Song | Audio | Model | Missing | Matched >0.5s | Extra | <=2s |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        md.append(f"| {r['song']} | {r['view']} | {r['model']} | {len(r['missing'])} | {sum(abs(p['error'])>.5 for p in r['pairs'])} | {len(r['extra'])} | {sum(abs(p['error'])<=2 for p in r['pairs'])} |")
    for r in rows:
        if r['model'] != 'v3': continue
        base = next(b for b in rows if (b['song'], b['view'], b['model']) == (r['song'], r['view'], 'v2'))
        before = {p['index']:p for p in base['pairs'] if abs(p['error'])<=2}
        after = {p['index']:p for p in r['pairs'] if abs(p['error'])<=2}
        r['near_gains'] = [after[i] for i in sorted(after.keys()-before.keys())]
        r['near_losses'] = [before[i] for i in sorted(before.keys()-after.keys())]
        md += ['', f"## {r['song']} / {r['view']}"]
        for label in ('near_gains', 'near_losses'):
            for p in r[label]:
                md.append(f"- {label}: {p['reference']['start']:.2f} {p['reference']['text']} (error {p['error']:+.2f}s)")
    save_json(out/'summary.json', rows)
    atomic_write(out/'summary.md', '\n'.join(md)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--songs', nargs='+', default=['E06', 'E08', 'E18'],
                        choices=['E01', 'E02', 'E05', 'E06', 'E07', 'E08', 'E14', 'E18'])
    args = parser.parse_args()
    out = args.output
    save_json(out/'experiment.json', dict(model=str(args.model.resolve()),
        revision='edaa852ec7e145841d8ffdb056a99866b5f0a478', options=WHISPER_OPTIONS,
        songs=args.songs, parameter_scan=False, reference_used_for_inference=False,
        source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'pyroller/hybrid').glob('*.py')}))
    rows = []
    for sid in args.songs:
        for view in views(sid):
            inputs, provenance, reference = load(sid, view)
            old_windows = inputs[1]
            for model in ('v2', 'v3'):
                folder = out/sid/view/model
                if (folder/'result.json').exists():
                    result = read_json(folder/'result.json')
                else:
                    windows = old_windows
                    run = dict(provenance, model=model)
                    if model == 'v3':
                        samples = decode_audio(provenance['audio'])
                        if sha256(provenance['audio']) != provenance['audio_sha256']:
                            raise ValueError('Audio changed')
                        stream = WhisperStream(samples, provenance['audio_sha256'], args.model, EngineConfig())
                        windows = []
                        try:
                            for window in stream:
                                expected = old_windows[window['index']]
                                for key in ('start', 'end', 'core_start', 'core_end'):
                                    if abs(window[key]-expected[key]) > 1e-4:
                                        raise ValueError('Window layout differs from baseline')
                                windows.append(window)
                                print('WINDOW', sid, view, window['index'], window['end'], flush=True)
                        finally:
                            stream.close()
                        run.update(whisper_cache=str(stream.cache), whisper_inference_seconds=stream.inference_seconds,
                            whisper_load_seconds=stream.load_seconds, cache_hits=stream.cache_hits)
                    result = align_evidence(inputs[0], windows, *inputs[2:], settings=AlignmentSettings())
                    write_replay(result, run, folder)
                    save_json(folder/'raw-windows.json', windows)
                    atomic_write(folder/'raw-transcript.txt', '\n\n'.join(
                        f"[{w['start']:.2f}-{w['end']:.2f}] " + ' '.join(s['text'] for s in w['segments']) for w in windows)+'\n')
                rows.append(dict(song=sid, view=view, model=model, metrics=result['metrics'],
                    **evaluate(result['occurrences'], read_lrc(reference))))
                summarize(out, rows)
                print('DONE', sid, view, model, len(rows[-1]['missing']), flush=True)


if __name__ == '__main__': main()
