"""Check release refactors against saved v3 evidence, without model inference."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pyroller.hybrid.core import align_evidence
from pyroller.hybrid.io import read_json, save_json
from pyroller.backends.catalog import resolve_model
from pyroller.config import EngineConfig
from pyroller.protocol import alignment_document


def main():
    ctc = resolve_model('ctc',EngineConfig(offline=True))
    vocabulary = read_json(ctc/'vocab.json')
    results=[]
    for name in ('whisper-v3-validation','whisper-v3-expanded'):
        root=ROOT/'py-roller-artifacts'/name
        paths = sorted(root.glob('*/*/v3/result.json'))
        if not paths:
            raise FileNotFoundError(f'Missing release replay dataset: {root}')
        for path in paths:
            run=read_json(path.parent/'run.json')
            old=read_json(path)
            ctc_cache=run.get('ctc_cache')
            if not ctc_cache: ctc_cache=read_json(Path(run['source_run']))['ctc_cache']
            logp=np.concatenate([np.load(p,allow_pickle=False) for p in sorted(Path(ctc_cache).glob('*.npy'))])
            if 'lyrics' in run:
                lyrics=Path(run['lyrics']).read_text(encoding='utf-8-sig')
                duration=run['duration']
            else:
                original=read_json(Path(run['source_run']))
                lyrics=Path(original['lyrics']).read_text(encoding='utf-8-sig')
                duration=original['duration']
            new=align_evidence(lyrics,read_json(path.parent/'raw-windows.json'),logp,vocabulary,.0125,.02,duration)
            new=json.loads(json.dumps(new,allow_nan=False))
            for key in old.keys()-{'metrics'}:
                assert new[key]==old[key],f'{path}: {key} changed'
            public=alignment_document(new,'release-check',duration)
            assert all(0<=r['start']<=r['end']<=duration for r in public['lines'])
            results.append(dict(sample=str(path.parent.relative_to(root)),dataset=name,identical=True,lines=len(public['lines'])))
            print('IDENTICAL',results[-1],flush=True)
    if len(results) != 10:
        raise ValueError(f'Expected 10 release replay arms, found {len(results)}')
    save_json(ROOT/'py-roller-artifacts/release-0.10.0/replay.json',results)


if __name__=='__main__':main()
