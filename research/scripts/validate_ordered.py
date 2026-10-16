"""Replay two ordered transcripts. Reference timestamps never enter inference."""
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pyroller.hybrid.core import align_evidence
from pyroller.hybrid.config import AlignmentSettings
from pyroller.hybrid.io import read_json, save_json, atomic_write, lrc_text
from pyroller.backends.catalog import resolve_model
from pyroller.config import EngineConfig
from evaluation import read_lrc
from validate_scoring import evaluate


def main():
    out=ROOT/'py-roller-artifacts/ordered-validation'
    vocabulary=read_json(resolve_model('ctc',EngineConfig(offline=True))/'vocab.json')
    summary=[]
    for sid,view,dataset,reference in [
        ('E08','vocals','whisper-v3-validation','extended-vocal-validation/E08/reference.lrc'),
        ('E06','mix','whisper-v3-validation','whisper-extended/references/E06.lrc')]:
        source=ROOT/'py-roller-artifacts'/dataset/sid/view/'v3'
        run=read_json(source/'run.json');refs=read_lrc(ROOT/'src-reconstruct/artifacts'/reference)
        # Only text and sequence are passed to the engine, never the LRC times.
        text='\n'.join(r['text'] for r in refs)
        windows=read_json(source/'raw-windows.json')
        logp=np.concatenate([np.load(p,allow_pickle=False) for p in sorted(Path(run['ctc_cache']).glob('*.npy'))])
        for mode in ('unordered','ordered'):
            result=align_evidence(text,windows,logp,vocabulary,.0125,.02,run['duration'],
                settings=AlignmentSettings(alignment_mode=mode))
            folder=out/sid/mode
            save_json(folder/'result.json',result)
            atomic_write(folder/'result.lrc',lrc_text(result['occurrences']))
            atomic_write(folder/'input.txt',text+'\n')
            if mode=='ordered':
                located={r['input_line_index']:r for r in result['occurrences']}
                pairs=[dict(index=i,reference=refs[i],predicted=r,error=r['start']-refs[i]['start']) for i,r in located.items()]
                missing=[dict(index=i,**r) for i,r in enumerate(refs) if i not in located]
                evaluation=dict(pairs=pairs,missing=missing,extra=[])
            else:evaluation=evaluate(result['occurrences'],refs)
            row=dict(song=sid,view=view,mode=mode,reference_lines=len(refs),
                located=len(result['occurrences']),missing=len(evaluation['missing']),
                over_half_second=sum(abs(p['error'])>.5 for p in evaluation['pairs']),
                within_half_second=sum(abs(p['error'])<=.5 for p in evaluation['pairs']),
                over_two_seconds=sum(abs(p['error'])>2 for p in evaluation['pairs']),
                seconds=result['metrics']['processing_seconds'],evaluation=evaluation)
            summary.append(row);save_json(out/'summary.json',summary)
            print({k:v for k,v in row.items() if k!='evaluation'},flush=True)


if __name__=='__main__':main()
