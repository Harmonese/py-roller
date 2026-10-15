"""Separate exact-text occurrence deficits from timing errors, without time gating."""
from collections import defaultdict
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from evaluation import read_lrc,normalize
from pyroller.hybrid.io import read_json,save_json,atomic_write


def pair_identity(pred,ref,pi,rj):
    # Maximize paired occurrence count first, then minimize total absolute error.
    # Sorted equal-text instances admit a minimum-cost noncrossing assignment.
    n,m=len(pi),len(rj);dp={(0,0):(0,0.)};parent={}
    for i in range(n+1):
        for j in range(m+1):
            cost=dp[i,j];choices=[]
            if i<n:choices.append((i+1,j,(cost[0]+1,cost[1]),'extra'))
            if j<m:choices.append((i,j+1,(cost[0]+1,cost[1]),'missing'))
            if i<n and j<m:
                choices.append((i+1,j+1,(cost[0],cost[1]+abs(pred[pi[i]]['start']-ref[rj[j]]['start'])),'pair'))
            for a,b,c,kind in choices:
                if (a,b) not in dp or c<dp[a,b]:dp[a,b]=c;parent[a,b]=(i,j,kind)
    pairs=[];missing=[];extra=[];i,j=n,m
    while i or j:
        a,b,kind=parent[i,j]
        if kind=='pair':pairs.append((pi[a],rj[b]))
        elif kind=='missing':missing.append(rj[b])
        else:extra.append(pi[a])
        i,j=a,b
    return pairs,missing,extra


def main():
    root=Path(sys.argv[1]);summary=[]
    for sid in ('E01','E06','E08'):
        refs=read_lrc(ROOT/'src-reconstruct/artifacts'/('extended-vocal-validation/E08/reference.lrc'
            if sid=='E08' else f'whisper-extended/references/{sid}.lrc'))
        for view in ('mix','vocals'):
            for mode in ('legacy','posterior'):
                data=read_json(root/mode/view/sid/'result.json');pred=data['occurrences']
                pg=defaultdict(list);rg=defaultdict(list)
                for i,r in enumerate(pred):pg[normalize(r['text'])].append(i)
                for j,r in enumerate(refs):rg[normalize(r['text'])].append(j)
                pairs=[];missing=[];extra=[]
                for identity in sorted(pg.keys()|rg.keys()):
                    p,m,e=pair_identity(pred,refs,sorted(pg[identity],key=lambda i:pred[i]['start']),
                        sorted(rg[identity],key=lambda j:refs[j]['start']))
                    pairs+=p;missing+=m;extra+=e
                paired=[dict(predicted_index=i,reference_index=j,text=refs[j]['text'],
                    reference_start=refs[j]['start'],predicted_start=pred[i]['start'],
                    absolute_error=abs(pred[i]['start']-refs[j]['start'])) for i,j in pairs]
                row=dict(song=sid,view=view,mode=mode,reference=len(refs),output=len(pred),
                    missing_count=len(missing),paired=len(paired),extra_count=len(extra),
                    within_half=sum(p['absolute_error']<=.5 for p in paired),
                    half_to_two=sum(.5<p['absolute_error']<=2 for p in paired),
                    over_two=sum(p['absolute_error']>2 for p in paired),
                    pairs=sorted(paired,key=lambda p:p['reference_start']),
                    missing=[refs[j] for j in sorted(missing)],extra=[pred[i] for i in sorted(extra)])
                assert row['reference']==row['missing_count']+row['within_half']+row['half_to_two']+row['over_two']
                summary.append(row)
    md=['# Text coverage and timing separately','',
        'Final lyric output, not raw ASR. Exact normalized whole-line text; repeated instances counted separately. No time gate. Maximum matched count, then minimum total absolute time error per identity. This measures text-count deficit, not proven audible occurrence recall; wrong repetitions can mask omissions. Existing references not re-audited.', '',
        '| Song/view | Mode | Reference | Missing text instances | 0.5–2s error | >2s error | <=0.5s | Extra |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        md.append(f"| {r['song']} {r['view']} | {r['mode']} | {r['reference']} | {r['missing_count']} | {r['half_to_two']} | {r['over_two']} | {r['within_half']} | {r['extra_count']} |")
    for r in summary:
        if r['mode']!='posterior':continue
        md+=['',f"## {r['song']} {r['view']}"]
        for x in r['missing']:md.append(f"- Missing: {x['start']:.2f} {x['text']}")
        for x in r['pairs']:
            if x['absolute_error']>2:md.append(f"- Timing: {x['reference_start']:.2f} -> {x['predicted_start']:.2f} ({x['absolute_error']:.2f}s): {x['text']}")
    save_json(root/'coverage-timing.json',summary);atomic_write(root/'coverage-timing.md','\n'.join(md)+'\n')
    print('\n'.join(md[:19]))


if __name__=='__main__':main()
