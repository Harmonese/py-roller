"""Reference-free inference from acoustic evidence and an untimed inventory."""
from dataclasses import asdict
import time
import numpy as np

from .alignment import choose_variants,refine
from .config import AlignmentSettings
from .matching import lyric_inventory,stitch_windows,discover_instances,select_instances


def align_evidence(lyrics,windows,logp,vocabulary,origin,step,duration,settings=None,final=True):
    settings=settings or AlignmentSettings();tick=time.perf_counter()
    if (logp.ndim!=2 or not len(logp) or not np.isfinite(logp).all()
        or not np.isfinite([origin,step,duration]).all() or step<=0 or duration<=0
        or not 0<=vocabulary.get('<pad>',-1)<logp.shape[1]):
        raise ValueError('Invalid CTC evidence or timing metadata')
    inventory=lyric_inventory(lyrics)
    if settings.stitching_mode=='evidence':
        from .stitching import stitch_evidence
        words,seams=stitch_evidence(windows,inventory,settings)
    else:
        words,seams=stitch_windows(windows,settings)
    stitched=time.perf_counter()
    if settings.matching_mode == 'robust':
        from .discovery import discover
        candidates, discovery = discover(words,inventory,settings)
    else:
        discovery={}
        candidates=discover_instances(words,inventory,settings,diagnostics=discovery)
    paths=select_instances(words,candidates,settings)
    selected=[dict(candidates[i]) for i in paths[0]['candidates']];matched=time.perf_counter()
    chosen=set(paths[0]['candidates'])
    decisions=[]
    for i,candidate in enumerate(candidates):
        conflicts=[j for j in chosen if max(candidate['token_start'],candidates[j]['token_start'])<
                   min(candidate['token_end'],candidates[j]['token_end'])]
        decisions.append(dict(candidate=i,status='selected' if i in chosen else 'lost_text_competition',
                              conflicting_selected=sorted(conflicts)))
    coarse=[dict(r) for r in selected]
    selected,variants=choose_variants(selected,inventory,words,logp,vocabulary,origin,step,duration,settings)
    for i,row in enumerate(selected):
        row.setdefault('content_status','text_matched_unverified')
        row.update(occurrence_id=f'occ-{i+1:06d}',status='needs_review' if final else 'tentative',finalized=final)
    if settings.timing_mode == 'posterior':
        from .posterior import refine_posteriors
        rows,groups=refine_posteriors(selected,words,logp,vocabulary,origin,step,duration,settings)
    elif settings.timing_mode == 'conditional':
        from .conditional import refine_conditional
        rows,groups=refine_conditional(selected,words,logp,vocabulary,origin,step,duration,settings)
    elif settings.timing_mode == 'joint':
        from .joint import refine_joint
        rows,groups=refine_joint(selected,words,logp,vocabulary,origin,step,duration,settings)
    elif settings.timing_mode == 'profile':
        from .timing import refine_profiles
        rows,groups=refine_profiles(selected,words,logp,vocabulary,origin,step,duration,settings)
    else:
        rows,groups=refine(selected,words,logp,vocabulary,origin,step,duration,settings)
    finished=time.perf_counter()
    used={i for row in rows for i in range(row['token_start'],row['token_end'])}
    return dict(schema_version=1,engine='whisper_ctc',occurrences=rows,coarse_occurrences=coarse,
        inventory=inventory,stitched_words=words,seams=seams,candidates=candidates,alternatives=paths,
        variant_decisions=variants,timing_groups=groups,discovery_diagnostics=discovery,
        candidate_decisions=decisions,
        unmatched_words=[dict(index=i,**word) for i,word in enumerate(words) if i not in used],
        settings=asdict(settings),reference_used=False,reference_order_used=False,
        metrics=dict(stitch_seconds=stitched-tick,matching_seconds=matched-stitched,
                     alignment_seconds=finished-matched,processing_seconds=finished-tick))
