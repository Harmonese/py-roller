"""Instance-local paths scored against one shared background, then interval DP.

Every proposal covers its own fixed evidence support. Subtracting the background
score yields an additive replacement gain: disjoint proposals plus background
have the same objective regardless of how computation is batched.
"""
import numpy as np
from ._alignment import forced_path
from .alignment import encode,frame_slice,word_evidence


def select_paths(candidate_sets,coarse_starts=None):
    # Skipping an acoustically unsupported instance is allowed; its text remains
    # in the output with explicitly unconfirmed coarse timing.
    states=[(0.,-1.,[],float('-inf'))]
    for index,candidates in enumerate(candidate_sets):
        fallback=coarse_starts[index] if coarse_starts is not None else None
        expanded=[(score,end,path+[None],last if fallback is None else fallback)
                  for score,end,path,last in states if fallback is None or last<=fallback]
        for c in candidates:
            eligible=[s for s in states if max(s[1],s[3])<=c['start']]
            if eligible:
                score,_,path,_=max(eligible,key=lambda s:s[0])
                expanded.append((score+c['gain'],c['end'],path+[c],c['start']))
        expanded.sort(key=lambda s:(s[1],s[3],-s[0]))
        states=[]
        for s in expanded:
            if not any(p[3]<=s[3] and p[0]>=s[0] for p in states):states.append(s)
    return max(states,key=lambda s:s[0])[:3]


def time_entry_costs(row,words,spans,times,shift,settings):
    costs={}
    operations=row.get('operations') or [dict(lyric_start=m['lyric_word'],lyric_end=m['lyric_word']+1,
        asr_start=m['asr_word'],asr_end=m['asr_word']+1,cost=m['cost']) for m in row['matched_words']]
    for op in operations:
        if op['cost']>=1 or op['lyric_end']<=op['lyric_start'] or op['asr_end']<=op['asr_start']:continue
        observed=words[op['asr_start']:op['asr_end']]
        lo=min(w['start'] for w in observed)+shift
        hi=max(w['end'] for w in observed)+shift
        distance=np.maximum(np.maximum(lo-times,times-hi)-settings.anchor_tolerance,0.)
        # One phrase contributes once, never once per Cartesian word pair.
        costs[spans[op['lyric_start']][0]]=(np.minimum(distance,settings.joint_time_cost_cap)*(1-op['cost'])).tolist()
    return costs


def generate_proposals(rows,words,logp,vocabulary,origin,step,duration,settings):
    blank=vocabulary['<pad>']
    background=np.maximum(logp[:,blank],np.max(np.delete(logp,blank,axis=1),axis=1)-settings.unknown_cost)
    all_candidates=[];reports=[]
    for index,row in enumerate(rows):
        start=max(0.,row['start']-settings.joint_padding_seconds)
        end=min(duration,row['end']+settings.joint_padding_seconds)
        a,b=frame_slice(start,end,origin,step,len(logp))
        report=dict(indices=[index],start=start,end=end,candidates=[],status='searched')
        candidates=[]
        try:
            if end-start>settings.group_seconds:raise ValueError('instance evidence budget exceeded')
            labels,spans=encode(row['text'],vocabulary)
            times=origin+(a+np.arange(b-a))*step
            shifts=(0.,) if settings.joint_stage=='structure' else settings.joint_shift_seconds
            for shift in shifts:
                costs={}
                if settings.joint_stage!='structure':
                    costs=time_entry_costs(row,words,spans,times,shift,settings)
                bounds_list=[{}]
                # A coarse-near and an unrestricted explanation share evidence/objective.
                near=np.flatnonzero(np.abs(times-row['start'])<=.5)
                if len(near):bounds_list.append({0:[int(near[0]),int(near[-1])+1]})
                if settings.timing_mode=='conditional' and settings.conditional_candidates in ('dense','endpoints'):
                    # Fixed onset bins retain alternative placements before neighbor competition.
                    for low in np.arange(start,end,1.):
                        cells=np.flatnonzero((times>=low)&(times<low+1.))
                        if len(cells):bounds_list.append({0:[int(cells[0]),int(cells[-1])+1]})
                end_ranges=[[]]
                if settings.timing_mode=='conditional' and settings.conditional_candidates=='endpoints':
                    ends=origin+(a+np.arange(1,b-a+1))*step
                    for low in np.arange(max(start,row['end']-4.),end,2.):
                        cells=np.flatnonzero((ends>=low)&(ends<low+2.))
                        if len(cells):end_ranges.append([int(cells[0])+1,int(cells[-1])+1])
                for bounds,end_range in ((bounds,end_range) for bounds in bounds_list for end_range in end_ranges):
                    try:path=forced_path(logp[a:b],labels,blank,settings.unknown_cost,bounds,costs,end_range)
                    except RuntimeError:continue
                    evidence=word_evidence(path,labels,spans,logp[a:b],a,origin,step)
                    gain=path['score']-float(background[a:b].astype(np.float64).sum())-abs(shift)*settings.joint_shift_cost
                    candidate=dict(start=evidence[0]['start'],end=evidence[-1]['end'],gain=gain,shift=shift,
                        path=path,word_evidence=evidence,frame_start=a,labels=labels,
                        entry_prior=sum(costs[i][t] for i,t in enumerate(path['starts']) if i in costs))
                    candidates.append(candidate)
            unique={}
            for c in candidates:
                key=(round(c['start'],6),round(c['end'],6))
                if key not in unique or c['gain']>unique[key]['gain']:unique[key]=c
            candidates=sorted(unique.values(),key=lambda c:-c['gain'])
            report['candidates']=[{k:c[k] for k in ('start','end','gain','shift','entry_prior')} for c in candidates]
        except (ValueError,RuntimeError) as exc:report.update(status='budget_or_path_failure',reason=str(exc))
        all_candidates.append(candidates);reports.append(report)
    return all_candidates,reports


def refine_joint(rows,words,logp,vocabulary,origin,step,duration,settings):
    all_candidates,reports=generate_proposals(rows,words,logp,vocabulary,origin,step,duration,settings)
    coarse_starts=[r['start'] for r in rows]
    score,_,chosen=select_paths(all_candidates,coarse_starts)
    # Filter ambiguous explanations, then solve again with explicit coarse
    # display alternatives; do not revert unrelated instances on one conflict.
    filtered=all_candidates
    if settings.joint_stage=='decision':
        filtered=[]
        for i,candidates in enumerate(all_candidates):
            retained=[]
            for c in candidates:
                alternatives=list(all_candidates)
                alternatives[i]=[x for x in candidates if abs(x['start']-c['start'])>.5]
                other_score,_,_=select_paths(alternatives,coarse_starts)
                margin=score-other_score
                edge=min(c['start']-reports[i]['start'],reports[i]['end']-c['end'])<.1
                if margin>settings.joint_margin and not edge:retained.append(c)
            filtered.append(retained)
        _,_,chosen=select_paths(filtered,coarse_starts)
    output=[]
    for i,(row,c) in enumerate(zip(rows,chosen)):
        result=dict(row,coarse_start=row['start'],coarse_end=row['end'],timing_method='whisper_matched_words',
                    timing_status='coarse_unconfirmed',character_path=None,end_status='coarse')
        if c is not None:
            alternatives=list(all_candidates)
            alternatives[i]=[x for x in all_candidates[i] if abs(x['start']-c['start'])>.5]
            other_score,_,_=select_paths(alternatives,coarse_starts)
            margin=score-other_score
            edge=min(c['start']-reports[i]['start'],reports[i]['end']-c['end'])<.1
            result['onset_decision']=dict(gain=c['gain'],global_alternative_margin=margin,edge_limited=edge,
                shift=c['shift'],accepted=True,score_kind='background_replacement_with_time_prior_not_probability')
            result['ctc_proposal']=dict(start=c['start'],end=c['end'])
            result['character_path']=dict(tokens=c['labels'],starts=[c['frame_start']+x for x in c['path']['starts']],
                                          ends=[c['frame_start']+x for x in c['path']['ends']])
            result['character_path_role']='joint_proposal'
            result['word_evidence']=c['word_evidence']
            result.update(start=c['start'],end=c['end'],timing_method='joint_ctc',timing_status='ctc_joint_unverified',end_status='ctc_unverified')
        else:
            reason=('budget_or_path_failure' if not all_candidates[i] else
                    'background_preferred' if max(c['gain'] for c in all_candidates[i])<=0 else
                    'ambiguous_or_edge_limited' if not filtered[i] else 'joint_competition_lost')
            result['onset_decision']=dict(accepted=False,reason=reason)
        output.append(result)
    for i,r in enumerate(output):
        r['acoustic_end']=r['end']
        r['end']=max(r['start'],min(r['end'],output[i+1]['start'] if i+1<len(output) else duration))
        if r['end']!=r['acoustic_end']:r['end_status']='bounded_for_display'
    return output,reports
