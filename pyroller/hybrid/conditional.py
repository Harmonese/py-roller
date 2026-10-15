"""Timing conditional on the selected identities, not an existence classifier."""
from .joint import generate_proposals


def complete_chain(candidate_sets):
    states=[(0.,float('-inf'),[])]
    for candidates in candidate_sets:
        following=[]
        for c in candidates:
            eligible=[s for s in states if s[1]<=c['start']]
            if eligible:
                score,_,path=max(eligible,key=lambda s:s[0])
                following.append((score+c['gain'],c['end'],path+[c]))
        if not following:return None
        states=[];best=float('-inf')
        for state in sorted(following,key=lambda s:s[1]):
            if state[0]>best:states.append(state);best=state[0]
    return max(states,key=lambda s:s[0])


def refine_conditional(rows,words,logp,vocabulary,origin,step,duration,settings):
    candidates,reports=generate_proposals(rows,words,logp,vocabulary,origin,step,duration,settings)
    output=[dict(r,coarse_start=r['start'],coarse_end=r['end'],timing_status='coarse_unconfirmed',
        timing_method='whisper_matched_words',character_path=None,end_status='coarse') for r in rows]
    components=[]
    for i,report in enumerate(reports):
        if components and report['start']<max(reports[j]['end'] for j in components[-1]):components[-1].append(i)
        else:components.append([i])
    for ids in components:
        sets=[candidates[i] for i in ids]
        solution=complete_chain(sets)
        for i in ids:reports[i]['component_indices']=ids
        if solution is None:
            for i in ids:
                output[i]['timing_status']='coarse_no_complete_conditional_chain'
                reports[i]['decision']='no_complete_chain'
            continue
        score,_,selected=solution
        for local,(i,c) in enumerate(zip(ids,selected)):
            alternatives=list(sets)
            alternatives[local]=[x for x in sets[local] if abs(x['start']-c['start'])>.5]
            other=complete_chain(alternatives)
            margin=score-other[0] if other else None
            r=output[i]
            r.update(start=c['start'],end=c['end'],timing_method='conditional_ctc',
                timing_status='ctc_conditional_unverified',end_status='ctc_unverified',
                ctc_proposal=dict(start=c['start'],end=c['end']),word_evidence=c['word_evidence'],
                onset_decision=dict(accepted=True,conditional_on_text=True,
                    alternative_margin=margin,alternatives_exhaustive=False,gain=c['gain'],
                    edge_limited=min(c['start']-reports[i]['start'],reports[i]['end']-c['end'])<.1),
                character_path=dict(tokens=c['labels'],starts=[c['frame_start']+t for t in c['path']['starts']],
                    ends=[c['frame_start']+t for t in c['path']['ends']]),character_path_role='accepted_conditional_path')
            reports[i]['decision']='conditional_path'
    for i,r in enumerate(output):
        r['acoustic_end']=r['end']
        r['end']=max(r['start'],min(r['end'],output[i+1]['start'] if i+1<len(output) else duration))
        if r['end']!=r['acoustic_end']:r['end_status']='bounded_for_display'
    return output,reports
