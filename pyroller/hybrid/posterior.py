"""Joint ordered CTC graph, summed onset support, explicit timing abstention.

The 90%-mass/1-second rule is a fixed experimental concentration criterion,
not a calibrated 90% probability that a lyric timestamp is correct.
"""
import numpy as np
from ._alignment import forced_path, onset_posteriors
from .alignment import encode, frame_slice, groups_for, select_onsets, word_evidence


def support_decision(weights, times, settings):
    p = np.asarray(weights, dtype=float)
    if len(p) != len(times) or not len(p) or not np.isfinite(p).all() or np.any(p < 0) or p.sum() <= 0:
        raise ValueError('Invalid onset support')
    p = p / p.sum()
    cumulative = np.r_[0., np.cumsum(p)]
    best = None
    for left in range(len(p)):
        right = int(np.searchsorted(cumulative, cumulative[left]+settings.posterior_mass, side='left'))
        if right > len(p): break
        if right <= left: continue
        candidate = (float(times[right-1]-times[left]), left, right)
        if best is None or candidate < best: best = candidate
    if best is None: raise ValueError('No support interval')
    width,left,right = best
    edge = min(times[left]-times[0], times[-1]-times[right-1]) < .1
    return dict(status='boundary_limited' if edge else 'diffuse_onset' if width > settings.posterior_max_width else 'concentrated',
        interval_start=float(times[left]), interval_end=float(times[right-1]),
        interval_frames=[left,right], interval_width=width,
        interval_mass=float(p[left:right].sum()),
        median=float(times[min(len(p)-1, int(np.searchsorted(cumulative[1:], .5)))]),
        score_kind='normalized_complete_path_mass_not_calibrated_accuracy')


def refine_posteriors(rows, words, logp, vocabulary, origin, step, duration, settings):
    output = [dict(r,coarse_start=r['start'],coarse_end=r['end'],timing_method='whisper_matched_words',
        timing_status='coarse_unconfirmed',end_status='coarse',character_path=None) for r in rows]
    options = [[dict(start=r['start'],rank=0.,method='whisper_matched_words')] for r in rows]
    reports = []
    for region in groups_for(rows,duration,settings):
        a,b = frame_slice(region['start'],region['end'],origin,step,len(logp))
        report = dict(region,frame_start=a,frame_end=b,onsets=[],profile_origin=origin+a*step,profile_step=step)
        try:
            if b<=a or region['end']-region['start']>settings.group_seconds:
                raise ValueError('region budget or empty audio')
            labels,offsets = [],[]
            for i in region['indices']:
                encoded,spans = encode(rows[i]['text'],vocabulary);begin=len(labels);labels.extend(encoded)
                offsets.append((i,begin,len(labels),[(begin+x,begin+y) for x,y in spans]))
            block = logp[a:b]
            profiles = onset_posteriors(block,labels,[x[1] for x in offsets],vocabulary['<pad>'],settings.unknown_cost)
            times = origin+(a+np.arange(b-a))*step
            bounds = {};decisions=[]
            for (i,begin,_,_),weights in zip(offsets,profiles):
                decision = support_decision(weights,times,settings)
                decisions.append(decision)
                output[i]['onset_decision'] = dict(decision)
                output[i]['timing_status'] = 'coarse_'+decision['status']
                report['onsets'].append(dict(decision,index=i,weights=weights.tolist()))
                if decision['status']=='concentrated': bounds[begin]=decision['interval_frames']
            if not bounds:
                report['status']='all_onsets_uncertain';reports.append(report);continue
            # Only concentrated intervals constrain the joint character path.
            # Uncertain Whisper times are display estimates, not fake CTC anchors.
            path = forced_path(block,labels,vocabulary['<pad>'],settings.unknown_cost,bounds)
            report.update(status='aligned',path=path,onset_bounds=bounds)
            for (i,begin,end,spans),decision in zip(offsets,decisions):
                evidence = word_evidence(path,labels,spans,block,a,origin,step)
                row = output[i]
                row.update(ctc_proposal=dict(start=evidence[0]['start'],end=evidence[-1]['end']),
                    word_evidence=evidence,character_path=dict(tokens=labels[begin:end],
                        starts=[a+t for t in path['starts'][begin:end]],ends=[a+t for t in path['ends'][begin:end]]),
                    character_path_role='joint_timing_proposal')
                if decision['status']=='concentrated':
                    options[i].append(dict(start=evidence[0]['start'],rank=decision['interval_mass'],method='posterior_ctc'))
        except (RuntimeError,ValueError) as exc:
            report.update(status='coarse_path_or_budget_failure',reason=str(exc))
            for i in region['indices']:
                output[i]['timing_status']='coarse_path_or_budget_failure'
        reports.append(report)
    for row,choice,choices in zip(output,select_onsets(options),options):
        row.update(start=choice['start'],timing_method=choice['method'])
        if choice['method']=='posterior_ctc':
            row.update(timing_status='ctc_posterior_unverified',end=row['ctc_proposal']['end'],end_status='ctc_unverified',
                character_path_role='accepted_joint_timing_proposal')
        elif len(choices)>1: row['timing_status']='coarse_monotonic_conflict'
        row['acoustic_end']=row['end']
    for i,row in enumerate(output):
        end=max(row['start'],min(row['end'],output[i+1]['start'] if i+1<len(output) else duration))
        if end!=row['end']:row['end_status']='bounded_for_display'
        row['end']=end
    return output,reports
