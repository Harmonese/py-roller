"""P2: same-evidence complete-path onset comparisons, with abstention.

Profiles are maximum COMPLETE path scores conditional on each onset, not
probabilities. No cross-model confidence or absolute scores are combined.
"""
import numpy as np

from ._alignment import forced_path, onset_profiles
from .alignment import encode, frame_slice, groups_for, select_onsets, word_evidence


def profile_decision(scores, coarse, absolute_times, settings):
    scores = np.asarray(scores)
    best_index = int(np.argmax(scores))
    best = float(scores[best_index]); start = float(absolute_times[best_index])
    radius = settings.profile_time_tolerance
    near = np.abs(absolute_times-coarse) <= radius
    outside = np.abs(absolute_times-start) > radius
    def maximum(mask):
        values = scores[mask & np.isfinite(scores)]
        return float(values.max()) if len(values) else None
    coarse_score, competing = maximum(near), maximum(outside)
    margin = best-competing if competing is not None else None
    plausible = absolute_times[np.isfinite(scores) & (scores >= best-settings.profile_score_tolerance)]
    at_edge = min(start-absolute_times[0], absolute_times[-1]-start) < .1
    distinct = margin is None or margin > settings.profile_score_tolerance
    status = ('boundary_limited' if at_edge else 'ambiguous_onset' if not distinct else 'ctc_preferred')
    return dict(status=status, proposed_start=start, best_score=best,
        coarse_start=coarse, coarse_band_score=coarse_score,
        score_gain_over_coarse=best-coarse_score if coarse_score is not None else None,
        competing_score=competing, separated_peak_margin=margin,
        plausible_start=float(plausible.min()), plausible_end=float(plausible.max()),
        coarse_band_has_legal_path=coarse_score is not None,
        score_kind='complete_path_viterbi_max_marginal_not_probability')


def refine_profiles(rows, words, logp, vocabulary, origin, step, duration, settings):
    output = [dict(row, coarse_start=row['start'], coarse_end=row['end'],
        timing_status='coarse_unverified', timing_method='whisper_matched_words',
        end_status='coarse', character_path=None) for row in rows]
    options = [[dict(start=row['start'],rank=0.,method='whisper_matched_words')] for row in rows]
    reports = []
    for region in groups_for(rows,duration,settings):
        a,b = frame_slice(region['start'],region['end'],origin,step,len(logp))
        block = logp[a:b]
        report = dict(region,frame_start=a,frame_end=b,score_source='same_group_evidence',onsets=[],
                      profile_origin=origin+a*step,profile_step=step)
        try:
            if b<=a or region['end']-region['start']>settings.group_seconds:
                raise ValueError('region budget or empty audio')
            labels, offsets = [], []
            for index in region['indices']:
                encoded, spans = encode(rows[index]['text'],vocabulary)
                start = len(labels); labels.extend(encoded)
                offsets.append((index,start,len(labels),[(start+x,start+y) for x,y in spans]))
            path = forced_path(block,labels,vocabulary['<pad>'],settings.unknown_cost)
            probes = [begin for _,begin,_,_ in offsets]
            profiles = onset_profiles(block,labels,probes,vocabulary['<pad>'],settings.unknown_cost)
            times = origin+(a+np.arange(len(block)))*step
            decisions = []
            for (index,begin,end,spans),profile in zip(offsets,profiles):
                decision = profile_decision(profile,rows[index]['start'],times,settings)
                decision['index'] = index
                decision['scores'] = [float(value) if np.isfinite(value) else None for value in profile]
                report['onsets'].append(decision)
                decisions.append(decision)
            # Accepted proposals come from ONE joint path, not independent maxima.
            # For an ambiguous line, constrain its onset near the coarse estimate
            # and reconstruct the whole group to check left/right feasibility.
            bounds = {}
            for (_,begin,_,_),decision in zip(offsets,decisions):
                center = (decision['proposed_start'] if decision['status']=='ctc_preferred'
                          else decision['coarse_start'])
                indices = np.flatnonzero(np.abs(times-center)<=settings.profile_time_tolerance)
                if not len(indices): raise RuntimeError('coarse onset outside local evidence')
                bounds[begin] = [int(indices[0]),int(indices[-1])+1]
            try:
                joint = forced_path(block,labels,vocabulary['<pad>'],settings.unknown_cost,bounds)
            except (RuntimeError,ValueError) as exc:
                report.update(status='connection_failed',reason=str(exc),unconstrained_path=path)
                for (index,_,_,_),decision in zip(offsets,decisions):
                    output[index].update(timing_status='coarse_connection_failure',
                        onset_decision={k:v for k,v in decision.items() if k!='scores'})
                reports.append(report)
                continue
            report.update(status='aligned',path=joint,unconstrained_score=path['score'],
                          constrained_score=joint['score'],onset_bounds=bounds)
            for (index,begin,end,spans),decision in zip(offsets,decisions):
                row = output[index]
                evidence = word_evidence(joint,labels,spans,block,a,origin,step)
                start,end_time = evidence[0]['start'],evidence[-1]['end']
                row.update(ctc_proposal=dict(start=start,end=end_time),word_evidence=evidence,
                    onset_decision={k:v for k,v in decision.items() if k!='scores'},
                    character_path=dict(tokens=labels[begin:end],starts=[a+x for x in joint['starts'][begin:end]],
                                        ends=[a+x for x in joint['ends'][begin:end]]),
                    character_path_role='timing_proposal')
                if decision['status']=='ctc_preferred':
                    # Both choices were assessed on complete paths; no first-letter
                    # argmax gate and no maximum allowed displacement from Whisper.
                    margin=decision['separated_peak_margin']
                    options[index].append(dict(start=start,rank=min(margin if margin is not None else 10.,10.),
                        method='profile_ctc'))
                else: row['timing_status']='coarse_'+decision['status']
                if evidence[-1]['character_support']>=.5:
                    row.update(end=end_time,end_status='ctc_unverified')
        except (ValueError,RuntimeError) as exc:
            report.update(status='coarse_fallback',reason=str(exc))
            for index in region['indices']:
                output[index].update(timing_status='coarse_path_or_budget_failure',
                                     onset_decision=dict(status='path_or_budget_failure',reason=str(exc)))
        reports.append(report)
    chosen = select_onsets(options)
    for row,choice,choices in zip(output,chosen,options):
        row.update(start=choice['start'],timing_method=choice['method'])
        if choice['method']=='profile_ctc': row['timing_status']='ctc_profile_unverified'
        elif len(choices)>1:
            row['timing_status']='coarse_neighbor_conflict'
            row['onset_decision']['connection_status']='monotonic_chain_rejected'
        row['acoustic_end']=row['end']
    for i,row in enumerate(output):
        limit=output[i+1]['start'] if i+1<len(output) else duration
        end=max(row['start'],min(row['end'],limit))
        if end!=row['end']:row['end_status']='bounded_for_display'
        row['end']=end
    return output,reports
