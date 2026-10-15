"""Bounded CTC decisions on already discovered lyric instances."""
import math
import numpy as np

from ._alignment import forced_path
from .matching import normalize


def encode(text, vocabulary):
    words=normalize(text).split();labels=[];spans=[]
    for word in words:
        start=len(labels)
        try: labels.extend(vocabulary[c] for c in word.upper())
        except KeyError as exc: raise ValueError(f'CTC vocabulary cannot represent {word!r}') from exc
        spans.append((start,len(labels)))
    return labels,spans


def groups_for(rows,duration,settings):
    groups=[]
    for i,row in enumerate(rows):
        if groups:
            first=rows[groups[-1][0]];last=rows[groups[-1][-1]]
            if (last['token_end']==row['token_start'] and row['start']-last['end']<=settings.max_word_gap_seconds
                and row['end']-first['start']+2*settings.padding_seconds<=settings.group_seconds):
                groups[-1].append(i);continue
        groups.append([i])
    regions=[]
    for ids in groups:
        first,last=ids[0],ids[-1]
        a=max(0.,rows[first]['start']-settings.padding_seconds)
        b=min(duration,rows[last]['end']+settings.padding_seconds)
        if first:a=max(a,(rows[first-1]['end']+rows[first]['start'])/2)
        if last+1<len(rows):b=min(b,(rows[last]['end']+rows[last+1]['start'])/2)
        regions.append(dict(indices=ids,start=a,end=b))
    return regions


def frame_slice(start,end,origin,step,length):
    return max(0,math.ceil((start-origin)/step)),min(length,math.ceil((end-origin)/step))


def word_evidence(path,labels,spans,logp,offset,origin,step):
    evidence=[]
    for begin,end in spans:
        support=[]
        for i in range(begin,end):
            a,b=path['starts'][i],path['ends'][i]
            frames=logp[a:b]
            support.append(bool(np.any(frames[:,labels[i]]>=frames.max(axis=1)-1e-5)))
        evidence.append(dict(start=origin+(offset+path['starts'][begin])*step,
            end=origin+(offset+path['ends'][end-1])*step,character_support=sum(support)/len(support)))
    return evidence


def choose_variants(rows,inventory,words,logp,vocabulary,origin,step,duration,settings):
    """Compare only existing prefix-related lyric versions on identical audio.

    CTC scores are compared within a region, never summed with Whisper scores.
    The original identity stays available when acoustic evidence is ambiguous.
    """
    current=[dict(r) for r in rows];reports=[]
    for index,row in enumerate(rows):
        target=normalize(row['text']).split()
        variants=[]
        for line in inventory:
            other=line['words'];shared=min(len(target),len(other))
            if shared>=3 and abs(len(target)-len(other))<=2 and target[:shared]==other[:shared]:
                variants.append(line)
        if len(variants)<2:continue
        a=max(0.,row['start']-settings.padding_seconds)
        b=min(duration,row['end']+settings.padding_seconds)
        if index:a=max(a,(rows[index-1]['end']+row['start'])/2)
        if index+1<len(rows):b=min(b,(row['end']+rows[index+1]['start'])/2)
        begin,end=frame_slice(a,b,origin,step,len(logp));block=logp[begin:end]
        record=dict(index=index,original=row['line_id'],start=a,end=b,variants=[])
        if b<=a or b-a>settings.group_seconds:
            reports.append(record|dict(status='budget_or_empty'));continue
        best=None;original=None
        for line in variants:
            try:
                labels,spans=encode(line['text'],vocabulary)
                path=forced_path(block,labels,vocabulary['<pad>'],settings.unknown_cost)
                evidence=word_evidence(path,labels,spans,block,begin,origin,step)
                shared=min(len(target),len(line['words']))
                prefix_supported=sum(w['character_support']>=.5 for w in evidence[:shared])>=min(2,shared)
                decision=dict(line_id=line['line_id'],score=path['score'],prefix_supported=prefix_supported,words=evidence)
                record['variants'].append(decision)
                if line['line_id']==row['line_id']:original=decision
                if prefix_supported and (best is None or decision['score']>best[0]['score']):best=(decision,line)
            except (RuntimeError,ValueError) as exc:
                record['variants'].append(dict(line_id=line['line_id'],error=str(exc)))
        if best and original and best[0]['line_id']!=row['line_id'] and best[0]['score']>original['score']+1e-6:
            choice,line=best
            # Weak/missing CTC suffix evidence is not proof that Whisper's word
            # was absent. Preserve identity and expose the competing reading.
            current[index].update(content_status='ambiguous_variant',
                alternative_line_id=line['line_id'], alternative_text=line['text'])
            record.update(status='alternative_pending_review',selected=row['line_id'],
                proposed=line['line_id'],reason='CTC score alone cannot establish a missing suffix')
        else:record.update(status='retained',selected=row['line_id'])
        reports.append(record)
    return current,reports


def select_onsets(options):
    """Small ordered chain: select coarse/CTC onsets without end-time vetoes."""
    if not options:return []
    paths=[(0.,[],float('-inf'))]
    for choices in options:
        following=[]
        for choice in choices:
            eligible=[(score,chosen) for score,chosen,last in paths if last<=choice['start']]
            if eligible:
                score,chosen=max(eligible,key=lambda p:p[0])
                following.append((score+choice['rank'],chosen+[choice],choice['start']))
        if not following:raise ValueError('No monotonic onset sequence; inconsistent ASR timestamps')
        paths=following
    return max(paths,key=lambda p:p[0])[1]


def refine(rows,words,logp,vocabulary,origin,step,duration,settings):
    output=[dict(r,coarse_start=r['start'],coarse_end=r['end'],timing_status='coarse_unverified',
        timing_method='whisper_matched_words',end_status='coarse',character_path=None) for r in rows]
    options=[[dict(start=r['start'],rank=0.,method='whisper_matched_words')] for r in rows]
    reports=[]
    for region in groups_for(rows,duration,settings):
        a,b=frame_slice(region['start'],region['end'],origin,step,len(logp));block=logp[a:b]
        labels=[];offsets=[]
        record=dict(region,frame_start=a,frame_end=b)
        try:
            for index in region['indices']:
                encoded,spans=encode(rows[index]['text'],vocabulary);begin=len(labels)
                labels.extend(encoded);offsets.append((index,begin,len(labels),[(begin+x,begin+y) for x,y in spans]))
            if b<=a or region['end']-region['start']>settings.group_seconds:raise ValueError('region budget or empty audio')
            path=forced_path(block,labels,vocabulary['<pad>'],settings.unknown_cost);record['path']=path
            for index,begin,end,spans in offsets:
                row=output[index];evidence=word_evidence(path,labels,spans,block,a,origin,step)
                anchors=[]
                for match in row['matched_words']:
                    if match['cost']>=1 or match['lyric_word']>=len(evidence):continue
                    aligned=evidence[match['lyric_word']];observed=words[match['asr_word']]
                    overlap=(aligned['start']<=observed['end']+settings.anchor_tolerance and
                             aligned['end']>=observed['start']-settings.anchor_tolerance)
                    if overlap and aligned['character_support']>=.5:anchors.append(match['lyric_word'])
                start_time=evidence[0]['start'];end_time=evidence[-1]['end']
                anchored=len(set(anchors))>=min(2,len(evidence)) and evidence[0]['character_support']>=.5
                ordinary=abs(start_time-row['coarse_start'])<=settings.ordinary_onset_shift
                supported=ordinary or anchored
                row.update(ctc_proposal=dict(start=start_time,end=end_time),word_evidence=evidence,
                    onset_decision=dict(supported=supported,ordinary_shift=ordinary,anchored_shift=anchored,
                        anchors=anchors,first_word_support=evidence[0]['character_support']),
                    character_path=dict(tokens=labels[begin:end],starts=[a+x for x in path['starts'][begin:end]],
                                        ends=[a+x for x in path['ends'][begin:end]]))
                if supported:
                    rank=len(set(anchors))/len(evidence)+sum(w['character_support'] for w in evidence)/len(evidence)
                    options[index].append(dict(start=start_time,rank=rank,method='local_ordered_ctc'))
                else:row['timing_status']='needs_review_weak_onset_support'
                if evidence[-1]['character_support']>=.5:
                    row.update(end=end_time,end_status='ctc_unverified')
            record['status']='aligned'
        except (RuntimeError,ValueError) as exc:
            record.update(status='coarse_fallback',reason=str(exc))
        reports.append(record)
    selected=select_onsets(options)
    for row,choice in zip(output,selected):
        row.update(start=choice['start'],timing_method=choice['method'])
        if choice['method']=='local_ordered_ctc':row['timing_status']='ctc_supported_unverified'
        row['acoustic_end']=row['end']
    # End is a display extent in this LRC engine, not a reason to undo an onset.
    # Raw acoustic ends and character paths remain available in the diagnostics.
    for i,row in enumerate(output):
        limit=output[i+1]['start'] if i+1<len(output) else duration
        end=max(row['start'],min(row['end'],limit))
        if end!=row['end']:row['end_status']='bounded_for_display'
        row['end']=end
    return output,reports
