"""Monotonic transcript alignment with bounded, evidence-checked CTC timing."""
from dataclasses import asdict
import time

import numpy as np

from .matching import lyric_inventory
from .stitching import stitch_evidence
from .text_distance import phrase_cost, word_weight, phrase_comparator
from ..language import independent_support, is_han, normalize
from .alignment import encode, frame_slice, word_evidence, refine
from ._alignment import forced_path


def ordered_lines(text):
    lyric_inventory(text)  # Same input validation as inventory mode.
    lines = []
    for index, raw in enumerate(text.splitlines()):
        try:
            parsed = lyric_inventory(raw)
        except ValueError as exc:
            if str(exc) == 'Empty lyric inventory': continue
            raise
        lines.append(dict(parsed[0], input_line_index=index,
                          occurrence_id=f'input-{index+1:06d}'))
    return lines


def align_words(target, observed, final=True):
    """Weighted edit lattice; phrase edges retain contractions and segmentation.

    Previews align an observed prefix, without charging for future lyric words.
    A second, equally optimal traceback exposes ambiguous repeated-word anchors.
    """
    n, m = len(target), len(observed)
    if (n+1)*(m+1) > 4_000_000:
        raise ValueError('Ordered word alignment exceeds 4 million cells')
    costs = np.full((n+1, m+1), np.inf)
    costs[0, 0] = 0.
    backs = np.zeros((n+1, m+1), dtype=np.uint8)
    edges = ((1,1), (1,2), (2,1), (1,0), (0,1))
    compare=phrase_comparator(target,observed)
    for i in range(n+1):
        for j in range(m+1):
            if i == j == 0: continue
            options = []
            for k, (di,dj) in enumerate(edges):
                if i < di or j < dj: continue
                if not dj: penalty = word_weight(target[i-1])
                elif not di: penalty = word_weight(observed[j-1])
                else:
                    pair = compare(i,j,di,dj)
                    if pair is None: continue
                    penalty = pair[0]*max(sum(map(word_weight,target[i-di:i]))/di,
                                          sum(map(word_weight,observed[j-dj:j]))/dj)
                options.append((costs[i-di,j-dj]+penalty, k))
            best = min(v for v,k in options)
            costs[i,j] = best
            backs[i,j] = sum(1<<k for v,k in options if abs(v-best)<1e-8)
    end = n if final else min(range(n+1), key=lambda i: (costs[i,m], i))
    def trace(reverse=False):
        i,j=end,m; mapping=[]
        while i or j:
            choices=[k for k in range(len(edges)) if backs[i,j] & (1<<k)]
            k=choices[-1] if reverse else choices[0];di,dj=edges[k]
            if di and dj:
                cost,kind=compare(i,j,di,dj)
                if cost<1:
                    mapping.extend((li,wi,cost) for li in range(i-di,i) for wi in range(j-dj,j))
            i-=di;j-=dj
        return set(mapping)
    first,second=trace(),trace(True)
    return sorted(first & second), dict(cost=float(costs[end,m]),
        ambiguous_pairs=len(first ^ second), lyric_prefix_words=end)


def align_ordered(text, windows, logp, vocabulary, origin, step, duration, settings, final):
    tick=time.perf_counter(); lines=ordered_lines(text)
    words,seams=stitch_evidence(windows,lyric_inventory(text),settings)
    target=[]; spans=[]
    for line in lines:
        a=len(target);target.extend(line['words']);spans.append((a,len(target)))
    mapping,lattice=align_words(target,[w['token'] for w in words],final)
    anchors=[]
    for a,b in spans:
        matches=[(li-a,wi,c) for li,wi,c in mapping if a<=li<b]
        # Two supported lyric words (or the entire one-word line) establish a region.
        independent=independent_support([dict(asr_word=wi) for li,wi,c in matches],words)
        literal=any(c==0 for li,wi,c in matches)
        shared=any('source_id' in words[wi] for li,wi,c in matches)
        whole_phrase=(shared and independent==1 and
            {li for li,wi,c in matches if c==0}==set(range(b-a)) and
            any(normalize(words[wi].get('source_text','')).split()==target[a:b] for li,wi,c in matches))
        if len({li for li,wi,c in matches}) >= min(2,b-a) and (not shared or independent>=min(2,b-a) or whole_phrase) and (
            not any(is_han(t) for t in target[a:b]) or literal):
            anchors.append(matches)
        else: anchors.append([])
    coarse=[]
    for line,matches in zip(lines,anchors):
        if not matches: continue
        indices=[wi for li,wi,c in matches]
        a=min(words[wi]['start'] for wi in indices);b=max(words[wi]['end'] for wi in indices)
        if b-a>settings.max_line_seconds:
            matches.clear();continue
        coarse.append(dict(line,start=a,end=b,token_start=min(indices),token_end=max(indices)+1,
            matched_words=[dict(lyric_word=li,asr_word=wi,cost=c) for li,wi,c in matches],
            status='needs_review' if final else 'tentative',finalized=final,
            content_status='provided_ordered_text'))
    # Timestamp regressions at ASR seams cannot establish monotonic anchors.
    stable=[]
    for row in coarse:
        if stable and row['start']<stable[-1]['start']:
            anchors[next(i for i,l in enumerate(lines) if l['input_line_index']==row['input_line_index'])]=[]
        else:stable.append(row)
    timed,timing_groups=refine(stable,words,logp,vocabulary,origin,step,duration,settings)
    timed_by_line={r['input_line_index']:r for r in timed}
    rows=[]; unresolved=[];reports=[]; recovery_cache={}
    for index,line in enumerate(lines):
        if line['input_line_index'] in timed_by_line:
            rows.append(timed_by_line[line['input_line_index']]);continue
        own=anchors[index]
        previous=next((timed_by_line[lines[i]['input_line_index']] for i in range(index-1,-1,-1)
                       if lines[i]['input_line_index'] in timed_by_line),None)
        following=next((timed_by_line[lines[i]['input_line_index']] for i in range(index+1,len(lines))
                        if lines[i]['input_line_index'] in timed_by_line),None)
        left=previous['end'] if previous else 0.
        right=following['start'] if following else duration
        if own:
            a=max(left,min(words[wi]['start'] for li,wi,c in own)-settings.padding_seconds)
            b=min(right,max(words[wi]['end'] for li,wi,c in own)+settings.padding_seconds)
        else: a,b=left,right
        reason='insufficient_evidence'; report=dict(input_line_index=line['input_line_index'],start=a,end=b)
        # Do not speculate about missing lyrics beyond the latest preview anchor.
        if not final and not own and following is None: reason='pending_audio'
        elif b<=a or b-a>settings.group_seconds: reason='region_budget_or_empty'
        else:
            try:
                begin,end=frame_slice(a,b,origin,step,len(logp));block=logp[begin:end]
                # All consecutive missing lines compete in one ordered path;
                # independent forced paths could reuse the same clear syllable.
                first=index
                while first and lines[first-1]['input_line_index'] not in timed_by_line:first-=1
                last=index+1
                while last<len(lines) and lines[last]['input_line_index'] not in timed_by_line:last+=1
                key=(first,last,begin,end)
                if key not in recovery_cache:
                    labels=[];segments={}
                    for entry in range(first,last):
                        encoded,sp=encode(lines[entry]['text'],vocabulary);offset=len(labels)
                        labels.extend(encoded);segments[entry]=(offset,len(labels),[(offset+x,offset+y) for x,y in sp])
                    path=forced_path(block,labels,vocabulary['<pad>'],settings.unknown_cost)
                    recovery_cache[key]=(labels,segments,path)
                labels,segments,path=recovery_cache[key]
                token_a,token_b,word_spans=segments[index]
                evidence=word_evidence(path,labels,word_spans,block,begin,origin,step)
                supported=[e['character_support']>=.5 for e in evidence]
                agrees=sum(evidence[li]['start']<=words[wi]['end']+settings.anchor_tolerance and
                           evidence[li]['end']>=words[wi]['start']-settings.anchor_tolerance
                           for li,wi,c in own)
                report.update(word_evidence=evidence,score=path['score'],anchor_agreements=agrees)
                if supported[0] and sum(supported)>=max(1, int(np.ceil(.75*len(evidence)))) and (
                    not own or agrees>=min(2,len(own))):
                    rows.append(dict(line,start=evidence[0]['start'],end=evidence[-1]['end'],
                        status='needs_review' if final else 'tentative',finalized=final,
                        content_status='provided_ordered_text',timing_status='ctc_supported_unverified',
                        timing_method='ordered_local_ctc',character_path=dict(tokens=labels[token_a:token_b],
                            starts=[begin+x for x in path['starts'][token_a:token_b]],
                            ends=[begin+x for x in path['ends'][token_a:token_b]]),
                        word_evidence=evidence))
                    report['status']='aligned';reports.append(report);continue
            except (ValueError,RuntimeError) as exc: report['error']=str(exc)
        unresolved.append(dict(occurrence_id=line['occurrence_id'],line_id=line['line_id'],
            input_line_index=line['input_line_index'],text=line['text'],reason=reason))
        report['status']=reason;reports.append(report)
    # Missing neighbours can propose the same audio. Do not claim both located.
    conflicts=set()
    for i in range(1,len(rows)):
        if rows[i]['start']<rows[i-1]['start'] or (
            rows[i]['start']<rows[i-1]['end'] and
            rows[i]['timing_method']==rows[i-1]['timing_method']=='ordered_local_ctc'):
            conflicts.update((i-1,i))
    kept=[]
    for i,row in enumerate(rows):
        if i in conflicts:
            unresolved.append({k:row[k] for k in ('occurrence_id','line_id','input_line_index','text')} |
                              dict(reason='overlapping_assignment'))
        else: kept.append(row)
    for i,row in enumerate(kept):
        row['end']=max(row['start'],min(row['end'],kept[i+1]['start'] if i+1<len(kept) else duration))
    return dict(schema_version=1,engine='whisper_ctc',occurrences=kept,coarse_occurrences=stable,
        inventory=lines,stitched_words=words,seams=seams,unaligned_lines=sorted(unresolved,key=lambda r:r['input_line_index']),
        word_alignment=lattice,word_mapping=mapping,timing_groups=timing_groups,recovery_regions=reports,
        settings=asdict(settings),reference_used=False,reference_order_used=True,
        metrics=dict(processing_seconds=time.perf_counter()-tick))
