"""Choose one provenance-preserving window splice after scoring whole phrases.

Raw windows remain alternatives, never concatenated wholesale. The original
splice wins ties. Only evidence available in the current prefix is examined.
"""
from .matching import window_words,overlap_pairs,select_instances
from .discovery import discover


def stitch_evidence(windows,inventory,settings):
    stream=[];reports=[]
    for window in sorted(windows,key=lambda w:w['index']):
        incoming=[dict(w,source_token_index=i) for i,w in enumerate(window_words(window))]
        if not incoming:continue
        if not stream:stream=incoming;continue
        boundary=window['core_start'];tolerance=settings.overlap_time_tolerance
        # Include full lyric context before the overlap, without re-searching
        # the entire song for each proposed splice.
        begin=next((i for i,w in enumerate(stream) if w['end']>=window['start']-settings.max_line_seconds),len(stream))
        prefix,previous=stream[:begin],stream[begin:]
        if not previous:
            stream=prefix+incoming
            reports.append(dict(window=window['index'],method='no_recent_words_append',cut=boundary,matches=0))
            continue
        overlap_begin=next((i for i,w in enumerate(previous) if w['end']>=window['start']-tolerance),len(previous))
        cutoff=max(w['available_until'] for w in previous)
        head=[w for w in incoming if w['start']<=cutoff+tolerance]
        pairs=[(i+overlap_begin,j) for i,j in overlap_pairs(previous[overlap_begin:],head,tolerance)]
        if pairs:
            i,j=min(pairs,key=lambda ij:abs((previous[ij[0]]['start']+incoming[ij[1]]['start'])/2-boundary))
            original=previous[:i+1]+incoming[j+1:]
        else:
            original=[w for w in previous if (w['start']+w['end'])/2<boundary]+[
                w for w in incoming if (w['start']+w['end'])/2>=boundary]
        left_candidates,_=discover(previous,inventory,settings)
        endpoints={c['token_end'] for c in left_candidates
                   if window['start']-tolerance<=c['end']<=cutoff}
        endpoints.update(i+1 for i,w in enumerate(previous)
            if w['end']>=window['start']-tolerance and
            (i+1==len(previous) or (w['window'],w['segment'])!=(previous[i+1]['window'],previous[i+1]['segment'])))
        variants=[('legacy',original,None,None)]
        for stop in sorted(endpoints):
            last=previous[stop-1];mid=(last['start']+last['end'])/2
            right=next((j for j,w in enumerate(incoming) if (w['start']+w['end'])/2>mid),len(incoming))
            # An aligned word retained on the left cannot also survive on the
            # right, even when Whisper places its two copies differently.
            right=max([right]+[j+1 for i,j in pairs if i<stop])
            variant=previous[:stop]+incoming[right:]
            variants.append(('phrase_boundary',variant,stop,right))
        def word_key(w):return (w['window'],w['source_token_index'])
        # Discover once per source, then project intact candidate spans into
        # each splice. Do not rerun fuzzy DP for every proposed boundary.
        source_candidates=[]
        for source,cs in ((previous,left_candidates),(incoming,discover(incoming,inventory,settings)[0]),
                          (original,discover(original,inventory,settings)[0])):
            for c in cs:
                source_candidates.append((c,tuple(word_key(w) for w in source[c['token_start']:c['token_end']])))
        seen=set();choices=[];winner=None
        for policy,words,left,right in variants:
            key=tuple(word_key(w) for w in words)
            if key in seen:continue
            seen.add(key)
            positions={k:i for i,k in enumerate(key)};projected={}
            for c,span in source_candidates:
                start=positions.get(span[0])
                if start is not None and key[start:start+len(span)]==span:
                    projected[c['line_id'],start,start+len(span)]=dict(c,token_start=start,token_end=start+len(span))
            cs=list(projected.values());path=select_instances(words,cs,settings)[0]
            score=path['score']
            record=dict(policy=policy,score=score,left_words=left,right_skip=right,
                selected=[dict(text=cs[i]['text'],start=cs[i]['start'],end=cs[i]['end']) for i in path['candidates']])
            choices.append(record)
            if winner is None or score>winner[0]+1e-9:winner=(score,words,len(choices)-1)
        stream=prefix+winner[1]
        reports.append(dict(window=window['index'],method='phrase_evidence_splice',cut=boundary,
            alternatives=choices,selected_alternative=winner[2],matches=len(pairs)))
    return stream,reports
