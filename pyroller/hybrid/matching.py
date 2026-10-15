"""Text-only lyric inventory, overlap stitching and instance matching."""
from functools import lru_cache
import hashlib
import math
import re
import unicodedata

import numpy as np


def normalize(text):
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "\x27")
    return " ".join(re.findall(r"[a-z]+(?:\x27[a-z]+)*|[0-9]+", text))


def lyric_inventory(text):
    if re.search(r'\[\d+:\d', text):
        raise ValueError('Only untimed lyrics are allowed')
    inventory = {}
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw or re.fullmatch(r'\[[^\]]+\]', raw): continue
        normalized = normalize(raw)
        if not normalized: continue
        if any(c.isalpha() and not 'a' <= c.lower() <= 'z' for c in raw):
            raise ValueError('This version supports English lyrics only')
        if normalized not in inventory:
            inventory[normalized] = dict(text=raw, normalized=normalized,
                line_id='line-'+hashlib.sha256(normalized.encode()).hexdigest()[:12],
                words=normalized.split())
    if not inventory: raise ValueError('Empty lyric inventory')
    # Inventory order cannot leak a preferred performance order into inference.
    return [inventory[k] for k in sorted(inventory)]


def window_words(window):
    words = []
    for si, segment in enumerate(window['segments']):
        for wi, word in enumerate(segment.get('words') or []):
            for token in normalize(word['word']).split():
                # Fresh Whisper output can carry numpy scalars; cached JSON
                # carries Python scalars. Keep both paths identical at ingress.
                start = float(window['start'])+float(word['start'])
                end = float(window['start'])+float(word['end'])
                if not 0 <= start <= end or not np.isfinite([start, end]).all():
                    raise ValueError('Invalid ASR timestamp')
                start, end = min(start, window['end']), min(end, window['end'])
                words.append(dict(token=token, start=start, end=end,
                    probability=word.get('probability'), window=window['index'], segment=si,
                    word=wi, available_until=window['end']))
    if any(b['start'] < a['start'] for a, b in zip(words, words[1:])):
        raise ValueError('ASR window is not monotonic')
    return words


def overlap_pairs(left, right, tolerance):
    """Time-gated LCS: identical repeated words far apart cannot be duplicates."""
    n, m = len(left), len(right)
    score = np.zeros((n+1, m+1)); parent = {}
    for i in range(1, n+1):
        for j in range(1, m+1):
            options = [(score[i-1, j], 'left'), (score[i, j-1], 'right')]
            a, b = left[i-1], right[j-1]
            delta = abs((a['start']+a['end']-b['start']-b['end'])/2)
            if a['token'] == b['token'] and delta <= tolerance:
                options.append((score[i-1, j-1]+1-.1*delta/max(tolerance, .001), 'pair'))
            score[i, j], parent[i, j] = max(options, key=lambda x: x[0])
    i, j = n, m; pairs = []
    while i and j:
        op = parent[i, j]
        if op == 'pair': pairs.append((i-1, j-1)); i -= 1; j -= 1
        elif op == 'left': i -= 1
        else: j -= 1
    return pairs[::-1]


def stitch_windows(windows, settings):
    stream = []; diagnostics = []
    for window in sorted(windows, key=lambda w: w['index']):
        incoming = window_words(window)
        if not stream:
            stream = incoming; continue
        if not incoming: continue
        begin = next((i for i, w in enumerate(stream)
                      if w['end'] >= window['start']-settings.overlap_time_tolerance), len(stream))
        previous = stream[begin:]
        # The previous window's coverage, not all remaining words, defines overlap.
        cutoff = max((w['available_until'] for w in previous), default=window['start'])
        head = [w for w in incoming if w['start'] <= cutoff+settings.overlap_time_tolerance]
        pairs = overlap_pairs(previous, head, settings.overlap_time_tolerance)
        boundary = window['core_start']
        if pairs:
            i, j = min(pairs, key=lambda ij: abs((previous[ij[0]]['start']+head[ij[1]]['start'])/2-boundary))
            # One shared word is the splice point; do not independently crop both
            # copies by their noisy timestamps (which previously lost "cannot").
            cut = begin+i
            stream = stream[:cut+1]+incoming[j+1:]
            diagnostics.append(dict(window=window['index'], method='time_gated_word_anchor',
                token=previous[i]['token'], left_time=previous[i]['start'], right_time=head[j]['start'],
                matches=len(pairs), cut=boundary))
        else:
            # Explicitly uncertain fallback when the windows have no common text.
            stream = [w for w in stream if (w['start']+w['end'])/2 < boundary]+[
                w for w in incoming if (w['start']+w['end'])/2 >= boundary]
            diagnostics.append(dict(window=window['index'], method='unresolved_time_cut', matches=0, cut=boundary))
    # Keep word order; timestamp regressions at a splice are diagnosed, not sorted
    # into a new transcript. Coarse occurrence times use interval min/max below.
    return stream, diagnostics


@lru_cache(maxsize=8192)
def word_cost(a, b):
    if a == b: return 0.
    prev = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        row = [i]
        for j, y in enumerate(b, 1): row.append(min(row[-1]+1, prev[j]+1, prev[j-1]+(x != y)))
        prev = row
    ratio = prev[-1]/max(len(a), len(b))
    return .5 if ratio <= .34 else 1.


def discover_instances(words, inventory, settings, diagnostics=None):
    candidates = []
    rejected = {}; counts = {}
    def reject(reason,line,begin,length,cost,similarity):
        if diagnostics is None:return
        counts[reason]=counts.get(reason,0)+1
        chosen=words[begin:begin+length]
        if not chosen:return
        start=min(w['start'] for w in chosen);end=max(w['end'] for w in chosen)
        key=(line['line_id'],int(start/settings.diagnostic_bin_seconds))
        bucket=rejected.setdefault(key,[])
        bucket.append(dict(line_id=line['line_id'],text=line['text'],start=start,end=end,
            token_start=begin,token_end=begin+length,text_cost=cost,similarity=similarity,rejection=reason))
        bucket.sort(key=lambda row:(-row['similarity'],row['token_start'],row['token_end']))
        del bucket[2:]
    for line in inventory:
        target = line['words']; n = len(target)
        edits = math.floor(n*settings.edit_fraction)
        for begin in range(len(words)):
            available = words[begin:begin+n+edits]
            length = len(available)
            if length < max(1, n-edits): continue
            costs = np.zeros((n+1, length+1)); costs[:, 0] = np.arange(n+1); costs[0] = np.arange(length+1)
            back = np.zeros(costs.shape, dtype=np.uint8); back[1:, 0] = 1; back[0, 1:] = 2
            for i in range(1, n+1):
                for j in range(1, length+1):
                    options = (costs[i-1, j-1]+word_cost(target[i-1], available[j-1]['token']),
                               costs[i-1, j]+1, costs[i, j-1]+1)
                    op = min(range(3), key=options.__getitem__)
                    costs[i, j] = options[op]; back[i, j] = op
            for length_used in range(max(1, n-edits), length+1):
                cost = float(costs[n, length_used]); similarity = 1-cost/max(n, length_used)
                if similarity < settings.minimum_similarity:
                    reject('similarity',line,begin,length_used,cost,similarity);continue
                i, j = n, length_used; mapping = []
                while i or j:
                    op = back[i, j]
                    if op == 0:
                        mapping.append(dict(lyric_word=i-1, asr_word=begin+j-1,
                                            cost=word_cost(target[i-1], available[j-1]['token'])))
                        i -= 1; j -= 1
                    elif op == 1: i -= 1
                    else: j -= 1
                mapping.reverse()
                supported = [m for m in mapping if m['cost'] < 1]
                if not supported:
                    reject('unsupported_span_edges',line,begin,length_used,cost,similarity);continue
                # Do not consume unmatched prefix/suffix words from adjacent lyrics.
                a, b = supported[0]['asr_word'], supported[-1]['asr_word']+1
                if a != begin or b != begin+length_used:
                    reject('unsupported_span_edges',line,begin,length_used,cost,similarity);continue
                chosen = words[a:b]
                start, end = min(w['start'] for w in chosen), max(w['end'] for w in chosen)
                if not .15 <= end-start <= settings.max_line_seconds:
                    reject('duration',line,begin,length_used,cost,similarity);continue
                if any(y['start']-x['end'] > settings.max_word_gap_seconds for x, y in zip(chosen, chosen[1:])):
                    reject('word_gap',line,begin,length_used,cost,similarity);continue
                candidates.append(dict(line_id=line['line_id'], text=line['text'],
                    token_start=a, token_end=b, start=start, end=end, similarity=similarity,
                    text_cost=cost, score=n-settings.edit_penalty*cost-settings.instance_cost,
                    matched_words=mapping, lyric_words=n))
    if diagnostics is not None:
        counts['admitted']=len(candidates)
        diagnostics.update(mode='legacy',counts=counts,rejected=[c for bucket in rejected.values() for c in bucket],
            diagnostic_policy='best two rejected per lyric/time bin; length range remains legacy floor(0.3*n)')
    return sorted(candidates, key=lambda c: (c['token_end'], c['token_start'], c['line_id']))


def select_instances(words, candidates, settings):
    """K-best paths through word boundaries; unknown words cost zero.

    Candidate edges consume ASR words once. Repeated lyric identities are legal
    at distinct word positions. There is no preferred plain-lyrics ordering.
    """
    ends = {}
    for i, c in enumerate(candidates): ends.setdefault(c['token_end'], []).append(i)
    paths = [[(0., ())]]
    for end in range(1, len(words)+1):
        choices = list(paths[-1])
        for index in ends.get(end, []):
            c = candidates[index]
            choices.extend((score+c['score'], path+(index,)) for score, path in paths[c['token_start']])
        unique = {}
        for score, path in choices:
            if path not in unique or score > unique[path]: unique[path] = score
        paths.append(sorted(((score, path) for path, score in unique.items()),
                            key=lambda x: (-x[0], x[1]))[:settings.alternatives])
    best = paths[-1]
    return [dict(score=score, candidates=list(path)) for score, path in best]
