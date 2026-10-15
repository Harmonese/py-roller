"""Bounded phrase-aware candidate discovery with reference-free rejection traces."""
import math
from importlib.metadata import version
from collections import Counter, defaultdict

from .text_distance import phrase_cost, word_weight


def discover(words, inventory, settings):
    pair_cost = phrase_cost
    steps = ((1, 1), (1, 2), (2, 1))
    if settings.scoring_mode == 'power':
        from .power_distance import power_phrase_cost
        pair_cost = power_phrase_cost
        steps = tuple((i, j) for i in range(1, 4) for j in range(1, 4))
    candidates = []
    rejected = defaultdict(list)
    counts = Counter()
    for line in inventory:
        target = line['words']; n = len(target)
        weighted = settings.scoring_mode in ('weighted', 'power')
        consistent = settings.scoring_mode != 'original'
        target_weights = [word_weight(w) if weighted else 1. for w in target]
        # At least one insertion/deletion can be investigated even for short lines.
        slack = max(1, math.ceil(n*settings.edit_fraction))
        for begin in range(len(words)):
            available = words[begin:begin+n+slack]
            observed = [w['token'] for w in available]; m = len(observed)
            observed_weights = [word_weight(w) if weighted else 1. for w in observed]
            if m < max(1, n-slack):
                counts['length_range'] += 1
                continue
            costs = [[float('inf')]*(m+1) for _ in range(n+1)]
            back = {}
            costs[0][0] = 0.
            for i in range(n+1):
                for j in range(m+1):
                    if i == j == 0: continue
                    options = []
                    for di, dj in steps:
                        if i >= di and j >= dj:
                            pair = pair_cost(tuple(target[i-di:i]), tuple(observed[j-dj:j]))
                            if pair:
                                value, reason = pair
                                weight = (max(sum(target_weights[i-di:i])/di,
                                              sum(observed_weights[j-dj:j])/dj) if weighted else 1.)
                                penalty = value*weight
                                diagnostic_value = value/max(di, dj) if reason == 'power_phonetic_phrase' else value
                                options.append((costs[i-di][j-dj]+penalty, di, dj, diagnostic_value, reason, penalty))
                    if i:
                        penalty = target_weights[i-1]
                        options.append((costs[i-1][j]+penalty, 1, 0, 1., 'missing_lyric_word', penalty))
                    if j:
                        penalty = observed_weights[j-1]
                        options.append((costs[i][j-1]+penalty, 0, 1, 1., 'extra_asr_word', penalty))
                    winner = min(options, key=lambda item: item[0])
                    costs[i][j] = winner[0]; back[i, j] = winner[1:]
            for length in range(max(1, n-slack), m+1):
                cost = costs[n][length]
                # Weight editing errors, not successfully observed coverage.
                # Otherwise discounting a correct function word makes an unchanged
                # content-word error larger and can reject a previously valid line.
                scoring_mass = max(n, length) if consistent else n
                similarity = 1-cost/(scoring_mass if consistent else max(n, length))
                chosen = available[:length]
                start, end = min(w['start'] for w in chosen), max(w['end'] for w in chosen)
                reason = 'similarity' if similarity < settings.minimum_similarity else None
                i, j = n, length; operations = []
                while i or j:
                    di, dj, value, kind, penalty = back[i, j]
                    operations.append(dict(lyric_start=i-di, lyric_end=i,
                        asr_start=begin+j-dj, asr_end=begin+j, cost=value, penalty=penalty, kind=kind))
                    i -= di; j -= dj
                operations.reverse()
                support = [op for op in operations if op['lyric_end']>op['lyric_start']
                           and op['asr_end']>op['asr_start'] and op['cost']<1]
                # A paid substitution is allowed at an edge, but unrelated words
                # never become timing anchors. Free-standing edge insertions still
                # cannot consume neighbouring lyrics just to enlarge a candidate.
                edges = [op for op in operations if op['lyric_end']>op['lyric_start']
                         and op['asr_end']>op['asr_start']] if consistent else support
                if reason is None and (not support or not edges or edges[0]['asr_start'] != begin or
                                       edges[-1]['asr_end'] != begin+length): reason = 'unsupported_span_edges'
                if reason is None and not .15 <= end-start <= settings.max_line_seconds: reason = 'duration'
                if reason is None and any(b['start']-a['end']>settings.max_word_gap_seconds
                                          for a,b in zip(chosen,chosen[1:])): reason = 'word_gap'
                row = dict(line_id=line['line_id'], text=line['text'], token_start=begin,
                    token_end=begin+length, start=start, end=end, similarity=similarity,
                    text_cost=cost, score=scoring_mass-settings.edit_penalty*cost-settings.instance_cost,
                    scoring_mass=scoring_mass, scoring_mode=settings.scoring_mode,
                    lyric_words=n, operations=operations)
                if reason:
                    counts[reason] += 1
                    # Top rejected explanations per identity/time bin; never need reference times.
                    key = (line['line_id'], int(start/settings.diagnostic_bin_seconds))
                    bucket = rejected[key]
                    bucket.append(dict(row, rejection=reason))
                    bucket.sort(key=lambda c: (-c['similarity'], c['token_start'], c['token_end']))
                    del bucket[2:]
                    continue
                mapping = []
                for op in operations:
                    if op['lyric_end'] == op['lyric_start'] or op['asr_end'] == op['asr_start']: continue
                    for li in range(op['lyric_start'],op['lyric_end']):
                        for wi in range(op['asr_start'],op['asr_end']):
                            mapping.append(dict(lyric_word=li, asr_word=wi, cost=op['cost'], kind=op['kind']))
                row['matched_words'] = mapping
                candidates.append(row)
                counts['admitted'] += 1
    # Retain different observation positions; cap alternatives only at the SAME start.
    buckets = defaultdict(list)
    for row in candidates: buckets[row['line_id'],row['token_start']].append(row)
    kept = []
    for bucket in buckets.values():
        bucket.sort(key=lambda c: (-c['score'], -c['similarity'], c['token_end']))
        kept.extend(bucket[:settings.candidate_limit_per_line_start])
        counts['candidate_budget'] += max(0,len(bucket)-settings.candidate_limit_per_line_start)
        for row in bucket[settings.candidate_limit_per_line_start:]:
            key=(row['line_id'],int(row['start']/settings.diagnostic_bin_seconds))
            rejected[key].append(dict(row,rejection='candidate_budget'))
    return sorted(kept,key=lambda c:(c['token_end'],c['token_start'],c['line_id'])), dict(
        mode='robust', scoring_mode=settings.scoring_mode, cmudict_version=version('cmudict'), counts=dict(counts),
        rejected=[c for bucket in rejected.values() for c in bucket],
        diagnostic_policy='best two rejected per lyric and 10s bin, plus budget exits; not exhaustive')
