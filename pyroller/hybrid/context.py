"""Retrieve a small lyric vocabulary from the current unprompted ASR window."""
from .matching import window_words
from .text_distance import substitution


def retrieve_context(window, inventory, tokenizer, token_budget=192, line_budget=6):
    observed = [w['token'] for w in window_words(window)]
    ranked = []
    for line in inventory:
        target = line['words']
        # Local ordered coverage avoids selecting a long line from scattered words.
        best = 0.
        for begin in range(len(observed)):
            span = observed[begin:begin+len(target)+2]
            previous = [0.]*(len(span)+1)
            for word in target:
                current = [0.]
                for j, token in enumerate(span, 1):
                    support = max(0., 1.-substitution(word, token)[0])
                    current.append(max(previous[j], current[-1], previous[j-1]+support))
                previous = current
            best = max(best, previous[-1])
        coverage = best/len(target)
        if coverage >= .5 and best >= min(2.,len(target)):
            ranked.append((coverage, best, line))
    ranked.sort(key=lambda x: (-x[0],-x[1],x[2]['line_id']))
    selected = []; prompt = ''
    for coverage, support, line in ranked:
        candidate = '\n'.join([prompt,line['text']]).strip()
        if len(tokenizer.encode(candidate).ids) > token_budget: continue
        prompt = candidate
        selected.append(dict(line_id=line['line_id'],text=line['text'],coverage=coverage,support=support))
        if len(selected) >= line_budget: break
    return dict(prompt=prompt,selected=selected,token_count=len(tokenizer.encode(prompt).ids) if prompt else 0,
                policy='current_unprompted_window_ordered_coverage_v1',reference_used=False)
