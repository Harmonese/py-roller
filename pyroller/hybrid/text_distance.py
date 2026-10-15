"""Explainable text/phrase confusions; identities are never normalized away."""
from functools import lru_cache
import re


# Negation, quantifiers and pronouns keep full weight: short does not mean optional.
FUNCTION_WORDS = frozenset('a an the of to in into on at for from by with as '
                           'and or but my your our their his her its '
                           'am is are was were be been being have has had'.split())


def word_weight(word):
    return .35 if word in FUNCTION_WORDS else 1.


def edit_distance(left, right):
    previous = list(range(len(right)+1))
    for i, a in enumerate(left, 1):
        row = [i]
        for j, b in enumerate(right, 1):
            row.append(min(row[-1]+1, previous[j]+1, previous[j-1]+(a != b)))
        previous = row
    return previous[-1]


@lru_cache(maxsize=1)
def lexicon():
    import cmudict
    return cmudict.dict()


@lru_cache(maxsize=8192)
def phones(word):
    return tuple(tuple(re.sub(r'\d', '', phone) for phone in variant)
                 for variant in lexicon().get(word, ()))


def expansion(word):
    special = {"can't": ('can', 'not'), "won't": ('will', 'not'),
               "shan't": ('shall', 'not'), "let's": ('let', 'us')}
    if word in special: return special[word]
    if word.endswith("n't"): return (word[:-3], 'not')
    for suffix, auxiliary in [("'ve", 'have'), ("'re", 'are'), ("'ll", 'will'),
                              ("'m", 'am'), ("'d", 'would'), ("'s", 'is')]:
        if word.endswith(suffix): return (word[:-len(suffix)], auxiliary)
    return (word,)


@lru_cache(maxsize=32768)
def substitution(left, right):
    if left == right: return 0., 'exact'
    a, b = expansion(left), expansion(right)
    # Soft correspondence, not equivalence: do not erase negation or identities.
    if (len(a) == 2 and a[1] != 'not' and a[0] == right or
        len(b) == 2 and b[1] != 'not' and b[0] == left):
        return .25, 'auxiliary_contraction'
    spelling = edit_distance(left, right)/max(len(left), len(right))
    best = (.5, 'spelling') if spelling <= .34 else (1., 'substitution')
    pa, pb = phones(left), phones(right)
    if pa and pb:
        distance = min(edit_distance(x, y)/max(len(x), len(y)) for x in pa for y in pb)
        if distance == 0: return .25, 'dictionary_homophone'
        if distance <= .34 and best[0] > .5: return .5, 'dictionary_pronunciation'
    return best


@lru_cache(maxsize=16384)
def phrase_cost(left, right):
    if len(left) == len(right) == 1: return substitution(left[0], right[0])
    expanded_left = tuple(t for word in left for t in expansion(word))
    expanded_right = tuple(t for word in right for t in expansion(word))
    if expanded_left == expanded_right: return .15, 'contraction_expansion'
    if ''.join(left) == ''.join(right): return .25, 'word_segmentation'
    return None
