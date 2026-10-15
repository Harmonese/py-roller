"""Experimental POWER-kernel phrase costs, not the POWER error-rate metric."""
from functools import lru_cache

from .text_distance import phones, phrase_cost
from .vendor_power.levenshtein import Levenshtein
from .vendor_power.phonemes import Phonemes


@lru_cache(maxsize=32768)
def power_phrase_cost(left, right):
    original = phrase_cost(left, right)
    if original and original[0] == 0:
        return original
    variants = [phones(w) for w in left + right]
    if not all(variants):
        return original
    # Fixed first dictionary pronunciation; no per-song pronunciation selection.
    def sequence(words):
        result = ['|']
        for word in words:
            result.extend(p.lower() for p in phones(word)[0])
            result.append('|')
        return result
    a, b = sequence(left), sequence(right)
    lev = Levenshtein.align(a, b, reserve_list={'|', '#'},
        exclusive_sets=[Phonemes.vowels, Phonemes.consonants, {'r'} | Phonemes.r_vowels],
        weights=Levenshtein.wordAlignWeights)
    # POWER's native 4/3/3 path cost, normalized to word-edit units.
    # Boundary edits remain paid; boundary matches do not inflate evidence mass.
    mass = max(sum(p != '|' for p in a), sum(p != '|' for p in b))
    cost = min(1., lev.dist / (4 * mass)) * max(len(left), len(right))
    proposal = (cost, 'power_phonetic_phrase')
    return original if original and original[0] <= cost else proposal
