"""Checkpoint-compatible eSpeak units, never display text or guessed timestamps."""
from collections import defaultdict
from functools import lru_cache
from importlib.metadata import version
import unicodedata

from ..language import normalize, is_han

POLICY = 'espeak-unit-voices-exact-tokens-v1'


@lru_cache(maxsize=1)
def dependencies():
    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator
        from langid.langid import LanguageIdentifier, model
    except ImportError as exc:
        raise ImportError('mul requires py-roller[mul] and the native espeak-ng library') from exc
    if not EspeakBackend.is_available():
        raise ImportError('mul requires espeak-ng (macOS: brew install espeak-ng; Linux: install libespeak-ng1)')
    return EspeakBackend, Separator, LanguageIdentifier.from_modelstring(model, norm_probs=True)


def script(unit):
    names=[unicodedata.name(c,'') for c in unit if c.isalpha()]
    if is_han(unit):return 'han'
    for name in ('CYRILLIC','LATIN','GREEK','ARABIC','HEBREW','HIRAGANA','KATAKANA','HANGUL'):
        if any(name in n for n in names):return name.lower()
    return 'other'


@lru_cache(maxsize=64)
def phonemize_units(voice, units):
    backend,separator,_=dependencies()
    engine=backend(voice,with_stress=False,language_switch='remove-flags',words_mismatch='ignore')
    # The checkpoint tokenizer splits on spaces, not on IPA code points.
    output=engine.phonemize(list(units),separator=separator(phone=' ',word=' | ',syllable=''),strip=True)
    if len(output)!=len(units):raise ValueError('eSpeak changed the number of source units')
    return tuple(tuple(p for p in value.split() if p!='|') for value in output)


class PhonemeEncoder:
    @classmethod
    def from_metadata(cls, metadata):
        """Replay the original labels without re-running a changed G2P installation."""
        if metadata.get('policy')!=POLICY:raise ValueError('Unknown phoneme encoding policy')
        self=object.__new__(cls)
        self.metadata=metadata
        self.units={k:tuple(v) for k,v in metadata['units'].items()}
        self.voices=dict(metadata['voices']);self.errors=dict(metadata['errors'])
        return self

    def __init__(self, text):
        backend,_,identifier=dependencies()
        available=backend.supported_languages()
        units=list(dict.fromkeys(normalize(text).split()))
        default_code=identifier.classify(' '.join(units))[0] if units else 'en'
        by_script=defaultdict(list)
        for unit in units:by_script[script(unit)].append(unit)
        self.voices={};self.units={};self.errors={}
        for family,items in by_script.items():
            if family=='han':code='ja' if any(k in by_script for k in ('hiragana','katakana')) else 'zh'
            elif family in ('hiragana','katakana'):code='ja'
            elif family=='hangul':code='ko'
            elif all(u.isdigit() for u in items):code=default_code
            else:code=identifier.classify(' '.join(items))[0]
            voice={'en':'en-us','zh':'cmn'}.get(code,code)
            if voice not in available:
                for unit in items:self.errors[unit]=f'no eSpeak voice for detected language {code}'
                continue
            self.voices[family]=voice
            for unit,phones in zip(items,phonemize_units(voice,tuple(items))):self.units[unit]=phones
        self.metadata=dict(policy=POLICY,phonemizer=version('phonemizer'),langid=version('langid'),
                           espeak_version=str(backend.version()),voices=self.voices,
                           # Prevent changes in pronunciation resources from reusing incompatible caches.
                           units={k:list(v) for k,v in self.units.items()},errors=self.errors)

    def encode(self, text, vocabulary):
        labels=[];spans=[]
        for unit in normalize(text).split():
            if unit in self.errors:raise ValueError(f'Cannot phonemize {unit!r}: {self.errors[unit]}')
            phones=self.units.get(unit)
            if not phones:raise ValueError(f'No phoneme encoding for {unit!r}')
            unknown=[p for p in phones if p not in vocabulary or vocabulary[p] in
                     (vocabulary.blank,vocabulary.get('<unk>'))]
            if unknown:raise ValueError(f'CTC cannot encode phonemes for {unit!r}: {unknown}')
            start=len(labels);labels.extend(vocabulary[p] for p in phones)
            spans.append((start,len(labels)))
        return labels,spans
