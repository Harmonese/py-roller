"""Shared text units and provenance for English, Mandarin and code switching.

Display strings are never replaced by matching strings. Source offsets are
half-open Python character offsets in the original text.
"""
from functools import lru_cache
import re
import unicodedata
import regex

TEXT_POLICY = 'unicode-word-unspaced-char-t2s-v2'
HAN = r'\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0003134f'
# Unspaced scripts keep source-character units; other scripts keep whole words.
SINGLE = r'[\p{sc=Han}\p{sc=Hiragana}\p{sc=Katakana}\p{sc=Thai}\p{sc=Lao}\p{sc=Khmer}\p{sc=Myanmar}]'
LETTER = r'(?:(?!'+SINGLE+r')[\p{L}\p{M}])'
PATTERN = regex.compile(SINGLE+'|'+LETTER+"+(?:'"+LETTER+'+)*|[0-9]+')


def is_han(text):
    return bool(text) and all(re.fullmatch('['+HAN+']', c) for c in text)


@lru_cache(maxsize=1)
def converter():
    from opencc import OpenCC
    return OpenCC('t2s')


@lru_cache(maxsize=16384)
def text_units(text):
    expanded=[]; offsets=[]
    for index,c in enumerate(text):
        for char in unicodedata.normalize('NFKC',c).lower().replace('\u2019', "'"):
            expanded.append(char); offsets.append(index)
    value=''.join(expanded)
    canonical_chars=list(value)
    for run in re.finditer('['+HAN+']+',value):
        converted=converter().convert(run.group())
        if len(converted)!=len(run.group()):
            raise ValueError('Non-length-preserving Chinese conversion needs explicit text mapping')
        canonical_chars[run.start():run.end()]=converted
    canonical_value=''.join(canonical_chars)
    units=[]
    for match in PATTERN.finditer(value):
        raw=match.group(); canonical=canonical_value[match.start():match.end()]
        # Contextual t2s preserves one-to-one source character offsets. Unknown
        # conversion cardinality is reported rather than inventing positions.
        for token in ([*canonical] if is_han(canonical) else [canonical]):
            units.append((token,offsets[match.start()],offsets[match.end()-1]+1))
    return tuple(units)


def normalize(text):
    return ' '.join(t for t,a,b in text_units(text))


def resolve_language(text, requested='auto'):
    text='\n'.join(line for line in text.splitlines()
                   if not re.fullmatch(r'\s*\[[^\]]+\]\s*',line))
    tokens=normalize(text).split()
    unsupported=[c for c in text if c.isalpha() and not (is_han(c) or c.isascii())]
    if unsupported and requested not in ('auto','mul'):
        raise ValueError(f'Unsupported script for {requested}: {unsupported[0]!r}; use mul')
    han=any(is_han(t) for t in tokens)
    latin=any(re.search('[a-z]',t) for t in tokens)
    if requested=='en' and han: raise ValueError('English profile cannot process Han lyrics; use zh or auto')
    return ('mul' if unsupported else 'zh-en' if han and latin else 'zh' if han else 'en') if requested=='auto' else requested


def whisper_options(language, baseline):
    options=dict(baseline)
    if language=='zh': options['language']='zh'
    elif language in ('zh-en','mul'): options.update(language=None,multilingual=True)
    return options


def source_key(word, fallback):
    return tuple(word.get('source_id', ('independent', fallback)))


def independent_support(matches, words):
    return len({source_key(words[m['asr_word']],m['asr_word']) for m in matches})


def asr_units(words):
    """Reassemble Unicode/BPE pieces before choosing Han/Latin text units.

    Whisper's Chinese timestamp splitter can emit Latin subwords. The original
    leading whitespace, rather than an invented space between pieces, defines
    a Latin word. Each returned unit retains all contributing source indices.
    """
    chunks=[];offset=0
    for index,word in enumerate(words):
        text=word['word'];chunks.append((offset,offset+len(text),index,word));offset+=len(text)
    joined=''.join(w['word'] for w in words)
    units=text_units(joined)
    counts={i:sum(a<end and b>start for token,a,b in units) for start,end,i,w in chunks}
    output=[]
    for token,a,b in units:
        parents=[c for c in chunks if a<c[1] and b>c[0]]
        begin,end=parents[0][0],parents[-1][1]
        output.append(dict(token=token,start=min(float(c[3]['start']) for c in parents),
            end=max(float(c[3]['end']) for c in parents),word=parents[0][2],
            probability=parents[0][3].get('probability') if len(parents)==1 else None,
            source_indices=[c[2] for c in parents],source_text=joined[begin:end],
            source_span=[a-begin,b-begin],shared_time=any(counts[c[2]]>1 for c in parents)))
    return output


@lru_cache(maxsize=2048)
def pronunciations(units):
    """Resolve polyphones in contiguous Mandarin context, never across Latin."""
    from pypinyin import lazy_pinyin, Style
    output=['']*len(units);i=0
    while i<len(units):
        if not is_han(units[i]): i+=1;continue
        end=i+1
        while end<len(units) and is_han(units[end]):end+=1
        readings=lazy_pinyin(''.join(units[i:end]),style=Style.NORMAL,errors=lambda text:list(text))
        if len(readings)==end-i:output[i:end]=readings
        i=end
    return tuple(output)
