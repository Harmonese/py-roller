"""CTC labels and explicit frame geometry, independent of display spelling."""
from dataclasses import dataclass
import numpy as np

from ..language import normalize, text_units
import re


class Vocabulary(dict):
    def __init__(self, values, blank):
        super().__init__(values)
        self.blank=blank
        self['<pad>']=blank  # Internal compatibility alias; not a model assumption.


@dataclass(frozen=True)
class FrameGeometry:
    stride: int
    receptive: int
    sample_rate: int = 16000

    @classmethod
    def wav2vec2(cls, config):
        stride=receptive=1
        for kernel,hop in zip(config['conv_kernel'],config['conv_stride']):
            receptive+=(kernel-1)*stride;stride*=hop
        return cls(stride,receptive)

    @property
    def step(self): return self.stride/self.sample_rate

    @property
    def origin(self): return self.receptive/(2*self.sample_rate)


def encode_units(text, vocabulary):
    if getattr(vocabulary,'encoder',None) is not None:
        return vocabulary.encoder.encode(text,vocabulary)
    labels=[];spans=[];unsupported=[]
    for unit in normalize(text).split():
        start=len(labels)
        for char in unit.upper():
            label=vocabulary.get(char)
            if label is None or label==vocabulary.get('<unk>') or label==vocabulary['<pad>']:
                unsupported.append(char)
            else: labels.append(label)
        spans.append((start,len(labels)))
    if unsupported:
        raise ValueError('CTC cannot encode characters/digits: '+', '.join(sorted(set(unsupported))))
    return labels,spans


class Projection:
    """Keep required emissions and an unselectable background maximum.

    The extra column is max of ALL nonblank log probabilities, not a new
    normalized distribution. It preserves this engine's Viterbi/argmax tests.
    """
    def __init__(self, vocabulary, text):
        blank=vocabulary['<pad>']
        required={vocabulary[c] for c in normalize(text).upper() if c in vocabulary and c!=' '
                  and vocabulary[c] not in (blank,vocabulary.get('<unk>'))}
        self.columns=[blank,*sorted(required)]
        positions={old:new for new,old in enumerate(self.columns)}
        self.vocabulary=Vocabulary({k:positions[v] for k,v in vocabulary.items() if v in positions},0)
        self.metadata=dict(policy='required-labels-plus-nonblank-max-v1',columns=self.columns,blank=blank)

    def apply(self, logp):
        blank=self.columns[0]
        mask=np.arange(logp.shape[1])!=blank
        peak=logp[:,mask].max(axis=1,keepdims=True)
        return np.concatenate((logp[:,self.columns],peak),axis=1)


def text_diagnostics(text, vocabulary):
    issues=[]
    for index,line in enumerate(text.splitlines()):
        if re.fullmatch(r'\s*\[[^\]]+\]\s*',line):continue
        for unit,start,end in text_units(line):
            reasons=[]
            if unit.isdigit() and getattr(vocabulary,'encoder',None) is None:reasons.append('numeric_reading_not_expanded')
            try:encode_units(unit,vocabulary)
            except ValueError:reasons.append('outside_ctc_vocabulary')
            if reasons:issues.append(dict(input_line_index=index,unit=unit,start=start,end=end,reasons=reasons))
    return issues
