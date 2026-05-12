from __future__ import annotations

import re

from pyroller.domain import TimedUnit
from pyroller.transcriber.engine_types import EngineOutput
from pyroller.transcriber.protocol import build_unit_trace_metadata
from pyroller.transcriber.unitizers.base import TranscriptionAdapter
from pyroller.transcriber.unitizers.common import engine_spans_by_level
from pyroller.utils.ids import make_id
from pyroller.utils.text import normalize_ipa_phone, pinyin_syllable_to_ipa_tokens

_EXACT_PHONE_MAP: dict[str, list[str]] = {
    "tS": ["tʃ"],
    "dZ": ["dʒ"],
    "S": ["ʃ"],
    "Z": ["ʒ"],
    "N": ["ŋ"],
    "T": ["θ"],
    "D": ["ð"],
    "X": ["χ"],
    "C": ["ç"],
    'u"': ["y"],
    "t[": ["ʈ"],
    "d[": ["ɖ"],
    "th": ["tʰ"],
    "kh": ["kʰ"],
    "ph": ["pʰ"],
    "tsh": ["tsʰ"],
    "ts.h": ["tsʰ"],
    "tɕh": ["tɕʰ"],
    "a.": ["a"],
    "a.ː": ["aː"],
    "i.": ["i"],
    "i.ː": ["iː"],
    "u.": ["u"],
    "u.ː": ["uː"],
    "r.": ["r"],
    "ee": ["eː"],
    "oe": ["ø"],
    "oe:": ["øː"],
    "ɡ": ["ɡ"],
}
_ASCII_TO_IPA_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("tS", "tʃ"),
    ("dZ", "dʒ"),
    ("S", "ʃ"),
    ("Z", "ʒ"),
    ("N", "ŋ"),
    ("T", "θ"),
    ("D", "ð"),
    ("X", "χ"),
    ("C", "ç"),
    ('u"', "y"),
)
_PUNCT_ONLY_RE = re.compile(r"^[0-9?.]+$")
_PY_TONE_SUFFIX_RE = re.compile(r"([1245]|ɜ)$")
_LONG_VOWEL_ASCII_RE = re.compile(r"^([aeiouyøɑɔɛɨɯɐəɪʊɜœɵɒæɚ])[:ː]$")
_RHOTIC_VOWEL_RE = re.compile(r"^(ɑː|ɔː|oː|ɛ|ɪ|ʊ|i|ə)ɹ$")
_PY_PINYINISH_RE = re.compile(r"^[a-zəɛɔɑøyɨʊɯɐɜɡ\.]+(?:[1245]|ɜ)?$")
_PY_PRE_CANONICAL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("ɡ", "g"),
    (".", ""),
    ("ə", "e"),
    ("ɛ", "e"),
    ("ɔ", "o"),
    ("ɑ", "a"),
)


def _strip_pinyin_tone_marker(token: str) -> str:
    return _PY_TONE_SUFFIX_RE.sub("", token)


def _looks_like_model_pinyin_label(token: str) -> bool:
    if not _PY_PINYINISH_RE.match(token):
        return False
    has_pinyin_signal = bool(_PY_TONE_SUFFIX_RE.search(token) or any(ch in token for ch in ("ə", "ɛ", ".")))
    if has_pinyin_signal:
        return True
    return len(token) > 1 and "ɡ" in token


def _pinyinish_token_to_ipa_units(raw_token: str) -> list[str]:
    token = raw_token.strip()
    if not token:
        return []
    token = _strip_pinyin_tone_marker(token)
    for src, dst in _PY_PRE_CANONICAL_REPLACEMENTS:
        token = token.replace(src, dst)
    if not token:
        return []
    return [phone for phone in pinyin_syllable_to_ipa_tokens(token) if phone]


def _apply_ascii_phone_replacements(token: str) -> str:
    value = token
    for src, dst in _ASCII_TO_IPA_REPLACEMENTS:
        value = value.replace(src, dst)
    return value


def _canonicalize_model_phone(raw_token: str) -> list[str]:
    token = raw_token.strip()
    if not token or _PUNCT_ONLY_RE.match(token):
        return []

    if token in _EXACT_PHONE_MAP:
        return _EXACT_PHONE_MAP[token]

    if _looks_like_model_pinyin_label(token):
        phones = _pinyinish_token_to_ipa_units(token)
        if phones:
            return phones

    value = _apply_ascii_phone_replacements(token)
    value = value.replace(".", "")
    value = _strip_pinyin_tone_marker(value)

    long_match = _LONG_VOWEL_ASCII_RE.match(value)
    if long_match is not None:
        return [f"{long_match.group(1)}ː"]

    rhotic_match = _RHOTIC_VOWEL_RE.match(value)
    if rhotic_match is not None:
        return [rhotic_match.group(1), "ɹ"]

    if value in {"ts", "tsʰ"}:
        return [value]

    value = normalize_ipa_phone(value)
    return [value] if value else []


class MulIpaUnitizer(TranscriptionAdapter):
    name = "mul_ipa"
    unit_timing_semantics = "interpolated_from_token_segment"

    def __init__(self, *, backend: str = "wav2vec2_phoneme") -> None:
        self.backend = backend

    def _raw_segment_spans(self, engine_output: EngineOutput):
        return engine_spans_by_level(engine_output, "token")

    def _unitize(self, engine_output: EngineOutput, *, language: str, tone_mode: str) -> list[TimedUnit]:
        token_spans = engine_spans_by_level(engine_output, "token")
        units: list[TimedUnit] = []
        for index, span in enumerate(token_spans):
            phones = _canonicalize_model_phone(str(span.token or span.text or ""))
            if not phones:
                continue
            duration = max(float(span.end_time) - float(span.start_time), 0.0)
            step = duration / len(phones) if phones else 0.0
            for phone_index, phone in enumerate(phones):
                start = float(span.start_time) + (phone_index * step)
                end = float(span.end_time) if phone_index == len(phones) - 1 else float(span.start_time) + ((phone_index + 1) * step)
                units.append(
                    TimedUnit(
                        unit_id=make_id("timed_unit"),
                        symbol=phone,
                        normalized_symbol=phone,
                        unit_type="ipa_phone",
                        language=language,
                        tone=None,
                        start_time=start,
                        end_time=end,
                        confidence=None,
                        source_backend=self.backend,
                        raw_tokens=[str(span.token or span.text or "")],
                        metadata=build_unit_trace_metadata(
                            backend=self.backend,
                            source_segment_index=span.segment_index if span.segment_index is not None else index,
                            source_segment_level="token",
                            source_token_index=span.token_index if span.token_index is not None else index,
                            source_start_time=float(span.start_time),
                            source_end_time=float(span.end_time),
                            source_text=str(span.text or span.token or ""),
                            source_token=str(span.token or span.text or ""),
                            timing_mode="interpolated_from_token_segment",
                            extra={
                                "engine": engine_output.engine,
                                "sequence_index": index,
                            },
                        ),
                    )
                )
        return units
