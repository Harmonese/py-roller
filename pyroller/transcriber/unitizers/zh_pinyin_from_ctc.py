from __future__ import annotations

import re
from typing import Any

from pyroller.domain import TimedUnit
from pyroller.transcriber.engine_types import EngineOutput, EngineSpan
from pyroller.transcriber.protocol import build_unit_trace_metadata
from pyroller.transcriber.unitizers.base import TranscriptionAdapter
from pyroller.transcriber.unitizers.common import engine_spans_by_level
from pyroller.utils.ids import make_id
from pyroller.utils.text import split_pinyin_tone

INITIALS = [
    "ㄅ", "ㄆ", "ㄇ", "ㄈ", "ㄉ", "ㄊ", "ㄋ", "ㄌ", "ㄍ", "ㄎ", "ㄏ", "ㄐ", "ㄑ", "ㄒ",
    "ㄓ", "ㄔ", "ㄕ", "ㄖ", "ㄗ", "ㄘ", "ㄙ", "j",
]
FINALS = [
    "ㄚ", "ㄛ", "ㄜ", "ㄝ", "ㄞ", "ㄟ", "ㄠ", "ㄡ", "ㄢ", "ㄣ", "ㄤ", "ㄥ", "ㄦ", "ㄧ", "ㄧㄛ",
    "ㄧㄚ", "ㄧㄝ", "ㄧㄠ", "ㄧㄡ", "ㄧㄢ", "ㄧㄣ", "ㄧㄤ", "ㄧㄥ", "ㄨ", "ㄨㄚ", "ㄨㄛ", "ㄨㄞ",
    "ㄨㄟ", "ㄨㄢ", "ㄨㄣ", "ㄨㄤ", "ㄨㄥ", "ㄩ", "ㄩㄝ", "ㄩㄢ", "ㄩㄣ", "ㄩㄥ", "ㄭ+", "ㄭ-", "r",
]
INITIALS_SET = set(INITIALS)
FINALS_SET = set(FINALS)
INITIAL_MAP = {
    "ㄅ": "b", "ㄆ": "p", "ㄇ": "m", "ㄈ": "f", "ㄉ": "d", "ㄊ": "t", "ㄋ": "n", "ㄌ": "l",
    "ㄍ": "g", "ㄎ": "k", "ㄏ": "h", "ㄐ": "j", "ㄑ": "q", "ㄒ": "x", "ㄓ": "zh", "ㄔ": "ch",
    "ㄕ": "sh", "ㄖ": "r", "ㄗ": "z", "ㄘ": "c", "ㄙ": "s",
}
ZERO_INITIAL_FINAL_MAP = {
    "ㄚ": "a", "ㄛ": "o", "ㄜ": "e", "ㄝ": "e", "ㄞ": "ai", "ㄟ": "ei", "ㄠ": "ao", "ㄡ": "ou",
    "ㄢ": "an", "ㄣ": "en", "ㄤ": "ang", "ㄥ": "eng", "ㄦ": "er", "ㄧ": "yi", "ㄧㄛ": "yo",
    "ㄧㄚ": "ya", "ㄧㄝ": "ye", "ㄧㄠ": "yao", "ㄧㄡ": "you", "ㄧㄢ": "yan", "ㄧㄣ": "yin",
    "ㄧㄤ": "yang", "ㄧㄥ": "ying", "ㄨ": "wu", "ㄨㄚ": "wa", "ㄨㄛ": "wo", "ㄨㄞ": "wai",
    "ㄨㄟ": "wei", "ㄨㄢ": "wan", "ㄨㄣ": "wen", "ㄨㄤ": "wang", "ㄨㄥ": "weng", "ㄩ": "yu",
    "ㄩㄝ": "yue", "ㄩㄢ": "yuan", "ㄩㄣ": "yun", "ㄩㄥ": "yong",
}
GENERAL_FINAL_MAP = {
    "ㄚ": "a", "ㄛ": "o", "ㄜ": "e", "ㄝ": "e", "ㄞ": "ai", "ㄟ": "ei", "ㄠ": "ao", "ㄡ": "ou",
    "ㄢ": "an", "ㄣ": "en", "ㄤ": "ang", "ㄥ": "eng", "ㄦ": "er", "ㄧ": "i", "ㄧㄛ": "io",
    "ㄧㄚ": "ia", "ㄧㄝ": "ie", "ㄧㄠ": "iao", "ㄧㄡ": "iu", "ㄧㄢ": "ian", "ㄧㄣ": "in",
    "ㄧㄤ": "iang", "ㄧㄥ": "ing", "ㄨ": "u", "ㄨㄚ": "ua", "ㄨㄛ": "uo", "ㄨㄞ": "uai",
    "ㄨㄟ": "ui", "ㄨㄢ": "uan", "ㄨㄣ": "un", "ㄨㄤ": "uang", "ㄨㄥ": "ong", "ㄩ": "ü",
    "ㄩㄝ": "üe", "ㄩㄢ": "üan", "ㄩㄣ": "ün", "ㄩㄥ": "iong", "ㄭ+": "i", "ㄭ-": "i",
}
JQX_FINAL_MAP = dict(GENERAL_FINAL_MAP)
JQX_FINAL_MAP.update({
    "ㄩ": "u", "ㄩㄝ": "ue", "ㄩㄢ": "uan", "ㄩㄣ": "un", "ㄩㄥ": "iong",
})
TONE_RE = re.compile(r"^(.*?)([1-5])?$")


def split_tone(token: str) -> tuple[str, str]:
    match = TONE_RE.match(token)
    if not match:
        return token, ""
    base = match.group(1)
    tone = match.group(2) or ""
    return base, tone


def zhuyin_to_pinyin(initial: str | None, final: str) -> str:
    if initial is None or initial == "j":
        return ZERO_INITIAL_FINAL_MAP.get(final, final)
    init_py = INITIAL_MAP.get(initial, initial)
    if initial in {"ㄐ", "ㄑ", "ㄒ"}:
        final_py = JQX_FINAL_MAP.get(final, GENERAL_FINAL_MAP.get(final, final))
    else:
        final_py = GENERAL_FINAL_MAP.get(final, GENERAL_FINAL_MAP.get(final, final))
    return init_py + final_py


def merge_tokens_to_pinyin_syllables(token_spans: list[EngineSpan]) -> list[dict[str, Any]]:
    syllables: list[dict[str, Any]] = []
    index = 0
    while index < len(token_spans):
        seg = token_spans[index]
        token = seg.token or seg.text or ""
        base, tone = split_tone(token)

        if base in INITIALS_SET:
            if index + 1 < len(token_spans):
                next_seg = token_spans[index + 1]
                next_token = next_seg.token or next_seg.text or ""
                next_base, next_tone = split_tone(next_token)
                if next_base in FINALS_SET and next_base != "r":
                    pinyin = zhuyin_to_pinyin(base, next_base) + next_tone
                    raw_tokens = [token, next_token]
                    start_time = seg.start_time
                    end_time = next_seg.end_time
                    source_span_ids = [seg.span_id, next_seg.span_id]
                    index += 2
                    if index < len(token_spans):
                        maybe_r_span = token_spans[index]
                        maybe_r = maybe_r_span.token or maybe_r_span.text or ""
                        r_base, r_tone = split_tone(maybe_r)
                        if r_base == "r":
                            pinyin += "r" + r_tone
                            raw_tokens.append(maybe_r)
                            end_time = maybe_r_span.end_time
                            source_span_ids.append(maybe_r_span.span_id)
                            index += 1
                    syllables.append(
                        {
                            "raw_tokens": raw_tokens,
                            "pinyin": pinyin,
                            "start_time": start_time,
                            "end_time": end_time,
                            "source_spans": source_span_ids,
                            "sequence_index": len(syllables),
                        }
                    )
                    continue
            syllables.append(
                {
                    "raw_tokens": [token],
                    "pinyin": INITIAL_MAP.get(base, base) + tone,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "source_spans": [seg.span_id],
                    "sequence_index": len(syllables),
                }
            )
            index += 1
            continue

        if base in FINALS_SET:
            if base == "r":
                if syllables:
                    syllables[-1]["raw_tokens"].append(token)
                    syllables[-1]["pinyin"] += "r" + tone
                    syllables[-1]["end_time"] = seg.end_time
                    syllables[-1]["source_spans"].append(seg.span_id)
                else:
                    syllables.append(
                        {
                            "raw_tokens": [token],
                            "pinyin": "r" + tone,
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "source_spans": [seg.span_id],
                            "sequence_index": len(syllables),
                        }
                    )
                index += 1
                continue

            pinyin = zhuyin_to_pinyin(None, base) + tone
            raw_tokens = [token]
            start_time = seg.start_time
            end_time = seg.end_time
            source_span_ids = [seg.span_id]
            index += 1
            if index < len(token_spans):
                maybe_r_span = token_spans[index]
                maybe_r = maybe_r_span.token or maybe_r_span.text or ""
                r_base, r_tone = split_tone(maybe_r)
                if r_base == "r":
                    pinyin += "r" + r_tone
                    raw_tokens.append(maybe_r)
                    end_time = maybe_r_span.end_time
                    source_span_ids.append(maybe_r_span.span_id)
                    index += 1
            syllables.append(
                {
                    "raw_tokens": raw_tokens,
                    "pinyin": pinyin,
                    "start_time": start_time,
                    "end_time": end_time,
                    "source_spans": source_span_ids,
                    "sequence_index": len(syllables),
                }
            )
            continue

        syllables.append(
            {
                "raw_tokens": [token],
                "pinyin": token,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "source_spans": [seg.span_id],
                "sequence_index": len(syllables),
            }
        )
        index += 1
    return syllables


class ZhPinyinFromCTCUnitizer(TranscriptionAdapter):
    name = "zh_pinyin_from_ctc"
    unit_timing_semantics = "model_native_segment"

    def __init__(self, *, backend: str = "mms_phonetic") -> None:
        self.backend = backend

    def _raw_segment_spans(self, engine_output: EngineOutput):
        return engine_spans_by_level(engine_output, "token")

    def _unitize(self, engine_output: EngineOutput, *, language: str, tone_mode: str) -> list[TimedUnit]:
        token_spans = engine_spans_by_level(engine_output, "token")
        syllables = merge_tokens_to_pinyin_syllables(token_spans)
        units: list[TimedUnit] = []
        for index, syllable in enumerate(syllables):
            symbol = syllable["pinyin"]
            normalized_symbol, tone = split_pinyin_tone(symbol)
            if tone_mode != "ignore":
                normalized_symbol = symbol
            first_source = self._find_first_source_span(token_spans, syllable.get("source_spans", []))
            source_segment_index = first_source.segment_index if first_source and first_source.segment_index is not None else index
            units.append(
                TimedUnit(
                    unit_id=make_id("timed_unit"),
                    symbol=symbol or normalized_symbol,
                    normalized_symbol=normalized_symbol or symbol,
                    unit_type="pinyin_syllable",
                    language=language,
                    tone=tone,
                    start_time=float(syllable["start_time"]),
                    end_time=float(syllable["end_time"]),
                    confidence=None,
                    source_backend=self.backend,
                    raw_tokens=[str(tok) for tok in syllable.get("raw_tokens", [])],
                    metadata=build_unit_trace_metadata(
                        backend=self.backend,
                        source_segment_index=source_segment_index,
                        source_segment_level="token",
                        source_token_index=source_segment_index,
                        source_start_time=float(syllable["start_time"]),
                        source_end_time=float(syllable["end_time"]),
                        source_text=syllable["pinyin"],
                        normalized_text=normalized_symbol or symbol,
                        timing_mode="model_native_segment",
                        extra={
                            "engine": engine_output.engine,
                            "sequence_index": index,
                            "raw_pinyin": syllable["pinyin"],
                            "engine_source_span_ids": list(syllable.get("source_spans", [])),
                        },
                    ),
                )
            )
        return units

    def _find_first_source_span(self, token_spans: list[EngineSpan], span_ids: list[str]) -> EngineSpan | None:
        span_lookup = {span.span_id: span for span in token_spans}
        for span_id in span_ids:
            span = span_lookup.get(span_id)
            if span is not None:
                return span
        return None
