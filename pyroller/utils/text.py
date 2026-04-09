from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Optional

try:
    import opencc
except ImportError:  # pragma: no cover
    opencc = None

try:
    from pypinyin import Style, lazy_pinyin
except ImportError:  # pragma: no cover
    Style = None
    lazy_pinyin = None

try:
    import pronouncing
except ImportError:  # pragma: no cover
    pronouncing = None

try:
    from g2p_en import G2p
except ImportError:  # pragma: no cover
    G2p = None

_CC_T2S = opencc.OpenCC("t2s") if opencc is not None else None
_CHINESE_RE = re.compile(r"[一-鿿]")
_NON_CHINESE_RE = re.compile(r"[^一-鿿]")
_ARABIC_NUMBER_RE = re.compile(r"\d+")
_CHINESE_DIGITS = "零一二三四五六七八九"
_SMALL_UNITS = ["", "十", "百", "千"]
_BIG_UNITS = ["", "万", "亿", "兆"]
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_ARPABET_TOKEN_RE = re.compile(r"^[A-Z]{1,5}[012]?$")
_ARPABET_STRESS_RE = re.compile(r"([A-Z]{1,5})([012])$")
_IPA_DIACRITIC_RE = re.compile(r"[ˈˌːˑ‿]")
_IPA_HINT_RE = re.compile(r"[ɐ-ʯəɚɝθðʃʒŋɲɾʔˈˌː]")
_FALLBACK_GRAPHEME_PREFIX = "GR_"
_KNOWN_ARPABET_BASES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P",
    "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
}
_G2P_INSTANCE = None

_JOINER_CHARS = {"'", "’", "-", "_"}

_GRUUT_LANGUAGE_MAP = {
    "en": "en-us",
    "ru": "ru-ru",
    "ar": "ar",
}
_ESPEAK_LANGUAGE_MAP = {
    "en": "en-us",
    "ru": "ru",
    "ar": "ar",
    "zh": "cmn",
}

_ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɚ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

_PINYIN_INITIAL_TO_IPA = {
    "b": "p",
    "p": "pʰ",
    "m": "m",
    "f": "f",
    "d": "t",
    "t": "tʰ",
    "n": "n",
    "l": "l",
    "g": "k",
    "k": "kʰ",
    "h": "x",
    "j": "tɕ",
    "q": "tɕʰ",
    "x": "ɕ",
    "zh": "ʈʂ",
    "ch": "ʈʂʰ",
    "sh": "ʂ",
    "r": "ʐ",
    "z": "ts",
    "c": "tsʰ",
    "s": "s",
}

_ZERO_PINYIN_TO_IPA = {
    "a": "a",
    "o": "o",
    "e": "ɤ",
    "ai": "ai",
    "ei": "ei",
    "ao": "au",
    "ou": "ou",
    "an": "an",
    "en": "ən",
    "ang": "aŋ",
    "eng": "əŋ",
    "er": "ɚ",
    "yi": "i",
    "ya": "ja",
    "yo": "jɔ",
    "yai": "jai",
    "yao": "jau",
    "ye": "jɛ",
    "you": "jou",
    "yan": "jɛn",
    "yin": "in",
    "yang": "jaŋ",
    "ying": "iŋ",
    "yong": "jʊŋ",
    "wu": "u",
    "wa": "wa",
    "wo": "wo",
    "wai": "wai",
    "wei": "wei",
    "wan": "wan",
    "wen": "wən",
    "wang": "waŋ",
    "weng": "wəŋ",
    "yu": "y",
    "yue": "yɛ",
    "yuan": "yɛn",
    "yun": "yn",
}

_PINYIN_FINAL_TO_IPA = {
    "a": "a",
    "o": "o",
    "e": "ɤ",
    "ai": "ai",
    "ei": "ei",
    "ao": "au",
    "ou": "ou",
    "an": "an",
    "en": "ən",
    "ang": "aŋ",
    "eng": "əŋ",
    "er": "ɚ",
    "i": "i",
    "ia": "ja",
    "iao": "jau",
    "ie": "jɛ",
    "iu": "jou",
    "iou": "jou",
    "ian": "jɛn",
    "in": "in",
    "iang": "jaŋ",
    "ing": "iŋ",
    "iong": "jʊŋ",
    "u": "u",
    "ua": "wa",
    "uo": "wo",
    "uai": "wai",
    "ui": "wei",
    "uei": "wei",
    "uan": "wan",
    "un": "wən",
    "uen": "wən",
    "uang": "waŋ",
    "ong": "ʊŋ",
    "ü": "y",
    "üe": "yɛ",
    "üan": "yɛn",
    "ün": "yn",
}

_SPECIAL_PINYIN_FINALS = {
    ("zh", "i"): "ʅ",
    ("ch", "i"): "ʅ",
    ("sh", "i"): "ʅ",
    ("r", "i"): "ʅ",
    ("z", "i"): "ɿ",
    ("c", "i"): "ɿ",
    ("s", "i"): "ɿ",
}


# ---------------------------------------------------------------------------
# General text normalization helpers
# ---------------------------------------------------------------------------

def to_simplified(text: str) -> str:
    if _CC_T2S is None:
        return text
    return _CC_T2S.convert(text)


def extract_chinese(text: str) -> str:
    return _NON_CHINESE_RE.sub("", text)



def integer_to_chinese_numerals(number: int) -> str:
    if number == 0:
        return _CHINESE_DIGITS[0]
    if number < 0:
        return "负" + integer_to_chinese_numerals(abs(number))

    def _convert_group(value: int) -> str:
        if value == 0:
            return ""
        pieces: list[str] = []
        zero_pending = False
        digits = list(map(int, f"{value:04d}"))
        for idx, digit in enumerate(digits):
            unit_idx = 3 - idx
            if digit == 0:
                if pieces:
                    zero_pending = True
                continue
            if zero_pending:
                pieces.append("零")
                zero_pending = False
            if not (digit == 1 and unit_idx == 1 and not pieces):
                pieces.append(_CHINESE_DIGITS[digit])
            pieces.append(_SMALL_UNITS[unit_idx])
        return "".join(pieces)

    groups: list[int] = []
    remaining = number
    while remaining > 0:
        groups.append(remaining % 10000)
        remaining //= 10000

    pieces: list[str] = []
    zero_between_groups = False
    for group_index in range(len(groups) - 1, -1, -1):
        group_value = groups[group_index]
        if group_value == 0:
            if pieces:
                zero_between_groups = True
            continue
        if zero_between_groups or (pieces and group_value < 1000):
            if pieces and pieces[-1] != "零":
                pieces.append("零")
        zero_between_groups = False
        pieces.append(_convert_group(group_value))
        pieces.append(_BIG_UNITS[group_index])

    result = "".join(pieces).rstrip("零")
    return result or _CHINESE_DIGITS[0]



def replace_arabic_numbers_with_chinese(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            value = int(raw)
        except ValueError:
            return raw
        return integer_to_chinese_numerals(value)

    return _ARABIC_NUMBER_RE.sub(_replace, text)



def normalize_chinese_text(text: str) -> str:
    simplified = to_simplified(text)
    normalized_numbers = replace_arabic_numbers_with_chinese(simplified)
    return extract_chinese(normalized_numbers)



def split_pinyin_tone(syllable: str) -> tuple[str, Optional[str]]:
    if syllable and syllable[-1].isdigit():
        return syllable[:-1], syllable[-1]
    return syllable, None



def chinese_text_to_pinyin_syllables(text: str, tone_mode: str = "ignore") -> list[dict[str, Optional[str]]]:
    if lazy_pinyin is None or Style is None:
        raise RuntimeError("pypinyin is required for Chinese parser/transcriber normalization.")

    normalized = normalize_chinese_text(text)
    if not normalized:
        return []

    style = Style.TONE3 if tone_mode == "keep" else Style.NORMAL
    syllables = lazy_pinyin(normalized, style=style, neutral_tone_with_five=True)

    results: list[dict[str, Optional[str]]] = []
    for char, syllable in zip(normalized, syllables):
        base, tone = split_pinyin_tone(syllable)
        results.append(
            {
                "symbol": syllable,
                "normalized_symbol": base if tone_mode == "ignore" else syllable,
                "tone": tone,
                "source_char": char,
            }
        )
    return results



def normalize_english_text(text: str) -> str:
    return " ".join(_ENGLISH_WORD_RE.findall(text.lower()))



def split_arpabet_stress(phone: str) -> tuple[str, Optional[str]]:
    match = _ARPABET_STRESS_RE.match(phone)
    if match is None:
        return phone.upper(), None
    return match.group(1).upper(), match.group(2)



def normalize_arpabet_phone(phone: str) -> str:
    base, _ = split_arpabet_stress(phone)
    return base



def text_looks_like_arpabet(text: str) -> bool:
    tokens = [token.strip() for token in text.split() if token.strip()]
    if not tokens:
        return False
    normalized = [normalize_arpabet_phone(token) for token in tokens]
    if not all(_ARPABET_TOKEN_RE.match(token.upper()) for token in tokens):
        return False
    return all(token in _KNOWN_ARPABET_BASES or token.startswith(_FALLBACK_GRAPHEME_PREFIX) for token in normalized)



def arpabet_text_to_phone_tokens(text: str) -> list[str]:
    return [token.upper() for token in text.split() if token.strip()]



def _get_g2p_instance():
    global _G2P_INSTANCE
    if G2p is None:
        return None
    if _G2P_INSTANCE is None:
        _G2P_INSTANCE = G2p()
    return _G2P_INSTANCE



def _fallback_word_to_grapheme_units(word: str) -> list[str]:
    return [f"{_FALLBACK_GRAPHEME_PREFIX}{char.upper()}" for char in word if char.isalpha()]



def english_word_to_arpabet(word: str) -> list[str]:
    normalized = word.lower()
    if not normalized:
        return []

    g2p = _get_g2p_instance()
    if g2p is not None:
        phones = [str(item).upper() for item in g2p(normalized) if isinstance(item, str)]
        phones = [item for item in phones if _ARPABET_TOKEN_RE.match(item)]
        if phones:
            return phones

    if pronouncing is not None:
        candidates = pronouncing.phones_for_word(normalized)
        if candidates:
            return [token.upper() for token in candidates[0].split() if token.strip()]

    return _fallback_word_to_grapheme_units(normalized)



def english_text_to_arpabet_units(text: str) -> list[dict[str, Optional[str]]]:
    if text_looks_like_arpabet(text):
        tokens = arpabet_text_to_phone_tokens(text)
        return [
            {
                "symbol": token,
                "normalized_symbol": normalize_arpabet_phone(token),
                "stress": split_arpabet_stress(token)[1],
                "source_word": None,
            }
            for token in tokens
        ]

    words = _ENGLISH_WORD_RE.findall(text)
    results: list[dict[str, Optional[str]]] = []
    for word in words:
        for token in english_word_to_arpabet(word):
            base, stress = split_arpabet_stress(token)
            results.append(
                {
                    "symbol": token,
                    "normalized_symbol": base,
                    "stress": stress,
                    "source_word": word,
                }
            )
    return results


# ---------------------------------------------------------------------------
# IPA helpers
# ---------------------------------------------------------------------------

def normalize_ipa_phone(phone: str) -> str:
    compact = _IPA_DIACRITIC_RE.sub("", phone).strip()
    return compact or phone.strip()



def text_looks_like_ipa(text: str) -> bool:
    if _IPA_HINT_RE.search(text):
        return True
    tokens = [token.strip() for token in text.split() if token.strip()]
    if not tokens:
        return False
    # Be conservative: only treat input as literal IPA when it contains explicit
    # IPA-only symbols/diacritics. Do not infer IPA merely because the text is
    # non-Latin; otherwise Cyrillic/Chinese/Arabic orthography gets misrouted as
    # ipa_literal instead of going through gruut/dedicated/espeak routing.
    ipaish_tokens = 0
    for token in tokens:
        if _IPA_HINT_RE.search(token):
            ipaish_tokens += 1
            continue
        if any(ch in token for ch in ("ː", "ˑ", "ʰ", "ʲ", "ʷ", "̃", "͡", "̥", "̬", "̩", "̯")):
            ipaish_tokens += 1
    return ipaish_tokens > 0 and ipaish_tokens == len(tokens)



def ipa_text_to_phone_tokens(text: str) -> list[str]:
    return [token for token in text.split() if token.strip()]


# ---------------------------------------------------------------------------
# Optional backends for multilingual text -> IPA
# ---------------------------------------------------------------------------

def _import_gruut():  # pragma: no cover - import wrapper
    try:
        from gruut import is_language_supported, sentences
    except ImportError:
        return None, None
    return is_language_supported, sentences



def _import_phonemizer():  # pragma: no cover - import wrapper
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
    except ImportError:
        return None, None
    return phonemize, Separator



def gruut_language_for(language: str) -> Optional[str]:
    return _GRUUT_LANGUAGE_MAP.get(language)



def espeak_language_for(language: str) -> Optional[str]:
    return _ESPEAK_LANGUAGE_MAP.get(language)



def language_supported_by_gruut(language: str) -> bool:
    is_language_supported, _ = _import_gruut()
    if is_language_supported is None:
        return False
    lang_code = gruut_language_for(language)
    if not lang_code:
        return False
    try:
        return bool(is_language_supported(lang_code))
    except Exception:
        return False



def language_supported_by_espeak(language: str) -> bool:
    return espeak_language_for(language) is not None and _import_phonemizer()[0] is not None



def phonemize_text_with_gruut(text: str, language: str) -> list[dict[str, Optional[str]]]:
    lang_code = gruut_language_for(language)
    _, gruut_sentences = _import_gruut()
    if gruut_sentences is None or not lang_code:
        return []

    results: list[dict[str, Optional[str]]] = []
    try:
        for sentence in gruut_sentences(
            text,
            lang=lang_code,
            ssml=False,
            espeak=False,
            phonemes=True,
            explicit_lang=True,
            punctuations=False,
            major_breaks=False,
            minor_breaks=False,
            break_phonemes=False,
            pos=False,
        ):
            for word in sentence:
                phonemes = list(getattr(word, "phonemes", None) or [])
                if not phonemes:
                    continue
                if getattr(word, "is_break", False) or getattr(word, "is_punctuation", False):
                    continue
                source_word = getattr(word, "text", None)
                source_lang = getattr(word, "lang", lang_code)
                for phone in phonemes:
                    phone_text = str(phone).strip()
                    if not phone_text or phone_text in {"|", "‖"}:
                        continue
                    results.append(
                        {
                            "symbol": phone_text,
                            "normalized_symbol": normalize_ipa_phone(phone_text),
                            "stress": None,
                            "source_word": source_word,
                            "source_language": source_lang,
                            "backend": "gruut",
                        }
                    )
    except Exception:
        return []
    return results



def phonemize_text_with_espeak(text: str, language: str) -> list[dict[str, Optional[str]]]:
    phonemize, Separator = _import_phonemizer()
    lang_code = espeak_language_for(language)
    if phonemize is None or Separator is None or not lang_code:
        return []

    try:
        phonemized = phonemize(
            [text],
            language=lang_code,
            backend="espeak",
            separator=Separator(phone=" ", word=" | "),
            strip=True,
            preserve_punctuation=False,
            with_stress=True,
            language_switch="remove-flags",
            words_mismatch="ignore",
            njobs=1,
        )
    except Exception:
        return []

    if not phonemized:
        return []

    results: list[dict[str, Optional[str]]] = []
    for word_chunk in str(phonemized[0]).split(" | "):
        word_chunk = word_chunk.strip()
        if not word_chunk:
            continue
        for phone in word_chunk.split():
            if not phone.strip():
                continue
            results.append(
                {
                    "symbol": phone,
                    "normalized_symbol": normalize_ipa_phone(phone),
                    "stress": None,
                    "source_word": None,
                    "source_language": lang_code,
                    "backend": "phonemizer_espeak",
                }
            )
    return results


# ---------------------------------------------------------------------------
# Dedicated fallback routes
# ---------------------------------------------------------------------------

def arpabet_text_to_ipa_units(text: str) -> list[dict[str, Optional[str]]]:
    results: list[dict[str, Optional[str]]] = []
    for unit in english_text_to_arpabet_units(text):
        base = unit["normalized_symbol"]
        phone = _ARPABET_TO_IPA.get(base, base.lower())
        results.append(
            {
                "symbol": phone,
                "normalized_symbol": normalize_ipa_phone(phone),
                "stress": unit.get("stress"),
                "source_word": unit.get("source_word"),
                "source_language": "en",
                "backend": "en_arpabet_dedicated",
            }
        )
    return results



def _normalize_pinyin_final(initial: str, final: str) -> str:
    if initial in {"j", "q", "x"} and final.startswith("u"):
        if final == "u":
            return "ü"
        if final.startswith("uan"):
            return final.replace("uan", "üan", 1)
        if final.startswith("ue"):
            return final.replace("ue", "üe", 1)
        if final.startswith("un"):
            return final.replace("un", "ün", 1)
    return final.replace("v", "ü")



def pinyin_syllable_to_ipa_tokens(syllable: str) -> list[str]:
    base, _ = split_pinyin_tone(syllable.lower())
    if not base:
        return []
    if base in _ZERO_PINYIN_TO_IPA:
        return [_ZERO_PINYIN_TO_IPA[base]]

    initial = ""
    final = base
    for candidate in ("zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s"):
        if base.startswith(candidate):
            initial = candidate
            final = base[len(candidate):]
            break

    final = _normalize_pinyin_final(initial, final)
    special = _SPECIAL_PINYIN_FINALS.get((initial, final))
    if special is not None:
        final_phone = special
    else:
        final_phone = _PINYIN_FINAL_TO_IPA.get(final, final)

    tokens: list[str] = []
    initial_phone = _PINYIN_INITIAL_TO_IPA.get(initial)
    if initial_phone:
        tokens.append(initial_phone)
    if final_phone:
        tokens.append(final_phone)
    return tokens



def chinese_text_to_ipa_units(text: str) -> list[dict[str, Optional[str]]]:
    syllables = chinese_text_to_pinyin_syllables(text, tone_mode="keep")
    results: list[dict[str, Optional[str]]] = []
    for syllable in syllables:
        raw_symbol = syllable["symbol"] or syllable["normalized_symbol"] or ""
        source_char = syllable.get("source_char")
        for phone in pinyin_syllable_to_ipa_tokens(raw_symbol):
            results.append(
                {
                    "symbol": phone,
                    "normalized_symbol": normalize_ipa_phone(phone),
                    "stress": syllable.get("tone"),
                    "source_word": source_char,
                    "source_language": "zh",
                    "backend": "zh_dedicated",
                }
            )
    return results


# ---------------------------------------------------------------------------
# Multilingual routing
# ---------------------------------------------------------------------------

def _char_script_language(ch: str) -> Optional[str]:
    code = ord(ch)
    if 0x4E00 <= code <= 0x9FFF:
        return "zh"
    if 0x3400 <= code <= 0x4DBF:
        return "zh"
    if 0x0600 <= code <= 0x06FF or 0x0750 <= code <= 0x077F or 0x08A0 <= code <= 0x08FF:
        return "ar"
    if 0x0400 <= code <= 0x04FF or 0x0500 <= code <= 0x052F:
        return "ru"
    if "LATIN" in unicodedata.name(ch, ""):
        return "en"
    return None



def split_multilingual_text_segments(text: str) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    buffer: list[str] = []
    current_language: Optional[str] = None
    seg_start: Optional[int] = None

    def flush(end_index: int) -> None:
        nonlocal buffer, current_language, seg_start
        if not buffer or current_language is None or seg_start is None:
            buffer = []
            current_language = None
            seg_start = None
            return
        segment_text = "".join(buffer).strip()
        if segment_text:
            segments.append(
                {
                    "text": segment_text,
                    "language": current_language,
                    "start": seg_start,
                    "end": end_index,
                }
            )
        buffer = []
        current_language = None
        seg_start = None

    for idx, ch in enumerate(text):
        if ch.isspace():
            flush(idx)
            continue

        lang = _char_script_language(ch)
        if lang is None:
            if buffer and current_language is not None and (ch.isdigit() or ch in _JOINER_CHARS):
                buffer.append(ch)
            else:
                flush(idx)
            continue

        if current_language is None:
            current_language = lang
            seg_start = idx
            buffer.append(ch)
            continue

        if lang == current_language:
            buffer.append(ch)
            continue

        flush(idx)
        current_language = lang
        seg_start = idx
        buffer.append(ch)

    flush(len(text))
    return segments



def _route_segment_to_ipa_units(segment_text: str, language: str) -> tuple[list[dict[str, Optional[str]]], str]:
    if not segment_text.strip():
        return [], "empty"

    if language_supported_by_gruut(language):
        units = phonemize_text_with_gruut(segment_text, language)
        if units:
            return units, "gruut"

    try:
        if language == "zh":
            units = chinese_text_to_ipa_units(segment_text)
            if units:
                return units, "dedicated"
        elif language == "en":
            units = arpabet_text_to_ipa_units(segment_text)
            if units:
                return units, "dedicated"
    except Exception:
        pass

    if language_supported_by_espeak(language):
        units = phonemize_text_with_espeak(segment_text, language)
        if units:
            return units, "phonemizer_espeak"

    return [], "empty"



def multilingual_text_to_ipa_units(
    text: str,
) -> list[dict[str, Optional[str]]]:

    if text_looks_like_ipa(text):
        tokens = ipa_text_to_phone_tokens(text)
        return [
            {
                "symbol": token,
                "normalized_symbol": normalize_ipa_phone(token),
                "stress": None,
                "source_word": None,
                "source_language": "ipa",
                "backend": "ipa_literal",
            }
            for token in tokens
            if token.strip()
        ]

    results: list[dict[str, Optional[str]]] = []
    for segment_index, segment in enumerate(split_multilingual_text_segments(text)):
        segment_text = str(segment["text"])
        segment_language = str(segment["language"])
        segment_units, route = _route_segment_to_ipa_units(segment_text, segment_language)
        for unit in segment_units:
            merged = dict(unit)
            merged.setdefault("source_language", segment_language)
            merged.setdefault("backend", route)
            merged["segment_language"] = segment_language
            merged["segment_text"] = segment_text
            merged["segment_index"] = segment_index
            results.append(merged)
    return results



def summarize_multilingual_routes(text: str) -> dict[str, object]:
    route_counter: Counter[str] = Counter()
    language_counter: Counter[str] = Counter()
    segments_summary: list[dict[str, object]] = []
    for segment in split_multilingual_text_segments(text):
        units, route = _route_segment_to_ipa_units(str(segment["text"]), str(segment["language"]))
        route_counter[route] += 1
        language_counter[str(segment["language"])] += 1
        segments_summary.append(
            {
                "text": segment["text"],
                "language": segment["language"],
                "route": route,
                "unit_count": len(units),
            }
        )
    return {
        "route_counts": dict(route_counter),
        "language_counts": dict(language_counter),
        "segments": segments_summary,
    }




# ---------------------------------------------------------------------------
# zh router: segmented text -> pinyin syllables
# ---------------------------------------------------------------------------

_ZH_ROUTER_FOREIGN_LEXICON: dict[str, list[str]] = {
    "ok": ["ou", "kei"],
    "okay": ["ou", "kei"],
    "okey": ["ou", "kei"],
    "oh": ["ou"],
    "yo": ["yo"],
    "yeah": ["ye"],
    "yah": ["ya"],
    "wow": ["wa"],
    "hi": ["hai"],
    "bye": ["bai"],
    "baby": ["bei", "bi"],
    "lover": ["la", "fo"],
    "love": ["la", "fu"],
    "rap": ["ra", "pu"],
    "rapper": ["ra", "pa"],
    "hiphop": ["xi", "ha"],
    "dj": ["di", "jie"],
    "mc": ["em", "xi"],
    "ktv": ["kei", "ti", "wei"],
    "bgm": ["bi", "ji", "em"],
    "mv": ["em", "wei"],
    "vip": ["wei", "ai", "pi"],
    "cpu": ["xi", "pi", "you"],
    "gpu": ["ji", "pi", "you"],
    "abc": ["ei", "bi", "xi"],
}

_ZH_ROUTER_LETTER_NAME_PINYIN: dict[str, list[str]] = {
    "A": ["ei"],
    "B": ["bi"],
    "C": ["xi"],
    "D": ["di"],
    "E": ["yi"],
    "F": ["ai", "fu"],
    "G": ["ji"],
    "H": ["ei", "qi"],
    "I": ["ai"],
    "J": ["jie"],
    "K": ["kei"],
    "L": ["ai", "lu"],
    "M": ["em"],
    "N": ["en"],
    "O": ["ou"],
    "P": ["pi"],
    "Q": ["qiu"],
    "R": ["a"],
    "S": ["ai", "si"],
    "T": ["ti"],
    "U": ["you"],
    "V": ["wei"],
    "W": ["da", "bu", "liu"],
    "X": ["ai", "ke", "si"],
    "Y": ["wai"],
    "Z": ["zi"],
}

_ZH_ROUTER_ARPABET_VOWELS = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
}

_ZH_ROUTER_ARPABET_INITIAL_TO_PINYIN = {
    "B": "b",
    "P": "p",
    "M": "m",
    "F": "f",
    "D": "d",
    "T": "t",
    "N": "n",
    "L": "l",
    "G": "g",
    "K": "k",
    "HH": "h",
    "JH": "j",
    "CH": "ch",
    "SH": "sh",
    "ZH": "r",
    "R": "r",
    "S": "s",
    "Z": "z",
    "TH": "s",
    "DH": "z",
    "V": "w",
    "W": "w",
    "Y": "y",
}

_ZH_ROUTER_ZERO_INITIAL_FINAL = {
    "a": "a",
    "ai": "ai",
    "an": "an",
    "ang": "ang",
    "ao": "ao",
    "e": "e",
    "ei": "ei",
    "en": "en",
    "eng": "eng",
    "er": "er",
    "i": "yi",
    "in": "yin",
    "ing": "ying",
    "o": "o",
    "ong": "weng",
    "ou": "ou",
    "u": "wu",
    "un": "wen",
}


def _zh_router_char_type(ch: str) -> str:
    if ch.isspace():
        return "space"
    code = ord(ch)
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "han"
    if ch.isdigit():
        return "digit"
    if ch in _JOINER_CHARS:
        return "joiner"
    if ch.isalpha() and "LATIN" in unicodedata.name(ch, ""):
        return "latin"
    return "symbol"



def split_zh_router_segments(text: str) -> list[dict[str, object]]:
    normalized_text = unicodedata.normalize("NFKC", text or "")
    segments: list[dict[str, object]] = []
    buffer: list[str] = []
    current_type: Optional[str] = None
    seg_start: Optional[int] = None

    def flush(end_index: int) -> None:
        nonlocal buffer, current_type, seg_start
        if not buffer or current_type is None or seg_start is None:
            buffer = []
            current_type = None
            seg_start = None
            return
        segment_text = "".join(buffer)
        if segment_text:
            segments.append({"text": segment_text, "segment_type": current_type, "start": seg_start, "end": end_index})
        buffer = []
        current_type = None
        seg_start = None

    for idx, ch in enumerate(normalized_text):
        segment_type = _zh_router_char_type(ch)
        if segment_type == "space":
            flush(idx)
            continue
        if segment_type == "joiner":
            if buffer and current_type == "latin":
                buffer.append(ch)
            else:
                flush(idx)
            continue
        if current_type is None:
            current_type = segment_type
            seg_start = idx
            buffer.append(ch)
            continue
        if segment_type == current_type:
            buffer.append(ch)
            continue
        flush(idx)
        current_type = segment_type
        seg_start = idx
        buffer.append(ch)

    flush(len(normalized_text))
    return segments



def _borrowed_pinyin_syllables_to_units(
    syllables: list[str],
    *,
    source_text: str,
    backend: str,
) -> list[dict[str, Optional[str]]]:
    return [
        {
            "symbol": syllable,
            "normalized_symbol": syllable,
            "tone": None,
            "source_text": source_text,
            "backend": backend,
        }
        for syllable in syllables
        if syllable
    ]



def _digit_text_to_pinyin_units(text: str, tone_mode: str) -> list[dict[str, Optional[str]]]:
    numeral_text = "".join(_CHINESE_DIGITS[int(ch)] for ch in text if ch.isdigit())
    units = chinese_text_to_pinyin_syllables(numeral_text, tone_mode=tone_mode)
    for unit, source_char in zip(units, text):
        unit["source_text"] = source_char
        unit["backend"] = "digit_by_digit"
    return units



def _choose_proxy_initial(onset: list[str]) -> str:
    for phone in onset:
        mapped = _ZH_ROUTER_ARPABET_INITIAL_TO_PINYIN.get(phone)
        if mapped:
            return mapped
    return ""



def _proxy_final_from_vowel(vowel: str, coda: list[str]) -> str:
    coda_set = set(coda)
    if "NG" in coda_set:
        if vowel in {"IH", "IY"}:
            return "ing"
        if vowel in {"UH", "UW", "OW"}:
            return "ong"
        return "ang"
    if "N" in coda_set:
        if vowel in {"IH", "IY", "EH"}:
            return "in"
        if vowel in {"UH", "UW", "OW"}:
            return "un"
        return "an"

    return {
        "AA": "a",
        "AE": "ai",
        "AH": "a",
        "AO": "ao",
        "AW": "ao",
        "AY": "ai",
        "EH": "ei",
        "ER": "er",
        "EY": "ei",
        "IH": "i",
        "IY": "i",
        "OW": "ou",
        "OY": "ou",
        "UH": "u",
        "UW": "u",
    }.get(vowel, "a")



def _compose_proxy_pinyin(initial: str, final: str) -> str:
    if not initial:
        return _ZH_ROUTER_ZERO_INITIAL_FINAL.get(final, final)
    return f"{initial}{final}"



def arpabet_phones_to_pinyin_proxy_syllables(phones: list[str]) -> list[str]:
    normalized = [normalize_arpabet_phone(phone) for phone in phones if normalize_arpabet_phone(phone)]
    if not normalized:
        return []

    syllables: list[str] = []
    onset: list[str] = []
    index = 0
    while index < len(normalized):
        phone = normalized[index]
        if phone not in _ZH_ROUTER_ARPABET_VOWELS:
            onset.append(phone)
            index += 1
            continue

        coda: list[str] = []
        lookahead = index + 1
        while lookahead < len(normalized) and normalized[lookahead] not in _ZH_ROUTER_ARPABET_VOWELS:
            if lookahead + 1 < len(normalized) and normalized[lookahead + 1] in _ZH_ROUTER_ARPABET_VOWELS:
                break
            coda.append(normalized[lookahead])
            lookahead += 1

        syllables.append(_compose_proxy_pinyin(_choose_proxy_initial(onset), _proxy_final_from_vowel(phone, coda)))
        onset = []
        index = lookahead

    return [syllable for syllable in syllables if syllable]



def english_word_to_pinyin_proxy_syllables(word: str) -> tuple[list[str], str]:
    normalized = unicodedata.normalize("NFKC", word or "").strip()
    if not normalized:
        return [], "empty"

    lowered = normalized.lower()
    collapsed = re.sub(r"[^a-z]", "", lowered)
    if not collapsed:
        return [], "empty"

    if collapsed in _ZH_ROUTER_FOREIGN_LEXICON:
        return list(_ZH_ROUTER_FOREIGN_LEXICON[collapsed]), "lexicon"

    if normalized.isalpha() and normalized.isupper():
        syllables: list[str] = []
        for letter in normalized:
            syllables.extend(_ZH_ROUTER_LETTER_NAME_PINYIN.get(letter.upper(), [letter.lower()]))
        return syllables, "acronym_letter_name"

    phones = english_word_to_arpabet(collapsed)
    if phones and not all(phone.startswith(_FALLBACK_GRAPHEME_PREFIX) for phone in phones):
        proxy = arpabet_phones_to_pinyin_proxy_syllables(phones)
        if proxy:
            return proxy, "arpabet_proxy"

    syllables = []
    for letter in collapsed.upper():
        syllables.extend(_ZH_ROUTER_LETTER_NAME_PINYIN.get(letter, [letter.lower()]))
    return syllables, "grapheme_letter_name"



def foreign_text_to_pinyin_proxy_units(text: str, tone_mode: str = "ignore") -> tuple[list[dict[str, Optional[str]]], str]:
    del tone_mode
    normalized = unicodedata.normalize("NFKC", text or "").strip()
    if not normalized:
        return [], "empty"

    words = _ENGLISH_WORD_RE.findall(normalized)
    if not words and normalized.isalpha():
        words = [normalized]
    all_units: list[dict[str, Optional[str]]] = []
    routes: list[str] = []
    for word in words:
        syllables, route = english_word_to_pinyin_proxy_syllables(word)
        routes.append(route)
        all_units.extend(_borrowed_pinyin_syllables_to_units(syllables, source_text=word, backend=route))
    route_name = routes[0] if len(set(routes)) == 1 and routes else "mixed_foreign"
    return all_units, route_name or "empty"



def route_zh_segment_to_pinyin_units(
    segment_text: str,
    segment_type: str,
    tone_mode: str = "ignore",
) -> tuple[list[dict[str, Optional[str]]], str]:
    if not segment_text:
        return [], "empty"
    if segment_type == "han":
        units = chinese_text_to_pinyin_syllables(to_simplified(segment_text), tone_mode=tone_mode)
        for unit, source_char in zip(units, to_simplified(segment_text)):
            unit["source_text"] = source_char
            unit["backend"] = "han_pinyin"
        return units, "han_pinyin"
    if segment_type == "digit":
        return _digit_text_to_pinyin_units(segment_text, tone_mode=tone_mode), "digit_by_digit"
    if segment_type == "latin":
        return foreign_text_to_pinyin_proxy_units(segment_text, tone_mode=tone_mode)
    return [], "symbol_skip"



def segmented_zh_text_to_pinyin_units(
    text: str,
    tone_mode: str = "ignore",
) -> tuple[list[dict[str, Optional[str]]], dict[str, object]]:
    route_counter: Counter[str] = Counter()
    segment_counter: Counter[str] = Counter()
    segments_summary: list[dict[str, object]] = []
    results: list[dict[str, Optional[str]]] = []

    for segment_index, segment in enumerate(split_zh_router_segments(text)):
        segment_text = str(segment["text"])
        segment_type = str(segment["segment_type"])
        segment_counter[segment_type] += 1
        units, route = route_zh_segment_to_pinyin_units(segment_text, segment_type, tone_mode=tone_mode)
        route_counter[route] += 1
        segments_summary.append(
            {
                "text": segment_text,
                "segment_type": segment_type,
                "route": route,
                "unit_count": len(units),
            }
        )
        for unit in units:
            merged = dict(unit)
            merged.setdefault("source_text", segment_text)
            merged.setdefault("backend", route)
            merged["segment_type"] = segment_type
            merged["segment_text"] = segment_text
            merged["segment_index"] = segment_index
            results.append(merged)

    summary: dict[str, object] = {
        "route_counts": dict(route_counter),
        "segment_type_counts": dict(segment_counter),
        "segments": segments_summary,
        "foreign_segment_count": sum(
            1 for item in segments_summary if str(item["segment_type"]) in {"latin", "digit"} and int(item["unit_count"]) > 0
        ),
    }
    return results, summary



def summarize_zh_router_routes(text: str, tone_mode: str = "ignore") -> dict[str, object]:
    _, summary = segmented_zh_text_to_pinyin_units(text, tone_mode=tone_mode)
    return summary
