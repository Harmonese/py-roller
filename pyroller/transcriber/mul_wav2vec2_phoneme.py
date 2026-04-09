from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pyroller.domain import AudioArtifact, TimedUnit, TranscriptionResult
from pyroller.progress import ProgressReporter
from pyroller.transcriber.base import Transcriber
from pyroller.transcriber.mms_phonetic import ctc_token_segments
from pyroller.utils.ids import make_id
from pyroller.utils.text import normalize_ipa_phone, pinyin_syllable_to_ipa_tokens

logger = logging.getLogger("pyroller.transcriber")

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
TARGET_SR = 16000

# The model vocabulary mixes IPA-like labels, eSpeak/Kirshenbaum mnemonics,
# and language-specific labels (notably Mandarin pinyin-like finals with tone markers).
# We canonicalize those raw labels into a parser-facing IPA-ish phone space so the
# multilingual parser and aligner can operate on comparable units.
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
    "u\"": ["y"],
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


def _strip_pinyin_tone_marker(token: str) -> str:
    return _PY_TONE_SUFFIX_RE.sub("", token)


# Some labels in the model vocabulary are Mandarin pinyin-like labels with tone suffixes,
# but use IPA-ish glyphs (notably ɡ) and occasional dot separators.
# Convert those to pinyin-like spellings first, then reuse the existing pinyin->IPA helper.
_PY_PRE_CANONICAL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("ɡ", "g"),
    (".", ""),
    ("ə", "e"),  # yə5 -> ye ; uə5 -> ue ; aligns better with existing pinyin helper
    ("ɛ", "e"),  # iɛ5 -> ie ; yɛ5 -> yue
    ("ɔ", "o"),
    ("ɑ", "a"),
)


def _looks_like_model_pinyin_label(token: str) -> bool:
    if not _PY_PINYINISH_RE.match(token):
        return False
    # Be conservative. A bare IPA phone like "ɡ" should not be treated as a
    # Mandarin pinyin-ish label just because it contains the glyph ɡ. Reserve the
    # pinyin helper for labels that actually look like the model's Mandarin-style
    # finals/syllables (tone suffixes, dot separators, central vowels, or longer
    # multi-character labels containing ɡ).
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
    # remove model-specific boundary dots and trailing tone/stress digits where they survive
    value = value.replace(".", "")
    value = _strip_pinyin_tone_marker(value)

    long_match = _LONG_VOWEL_ASCII_RE.match(value)
    if long_match is not None:
        return [f"{long_match.group(1)}ː"]

    rhotic_match = _RHOTIC_VOWEL_RE.match(value)
    if rhotic_match is not None:
        return [rhotic_match.group(1), "ɹ"]

    # Split affricate/cluster-like composites only when we know the parser side usually
    # expects multiple phones rather than a monolithic token.
    if value in {"ts", "tsʰ"}:
        return [value]

    return [value] if value else []


class MultilingualWav2Vec2PhonemeTranscriber(Transcriber):
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = "cpu",
        target_sample_rate: int = TARGET_SR,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.trust_remote_code = trust_remote_code

    def transcribe(self, audio_artifact: AudioArtifact, language: str, tone_mode: str, progress: ProgressReporter | None = None) -> TranscriptionResult:
        stage = progress.stage("transcriber", total=5, unit="phase") if progress is not None else None
        if stage is not None:
            stage.phase("loading wav2vec2 backend")
        try:
            import librosa  # type: ignore
            import torch  # type: ignore
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "wav2vec2_phoneme dependencies are not installed. Install with: pip install librosa transformers torch"
            ) from exc

        audio_path = Path(audio_artifact.path) if audio_artifact.path is not None else None
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for wav2vec2_phoneme backend: {audio_artifact.path}")

        logger.info("Loading multilingual Wav2Vec2Phoneme model=%s device=%s", self.model_name, self.device)
        if stage is not None:
            stage.phase("loading wav2vec2 model")
        processor = Wav2Vec2Processor.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code)
        model = Wav2Vec2ForCTC.from_pretrained(self.model_name, trust_remote_code=self.trust_remote_code).to(self.device)
        model.eval()

        if stage is not None:
            stage.phase("loading audio")
        speech, sampling_rate = librosa.load(str(audio_path), sr=self.target_sample_rate, mono=True)
        audio_duration = float(len(speech) / sampling_rate) if len(speech) else 0.0

        if stage is not None:
            stage.phase("running phonetic inference")
        inputs = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if "attention_mask" in inputs else None

        with torch.no_grad():
            if attention_mask is not None:
                logits = model(input_values=input_values, attention_mask=attention_mask).logits
            else:
                logits = model(input_values=input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)[0]
        raw_transcription = processor.batch_decode(predicted_ids.unsqueeze(0), skip_special_tokens=True)[0]
        time_offset = model.config.inputs_to_logits_ratio / sampling_rate
        blank_id = processor.tokenizer.pad_token_id
        if blank_id is None:
            blank_id = model.config.pad_token_id
        if blank_id is None:
            raise RuntimeError("Unable to determine CTC blank token id for wav2vec2_phoneme backend")

        if stage is not None:
            stage.phase("post-processing transcription")
        token_segments = ctc_token_segments(
            pred_ids=predicted_ids,
            tokenizer=processor.tokenizer,
            blank_id=int(blank_id),
            time_offset=float(time_offset),
        )
        units, kept_token_count, expanded_unit_count, dropped_token_count = self._token_segments_to_units(
            token_segments,
            language=language,
        )

        raw_segments = [
            {
                "text": segment["token"],
                "start": segment["start_time"],
                "end": segment["end_time"],
                "start_frame": segment["start_frame"],
                "end_frame": segment["end_frame"],
            }
            for segment in token_segments
        ]

        logger.info(
            "Multilingual Wav2Vec2Phoneme transcription complete: %d token segments -> %d kept canonical tokens -> %d timed units (%d dropped)",
            len(token_segments),
            kept_token_count,
            len(units),
            dropped_token_count,
        )
        if stage is not None:
            stage.close("transcriber complete")
        return TranscriptionResult(
            language=language,
            backend="wav2vec2_phoneme",
            units=units,
            raw_text=raw_transcription or None,
            raw_segments=raw_segments,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "target_sample_rate": self.target_sample_rate,
                "audio_duration": audio_duration,
                "token_segment_count": len(token_segments),
                "canonical_token_count": kept_token_count,
                "expanded_unit_count": expanded_unit_count,
                "dropped_token_count": dropped_token_count,
                "source_audio": str(audio_artifact.path) if audio_artifact.path else None,
                "source_role": audio_artifact.role,
                "phoneme_standard": "canonical_ipa_from_model_labels",
            },
        )

    def _token_segments_to_units(
        self,
        token_segments: list[dict[str, Any]],
        language: str,
    ) -> tuple[list[TimedUnit], int, int, int]:
        units: list[TimedUnit] = []
        kept_token_count = 0
        expanded_unit_count = 0
        dropped_token_count = 0

        for token_index, segment in enumerate(token_segments):
            raw_phone = str(segment["token"])
            canonical_phones = _canonicalize_model_phone(raw_phone)
            if not canonical_phones:
                dropped_token_count += 1
                continue

            kept_token_count += 1
            expanded_unit_count += len(canonical_phones)
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
            duration = max(0.0, end_time - start_time)
            step = duration / max(1, len(canonical_phones))

            for split_index, canonical_phone in enumerate(canonical_phones):
                normalized = normalize_ipa_phone(canonical_phone)
                if not normalized:
                    continue
                unit_start = start_time + (step * split_index)
                unit_end = end_time if split_index == len(canonical_phones) - 1 else start_time + (step * (split_index + 1))
                units.append(
                    TimedUnit(
                        unit_id=make_id("timed_unit"),
                        symbol=canonical_phone,
                        normalized_symbol=normalized,
                        unit_type="ipa_phone",
                        language=language,
                        tone=None,
                        start_time=float(unit_start),
                        end_time=float(unit_end),
                        confidence=None,
                        source_backend="wav2vec2_phoneme",
                        raw_tokens=[raw_phone],
                        metadata={
                            "token_index": token_index,
                            "split_index": split_index,
                            "raw_phone": raw_phone,
                            "canonical_phone": canonical_phone,
                            "start_frame": segment.get("start_frame"),
                            "end_frame": segment.get("end_frame"),
                        },
                    )
                )
        return units, kept_token_count, expanded_unit_count, dropped_token_count
