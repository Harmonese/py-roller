from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from pyroller.domain import AudioArtifact, TimedUnit, TranscriptionResult
from pyroller.progress import ProgressReporter
from pyroller.transcriber.base import Transcriber
from pyroller.utils.ids import make_id
from pyroller.utils.text import split_pinyin_tone

logger = logging.getLogger("pyroller.transcriber")

MODEL_NAME = "Chuatury/wav2vec2-mms-1b-cmn-phonetic"
TARGET_SR = 16000
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


def ctc_token_segments(pred_ids: Any, tokenizer: Any, blank_id: int, time_offset: float) -> list[dict[str, Any]]:
    if hasattr(pred_ids, "tolist"):
        pred_ids = pred_ids.tolist()
    segments: list[dict[str, Any]] = []
    if not pred_ids:
        return segments

    def normalize_token(token_id: int) -> str | None:
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        if token is None:
            return None
        if token in tokenizer.all_special_tokens:
            return None
        if token.strip() == "":
            return None
        return token

    run_token_id = pred_ids[0]
    run_start = 0
    for index in range(1, len(pred_ids) + 1):
        end_of_run = (index == len(pred_ids)) or (pred_ids[index] != run_token_id)
        if end_of_run:
            if run_token_id != blank_id:
                token = normalize_token(run_token_id)
                if token is not None:
                    segments.append(
                        {
                            "token": token,
                            "start_time": run_start * time_offset,
                            "end_time": index * time_offset,
                            "start_frame": run_start,
                            "end_frame": index,
                        }
                    )
            if index < len(pred_ids):
                run_token_id = pred_ids[index]
                run_start = index
    return segments



def zhuyin_to_pinyin(initial: str | None, final: str) -> str:
    if initial is None or initial == "j":
        return ZERO_INITIAL_FINAL_MAP.get(final, final)
    init_py = INITIAL_MAP.get(initial, initial)
    if initial in {"ㄐ", "ㄑ", "ㄒ"}:
        final_py = JQX_FINAL_MAP.get(final, GENERAL_FINAL_MAP.get(final, final))
    else:
        final_py = GENERAL_FINAL_MAP.get(final, GENERAL_FINAL_MAP.get(final, final))
    return init_py + final_py



def merge_tokens_to_pinyin_syllables(token_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    syllables: list[dict[str, Any]] = []
    index = 0
    while index < len(token_segments):
        seg = token_segments[index]
        token = seg["token"]
        base, tone = split_tone(token)

        if base in INITIALS_SET:
            if index + 1 < len(token_segments):
                next_seg = token_segments[index + 1]
                next_token = next_seg["token"]
                next_base, next_tone = split_tone(next_token)
                if next_base in FINALS_SET and next_base != "r":
                    pinyin = zhuyin_to_pinyin(base, next_base) + next_tone
                    raw_tokens = [token, next_token]
                    start_time = seg["start_time"]
                    end_time = next_seg["end_time"]
                    index += 2
                    if index < len(token_segments):
                        maybe_r = token_segments[index]["token"]
                        r_base, r_tone = split_tone(maybe_r)
                        if r_base == "r":
                            pinyin += "r" + r_tone
                            raw_tokens.append(maybe_r)
                            end_time = token_segments[index]["end_time"]
                            index += 1
                    syllables.append(
                        {
                            "raw_tokens": raw_tokens,
                            "pinyin": pinyin,
                            "start_time": start_time,
                            "end_time": end_time,
                        }
                    )
                    continue
            syllables.append(
                {
                    "raw_tokens": [token],
                    "pinyin": INITIAL_MAP.get(base, base) + tone,
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                }
            )
            index += 1
            continue

        if base in FINALS_SET:
            if base == "r":
                if syllables:
                    syllables[-1]["raw_tokens"].append(token)
                    syllables[-1]["pinyin"] += "r" + tone
                    syllables[-1]["end_time"] = seg["end_time"]
                else:
                    syllables.append(
                        {
                            "raw_tokens": [token],
                            "pinyin": "r" + tone,
                            "start_time": seg["start_time"],
                            "end_time": seg["end_time"],
                        }
                    )
                index += 1
                continue

            pinyin = zhuyin_to_pinyin(None, base) + tone
            raw_tokens = [token]
            start_time = seg["start_time"]
            end_time = seg["end_time"]
            index += 1
            if index < len(token_segments):
                maybe_r = token_segments[index]["token"]
                r_base, r_tone = split_tone(maybe_r)
                if r_base == "r":
                    pinyin += "r" + r_tone
                    raw_tokens.append(maybe_r)
                    end_time = token_segments[index]["end_time"]
                    index += 1
            syllables.append(
                {
                    "raw_tokens": raw_tokens,
                    "pinyin": pinyin,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
            continue

        syllables.append(
            {
                "raw_tokens": [token],
                "pinyin": token,
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
            }
        )
        index += 1
    return syllables


class Wav2Vec2MMSPhoneticTranscriber(Transcriber):
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
            stage.phase("loading MMS backend")
        try:
            import librosa  # type: ignore
            import torch  # type: ignore
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mms_phonetic dependencies are not installed. Install with: pip install librosa transformers torch"
            ) from exc

        audio_path = Path(audio_artifact.path) if audio_artifact.path is not None else None
        if audio_path is None or not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found for mms_phonetic backend: {audio_artifact.path}")

        logger.info("Loading MMS phonetic model=%s device=%s", self.model_name, self.device)
        if stage is not None:
            stage.phase("loading MMS model")
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
            raise RuntimeError("Unable to determine CTC blank token id for mms_phonetic backend")

        if stage is not None:
            stage.phase("post-processing transcription")
        token_segments = ctc_token_segments(
            pred_ids=predicted_ids,
            tokenizer=processor.tokenizer,
            blank_id=int(blank_id),
            time_offset=float(time_offset),
        )
        syllable_segments = merge_tokens_to_pinyin_syllables(token_segments)
        units = self._syllable_segments_to_units(syllable_segments, language=language, tone_mode=tone_mode)

        raw_segments = [
            {
                "text": segment["pinyin"],
                "start": segment["start_time"],
                "end": segment["end_time"],
                "raw_tokens": segment["raw_tokens"],
            }
            for segment in syllable_segments
        ]

        logger.info(
            "MMS phonetic transcription complete: %d token segments, %d syllable segments, %d timed units",
            len(token_segments),
            len(syllable_segments),
            len(units),
        )
        if stage is not None:
            stage.close("transcriber complete")
        return TranscriptionResult(
            language=language,
            backend="mms_phonetic",
            units=units,
            raw_text=raw_transcription or None,
            raw_segments=raw_segments,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "target_sample_rate": self.target_sample_rate,
                "audio_duration": audio_duration,
                "token_segment_count": len(token_segments),
                "syllable_segment_count": len(syllable_segments),
                "source_audio": str(audio_artifact.path) if audio_artifact.path else None,
                "source_role": audio_artifact.role,
            },
        )

    def _syllable_segments_to_units(
        self,
        syllable_segments: list[dict[str, Any]],
        language: str,
        tone_mode: str,
    ) -> list[TimedUnit]:
        units: list[TimedUnit] = []
        for index, segment in enumerate(syllable_segments):
            pinyin = segment["pinyin"]
            base, tone = split_pinyin_tone(pinyin)
            if tone_mode == "keep":
                symbol = pinyin
                normalized_symbol = pinyin
            else:
                symbol = base
                normalized_symbol = base
                tone = None
            units.append(
                TimedUnit(
                    unit_id=make_id("timed_unit"),
                    symbol=symbol,
                    normalized_symbol=normalized_symbol,
                    unit_type="pinyin_syllable",
                    language=language,
                    tone=tone,
                    start_time=float(segment["start_time"]),
                    end_time=float(segment["end_time"]),
                    confidence=None,
                    source_backend="mms_phonetic",
                    raw_tokens=list(segment.get("raw_tokens", [])),
                    metadata={
                        "syllable_index": index,
                        "raw_pinyin": pinyin,
                    },
                )
            )
        return units
