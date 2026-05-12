from __future__ import annotations


def _safe_seconds(seconds: float) -> float:
    return max(float(seconds), 0.0)


def format_lrc_timestamp(seconds: float, decimals: int = 3) -> str:
    safe_seconds = _safe_seconds(seconds)
    minutes = int(safe_seconds // 60)
    secs = safe_seconds - (minutes * 60)
    width = 2 + 1 + decimals
    return f"[{minutes:02d}:{secs:0{width}.{decimals}f}]"


def format_lrc_compact_timestamp(seconds: float, decimals: int = 2) -> str:
    return format_lrc_timestamp(seconds, decimals=decimals)


def format_ass_timestamp(seconds: float) -> str:
    safe_seconds = _safe_seconds(seconds)
    hours = int(safe_seconds // 3600)
    remainder = safe_seconds - (hours * 3600)
    minutes = int(remainder // 60)
    secs = remainder - (minutes * 60)
    return f"{hours:d}:{minutes:02d}:{secs:05.2f}"


def seconds_to_centiseconds(seconds: float) -> int:
    return max(0, int(round(float(seconds) * 100.0)))
