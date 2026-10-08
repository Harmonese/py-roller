from __future__ import annotations


def _safe_seconds(seconds: float) -> float:
    return max(float(seconds), 0.0)


def format_lrc_timestamp(seconds: float, decimals: int = 3) -> str:
    safe_seconds = _safe_seconds(seconds)
    multiplier = 10 ** decimals
    total_ticks = int(round(safe_seconds * multiplier))
    minutes = total_ticks // (60 * multiplier)
    remaining_ticks = total_ticks % (60 * multiplier)
    secs = remaining_ticks / multiplier
    width = 2 + 1 + decimals
    return f"[{minutes:02d}:{secs:0{width}.{decimals}f}]"


def format_lrc_compact_timestamp(seconds: float, decimals: int = 2) -> str:
    return format_lrc_timestamp(seconds, decimals=decimals)


def format_ass_timestamp(seconds: float) -> str:
    ticks = seconds_to_centiseconds(_safe_seconds(seconds))
    hours, remainder = divmod(ticks, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, fraction = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{fraction:02d}"


def seconds_to_centiseconds(seconds: float) -> int:
    return max(0, int(round(float(seconds) * 100.0)))
