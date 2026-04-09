from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path

from pyroller.domain import AlignmentLine, AlignmentResult, WriteResult
from pyroller.utils.time import format_lrc_compact_timestamp, format_lrc_timestamp
from pyroller.writer.base import Writer

logger = logging.getLogger("pyroller.writer")


class LRCWriter(Writer):
    def __init__(self, decimals: int = 3, compressed: bool = False, by_tag: str = "py-roller", reserve_spacing: bool = False) -> None:
        self.decimals = decimals
        self.compressed = compressed
        self.by_tag = by_tag
        self.reserve_spacing = reserve_spacing

    @property
    def backend_name(self) -> str:
        if self.compressed:
            return "lrc_compressed"
        return "lrc_ms" if self.decimals >= 3 else "lrc_cs"

    def _fmt(self, seconds: float) -> str:
        if self.decimals == 2:
            return format_lrc_compact_timestamp(seconds, decimals=2)
        return format_lrc_timestamp(seconds, decimals=self.decimals)

    def write(self, alignment: AlignmentResult, output_path: Path) -> WriteResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written_line_count = 0
        with output_path.open("w", encoding="utf-8") as f:
            f.write("[ti:]\n[ar:]\n[al:]\n")
            f.write(f"[by:{self.by_tag}]\n\n")
            if self.compressed:
                grouped: OrderedDict[str, list[float]] = OrderedDict()
                for line in alignment.lines:
                    if self._is_spacing_line(line):
                        continue
                    grouped.setdefault(line.raw_text, []).append(line.assigned_time)
                for raw_text, timestamps in grouped.items():
                    prefix = "".join(self._fmt(ts) for ts in timestamps)
                    f.write(f"{prefix} {raw_text}\n")
                    written_line_count += 1
            else:
                for index, line in enumerate(alignment.lines):
                    if self._is_spacing_line(line):
                        if not self.reserve_spacing:
                            continue
                        spacing_time = self._spacing_timestamp(alignment.lines, index)
                        f.write(f"{self._fmt(spacing_time)}\n")
                        written_line_count += 1
                        continue
                    f.write(f"{self._fmt(line.assigned_time)} {line.raw_text}\n")
                    written_line_count += 1

        logger.info("Wrote %d lines to %s using writer=%s", written_line_count, output_path, self.backend_name)
        return WriteResult(
            output_path=output_path,
            writer_backend=self.backend_name,
            metadata={
                "by_tag": self.by_tag,
                "line_count": written_line_count,
                "decimals": self.decimals,
                "compressed": self.compressed,
                "reserve_spacing": self.reserve_spacing,
            },
        )

    def _is_spacing_line(self, line: AlignmentLine) -> bool:
        return bool(line.metadata.get("is_spacing")) or not line.raw_text.strip()

    def _spacing_timestamp(self, lines: list[AlignmentLine], index: int) -> float:
        current = lines[index]
        prev_end = current.assigned_time
        next_start = current.assigned_time

        for prev_idx in range(index - 1, -1, -1):
            prev_line = lines[prev_idx]
            if self._is_spacing_line(prev_line):
                continue
            prev_end = prev_line.end_time if prev_line.end_time is not None else prev_line.assigned_time
            break

        for next_idx in range(index + 1, len(lines)):
            next_line = lines[next_idx]
            if self._is_spacing_line(next_line):
                continue
            next_start = next_line.start_time
            break

        if next_start < prev_end:
            return current.assigned_time
        return (prev_end + next_start) / 2.0
