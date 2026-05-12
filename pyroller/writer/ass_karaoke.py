from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from pyroller.domain import AlignmentLine, AlignmentResult, WriteResult
from pyroller.utils.time import format_ass_timestamp, seconds_to_centiseconds
from pyroller.writer.base import Writer

logger = logging.getLogger("pyroller.writer")


_ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,54,&H00FFFFFF,&H0000FFFF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,80,80,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


class ASSKaraokeWriter(Writer):
    def __init__(
        self,
        by_tag: str = "py-roller",
        tag_type: str = "kf",
        skip_structural_lines: bool = True,
        unmatched_line_duration: float = 0.6,
    ) -> None:
        self.by_tag = by_tag
        self.tag_type = tag_type
        self.skip_structural_lines = skip_structural_lines
        self.unmatched_line_duration = max(0.1, float(unmatched_line_duration))

    @property
    def backend_name(self) -> str:
        return "ass_karaoke"

    def write(self, alignment: AlignmentResult, output_path: Path) -> WriteResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        written_lines = 0
        with output_path.open("w", encoding="utf-8") as f:
            f.write(_ASS_HEADER)
            for index, line in enumerate(alignment.lines):
                if self._should_skip_line(line):
                    continue
                start = line.start_time
                end = self._display_end_time(alignment.lines, index)
                text = self._line_to_ass_text(line)
                f.write(
                    "Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n".format(
                        start=format_ass_timestamp(start),
                        end=format_ass_timestamp(end),
                        text=text,
                    )
                )
                written_lines += 1

        logger.info("Wrote %d ASS karaoke lines to %s", written_lines, output_path)
        return WriteResult(
            output_path=output_path,
            writer_backend=self.backend_name,
            metadata={
                "line_count": written_lines,
                "tag_type": self.tag_type,
                "by_tag": self.by_tag,
                "skip_structural_lines": self.skip_structural_lines,
                "unmatched_line_duration": self.unmatched_line_duration,
            },
        )

    def _line_to_ass_text(self, line: AlignmentLine) -> str:
        if not line.aligned_units:
            return line.raw_text
        chunks: list[str] = []
        for unit in line.aligned_units:
            duration_cs = seconds_to_centiseconds(max(unit.end_time - unit.start_time, 0.01))
            if duration_cs <= 0:
                duration_cs = 1
            chunks.append(f"{{\\{self.tag_type}{duration_cs}}}{unit.text}")
        return "".join(chunks)

    def _should_skip_line(self, line: AlignmentLine) -> bool:
        return self.skip_structural_lines and self._is_structural_line(line)

    def _is_structural_line(self, line: AlignmentLine) -> bool:
        return bool(line.metadata.get("is_spacing") or line.metadata.get("is_structural") or not line.raw_text.strip())

    def _next_non_structural_start(self, lines: list[AlignmentLine], index: int) -> Optional[float]:
        for next_index in range(index + 1, len(lines)):
            candidate = lines[next_index]
            if self._should_skip_line(candidate):
                continue
            return candidate.start_time
        return None

    def _display_end_time(self, lines: list[AlignmentLine], index: int) -> float:
        line = lines[index]
        start = line.start_time
        natural_end = start

        if line.aligned_units:
            natural_end = max(natural_end, line.aligned_units[-1].end_time)

        matched_end = line.metadata.get("matched_end_time")
        if matched_end is not None:
            try:
                natural_end = max(natural_end, float(matched_end))
            except (TypeError, ValueError):
                pass

        next_start = self._next_non_structural_start(lines, index)
        if line.end_time is not None and line.end_time > start:
            line_end = float(line.end_time)
            if next_start is None or line_end < next_start:
                natural_end = max(natural_end, line_end)

        if next_start is not None and next_start > start:
            guard = min(0.08, max((next_start - start) * 0.02, 0.01))
            ceiling = max(start, next_start - guard)
            if natural_end <= start:
                natural_end = min(start + self.unmatched_line_duration, ceiling)
            else:
                natural_end = min(natural_end, ceiling)
        elif natural_end <= start:
            natural_end = start + self.unmatched_line_duration

        if natural_end <= start:
            natural_end = start + 0.01
        return natural_end
