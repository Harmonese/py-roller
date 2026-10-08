from __future__ import annotations

import logging

from pyroller.i18n import _
import math
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Optional

from pyroller.domain import AlignedUnit, AlignmentLine, LyricLine, TranscriptionResult

logger = logging.getLogger("pyroller.aligner")


class SequenceAlignmentSupport:
    strategy_name = "sequence_alignment"

    def _segment_start(self, segment: dict[str, Any]) -> float | None:
        for key in ("start", "start_time"):
            value = segment.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _segment_end(self, segment: dict[str, Any]) -> float | None:
        for key in ("end", "end_time"):
            value = segment.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _estimate_alignment_window(
        self,
        transcription: TranscriptionResult,
        global_units: list[dict[str, Any]],
    ) -> tuple[float, float]:
        start_time = 0.0
        end_candidates: list[float] = []

        if global_units:
            start_time = float(global_units[0]["start_time"])
            end_candidates.append(float(global_units[-1]["end_time"]))

        if transcription.raw_segments:
            first_segment_start = self._segment_start(transcription.raw_segments[0])
            if first_segment_start is not None:
                start_time = min(start_time, first_segment_start) if global_units else first_segment_start
            segment_end = self._segment_end(transcription.raw_segments[-1])
            if segment_end is not None:
                end_candidates.append(segment_end)

        metadata_duration = transcription.metadata.get("audio_duration")
        if metadata_duration is not None:
            try:
                end_candidates.append(float(metadata_duration))
            except (TypeError, ValueError):
                pass

        end_time = max(end_candidates) if end_candidates else start_time
        if end_time < start_time:
            end_time = start_time
        return start_time, end_time

    def _build_global_unit_sequence(self, transcription: TranscriptionResult) -> tuple[list[dict[str, Any]], list[int]]:
        global_units: list[dict[str, Any]] = []
        for idx, unit in enumerate(transcription.units):
            seg_idx = unit.metadata.get("source_segment_index")
            if not isinstance(seg_idx, int):
                legacy_seg_idx = unit.metadata.get("sequence_index")
                seg_idx = legacy_seg_idx if isinstance(legacy_seg_idx, int) else -1
            global_units.append(
                {
                    "pos": len(global_units),
                    "unit_index": idx,
                    "symbol": unit.normalized_symbol,
                    "confidence": unit.confidence if unit.confidence is not None else 1.0,
                    "start_time": unit.start_time,
                    "end_time": unit.end_time,
                    "seg_idx": seg_idx if isinstance(seg_idx, int) else -1,
                }
            )
        return global_units, []

    def _line_symbols(self, line: LyricLine) -> list[str]:
        return [unit.normalized_symbol for unit in line.units if unit.normalized_symbol]

    def _sequence_similarity(self, left: list[str], right: list[str]) -> float:
        if not left or not right:
            return 0.0
        return float(SequenceMatcher(None, left, right).ratio())

    @staticmethod
    @lru_cache(maxsize=4096)
    def _symbol_similarity(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        if left == right:
            return 1.0
        return float(SequenceMatcher(None, left, right).ratio())

    def _interpolate_without_units(
        self,
        lyric_lines: list[LyricLine],
        min_time: float,
        max_time: float,
    ) -> list[AlignmentLine]:
        total = len(lyric_lines)
        lines: list[AlignmentLine] = []
        for idx, line in enumerate(lyric_lines):
            progress = (idx + 1) / (total + 1) if total else 0.0
            time_value = min_time + progress * (max_time - min_time)
            metadata = {"normalized_text": line.normalized_text, "unit_count": len(line.units), **dict(line.metadata), "unit_matches": []}
            lines.append(
                AlignmentLine(
                    line_index=line.line_index,
                    raw_text=line.raw_text,
                    assigned_time=time_value,
                    start_time=time_value,
                    end_time=None,
                    confidence=0.0,
                    method="interpolate",
                    lyric_unit_range=(0, len(line.units) - 1) if line.units else None,
                    aligned_units=self._build_aligned_units(
                        line=line,
                        line_start=time_value,
                        line_end=time_value,
                        confidence=0.0,
                        matched_range=None,
                        metadata=metadata,
                    ),
                    metadata=metadata,
                )
            )
        return lines

    def _ensure_monotonic(
        self,
        assignments: list[dict[str, Any]],
        min_time: float,
        max_time: float,
        min_gap: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not assignments:
            return [], []
        repaired = [dict(item) for item in assignments]
        repairs = []
        anchors = [i for i, item in enumerate(repaired) if item.get("pos", -1) >= 0]
        if any(float(repaired[a]["time"]) > float(repaired[b]["time"]) for a, b in zip(anchors, anchors[1:])):
            raise ValueError("Matched line anchors must be chronological")
        # Acoustic matches are immutable anchors. A display gap must never push
        # a fast lyric line past its already matched words.
        boundaries = [-1, *anchors, len(repaired)]
        for left, right in zip(boundaries, boundaries[1:]):
            count = right - left - 1
            if not count:
                continue
            lower = float(repaired[left]["time"]) if left >= 0 else min_time
            upper = float(repaired[right]["time"]) if right < len(repaired) else max_time
            gap = min(max(0.0, min_gap), max(0.0, upper - lower) / (count + 1))
            previous = lower
            for index in range(left + 1, right):
                old = float(repaired[index]["time"])
                new = min(max(old, previous + gap), upper - (right - index) * gap)
                repaired[index]["time"] = new
                previous = new
                if not math.isclose(old, new):
                    repairs.append({"lyric_idx": repaired[index]["lyric_idx"], "old_time": old,
                                    "new_time": new, "reason": "interpolated_min_gap"})
        return repaired, repairs

    def _assignment_to_alignment_line(self, line: LyricLine, assignment: dict[str, Any]) -> AlignmentLine:
        matched_range: Optional[tuple[int, int]] = None
        if assignment["pos"] >= 0 and assignment["end_pos"] >= assignment["pos"]:
            matched_range = (assignment["pos"], assignment["end_pos"])
        lyric_range = (0, len(line.units) - 1) if line.units else None
        metadata = {
            "normalized_text": line.normalized_text,
            "unit_count": len(line.units),
            **dict(line.metadata),
        }
        metadata.update(dict(assignment.get("metadata", {})))
        line_start = float(assignment.get("time", 0.0))
        matched_start = self._coerce_float(metadata.get("matched_start_time"), default=line_start)
        matched_end = self._coerce_float(metadata.get("matched_end_time"), default=matched_start)
        if matched_range is None:
            matched_start = matched_end = line_start
        end_time = matched_end if matched_end >= matched_start else matched_start
        aligned_units = self._build_aligned_units(
            line=line,
            line_start=matched_start,
            line_end=end_time,
            confidence=float(assignment.get("confidence", 0.0)),
            matched_range=matched_range,
            metadata=metadata,
        )
        if aligned_units:
            end_time = max(end_time, aligned_units[-1].end_time)
        return AlignmentLine(
            line_index=line.line_index,
            raw_text=line.raw_text,
            assigned_time=line_start,
            start_time=line_start,
            end_time=end_time,
            confidence=float(assignment.get("confidence", 0.0)),
            method=str(assignment.get("method", "unknown")),
            matched_audio_unit_range=matched_range,
            lyric_unit_range=lyric_range,
            aligned_units=aligned_units,
            metadata=metadata,
        )

    def _build_aligned_units(
        self,
        line: LyricLine,
        line_start: float,
        line_end: float,
        confidence: float,
        matched_range: Optional[tuple[int, int]],
        metadata: dict[str, Any],
    ) -> list[AlignedUnit]:
        if not line.units:
            return []

        unit_count = len(line.units)
        unit_matches = metadata.get("unit_matches") or []
        explicit_by_index: dict[int, dict[str, Any]] = {}
        for match in unit_matches:
            try:
                idx = int(match["unit_index_in_line"])
            except (KeyError, TypeError, ValueError):
                continue
            explicit_by_index[idx] = match

        base_start = min(line_start, line_end)
        base_end = max(line_start, line_end)
        unit_times: list[tuple[float, float, float, Optional[int]]] = []
        for idx, unit in enumerate(line.units):
            explicit = explicit_by_index.get(idx)
            if explicit is not None:
                start = self._coerce_float(explicit.get("start_time"), default=base_start)
                end = self._coerce_float(explicit.get("end_time"), default=start)
                if end < start:
                    end = start
                unit_conf = self._coerce_float(explicit.get("confidence"), default=confidence)
                source_audio_unit_index = explicit.get("audio_pos")
                try:
                    audio_pos = int(source_audio_unit_index) if source_audio_unit_index is not None else None
                except (TypeError, ValueError):
                    audio_pos = None
                unit_times.append((start, end, unit_conf, audio_pos))
                continue

            start = base_start + ((idx / unit_count) * (base_end - base_start))
            end = base_start + (((idx + 1) / unit_count) * (base_end - base_start))
            unit_times.append((start, end, 0.0, None))

        # Fill only the gaps between immutable matches, including zero-width
        # gaps. Never manufacture room by moving an observed timestamp.
        anchors = sorted(explicit_by_index)
        for left, right in zip([-1, *anchors], [*anchors, unit_count]):
            count = right - left - 1
            if not count:
                continue
            lower = unit_times[left][1] if left >= 0 else base_start
            upper = unit_times[right][0] if right < unit_count else base_end
            lower = min(lower, upper)
            for offset, index in enumerate(range(left + 1, right)):
                start = lower + (upper - lower) * offset / count
                end = lower + (upper - lower) * (offset + 1) / count
                unit_times[index] = (start, end, 0.0, None)

        aligned_units: list[AlignedUnit] = []
        text_cursor = 0
        for unit, timing in zip(line.units, unit_times):
            start, end, unit_conf, audio_pos = timing
            display_text = str(unit.metadata.get("source_char") or unit.metadata.get("display_text") or unit.symbol or unit.normalized_symbol)
            if unit.source_text_span is not None:
                span_start, span_end = unit.source_text_span
                display_text = line.raw_text[text_cursor:span_end] if span_end > text_cursor else ""
                text_cursor = max(text_cursor, span_end)
            aligned_units.append(
                AlignedUnit(
                    unit_id=unit.unit_id,
                    unit_index_in_line=unit.unit_index_in_line,
                    text=display_text,
                    normalized_symbol=unit.normalized_symbol,
                    unit_type=unit.unit_type,
                    language=unit.language,
                    start_time=start,
                    end_time=end,
                    confidence=unit_conf,
                    source_audio_unit_index=audio_pos,
                    source_audio_unit_range=(audio_pos, audio_pos) if audio_pos is not None else None,
                    metadata={
                        "tone": unit.tone,
                        "source_text_span": unit.source_text_span,
                        **dict(unit.metadata),
                    },
                )
            )
        if text_cursor and text_cursor < len(line.raw_text):
            aligned_units[-1].text += line.raw_text[text_cursor:]
        return aligned_units

    def _finalize_line_end_times(self, lines: list[AlignmentLine], max_time: float) -> None:
        if not lines:
            return
        for line in lines:
            candidate_end = line.end_time if line.end_time is not None else line.start_time
            if line.aligned_units:
                candidate_end = max(candidate_end, *(unit.end_time for unit in line.aligned_units))
            if candidate_end < line.start_time:
                candidate_end = line.start_time
            line.end_time = candidate_end

    def _coerce_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _build_report(
        self,
        lines: list[AlignmentLine],
        anchors: list[dict[str, Any]] | None = None,
        candidates: list[dict[str, Any]] | None = None,
        skipped_segments: list[int] | None = None,
        repairs: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        methods = Counter(line.method for line in lines)
        confidences = [line.confidence for line in lines if line.raw_text.strip() and not line.metadata.get("is_structural")]
        average_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
        confidence_buckets = {
            "high": sum(1 for c in confidences if c > 0.7),
            "medium": sum(1 for c in confidences if 0.5 < c <= 0.7),
            "low": sum(1 for c in confidences if c <= 0.5),
        }
        report = {
            "method_counts": dict(methods),
            "candidate_count": len(candidates or []),
            "skipped_segments": list(skipped_segments or []),
            "repairs": list(repairs or []),
            "average_confidence": average_confidence,
            "confidence_buckets": confidence_buckets,
            "candidates": list(candidates or []),
            "line_diagnostics": [
                {
                    "line_index": line.line_index,
                    "time": line.assigned_time,
                    "end_time": line.end_time,
                    "confidence": line.confidence,
                    "method": line.method,
                    "matched_audio_unit_range": line.matched_audio_unit_range,
                    "aligned_unit_count": len(line.aligned_units),
                    "raw_text": line.raw_text,
                }
                for line in lines
            ],
        }
        if extra:
            report.update(extra)
        return report

    def _log_phase_heading(self, title: str) -> None:
        logger.info("%s", "=" * 58)
        logger.info("%s", title)
        logger.info("%s", "=" * 58)

    def _log_alignment_report(self, strategy: str, result_report: dict[str, Any], lines: list[AlignmentLine]) -> None:
        logger.info(_("Alignment strategy: %s"), strategy)
        logger.info(
            _("Alignment methods: %s"),
            ", ".join(f"{method}={count}" for method, count in sorted(result_report.get("method_counts", {}).items())),
        )
        logger.info(
            _("Average confidence=%.3f | candidates=%d | repairs=%d"),
            float(result_report.get("average_confidence", 0.0)),
            int(result_report.get("candidate_count", 0)),
            len(result_report.get("repairs", [])),
        )
        if result_report.get("confidence_buckets"):
            buckets = result_report["confidence_buckets"]
            logger.info(
                _("Confidence buckets (matched only): high=%d medium=%d low=%d"),
                int(buckets.get("high", 0)),
                int(buckets.get("medium", 0)),
                int(buckets.get("low", 0)),
            )
        for line in lines:
            logger.debug(
                _("L%02d @ %.3fs→%.3fs conf=%.3f [%s] range=%s units=%d text=%r"),
                line.line_index + 1,
                line.assigned_time,
                line.end_time if line.end_time is not None else line.assigned_time,
                line.confidence,
                line.method,
                line.matched_audio_unit_range,
                len(line.aligned_units),
                line.raw_text[:80],
            )
