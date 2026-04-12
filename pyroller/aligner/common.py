from __future__ import annotations

import logging
import math
from collections import Counter
from difflib import SequenceMatcher
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
                start_time = first_segment_start
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

    def _symbol_similarity(self, left: str, right: str) -> float:
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
        repairs: list[dict[str, Any]] = []
        for item in repaired:
            original = float(item["time"])
            clamped = min(max(original, min_time), max_time)
            if not math.isclose(original, clamped):
                repairs.append(
                    {
                        "lyric_idx": item["lyric_idx"],
                        "old_time": original,
                        "new_time": clamped,
                        "reason": "clamp_to_window",
                    }
                )
                item["time"] = clamped

        if len(repaired) == 1:
            return repaired, repairs

        available_span = max(max_time - min_time, 0.0)
        effective_gap = min(min_gap, available_span / (len(repaired) - 1)) if len(repaired) > 1 else min_gap

        prev_time = float(repaired[0]["time"])
        for idx in range(1, len(repaired)):
            current = float(repaired[idx]["time"])
            minimum_allowed = prev_time + effective_gap
            if current < minimum_allowed:
                repairs.append(
                    {
                        "lyric_idx": repaired[idx]["lyric_idx"],
                        "old_time": current,
                        "new_time": minimum_allowed,
                        "reason": "forward_min_gap",
                    }
                )
                repaired[idx]["time"] = minimum_allowed
                current = minimum_allowed
            prev_time = current

        if float(repaired[-1]["time"]) > max_time:
            old_last = float(repaired[-1]["time"])
            repaired[-1]["time"] = max_time
            repairs.append(
                {
                    "lyric_idx": repaired[-1]["lyric_idx"],
                    "old_time": old_last,
                    "new_time": max_time,
                    "reason": "clamp_tail_to_end",
                }
            )
            for idx in range(len(repaired) - 2, -1, -1):
                current = float(repaired[idx]["time"])
                maximum_allowed = float(repaired[idx + 1]["time"]) - effective_gap
                if current > maximum_allowed:
                    repairs.append(
                        {
                            "lyric_idx": repaired[idx]["lyric_idx"],
                            "old_time": current,
                            "new_time": maximum_allowed,
                            "reason": "backward_min_gap",
                        }
                    )
                    repaired[idx]["time"] = maximum_allowed

            if float(repaired[0]["time"]) < min_time:
                old_first = float(repaired[0]["time"])
                repaired[0]["time"] = min_time
                repairs.append(
                    {
                        "lyric_idx": repaired[0]["lyric_idx"],
                        "old_time": old_first,
                        "new_time": min_time,
                        "reason": "clamp_head_to_start",
                    }
                )
                prev_time = min_time
                for idx in range(1, len(repaired)):
                    current = float(repaired[idx]["time"])
                    minimum_allowed = prev_time + effective_gap
                    if current < minimum_allowed:
                        repairs.append(
                            {
                                "lyric_idx": repaired[idx]["lyric_idx"],
                                "old_time": current,
                                "new_time": minimum_allowed,
                                "reason": "forward_min_gap",
                            }
                        )
                        repaired[idx]["time"] = minimum_allowed
                        current = minimum_allowed
                    prev_time = current

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
        if math.isclose(base_end, base_start):
            default_duration = max(float(metadata.get("fallback_unit_duration", 0.12)), 0.01)
            base_end = base_start + (default_duration * unit_count)

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
            unit_times.append((start, end, confidence if matched_range else 0.0, None))

        self._normalize_unit_times(unit_times, base_start, base_end)

        aligned_units: list[AlignedUnit] = []
        for unit, timing in zip(line.units, unit_times):
            start, end, unit_conf, audio_pos = timing
            display_text = str(unit.metadata.get("source_char") or unit.metadata.get("display_text") or unit.symbol or unit.normalized_symbol)
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
                    source_audio_unit_range=(audio_pos, audio_pos) if audio_pos is not None else matched_range,
                    metadata={
                        "tone": unit.tone,
                        **dict(unit.metadata),
                    },
                )
            )
        return aligned_units

    def _normalize_unit_times(
        self,
        unit_times: list[tuple[float, float, float, Optional[int]]],
        line_start: float,
        line_end: float,
    ) -> None:
        if not unit_times:
            return
        repaired: list[tuple[float, float, float, Optional[int]]] = []
        prev_end = line_start
        total = len(unit_times)
        fallback_duration = max((line_end - line_start) / max(1, total), 0.01)
        for idx, (start, end, conf, audio_pos) in enumerate(unit_times):
            start = max(start, prev_end)
            if end <= start:
                remaining_slots = max(1, total - idx)
                remaining_span = max(line_end - start, fallback_duration)
                end = start + max(remaining_span / remaining_slots, 0.01)
            repaired.append((start, end, conf, audio_pos))
            prev_end = end
        if repaired[-1][1] < line_end:
            start, _, conf, audio_pos = repaired[-1]
            repaired[-1] = (start, line_end, conf, audio_pos)
        elif repaired[-1][1] > line_end:
            start, _, conf, audio_pos = repaired[-1]
            repaired[-1] = (min(start, line_end), line_end, conf, audio_pos)
        unit_times[:] = repaired

    def _finalize_line_end_times(self, lines: list[AlignmentLine], max_time: float) -> None:
        if not lines:
            return
        for idx, line in enumerate(lines):
            next_start = lines[idx + 1].start_time if idx + 1 < len(lines) else max_time
            candidate_end = line.end_time if line.end_time is not None else line.start_time
            if candidate_end < line.start_time:
                candidate_end = line.start_time
            if next_start > line.start_time:
                candidate_end = max(candidate_end, next_start)
            line.end_time = candidate_end
            if line.aligned_units:
                if line.aligned_units[-1].end_time < candidate_end:
                    last = line.aligned_units[-1]
                    last.end_time = candidate_end
                prev_end = line.start_time
                for unit in line.aligned_units:
                    if unit.start_time < prev_end:
                        unit.start_time = prev_end
                    if unit.end_time < unit.start_time:
                        unit.end_time = unit.start_time
                    prev_end = unit.end_time

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
        confidences = [line.confidence for line in lines if line.confidence > 0]
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
        logger.info("Alignment strategy: %s", strategy)
        logger.info(
            "Alignment methods: %s",
            ", ".join(f"{method}={count}" for method, count in sorted(result_report.get("method_counts", {}).items())),
        )
        logger.info(
            "Average confidence=%.3f | candidates=%d | repairs=%d",
            float(result_report.get("average_confidence", 0.0)),
            int(result_report.get("candidate_count", 0)),
            len(result_report.get("repairs", [])),
        )
        if result_report.get("confidence_buckets"):
            buckets = result_report["confidence_buckets"]
            logger.info(
                "Confidence buckets (matched only): high=%d medium=%d low=%d",
                int(buckets.get("high", 0)),
                int(buckets.get("medium", 0)),
                int(buckets.get("low", 0)),
            )
        for line in lines:
            logger.debug(
                "L%02d @ %.3fs→%.3fs conf=%.3f [%s] range=%s units=%d text=%r",
                line.line_index + 1,
                line.assigned_time,
                line.end_time if line.end_time is not None else line.assigned_time,
                line.confidence,
                line.method,
                line.matched_audio_unit_range,
                len(line.aligned_units),
                line.raw_text[:80],
            )
