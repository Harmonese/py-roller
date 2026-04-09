from __future__ import annotations

from typing import Any, Optional

from pyroller.aligner.base import Aligner
from pyroller.aligner.common import SequenceAlignmentSupport, logger
from pyroller.domain import AlignmentResult, ParsedLyrics, TranscriptionResult
from pyroller.progress import NullProgressReporter, ProgressReporter


class GlobalDPAligner(Aligner, SequenceAlignmentSupport):
    name = "global_dp_v1"
    strategy_name = "global_dp_v1"

    def __init__(
        self,
        match_score: float = 2.0,
        mismatch_penalty: float = -1.05,
        audio_gap_penalty: float = -0.35,
        lyric_gap_penalty: float = -0.70,
        min_match_similarity: float = 0.55,
        min_gap: float = 0.5,
    ) -> None:
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.audio_gap_penalty = audio_gap_penalty
        self.lyric_gap_penalty = lyric_gap_penalty
        self.min_match_similarity = min_match_similarity
        self.min_gap = min_gap

    def align(self, transcription: TranscriptionResult, parsed_lyrics: ParsedLyrics, progress: ProgressReporter | None = None) -> AlignmentResult:
        progress = progress or NullProgressReporter()
        global_units, skipped_segments = self._build_global_unit_sequence(transcription)
        lyric_lines = parsed_lyrics.lines
        min_time, max_time = self._estimate_alignment_window(transcription, global_units)

        self._log_phase_heading("ALIGNER: GLOBAL DP (DEFAULT)")
        logger.info(
            "Lyric lines=%d | audio units=%d | window=[%.3fs, %.3fs]",
            len(lyric_lines),
            len(global_units),
            min_time,
            max_time,
        )

        if not global_units:
            stage_progress = progress.stage("aligner", total=2, unit="phase")
            stage_progress.phase("fallback interpolation (no audio units)")
            alignment_lines = self._interpolate_without_units(lyric_lines, min_time=min_time, max_time=max_time)
            self._finalize_line_end_times(alignment_lines, max_time=max_time)
            stage_progress.phase("finalizing alignment result")
            report = self._build_report(
                alignment_lines,
                skipped_segments=skipped_segments,
                repairs=[],
                extra={
                    "dp_stats": {
                        "final_score": 0.0,
                        "accepted_matches": 0,
                        "diag_steps": 0,
                        "audio_gaps": 0,
                        "lyric_gaps": 0,
                    }
                },
            )
            self._log_alignment_report(self.strategy_name, report, alignment_lines)
            stage_progress.close("aligner complete")
            return AlignmentResult(
                language=parsed_lyrics.language,
                unit_type=parsed_lyrics.unit_type,
                lines=alignment_lines,
                overall_confidence=0.0,
                report=report,
                metadata={
                    "strategy": self.strategy_name,
                    "backend": self.name,
                    "fallback": "full_interpolation_no_units",
                    "alignment_window": {"start": min_time, "end": max_time},
                },
            )

        lyric_units, line_ranges = self._flatten_lyric_units(parsed_lyrics)
        phase_budget = 4
        stage_progress = progress.stage("aligner", total=phase_budget + max(len(lyric_units), 1), unit="step")
        stage_progress.phase("building alignment inputs")
        logger.info("Flattened lyric units=%d across %d lines", len(lyric_units), len(lyric_lines))
        if not lyric_units:
            alignment_lines = self._interpolate_without_units(lyric_lines, min_time=min_time, max_time=max_time)
            self._finalize_line_end_times(alignment_lines, max_time=max_time)
            stage_progress.phase("finalizing alignment result")
            report = self._build_report(alignment_lines, skipped_segments=skipped_segments, repairs=[])
            self._log_alignment_report(self.strategy_name, report, alignment_lines)
            stage_progress.close("aligner complete")
            return AlignmentResult(
                language=parsed_lyrics.language,
                unit_type=parsed_lyrics.unit_type,
                lines=alignment_lines,
                overall_confidence=0.0,
                report=report,
                metadata={
                    "strategy": self.strategy_name,
                    "backend": self.name,
                    "fallback": "full_interpolation_no_lyric_units",
                    "alignment_window": {"start": min_time, "end": max_time},
                },
            )

        traceback_pairs, dp_stats = self._run_global_dp(lyric_units, global_units, stage_progress=stage_progress)
        stage_progress.phase("recovering line assignments")
        assignments = self._recover_line_assignments(
            lyric_lines=lyric_lines,
            lyric_units=lyric_units,
            line_ranges=line_ranges,
            global_units=global_units,
            traceback_pairs=traceback_pairs,
            min_time=min_time,
            max_time=max_time,
        )
        stage_progress.phase("repairing monotonic timestamps")
        repaired, repairs = self._ensure_monotonic(assignments, min_time=min_time, max_time=max_time, min_gap=self.min_gap)
        alignment_lines = [self._assignment_to_alignment_line(lyric_lines[item["lyric_idx"]], item) for item in repaired]
        self._finalize_line_end_times(alignment_lines, max_time=max_time)
        stage_progress.phase("finalizing alignment result")

        confidences = [line.confidence for line in alignment_lines if line.confidence > 0]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        report = self._build_report(
            alignment_lines,
            skipped_segments=skipped_segments,
            repairs=repairs,
            extra={
                "dp_stats": dp_stats,
                "unit_count": len(lyric_units),
                "global_unit_count": len(global_units),
            },
        )
        self._log_alignment_report(self.strategy_name, report, alignment_lines)
        logger.info(
            "DP stats: final_score=%.3f accepted_matches=%d diag_steps=%d audio_gaps=%d lyric_gaps=%d",
            float(dp_stats["final_score"]),
            int(dp_stats["accepted_matches"]),
            int(dp_stats["diag_steps"]),
            int(dp_stats["audio_gaps"]),
            int(dp_stats["lyric_gaps"]),
        )
        stage_progress.close("aligner complete")
        return AlignmentResult(
            language=parsed_lyrics.language,
            unit_type=parsed_lyrics.unit_type,
            lines=alignment_lines,
            overall_confidence=overall_confidence,
            report=report,
            metadata={
                "strategy": self.strategy_name,
                "backend": self.name,
                "global_unit_count": len(global_units),
                "lyric_unit_count": len(lyric_units),
                "match_score": self.match_score,
                "mismatch_penalty": self.mismatch_penalty,
                "audio_gap_penalty": self.audio_gap_penalty,
                "lyric_gap_penalty": self.lyric_gap_penalty,
                "min_match_similarity": self.min_match_similarity,
                "min_gap": self.min_gap,
                "alignment_window": {"start": min_time, "end": max_time},
            },
        )

    def _flatten_lyric_units(self, parsed_lyrics: ParsedLyrics) -> tuple[list[dict[str, Any]], dict[int, tuple[int, int] | None]]:
        flat_units: list[dict[str, Any]] = []
        line_ranges: dict[int, tuple[int, int] | None] = {}
        for line in parsed_lyrics.lines:
            start = len(flat_units)
            for unit in line.units:
                if not unit.normalized_symbol:
                    continue
                flat_units.append(
                    {
                        "flat_index": len(flat_units),
                        "line_index": line.line_index,
                        "unit_index_in_line": unit.unit_index_in_line,
                        "symbol": unit.normalized_symbol,
                    }
                )
            end = len(flat_units) - 1
            line_ranges[line.line_index] = (start, end) if end >= start else None
        return flat_units, line_ranges

    def _run_global_dp(
        self,
        lyric_units: list[dict[str, Any]],
        global_units: list[dict[str, Any]],
        stage_progress,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._log_phase_heading("PHASE 1: GLOBAL DYNAMIC PROGRAMMING")
        m = len(lyric_units)
        n = len(global_units)
        dp = [[0.0] * (n + 1) for _ in range(m + 1)]
        back = [["none"] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            dp[i][0] = dp[i - 1][0] + self.lyric_gap_penalty
            back[i][0] = "up"
        for j in range(1, n + 1):
            dp[0][j] = dp[0][j - 1] + self.audio_gap_penalty
            back[0][j] = "left"

        for i in range(1, m + 1):
            lyric_symbol = lyric_units[i - 1]["symbol"]
            for j in range(1, n + 1):
                audio_symbol = global_units[j - 1]["symbol"]
                similarity = self._symbol_similarity(lyric_symbol, audio_symbol)
                diag_score = dp[i - 1][j - 1] + self._pair_score(similarity)
                up_score = dp[i - 1][j] + self.lyric_gap_penalty
                left_score = dp[i][j - 1] + self.audio_gap_penalty

                if diag_score >= left_score and diag_score >= up_score:
                    dp[i][j] = diag_score
                    back[i][j] = "diag"
                elif left_score >= up_score:
                    dp[i][j] = left_score
                    back[i][j] = "left"
                else:
                    dp[i][j] = up_score
                    back[i][j] = "up"
            stage_progress.update(1, message=f"dp row {i}/{m}")

        traceback_pairs: list[dict[str, Any]] = []
        i = m
        j = n
        diag_steps = 0
        audio_gaps = 0
        lyric_gaps = 0
        accepted_matches = 0
        while i > 0 or j > 0:
            move = back[i][j]
            if move == "diag" and i > 0 and j > 0:
                lyric_symbol = lyric_units[i - 1]["symbol"]
                audio_symbol = global_units[j - 1]["symbol"]
                similarity = self._symbol_similarity(lyric_symbol, audio_symbol)
                accepted = similarity >= self.min_match_similarity
                if accepted:
                    accepted_matches += 1
                traceback_pairs.append(
                    {
                        "lyric_flat_idx": i - 1,
                        "audio_pos": j - 1,
                        "similarity": similarity,
                        "accepted": accepted,
                    }
                )
                diag_steps += 1
                i -= 1
                j -= 1
            elif move == "up" and i > 0:
                traceback_pairs.append(
                    {
                        "lyric_flat_idx": i - 1,
                        "audio_pos": None,
                        "similarity": 0.0,
                        "accepted": False,
                    }
                )
                lyric_gaps += 1
                i -= 1
            else:
                audio_gaps += 1
                j -= 1

        traceback_pairs.reverse()
        logger.info(
            "DP completed for %d lyric units x %d audio units | final_score=%.3f",
            m,
            n,
            dp[m][n],
        )
        return traceback_pairs, {
            "final_score": dp[m][n],
            "accepted_matches": accepted_matches,
            "diag_steps": diag_steps,
            "audio_gaps": audio_gaps,
            "lyric_gaps": lyric_gaps,
        }

    def _pair_score(self, similarity: float) -> float:
        return self.mismatch_penalty + ((self.match_score - self.mismatch_penalty) * similarity)

    def _recover_line_assignments(
        self,
        lyric_lines: list,
        lyric_units: list[dict[str, Any]],
        line_ranges: dict[int, tuple[int, int] | None],
        global_units: list[dict[str, Any]],
        traceback_pairs: list[dict[str, Any]],
        min_time: float,
        max_time: float,
    ) -> list[dict[str, Any]]:
        self._log_phase_heading("PHASE 2: LINE ASSIGNMENT RECOVERY")
        matched_positions_by_line: dict[int, list[int]] = {line.line_index: [] for line in lyric_lines}
        similarity_by_line: dict[int, list[float]] = {line.line_index: [] for line in lyric_lines}
        matched_units_by_line: dict[int, int] = {line.line_index: 0 for line in lyric_lines}
        unit_matches_by_line: dict[int, list[dict[str, Any]]] = {line.line_index: [] for line in lyric_lines}

        for pair in traceback_pairs:
            if pair["audio_pos"] is None or not pair["accepted"]:
                continue
            lyric_unit = lyric_units[pair["lyric_flat_idx"]]
            line_index = lyric_unit["line_index"]
            audio_pos = int(pair["audio_pos"])
            matched_positions_by_line[line_index].append(audio_pos)
            similarity_by_line[line_index].append(float(pair["similarity"]))
            matched_units_by_line[line_index] += 1
            unit_matches_by_line[line_index].append(
                {
                    "unit_index_in_line": lyric_unit["unit_index_in_line"],
                    "audio_pos": audio_pos,
                    "start_time": float(global_units[audio_pos]["start_time"]),
                    "end_time": float(global_units[audio_pos]["end_time"]),
                    "confidence": float(pair["similarity"]),
                }
            )

        assignments: list[dict[str, Any]] = []
        unresolved_indices: list[int] = []
        for line in lyric_lines:
            line_index = line.line_index
            positions = matched_positions_by_line[line_index]
            line_symbols = self._line_symbols(line)
            if positions:
                first_pos = min(positions)
                last_pos = max(positions)
                coverage = matched_units_by_line[line_index] / max(1, len(line_symbols))
                confidence = (sum(similarity_by_line[line_index]) / len(similarity_by_line[line_index])) * min(1.0, coverage)
                method = "dp_match" if matched_units_by_line[line_index] >= len(line_symbols) else "dp_partial"
                line_unit_matches = sorted(unit_matches_by_line[line_index], key=lambda item: int(item["unit_index_in_line"]))
                assignment = {
                    "lyric_idx": line_index,
                    "text": line.raw_text,
                    "time": float(global_units[first_pos]["start_time"]),
                    "confidence": float(confidence),
                    "pos": first_pos,
                    "end_pos": last_pos,
                    "method": method,
                    "metadata": {
                        "matched_unit_count": matched_units_by_line[line_index],
                        "line_unit_count": len(line_symbols),
                        "coverage": coverage,
                        "matched_start_time": float(global_units[first_pos]["start_time"]),
                        "matched_end_time": float(global_units[last_pos]["end_time"]),
                        "unit_matches": line_unit_matches,
                    },
                }
            else:
                unresolved_indices.append(line_index)
                line_range = line_ranges.get(line_index)
                unit_count = 0
                if line_range is not None:
                    unit_count = (line_range[1] - line_range[0]) + 1
                assignment = {
                    "lyric_idx": line_index,
                    "text": line.raw_text,
                    "time": None,
                    "confidence": 0.0,
                    "pos": -1,
                    "end_pos": -1,
                    "method": "pending_interpolate",
                    "metadata": {
                        "matched_unit_count": 0,
                        "line_unit_count": unit_count,
                        "coverage": 0.0,
                        "unit_matches": [],
                    },
                }
            assignments.append(assignment)

        logger.info(
            "Recovered direct line assignments=%d | pending interpolation=%d",
            len(assignments) - len(unresolved_indices),
            len(unresolved_indices),
        )
        self._fill_unresolved_times(assignments, min_time=min_time, max_time=max_time)
        for assignment in assignments:
            logger.debug(
                "RECOVERED L%02d @ %.3fs conf=%.3f [%s] range=%s text=%r",
                assignment["lyric_idx"] + 1,
                float(assignment["time"]),
                float(assignment["confidence"]),
                assignment["method"],
                None if assignment["pos"] < 0 else (assignment["pos"], assignment["end_pos"]),
                assignment["text"][:80],
            )
        return assignments

    def _fill_unresolved_times(self, assignments: list[dict[str, Any]], min_time: float, max_time: float) -> None:
        known_indices = [idx for idx, item in enumerate(assignments) if item["time"] is not None]
        if not known_indices:
            total = len(assignments)
            for idx, item in enumerate(assignments):
                progress = (idx + 1) / (total + 1) if total else 0.0
                interpolated_time = min_time + progress * (max_time - min_time)
                item["time"] = interpolated_time
                item["method"] = "interpolate"
                metadata = item.setdefault("metadata", {})
                metadata.setdefault("matched_start_time", interpolated_time)
                metadata.setdefault("matched_end_time", interpolated_time)
                metadata.setdefault("unit_matches", [])
            return

        idx = 0
        while idx < len(assignments):
            if assignments[idx]["time"] is not None:
                idx += 1
                continue
            start = idx
            while idx < len(assignments) and assignments[idx]["time"] is None:
                idx += 1
            end = idx - 1
            prev_idx = start - 1
            next_idx = idx if idx < len(assignments) else None
            prev_time = float(assignments[prev_idx]["time"]) if prev_idx >= 0 else min_time
            next_time = float(assignments[next_idx]["time"]) if next_idx is not None else max_time
            block_len = end - start + 1
            for offset, assign_idx in enumerate(range(start, end + 1), start=1):
                progress = offset / (block_len + 1)
                interpolated_time = prev_time + progress * (next_time - prev_time)
                assignments[assign_idx]["time"] = interpolated_time
                assignments[assign_idx]["method"] = "interpolate"
                assignments[assign_idx]["confidence"] = 0.0
                metadata = assignments[assign_idx].setdefault("metadata", {})
                metadata.setdefault("matched_start_time", interpolated_time)
                metadata.setdefault("matched_end_time", interpolated_time)
                metadata.setdefault("unit_matches", [])

