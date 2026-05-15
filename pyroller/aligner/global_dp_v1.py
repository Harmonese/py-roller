from __future__ import annotations

from typing import Any, Optional

from pyroller.aligner.base import Aligner
from pyroller.aligner.common import SequenceAlignmentSupport, logger
from pyroller.aligner.repetition import (
    LineCandidate,
    analyze_repeat_profile,
    assignment_trust,
    candidate_to_assignment,
    find_line_candidates,
    select_anchor_chain,
    select_best_candidate_path,
)
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
        repetition: str = "none",
    ) -> None:
        if repetition not in {"none", "few", "full"}:
            raise ValueError("repetition must be one of: none, few, full")
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.audio_gap_penalty = audio_gap_penalty
        self.lyric_gap_penalty = lyric_gap_penalty
        self.min_match_similarity = min_match_similarity
        self.min_gap = min_gap
        self.repetition = repetition

    def align(self, transcription: TranscriptionResult, parsed_lyrics: ParsedLyrics, progress: ProgressReporter | None = None) -> AlignmentResult:
        progress = progress or NullProgressReporter()
        global_units, skipped_segments = self._build_global_unit_sequence(transcription)
        lyric_lines = parsed_lyrics.lines
        min_time, max_time = self._estimate_alignment_window(transcription, global_units)

        self._log_phase_heading(f"ALIGNER: GLOBAL DP (repetition={self.repetition})")
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
                    "repetition": self.repetition,
                },
            )

        lyric_units, line_ranges = self._flatten_lyric_units(parsed_lyrics)
        if self.repetition == "full":
            # No DP row-by-row updates; only phase-level progress.
            stage_progress = progress.stage("aligner", total=4, unit="phase")
        elif self.repetition == "few":
            # DP rows + 5 phases (inputs, recovery, repair, monotonic, finalize).
            stage_progress = progress.stage("aligner", total=5 + max(len(lyric_units), 1), unit="step")
        else:
            # DP rows + 4 phases (inputs, recovery, monotonic, finalize).
            stage_progress = progress.stage("aligner", total=4 + max(len(lyric_units), 1), unit="step")
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
                    "repetition": self.repetition,
                },
            )

        repeat_stats: dict[str, Any] = {
            "mode": self.repetition,
            "profile": self._repeat_profile_dict([self._line_symbols(line) for line in lyric_lines]),
        }
        if self.repetition == "full":
            stage_progress.phase("building full-repetition candidate lattice")
            assignments, repeat_stats = self._align_full_repetition_lattice(
                lyric_lines=lyric_lines,
                global_units=global_units,
                min_time=min_time,
                max_time=max_time,
            )
            dp_stats = {
                "skipped": True,
                "reason": "repetition_full",
                "final_score": 0.0,
                "accepted_matches": 0,
                "diag_steps": 0,
                "audio_gaps": 0,
                "lyric_gaps": 0,
            }
        else:
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
                fill_unresolved=(self.repetition == "none"),
            )
            if self.repetition == "few":
                stage_progress.phase("repairing repeated/omitted regions")
                repeat_stats = self._repair_few_repetition_regions(
                    assignments=assignments,
                    lyric_lines=lyric_lines,
                    global_units=global_units,
                    min_time=min_time,
                    max_time=max_time,
                )
                self._fill_unresolved_times(assignments, min_time=min_time, max_time=max_time)

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
                "repetition_stats": repeat_stats,
            },
        )
        self._log_alignment_report(self.strategy_name, report, alignment_lines)
        if dp_stats.get("skipped"):
            logger.info("DP stats: skipped (%s)", dp_stats.get("reason", "unknown"))
        else:
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
                "repetition": self.repetition,
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
        fill_unresolved: bool = True,
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
        if fill_unresolved:
            self._fill_unresolved_times(assignments, min_time=min_time, max_time=max_time)
        for assignment in assignments:
            log_time = assignment["time"] if assignment["time"] is not None else -1.0
            logger.debug(
                "RECOVERED L%02d @ %.3fs conf=%.3f [%s] range=%s text=%r",
                assignment["lyric_idx"] + 1,
                float(log_time),
                float(assignment["confidence"]),
                assignment["method"],
                None if assignment["pos"] < 0 else (assignment["pos"], assignment["end_pos"]),
                assignment["text"][:80],
            )
        return assignments


    def _repeat_profile_dict(self, line_symbols: list[list[str]]) -> dict[str, Any]:
        profile = analyze_repeat_profile(line_symbols)
        return {
            "line_count": profile.line_count,
            "unique_signature_count": profile.unique_signature_count,
            "repeat_density": profile.repeat_density,
            "max_signature_count": profile.max_signature_count,
            "anchorless": profile.anchorless,
        }

    def _pending_assignment(self, line, unit_count: int, method: str = "pending_interpolate") -> dict[str, Any]:
        return {
            "lyric_idx": line.line_index,
            "text": line.raw_text,
            "time": None,
            "confidence": 0.0,
            "pos": -1,
            "end_pos": -1,
            "method": method,
            "metadata": {
                "matched_unit_count": 0,
                "line_unit_count": unit_count,
                "coverage": 0.0,
                "unit_matches": [],
            },
        }

    def _find_candidates_for_lines(
        self,
        lyric_lines: list,
        global_units: list[dict[str, Any]],
        *,
        audio_start: int,
        audio_end: int,
        top_k: int,
        min_score: float,
    ) -> tuple[list[list[LineCandidate]], int]:
        candidates_by_line: list[list[LineCandidate]] = []
        candidate_count = 0
        for line in lyric_lines:
            candidates = find_line_candidates(
                lyric_idx=line.line_index,
                lyric_symbols=self._line_symbols(line),
                global_units=global_units,
                symbol_similarity=self._symbol_similarity,
                min_match_similarity=self.min_match_similarity,
                audio_start=audio_start,
                audio_end=audio_end,
                top_k=top_k,
                min_score=min_score,
            )
            candidates_by_line.append(candidates)
            candidate_count += len(candidates)
        return candidates_by_line, candidate_count

    def _align_full_repetition_lattice(
        self,
        lyric_lines: list,
        global_units: list[dict[str, Any]],
        min_time: float,
        max_time: float,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._log_phase_heading("PHASE 1: FULL REPETITION CANDIDATE LATTICE")
        line_symbols = [self._line_symbols(line) for line in lyric_lines]
        top_k = min(128, max(24, len(lyric_lines) + 8))
        candidates_by_line, candidate_count = self._find_candidates_for_lines(
            lyric_lines=lyric_lines,
            global_units=global_units,
            audio_start=0,
            audio_end=len(global_units) - 1,
            top_k=top_k,
            min_score=0.25,
        )
        path = select_best_candidate_path(
            candidates_by_line=candidates_by_line,
            line_count=len(lyric_lines),
            min_time=min_time,
            max_time=max_time,
            beam_width=96,
            skip_penalty=-1.10,
        )
        assignments: list[dict[str, Any]] = []
        selected_count = 0
        for line, symbols, candidate in zip(lyric_lines, line_symbols, path):
            if candidate is None:
                assignments.append(self._pending_assignment(line, unit_count=len(symbols)))
                continue
            selected_count += 1
            assignments.append(
                candidate_to_assignment(
                    candidate=candidate,
                    text=line.raw_text,
                    line_unit_count=len(symbols),
                    mode="full",
                    method="repeat_full_lattice",
                    existing_metadata={"normalized_text": line.normalized_text},
                )
            )
        if len(assignments) < len(lyric_lines):
            for line, symbols in zip(lyric_lines[len(assignments) :], line_symbols[len(assignments) :]):
                assignments.append(self._pending_assignment(line, unit_count=len(symbols)))
        self._fill_unresolved_times(assignments, min_time=min_time, max_time=max_time)
        stats = {
            "mode": "full",
            "profile": self._repeat_profile_dict(line_symbols),
            "candidate_lines": sum(1 for candidates in candidates_by_line if candidates),
            "candidate_count": candidate_count,
            "selected_count": selected_count,
            "beam_width": 96,
            "top_k": top_k,
            "unresolved_count": sum(1 for item in assignments if item.get("method") == "interpolate"),
        }
        logger.info(
            "Full repetition lattice: candidates=%d selected=%d interpolated=%d",
            candidate_count,
            selected_count,
            int(stats["unresolved_count"]),
        )
        return assignments, stats

    def _repair_few_repetition_regions(
        self,
        assignments: list[dict[str, Any]],
        lyric_lines: list,
        global_units: list[dict[str, Any]],
        min_time: float,
        max_time: float,
    ) -> dict[str, Any]:
        self._log_phase_heading("PHASE 3: FEW REPETITION LOCAL LATTICE REPAIR")
        line_symbols = [self._line_symbols(line) for line in lyric_lines]
        anchors = select_anchor_chain(assignments=assignments, line_symbols=line_symbols)
        anchor_by_line = {anchor.lyric_idx: anchor for anchor in anchors}
        trust_by_line = {int(item["lyric_idx"]): assignment_trust(item) for item in assignments}
        weak_lines = {
            int(item["lyric_idx"])
            for item in assignments
            if int(item["lyric_idx"]) not in anchor_by_line and assignment_trust(item) in {"weak", "unresolved"}
        }
        segments: list[dict[str, Any]] = []
        repaired_count = 0
        candidate_count = 0
        selected_count = 0

        anchor_points: list[Optional[object]] = [None, *anchors, None]
        for left_anchor, right_anchor in zip(anchor_points, anchor_points[1:]):
            left_line = -1 if left_anchor is None else int(left_anchor.lyric_idx)
            right_line = len(lyric_lines) if right_anchor is None else int(right_anchor.lyric_idx)
            line_start = left_line + 1
            line_end = right_line - 1
            if line_start > line_end:
                continue
            if not any(line_idx in weak_lines for line_idx in range(line_start, line_end + 1)):
                continue

            audio_start = 0 if left_anchor is None else int(left_anchor.audio_end) + 1
            audio_end = len(global_units) - 1 if right_anchor is None else int(right_anchor.audio_start) - 1
            if audio_end < audio_start:
                segments.append(
                    {
                        "line_start": line_start,
                        "line_end": line_end,
                        "audio_start": audio_start,
                        "audio_end": audio_end,
                        "status": "skipped_empty_audio_window",
                    }
                )
                continue

            segment_lines = lyric_lines[line_start : line_end + 1]
            top_k = min(64, max(16, len(segment_lines) + 8))
            local_candidates, local_candidate_count = self._find_candidates_for_lines(
                lyric_lines=segment_lines,
                global_units=global_units,
                audio_start=audio_start,
                audio_end=audio_end,
                top_k=top_k,
                min_score=0.28,
            )
            candidate_count += local_candidate_count
            path = select_best_candidate_path(
                candidates_by_line=local_candidates,
                line_count=len(segment_lines),
                min_time=float(global_units[audio_start]["start_time"]),
                max_time=float(global_units[audio_end]["end_time"]),
                beam_width=64,
                skip_penalty=-1.20,
            )
            segment_selected = 0
            segment_repaired = 0
            for offset, candidate in enumerate(path):
                if candidate is None:
                    continue
                assign_idx = line_start + offset
                if assign_idx in anchor_by_line:
                    continue
                old_assignment = assignments[assign_idx]
                old_confidence = float(old_assignment.get("confidence") or 0.0)
                old_trust = trust_by_line.get(assign_idx, assignment_trust(old_assignment))
                should_update = old_trust in {"weak", "unresolved"} or candidate.score >= old_confidence + 0.05
                if not should_update:
                    continue
                line = lyric_lines[assign_idx]
                selected_count += 1
                segment_selected += 1
                assignments[assign_idx] = candidate_to_assignment(
                    candidate=candidate,
                    text=line.raw_text,
                    line_unit_count=len(line_symbols[assign_idx]),
                    mode="few",
                    method="repeat_lattice_repair",
                    existing_metadata=dict(old_assignment.get("metadata", {})),
                )
                repaired_count += 1
                segment_repaired += 1
            segments.append(
                {
                    "line_start": line_start,
                    "line_end": line_end,
                    "audio_start": audio_start,
                    "audio_end": audio_end,
                    "candidate_count": local_candidate_count,
                    "selected_count": segment_selected,
                    "repaired_count": segment_repaired,
                    "status": "ok",
                }
            )

        stats = {
            "mode": "few",
            "profile": self._repeat_profile_dict(line_symbols),
            "anchor_count": len(anchors),
            "anchors": [
                {
                    "lyric_idx": anchor.lyric_idx,
                    "audio_start": anchor.audio_start,
                    "audio_end": anchor.audio_end,
                    "confidence": anchor.confidence,
                    "coverage": anchor.coverage,
                }
                for anchor in anchors
            ],
            "trust_counts": {trust: list(trust_by_line.values()).count(trust) for trust in sorted(set(trust_by_line.values()))},
            "weak_line_count": len(weak_lines),
            "segments": segments,
            "candidate_count": candidate_count,
            "selected_count": selected_count,
            "repaired_count": repaired_count,
        }
        logger.info(
            "Few repetition repair: anchors=%d weak_lines=%d segments=%d repaired=%d candidates=%d",
            len(anchors),
            len(weak_lines),
            len(segments),
            repaired_count,
            candidate_count,
        )
        return stats

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

