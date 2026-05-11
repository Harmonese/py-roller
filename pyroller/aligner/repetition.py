from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Sequence


SymbolSimilarity = Callable[[str, str], float]


@dataclass(frozen=True)
class LineCandidate:
    lyric_idx: int
    audio_start: int
    audio_end: int
    start_time: float
    end_time: float
    score: float
    coverage: float
    avg_similarity: float
    unit_matches: list[dict[str, Any]]
    method: str = "repeat_lattice"

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_diagnostic(self) -> dict[str, Any]:
        return {
            "lyric_idx": self.lyric_idx,
            "audio_start": self.audio_start,
            "audio_end": self.audio_end,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "score": self.score,
            "coverage": self.coverage,
            "avg_similarity": self.avg_similarity,
            "method": self.method,
        }


@dataclass(frozen=True)
class RepetitionProfile:
    line_count: int
    unique_signature_count: int
    repeat_density: float
    max_signature_count: int
    anchorless: bool


@dataclass(frozen=True)
class BeamState:
    score: float
    path: tuple[LineCandidate | None, ...]
    last_audio_end: int
    last_time: float | None
    last_duration: float | None


@dataclass(frozen=True)
class Anchor:
    lyric_idx: int
    audio_start: int
    audio_end: int
    confidence: float
    coverage: float


def line_signature(symbols: Sequence[str]) -> tuple[str, ...]:
    return tuple(symbols)


def analyze_repeat_profile(line_symbols: Sequence[Sequence[str]]) -> RepetitionProfile:
    signatures = [line_signature(symbols) for symbols in line_symbols if symbols]
    line_count = len(line_symbols)
    if not signatures:
        return RepetitionProfile(
            line_count=line_count,
            unique_signature_count=0,
            repeat_density=0.0,
            max_signature_count=0,
            anchorless=True,
        )
    counts: dict[tuple[str, ...], int] = {}
    for signature in signatures:
        counts[signature] = counts.get(signature, 0) + 1
    unique_count = len(counts)
    max_count = max(counts.values()) if counts else 0
    repeat_density = 1.0 - (unique_count / max(1, len(signatures)))
    anchorless = repeat_density >= 0.65 or max_count >= max(3, int(len(signatures) * 0.4))
    return RepetitionProfile(
        line_count=line_count,
        unique_signature_count=unique_count,
        repeat_density=repeat_density,
        max_signature_count=max_count,
        anchorless=anchorless,
    )


def _sequence_ratio(left: Sequence[str], right: Sequence[str]) -> float:
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, list(left), list(right)).ratio())


def _length_bounds(line_len: int, window_len: int) -> tuple[int, int]:
    if line_len <= 0 or window_len <= 0:
        return 0, -1
    min_len = max(1, int(round(line_len * 0.55)))
    max_len = max(min_len, int(round(line_len * 1.75)))
    if line_len <= 2:
        max_len = max(max_len, line_len + 2)
    else:
        max_len = min(max_len, line_len + 6)
    return min(min_len, window_len), min(max_len, window_len)


def _local_align_span(
    *,
    lyric_idx: int,
    lyric_symbols: Sequence[str],
    global_units: Sequence[dict[str, Any]],
    audio_start: int,
    audio_end: int,
    symbol_similarity: SymbolSimilarity,
    min_match_similarity: float,
) -> LineCandidate | None:
    audio_symbols = [str(unit.get("symbol", "")) for unit in global_units[audio_start : audio_end + 1]]
    m = len(lyric_symbols)
    n = len(audio_symbols)
    if m == 0 or n == 0:
        return None

    gap_penalty = -0.35
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    back = [["none"] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        back[i][0] = "up"
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        back[0][j] = "left"

    for i in range(1, m + 1):
        left_symbol = str(lyric_symbols[i - 1])
        for j in range(1, n + 1):
            right_symbol = audio_symbols[j - 1]
            similarity = symbol_similarity(left_symbol, right_symbol)
            diag = dp[i - 1][j - 1] + ((2.0 * similarity) - 0.75)
            up = dp[i - 1][j] + gap_penalty
            left = dp[i][j - 1] + gap_penalty
            if diag >= left and diag >= up:
                dp[i][j] = diag
                back[i][j] = "diag"
            elif left >= up:
                dp[i][j] = left
                back[i][j] = "left"
            else:
                dp[i][j] = up
                back[i][j] = "up"

    matches: list[dict[str, Any]] = []
    similarities: list[float] = []
    i = m
    j = n
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag" and i > 0 and j > 0:
            similarity = symbol_similarity(str(lyric_symbols[i - 1]), audio_symbols[j - 1])
            if similarity >= min_match_similarity:
                audio_pos = audio_start + j - 1
                unit = global_units[audio_pos]
                matches.append(
                    {
                        "unit_index_in_line": i - 1,
                        "audio_pos": audio_pos,
                        "start_time": float(unit["start_time"]),
                        "end_time": float(unit["end_time"]),
                        "confidence": float(similarity),
                    }
                )
                similarities.append(float(similarity))
            i -= 1
            j -= 1
        elif move == "up" and i > 0:
            i -= 1
        elif j > 0:
            j -= 1
        else:
            break

    if not matches:
        return None
    matches.reverse()
    similarities.reverse()
    first_audio = min(int(item["audio_pos"]) for item in matches)
    last_audio = max(int(item["audio_pos"]) for item in matches)
    coverage = len({int(item["unit_index_in_line"]) for item in matches}) / max(1, m)
    avg_similarity = sum(similarities) / len(similarities)
    span_ratio = _sequence_ratio(lyric_symbols, audio_symbols)
    # coverage keeps short accidental matches from winning; span_ratio keeps the
    # candidate sensitive to the whole phrase, not only isolated accepted units.
    score = (0.65 * avg_similarity * min(1.0, coverage)) + (0.35 * span_ratio)
    start_time = float(global_units[first_audio]["start_time"])
    end_time = float(global_units[last_audio]["end_time"])
    return LineCandidate(
        lyric_idx=lyric_idx,
        audio_start=first_audio,
        audio_end=last_audio,
        start_time=start_time,
        end_time=end_time,
        score=float(score),
        coverage=float(coverage),
        avg_similarity=float(avg_similarity),
        unit_matches=matches,
    )


def find_line_candidates(
    *,
    lyric_idx: int,
    lyric_symbols: Sequence[str],
    global_units: Sequence[dict[str, Any]],
    symbol_similarity: SymbolSimilarity,
    min_match_similarity: float,
    audio_start: int = 0,
    audio_end: int | None = None,
    top_k: int = 24,
    min_score: float = 0.30,
) -> list[LineCandidate]:
    if audio_end is None:
        audio_end = len(global_units) - 1
    audio_start = max(0, int(audio_start))
    audio_end = min(len(global_units) - 1, int(audio_end))
    if not lyric_symbols or not global_units or audio_end < audio_start:
        return []

    window_len = audio_end - audio_start + 1
    min_len, max_len = _length_bounds(len(lyric_symbols), window_len)
    if max_len < min_len:
        return []

    candidates_by_range: dict[tuple[int, int], LineCandidate] = {}
    all_candidates: list[LineCandidate] = []
    for start in range(audio_start, audio_end + 1):
        max_actual_len = min(max_len, audio_end - start + 1)
        if max_actual_len < min_len:
            break
        for span_len in range(min_len, max_actual_len + 1):
            end = start + span_len - 1
            audio_symbols = [str(unit.get("symbol", "")) for unit in global_units[start : end + 1]]
            coarse = _sequence_ratio(lyric_symbols, audio_symbols)
            if coarse < 0.15 and len(lyric_symbols) > 1:
                continue
            candidate = _local_align_span(
                lyric_idx=lyric_idx,
                lyric_symbols=lyric_symbols,
                global_units=global_units,
                audio_start=start,
                audio_end=end,
                symbol_similarity=symbol_similarity,
                min_match_similarity=min_match_similarity,
            )
            if candidate is None:
                continue
            all_candidates.append(candidate)
            key = (candidate.audio_start, candidate.audio_end)
            previous = candidates_by_range.get(key)
            if previous is None or candidate.score > previous.score:
                candidates_by_range[key] = candidate

    candidates = list(candidates_by_range.values())
    if not candidates:
        return []
    qualified = [candidate for candidate in candidates if candidate.score >= min_score]
    pool = qualified if qualified else candidates
    pool.sort(key=lambda item: (-item.score, item.audio_start, item.audio_end))

    # Keep temporal diversity, otherwise a long line can produce many nearly
    # identical spans around the same occurrence and hide later repetitions.
    selected: list[LineCandidate] = []
    for candidate in pool:
        overlaps_existing = False
        for existing in selected:
            overlap = max(0, min(candidate.audio_end, existing.audio_end) - max(candidate.audio_start, existing.audio_start) + 1)
            shorter = max(1, min(candidate.audio_end - candidate.audio_start + 1, existing.audio_end - existing.audio_start + 1))
            if overlap / shorter >= 0.75:
                overlaps_existing = True
                break
        if overlaps_existing:
            continue
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    if len(selected) < min(top_k, len(pool)):
        seen = {(item.audio_start, item.audio_end) for item in selected}
        for candidate in pool:
            key = (candidate.audio_start, candidate.audio_end)
            if key in seen:
                continue
            selected.append(candidate)
            seen.add(key)
            if len(selected) >= top_k:
                break
    selected.sort(key=lambda item: (item.audio_start, item.audio_end, -item.score))
    return selected


def _transition_score(
    previous: LineCandidate | None,
    current: LineCandidate,
    *,
    line_idx: int,
    line_count: int,
    min_time: float,
    max_time: float,
) -> float:
    score = 0.0
    if previous is not None:
        if current.audio_start <= previous.audio_end:
            return -1_000_000.0
        gap = current.start_time - previous.end_time
        if gap < 0.0:
            return -1_000_000.0
        if gap < 0.03:
            score -= 0.20
        if previous.duration > 0.02 and current.duration > 0.02:
            ratio = current.duration / max(previous.duration, 0.02)
            if ratio > 3.5 or ratio < 0.28:
                score -= 0.25
    duration = max(max_time - min_time, 0.001)
    audio_progress = (current.start_time - min_time) / duration
    lyric_progress = (line_idx + 1) / max(1, line_count + 1)
    score -= abs(audio_progress - lyric_progress) * 0.30
    return score


def select_best_candidate_path(
    *,
    candidates_by_line: Sequence[Sequence[LineCandidate]],
    line_count: int,
    min_time: float,
    max_time: float,
    beam_width: int = 64,
    skip_penalty: float = -1.25,
) -> list[LineCandidate | None]:
    if line_count <= 0:
        return []
    states = [BeamState(score=0.0, path=tuple(), last_audio_end=-1, last_time=None, last_duration=None)]
    for line_idx in range(line_count):
        candidates = list(candidates_by_line[line_idx]) if line_idx < len(candidates_by_line) else []
        next_states: list[BeamState] = []
        for state in states:
            previous = state.path[-1] if state.path else None
            previous_candidate = previous if isinstance(previous, LineCandidate) else None
            # Allow the line to remain unresolved. Interpolation can fill it
            # later, and this prevents one bad line from collapsing the beam.
            next_states.append(
                BeamState(
                    score=state.score + skip_penalty,
                    path=state.path + (None,),
                    last_audio_end=state.last_audio_end,
                    last_time=state.last_time,
                    last_duration=state.last_duration,
                )
            )
            for candidate in candidates:
                if candidate.audio_start <= state.last_audio_end:
                    continue
                transition = _transition_score(
                    previous_candidate,
                    candidate,
                    line_idx=line_idx,
                    line_count=line_count,
                    min_time=min_time,
                    max_time=max_time,
                )
                if transition <= -999_999.0:
                    continue
                candidate_score = (candidate.score * 4.0) + (candidate.coverage * 0.75) + transition
                next_states.append(
                    BeamState(
                        score=state.score + candidate_score,
                        path=state.path + (candidate,),
                        last_audio_end=candidate.audio_end,
                        last_time=candidate.end_time,
                        last_duration=candidate.duration,
                    )
                )
        if not next_states:
            next_states = [
                BeamState(
                    score=state.score + skip_penalty,
                    path=state.path + (None,),
                    last_audio_end=state.last_audio_end,
                    last_time=state.last_time,
                    last_duration=state.last_duration,
                )
                for state in states
            ]
        next_states.sort(key=lambda item: item.score, reverse=True)
        states = next_states[: max(1, beam_width)]
    best = max(states, key=lambda item: item.score)
    return list(best.path)


def candidate_to_assignment(
    *,
    candidate: LineCandidate,
    text: str,
    line_unit_count: int,
    mode: str,
    method: str | None = None,
    existing_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(existing_metadata or {})
    metadata.update(
        {
            "matched_unit_count": len(candidate.unit_matches),
            "line_unit_count": line_unit_count,
            "coverage": candidate.coverage,
            "matched_start_time": candidate.start_time,
            "matched_end_time": candidate.end_time,
            "unit_matches": candidate.unit_matches,
            "repetition_mode": mode,
            "candidate_score": candidate.score,
            "candidate_avg_similarity": candidate.avg_similarity,
        }
    )
    return {
        "lyric_idx": candidate.lyric_idx,
        "text": text,
        "time": candidate.start_time,
        "confidence": candidate.score,
        "pos": candidate.audio_start,
        "end_pos": candidate.audio_end,
        "method": method or candidate.method,
        "metadata": metadata,
    }


def assignment_trust(assignment: dict[str, Any]) -> str:
    metadata = assignment.get("metadata", {}) or {}
    confidence = float(assignment.get("confidence") or 0.0)
    coverage = float(metadata.get("coverage") or 0.0)
    pos = int(assignment.get("pos") if assignment.get("pos") is not None else -1)
    method = str(assignment.get("method") or "")
    if assignment.get("time") is None or pos < 0 or method in {"pending_interpolate", "interpolate"}:
        return "unresolved"
    if confidence >= 0.76 and coverage >= 0.72:
        return "anchor_candidate"
    if confidence >= 0.55 and coverage >= 0.45:
        return "ok"
    return "weak"


def select_anchor_chain(
    *,
    assignments: Sequence[dict[str, Any]],
    line_symbols: Sequence[Sequence[str]],
) -> list[Anchor]:
    signature_counts: dict[tuple[str, ...], int] = {}
    for symbols in line_symbols:
        signature = line_signature(symbols)
        if signature:
            signature_counts[signature] = signature_counts.get(signature, 0) + 1

    candidates: list[Anchor] = []
    for assignment in assignments:
        lyric_idx = int(assignment.get("lyric_idx", -1))
        if lyric_idx < 0 or lyric_idx >= len(line_symbols):
            continue
        if assignment_trust(assignment) != "anchor_candidate":
            continue
        signature = line_signature(line_symbols[lyric_idx])
        if signature_counts.get(signature, 0) > 1:
            # Repeated lines are useful matching templates, but they are unsafe
            # as hard anchors because they do not identify an occurrence.
            continue
        audio_start = int(assignment.get("pos", -1))
        audio_end = int(assignment.get("end_pos", -1))
        if audio_start < 0 or audio_end < audio_start:
            continue
        metadata = assignment.get("metadata", {}) or {}
        candidates.append(
            Anchor(
                lyric_idx=lyric_idx,
                audio_start=audio_start,
                audio_end=audio_end,
                confidence=float(assignment.get("confidence") or 0.0),
                coverage=float(metadata.get("coverage") or 0.0),
            )
        )

    if not candidates:
        return []
    candidates.sort(key=lambda item: (item.lyric_idx, item.audio_start))
    n = len(candidates)
    scores = [candidate.confidence + candidate.coverage for candidate in candidates]
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if candidates[j].lyric_idx < candidates[i].lyric_idx and candidates[j].audio_end < candidates[i].audio_start:
                score = scores[j] + candidates[i].confidence + candidates[i].coverage
                if score > scores[i]:
                    scores[i] = score
                    prev[i] = j
    best_idx = max(range(n), key=lambda idx: scores[idx])
    chain: list[Anchor] = []
    while best_idx >= 0:
        chain.append(candidates[best_idx])
        best_idx = prev[best_idx]
    chain.reverse()
    return chain
