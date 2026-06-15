from __future__ import annotations

import logging

from pyroller.domain import PipelineRequest
from pyroller.i18n import _

logger = logging.getLogger("pyroller.pipeline")

STAGE_ALIASES = {
    "s": "splitter",
    "split": "splitter",
    "splitter": "splitter",
    "f": "filter",
    "filter": "filter",
    "t": "transcriber",
    "transcribe": "transcriber",
    "transcriber": "transcriber",
    "p": "parser",
    "parse": "parser",
    "parser": "parser",
    "a": "aligner",
    "align": "aligner",
    "aligner": "aligner",
    "w": "writer",
    "write": "writer",
    "writer": "writer",
}
CANONICAL_STAGE_ORDER = ["splitter", "filter", "transcriber", "parser", "aligner", "writer"]
SUPPORTED_LANGUAGES = {"zh", "en", "mul"}


def resolve_execution_plan(request: PipelineRequest) -> list[str]:
    canonical: list[str] = []
    seen: set[str] = set()
    for raw in request.stages:
        try:
            stage = STAGE_ALIASES[raw.lower()]
        except KeyError as exc:
            raise ValueError(_("Unknown stage: {}").format(raw)) from exc
        if stage not in seen:
            seen.add(stage)
            canonical.append(stage)
    resolved = [stage for stage in CANONICAL_STAGE_ORDER if stage in canonical]
    if canonical != resolved:
        logger.info(
            _("Requested stages were normalized to canonical execution order: requested=%s resolved=%s"),
            " -> ".join(canonical),
            " -> ".join(resolved),
        )
    return resolved


def resolve_language(requested_language: str) -> str:
    normalized = (requested_language or "").strip().lower()
    if normalized in SUPPORTED_LANGUAGES:
        return normalized
    raise ValueError(
        _("Unsupported language={}. Supported values: zh, en, mul.").format(requested_language)
    )


def validate_contiguous_stage_chain(stages: list[str]) -> None:
    indices = [CANONICAL_STAGE_ORDER.index(stage) for stage in stages]
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        normalized = ",".join(stage[0] for stage in stages)
        raise ValueError(
            _("Selected stages must form a contiguous chain in canonical order s,f,t,p,a,w. "
              "Got normalized stages: {}. For example, use s,f,t or t,p,a,w, but not s,t,w.").format(normalized)
        )


def infer_input_audio_role(stages: list[str]) -> str:
    if not stages:
        raise ValueError(_("Cannot infer audio role without any stages."))
    first_stage = stages[0]
    if first_stage == "splitter":
        return "mixed_audio"
    if first_stage == "filter":
        return "vocal_audio"
    if first_stage == "transcriber":
        return "filtered_vocal_audio"
    return "mixed_audio"


def missing_inputs_for_stage(stage: str, available: set[str]) -> list[str]:
    if stage == "splitter":
        return [] if "mixed_audio" in available else ["mixed_audio"]
    if stage == "filter":
        return [] if "vocal_audio" in available else ["vocal_audio"]
    if stage == "transcriber":
        return [] if "filtered_vocal_audio" in available else ["filtered_vocal_audio"]
    if stage == "parser":
        return [] if "lyrics_text" in available else ["lyrics_text"]
    if stage == "aligner":
        missing: list[str] = []
        if "timed_units" not in available:
            missing.append("timed_units")
        if "parsed_lyrics" not in available:
            missing.append("parsed_lyrics")
        return missing
    if stage == "writer":
        return [] if "alignment_result" in available else ["alignment_result"]
    return []


def format_missing_inputs(stage: str, missing: list[str]) -> str:
    joined = ", ".join(missing)
    return _("Stage '{}' is missing required input artifact(s): {}. Provide them explicitly or add the producing stage(s).").format(stage, joined)
