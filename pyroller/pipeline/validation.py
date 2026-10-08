from __future__ import annotations

from pyroller.domain import PipelineRequest
from pyroller.config_contracts import validate_configs
from pyroller.i18n import _
from pyroller.pipeline.stages import (
    format_missing_inputs,
    infer_input_audio_role,
    missing_inputs_for_stage,
    resolve_language,
    validate_contiguous_stage_chain,
)


def validate_pipeline_request(request: PipelineRequest, stages: list[str]) -> None:
    if not stages:
        raise ValueError(_("At least one stage is required. Use --stages s,f,t,p,a,w or a subset."))
    resolve_language(request.language)
    if request.cleanup not in {"on-success", "never"}:
        raise ValueError("Invalid cleanup policy")
    validate_configs(request)
    validate_contiguous_stage_chain(stages)

    first_stage = stages[0]
    explicit_aligner_inputs: list[str] = []
    if request.timed_units_path is not None:
        explicit_aligner_inputs.append("--timed-units")
    if request.parsed_lyrics_path is not None:
        explicit_aligner_inputs.append("--parsed-lyrics")
    if first_stage != "aligner" and explicit_aligner_inputs:
        joined = " and ".join(explicit_aligner_inputs)
        raise ValueError(_("{} {} only allowed when the selected stage chain starts with 'a'/'aligner'.").format(joined, _("is") if len(explicit_aligner_inputs) == 1 else _("are")))
    if first_stage != "writer" and request.alignment_result_path is not None:
        raise ValueError(_("--alignment-result is only allowed when the selected stage chain starts with 'w'/'writer'."))
    if "parser" not in stages and request.lyrics_path is not None:
        raise ValueError(_("--lyrics is only allowed when the selected stage chain includes 'p'/'parser'."))
    if "parser" not in stages and request.parser_lyrics_encoding is not None:
        raise ValueError(_("--parser-lyrics-encoding is only allowed when the selected stage chain includes 'p'/'parser'."))
    if first_stage not in {"splitter", "filter", "transcriber"} and request.audio_path is not None:
        raise ValueError(_("--audio is only allowed when the selected stage chain starts with 's'/'splitter', 'f'/'filter', or 't'/'transcriber'."))

    if request.output_vocal_audio_path is not None and "splitter" not in stages:
        raise ValueError(_("--output-vocal-audio requires stage 's'/'splitter'."))
    if request.output_filtered_audio_path is not None and "filter" not in stages:
        raise ValueError(_("--output-filtered-audio requires stage 'f'/'filter'."))
    if request.output_timed_units_path is not None and "transcriber" not in stages:
        raise ValueError(_("--output-timed-units requires stage 't'/'transcriber'."))
    if request.output_parsed_lyrics_path is not None and "parser" not in stages:
        raise ValueError(_("--output-parsed-lyrics requires stage 'p'/'parser'."))
    if request.output_alignment_result_path is not None and "aligner" not in stages:
        raise ValueError(_("--output-alignment-result requires stage 'a'/'aligner'."))
    if request.output_roller_path is not None and "writer" not in stages:
        raise ValueError(_("--output-roller requires stage 'w'/'writer'."))
    if "writer" in stages and request.output_roller_path is None:
        raise ValueError(_("Stage 'writer' requires --output-roller."))

    validate_stage_specific_options(request, stages)

    available: set[str] = set()
    if request.audio_path is not None:
        available.add(infer_input_audio_role(stages))
    if request.lyrics_path is not None:
        available.add("lyrics_text")
    if request.timed_units_path is not None:
        available.add("timed_units")
    if request.parsed_lyrics_path is not None:
        available.add("parsed_lyrics")
    if request.alignment_result_path is not None:
        available.add("alignment_result")

    for stage in stages:
        missing = missing_inputs_for_stage(stage, available)
        if missing:
            raise ValueError(format_missing_inputs(stage, missing))
        if stage == "splitter":
            available.add("vocal_audio")
        elif stage == "filter":
            available.add("filtered_vocal_audio")
        elif stage == "transcriber":
            available.add("timed_units")
        elif stage == "parser":
            available.add("parsed_lyrics")
        elif stage == "aligner":
            available.add("alignment_result")
        elif stage == "writer":
            available.add("written_output")


def validate_stage_specific_options(request: PipelineRequest, stages: list[str]) -> None:
    for stage, config in request.backend_config.items():
        if stage != "quality" and config and stage not in stages:
            raise ValueError(f"{stage} options require the {stage} stage")
    writer = request.backend_config.get("writer", {})
    if "writer" in stages:
        if writer.get("backend", "lrc_ms") != "ass_karaoke" and any(key in writer for key in ("tag_type", "unmatched_line_duration")):
            raise ValueError("ASS timing options require writer backend ass_karaoke")
        from pyroller.writer.registry import build_writer
        build_writer(writer.get("backend"), writer)
