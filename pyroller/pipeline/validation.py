from __future__ import annotations

from pyroller.domain import PipelineRequest
from pyroller.i18n import _
from pyroller.pipeline.stages import (
    format_missing_inputs,
    infer_input_audio_role,
    missing_inputs_for_stage,
    resolve_language,
    validate_contiguous_stage_chain,
)
from pyroller.transcriber.registry import get_transcriber_config_keys, resolve_transcriber_backend


def validate_pipeline_request(request: PipelineRequest, stages: list[str]) -> None:
    if not stages:
        raise ValueError(_("At least one stage is required. Use --stages s,f,t,p,a,w or a subset."))
    resolve_language(request.language)
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
    splitter_cfg = dict(request.backend_config.get("splitter", {}))
    filter_cfg = dict(request.backend_config.get("filter", {}))
    transcriber_cfg = dict(request.backend_config.get("transcriber", {}))
    aligner_cfg = dict(request.backend_config.get("aligner", {}))
    writer_cfg = dict(request.backend_config.get("writer", {}))

    if "splitter" not in stages:
        splitter_flags = {
            "backend": "--splitter-backend",
            "model": "--splitter-demucs-model",
            "device": "--splitter-demucs-device",
            "jobs": "--splitter-demucs-jobs",
            "overlap": "--splitter-demucs-overlap",
            "segment": "--splitter-demucs-segment",
        }
        used = [flag for key, flag in splitter_flags.items() if key in splitter_cfg]
        if used:
            joined = ", ".join(used)
            raise ValueError(_("{} {} only allowed when the selected stage chain includes 's'/'splitter'.").format(joined, _("is") if len(used) == 1 else _("are")))
    if "filter" not in stages and "chain" in filter_cfg:
        raise ValueError(_("--filter-chain is only allowed when the selected stage chain includes 'f'/'filter'."))
    if "transcriber" not in stages:
        transcriber_flags = {
            "backend": "--transcriber-backend",
            "device": "--transcriber-device",
            "model_name": "--transcriber-model-name",
            "model_path": "--transcriber-model-path",
            "local_files_only": "--transcriber-local-files-only",
            "hf_xet": "--transcriber-hf-xet",
            "hf_proxy": "--transcriber-hf-proxy",
            "hf_etag_timeout": "--transcriber-hf-etag-timeout",
            "hf_download_timeout": "--transcriber-hf-download-timeout",
            "hf_max_workers": "--transcriber-hf-max-workers",
            "compute_type": "--transcriber-compute-type",
            "batch_size": "--transcriber-batch-size",
            "vad_filter": "--transcriber-vad-filter",
        }
        used = [flag for key, flag in transcriber_flags.items() if key in transcriber_cfg]
        if used:
            joined = ", ".join(used)
            raise ValueError(_("{} {} only allowed when the selected stage chain includes 't'/'transcriber'.").format(joined, _("is") if len(used) == 1 else _("are")))
    if "aligner" not in stages:
        aligner_flags = {
            "backend": "--aligner-backend",
            "min_gap": "--aligner-min-gap",
            "repetition": "--aligner-repetition",
        }
        used = [flag for key, flag in aligner_flags.items() if key in aligner_cfg]
        if used:
            joined = ", ".join(used)
            raise ValueError(_("{} {} only allowed when the selected stage chain includes 'a'/'aligner'.").format(joined, _("is") if len(used) == 1 else _("are")))
    if "writer" not in stages:
        writer_flags = {
            "backend": "--writer-backend",
            "spacing": "--writer-spacing",
            "by_tag": "--writer-by-tag",
            "tag_type": "--writer-ass-karaoke-tag-type",
        }
        used = [flag for key, flag in writer_flags.items() if key in writer_cfg]
        if used:
            joined = ", ".join(used)
            raise ValueError(_("{} {} only allowed when the selected stage chain includes 'w'/'writer'.").format(joined, _("is") if len(used) == 1 else _("are")))
    if "transcriber" in stages:
        _resolved_language, chosen_transcriber_backend = resolve_transcriber_backend(
            resolve_language(request.language),
            str(transcriber_cfg.get("backend")) if transcriber_cfg.get("backend") else None,
        )
        transcriber_flags = {
            "backend": "--transcriber-backend",
            "device": "--transcriber-device",
            "model_name": "--transcriber-model-name",
            "model_path": "--transcriber-model-path",
            "local_files_only": "--transcriber-local-files-only",
            "hf_xet": "--transcriber-hf-xet",
            "hf_proxy": "--transcriber-hf-proxy",
            "hf_etag_timeout": "--transcriber-hf-etag-timeout",
            "hf_download_timeout": "--transcriber-hf-download-timeout",
            "hf_max_workers": "--transcriber-hf-max-workers",
            "compute_type": "--transcriber-compute-type",
            "batch_size": "--transcriber-batch-size",
            "vad_filter": "--transcriber-vad-filter",
        }
        accepted_transcriber_keys = get_transcriber_config_keys(chosen_transcriber_backend)
        incompatible = [
            transcriber_flags[key]
            for key in transcriber_flags
            if key != "backend" and key in transcriber_cfg and key not in accepted_transcriber_keys
        ]
        if incompatible:
            joined = ", ".join(incompatible)
            raise ValueError(
                _("{} {} not supported by --transcriber-backend {!r}.").format(joined, _("is") if len(incompatible) == 1 else _("are"), chosen_transcriber_backend)
            )

    if "writer" in stages:
        chosen_writer_backend = str(writer_cfg.get("backend") or "lrc_ms")
        if writer_cfg.get("tag_type") is not None and chosen_writer_backend != "ass_karaoke":
            raise ValueError(_("--writer-ass-karaoke-tag-type is only allowed when --writer-backend is 'ass_karaoke'."))
