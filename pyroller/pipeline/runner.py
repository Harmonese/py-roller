from __future__ import annotations
from pyroller.i18n import _

import importlib.util
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from pyroller.aligner import build_aligner
from pyroller.domain import (
    AlignmentResult,
    AudioArtifact,
    LyricLine,
    LyricsDocument,
    ParsedLyrics,
    PipelineRequest,
    RunPipelineResult,
    TranscriptionResult,
)
from pyroller.filter import build_filter_chain, get_filter_requirements
from pyroller.parser import get_lyrics_parser
from pyroller.parser.registry import get_parser_requirements
from pyroller.pipeline.execution_context import PipelineExecutionContext
from pyroller.progress import NullProgressReporter, ProgressReporter
from pyroller.splitter import build_splitter, get_splitter_requirements
from pyroller.transcriber.registry import get_transcriber_config_keys, get_transcriber_requirements, resolve_transcriber_backend
from pyroller.utils.ids import make_id
from pyroller.writer import build_writer

logger = logging.getLogger("pyroller.pipeline")

_STAGE_ALIASES = {
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
_CANONICAL_STAGE_ORDER = ["splitter", "filter", "transcriber", "parser", "aligner", "writer"]
_INTERMEDIATE_OWNER_MARKER = ".py-roller-owner.json"
_AUTO_LYRICS_ENCODINGS = ("utf-8-sig", "utf-16", "gb18030", "shift_jis")
_LYRICS_ENCODING_ALIASES = {
    "shift-jis": "shift_jis",
    "shift_jis": "shift_jis",
    "utf-8": "utf-8",
    "utf-8-sig": "utf-8-sig",
    "utf-16": "utf-16",
    "gbk": "gbk",
    "gb18030": "gb18030",
    "auto": "auto",
}

class ComposablePipelineRunner:
    def __init__(
        self,
        progress_reporter: ProgressReporter | None = None,
        execution_context: PipelineExecutionContext | None = None,
    ) -> None:
        self.progress_reporter = progress_reporter or NullProgressReporter()
        self.execution_context = execution_context or PipelineExecutionContext()

    def close(self) -> None:
        self.execution_context.close()

    def run(self, request: PipelineRequest) -> RunPipelineResult:
        stages = self._resolve_execution_plan(request)
        effective_language = self._resolve_language(request.language)
        success = False
        result = RunPipelineResult(executed_stages=[])
        self.progress_reporter.event(
            "run_started",
            stage="run",
            stages=stages,
            language=effective_language,
            requested_language=request.language,
            message=_("Pipeline run started"),
        )
        try:
            self._validate_request(request, stages)
            self._preflight_environment(request, stages, effective_language)
            self._ensure_intermediate_dir_ownership(request, stages)

            registry: dict[str, Any] = {}

            if request.audio_path is not None:
                input_audio_role = self._infer_input_audio_role(stages)
                source_audio = AudioArtifact(
                    artifact_id=make_id("artifact"),
                    stage="input",
                    kind="audio",
                    path=request.audio_path,
                    role=input_audio_role,
                    metadata={"source": "cli", "inferred_role": input_audio_role},
                )
                registry[input_audio_role] = source_audio
                result.source_audio_artifact = source_audio
                result.current_audio_artifact = source_audio
                result.artifacts.append(source_audio)

            if request.lyrics_path is not None:
                lyrics_document = self._load_lyrics_document(
                    request.lyrics_path,
                    language=effective_language,
                    requested_encoding=request.parser_lyrics_encoding,
                )
                registry["lyrics_text"] = lyrics_document
                result.lyrics_document = lyrics_document

            if request.timed_units_path is not None:
                registry["timed_units"] = TranscriptionResult.load(request.timed_units_path)
                result.transcription = registry["timed_units"]

            if request.parsed_lyrics_path is not None:
                registry["parsed_lyrics"] = ParsedLyrics.load(request.parsed_lyrics_path)
                result.parsed_lyrics = registry["parsed_lyrics"]

            if request.alignment_result_path is not None:
                registry["alignment_result"] = AlignmentResult.load(request.alignment_result_path)
                result.alignment = registry["alignment_result"]

            splitter_cfg = dict(request.backend_config.get("splitter", {}))
            filter_cfg = dict(request.backend_config.get("filter", {}))
            transcriber_cfg = dict(request.backend_config.get("transcriber", {}))
            parser_cfg = dict(request.backend_config.get("parser", {}))
            aligner_cfg = dict(request.backend_config.get("aligner", {}))
            writer_cfg = dict(request.backend_config.get("writer", {}))

            logger.info(_("Execution plan resolved: %s"), " -> ".join(stages))
            if effective_language != request.language:
                logger.info(_("Effective pipeline language resolved to %s (requested=%s)"), effective_language, request.language)
            logger.info(_("Intermediate task dir: %s"), request.intermediate_dir)
            self._log_inputs(request, stages)

            if "splitter" in stages:
                input_audio = self._require_audio_role(registry, "mixed_audio", "splitter")
                splitter = build_splitter(
                    backend_name=str(splitter_cfg.get("backend")) if splitter_cfg.get("backend") else None,
                    output_dir=request.intermediate_dir / "splitter",
                    config=splitter_cfg,
                )
                split_artifact = splitter.split(input_audio.path, progress=self.progress_reporter)
                if request.output_vocal_audio_path is not None:
                    split_artifact = self._materialize_audio_artifact(
                        split_artifact,
                        request.output_vocal_audio_path,
                        stage_name="splitter",
                        role="vocal_audio",
                    )
                registry["vocal_audio"] = split_artifact
                result.current_audio_artifact = split_artifact
                result.artifacts.append(split_artifact)
                result.executed_stages.append("splitter")
                logger.info(_("Stage splitter complete -> %s"), split_artifact.path)

            if "filter" in stages:
                input_audio = self._require_audio_input(registry, "filter")
                chain_names = [str(item) for item in filter_cfg.get("chain", [])]
                filter_chain = build_filter_chain(
                    chain_names=chain_names,
                    output_dir=request.intermediate_dir / "filter",
                    config=filter_cfg,
                )
                logger.info(_("Using filter chain: %s"), ", ".join(chain_names) if chain_names else _("<empty>"))
                filtered_artifact = filter_chain.process(input_audio, progress=self.progress_reporter)
                if request.output_filtered_audio_path is not None:
                    filtered_artifact = self._materialize_audio_artifact(
                        filtered_artifact,
                        request.output_filtered_audio_path,
                        stage_name="filter",
                        role="filtered_vocal_audio",
                    )
                registry["filtered_vocal_audio"] = filtered_artifact
                result.current_audio_artifact = filtered_artifact
                result.artifacts.append(filtered_artifact)
                result.executed_stages.append("filter")
                logger.info(_("Stage filter complete -> %s"), filtered_artifact.path)

            if "transcriber" in stages:
                input_audio = self._require_audio_input(registry, "transcriber")
                transcriber = self._get_transcriber(effective_language, transcriber_cfg)
                transcription = transcriber.transcribe(input_audio, language=effective_language, tone_mode="ignore", progress=self.progress_reporter)
                registry["timed_units"] = transcription
                result.transcription = transcription
                if request.output_timed_units_path is not None:
                    transcription.save(request.output_timed_units_path)
                    self._emit_artifact_written("transcriber", "timed_units", request.output_timed_units_path)
                    logger.info(_("Wrote timed_units artifact to %s"), request.output_timed_units_path)
                result.executed_stages.append("transcriber")
                logger.info(_("Stage transcriber complete -> %d timed units"), len(transcription.units))

            if "parser" in stages:
                parser_stage = self.progress_reporter.stage("parser", total=2, unit="phase")
                parser_failed = True
                try:
                    parser_stage.phase(_("parsing lyrics"))
                    lyrics_document = self._require_registry_item(registry, "lyrics_text", "parser")
                    parser = get_lyrics_parser(effective_language, config=parser_cfg)
                    parsed_lyrics = parser.parse(lyrics_document, language=effective_language, tone_mode="ignore")
                    parser_stage.phase(_("parsed {} lyric lines").format(len(parsed_lyrics.lines)))
                    registry["parsed_lyrics"] = parsed_lyrics
                    result.parsed_lyrics = parsed_lyrics
                    if request.output_parsed_lyrics_path is not None:
                        parsed_lyrics.save(request.output_parsed_lyrics_path)
                        self._emit_artifact_written("parser", "parsed_lyrics", request.output_parsed_lyrics_path)
                        logger.info(_("Wrote parsed_lyrics artifact to %s"), request.output_parsed_lyrics_path)
                    parser_failed = False
                finally:
                    if parser_failed:
                        parser_stage.fail(_("parser failed"))
                    else:
                        parser_stage.close(_("parser complete"))
                result.executed_stages.append("parser")
                logger.info(_("Stage parser complete -> %d lines"), len(parsed_lyrics.lines))

            if "aligner" in stages:
                transcription = self._require_registry_item(registry, "timed_units", "aligner")
                parsed_lyrics = self._require_registry_item(registry, "parsed_lyrics", "aligner")
                aligner = build_aligner(
                    backend_name=str(aligner_cfg.get("backend")) if aligner_cfg.get("backend") else None,
                    config=aligner_cfg,
                )
                logger.info(_("Using aligner backend: %s"), getattr(aligner, "name", aligner.__class__.__name__))
                alignment = aligner.align(transcription, parsed_lyrics, progress=self.progress_reporter)
                registry["alignment_result"] = alignment
                result.alignment = alignment
                if request.output_alignment_result_path is not None:
                    alignment.save(request.output_alignment_result_path)
                    self._emit_artifact_written("aligner", "alignment_result", request.output_alignment_result_path)
                    logger.info(_("Wrote alignment_result artifact to %s"), request.output_alignment_result_path)
                result.executed_stages.append("aligner")
                logger.info(_("Stage aligner complete -> %d aligned lines"), len(alignment.lines))

            if "writer" in stages:
                writer_stage = self.progress_reporter.stage("writer", total=2, unit="phase")
                writer_failed = True
                try:
                    writer_stage.phase(_("writing output"))
                    alignment = self._require_registry_item(registry, "alignment_result", "writer")
                    output_path = request.output_roller_path
                    if output_path is None:
                        raise ValueError(_("Writer stage requires --output-roller."))
                    writer = build_writer(
                        backend_name=str(writer_cfg.get("backend")) if writer_cfg.get("backend") else None,
                        config=writer_cfg,
                    )
                    logger.info(_("Using writer backend: %s"), getattr(writer, "backend_name", getattr(writer, "name", writer.__class__.__name__)))
                    result.write_result = writer.write(alignment, output_path)
                    writer_stage.phase(_("output written"))
                    self._emit_artifact_written("writer", "roller", result.write_result.output_path)
                    writer_failed = False
                finally:
                    if writer_failed:
                        writer_stage.fail(_("writer failed"))
                    else:
                        writer_stage.close(_("writer complete"))
                result.executed_stages.append("writer")
                logger.info(_("Stage writer complete -> %s"), result.write_result.output_path)

            success = True
            self.progress_reporter.event(
                "run_completed",
                stage="run",
                stages=stages,
                executed_stages=result.executed_stages,
                message=_("Pipeline run completed"),
                done=True,
            )
            return result
        except Exception as exc:
            self.progress_reporter.event(
                "run_failed",
                stage="run",
                stages=stages,
                message=_("Pipeline run failed: {}: {}").format(exc.__class__.__name__, exc),
                error_type=exc.__class__.__name__,
                error=str(exc),
                failed=True,
            )
            raise
        finally:
            if success:
                self._cleanup_intermediate_dir(request)

    def _emit_artifact_written(self, stage: str, artifact_type: str, path: Path | None) -> None:
        self.progress_reporter.event(
            "artifact_written",
            stage=stage,
            artifact_type=artifact_type,
            path=str(path) if path is not None else None,
            message=_("Wrote {} artifact").format(artifact_type),
        )

    def _cleanup_intermediate_dir(self, request: PipelineRequest) -> None:
        if request.cleanup != "on-success":
            return
        if not request.intermediate_dir.exists():
            return
        marker_path = request.intermediate_dir / _INTERMEDIATE_OWNER_MARKER
        if not marker_path.exists():
            logger.warning(_("Refusing to remove intermediate dir without ownership marker: %s"), request.intermediate_dir)
            return
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning(_("Refusing to remove intermediate dir with unreadable ownership marker: %s"), request.intermediate_dir)
            return
        if not marker.get("owned_by") == "py-roller" or marker.get("intermediate_dir") != str(request.intermediate_dir.resolve()):
            logger.warning(_("Refusing to remove intermediate dir with mismatched ownership marker: %s"), request.intermediate_dir)
            return
        shutil.rmtree(request.intermediate_dir, ignore_errors=True)

    def _resolve_execution_plan(self, request: PipelineRequest) -> list[str]:
        canonical: list[str] = []
        seen: set[str] = set()
        for raw in request.stages:
            try:
                stage = _STAGE_ALIASES[raw.lower()]
            except KeyError as exc:
                raise ValueError(_("Unknown stage: {}").format(raw)) from exc
            if stage not in seen:
                seen.add(stage)
                canonical.append(stage)
        return [stage for stage in _CANONICAL_STAGE_ORDER if stage in canonical]

    def _resolve_language(self, requested_language: str) -> str:
        normalized = (requested_language or "").strip().lower()
        if normalized in {"zh", "en", "mul"}:
            return normalized
        logger.error(_("Unsupported language=%s. Falling back to language=mul."), requested_language)
        return "mul"

    def _validate_request(self, request: PipelineRequest, stages: list[str]) -> None:
        if not stages:
            raise ValueError(_("At least one stage is required. Use --stages s,f,t,p,a,w or a subset."))
        self._validate_contiguous_stage_chain(stages)

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

        self._validate_stage_specific_options(request, stages)

        available: set[str] = set()
        if request.audio_path is not None:
            available.add(self._infer_input_audio_role(stages))
        if request.lyrics_path is not None:
            available.add("lyrics_text")
        if request.timed_units_path is not None:
            available.add("timed_units")
        if request.parsed_lyrics_path is not None:
            available.add("parsed_lyrics")
        if request.alignment_result_path is not None:
            available.add("alignment_result")

        for stage in stages:
            missing = self._missing_inputs_for_stage(stage, available)
            if missing:
                raise ValueError(self._format_missing_inputs(stage, missing))
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

    def _validate_stage_specific_options(self, request: PipelineRequest, stages: list[str]) -> None:
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
                "by_tag": "--writer-by-tag",
                "tag_type": "--writer-ass-karaoke-tag-type",
            }
            used = [flag for key, flag in writer_flags.items() if key in writer_cfg]
            if used:
                joined = ", ".join(used)
                raise ValueError(_("{} {} only allowed when the selected stage chain includes 'w'/'writer'.").format(joined, _("is") if len(used) == 1 else _("are")))
        if "transcriber" in stages:
            _, chosen_transcriber_backend = resolve_transcriber_backend(
                self._resolve_language(request.language),
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

    def _validate_contiguous_stage_chain(self, stages: list[str]) -> None:
        indices = [_CANONICAL_STAGE_ORDER.index(stage) for stage in stages]
        expected = list(range(indices[0], indices[0] + len(indices)))
        if indices != expected:
            normalized = ",".join(stage[0] for stage in stages)
            raise ValueError(
                _("Selected stages must form a contiguous chain in canonical order s,f,t,p,a,w. "
                  "Got normalized stages: {}. For example, use s,f,t or t,p,a,w, but not s,t,w.").format(normalized)
            )

    def _infer_input_audio_role(self, stages: list[str]) -> str:
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

    def _missing_inputs_for_stage(self, stage: str, available: set[str]) -> list[str]:
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

    def _format_missing_inputs(self, stage: str, missing: list[str]) -> str:
        joined = ", ".join(missing)
        return _("Stage '{}' is missing required input artifact(s): {}. Provide them explicitly or add the producing stage(s).").format(stage, joined)

    def _get_transcriber(self, language: str, config: dict[str, Any]):
        backend_name = str(config.get("backend")) if config.get("backend") else None
        return self.execution_context.get_transcriber(
            language=language,
            backend_name=backend_name,
            config=config,
        )

    def _preflight_environment(self, request: PipelineRequest, stages: list[str], effective_language: str) -> None:
        required_modules: dict[str, str] = {}

        if "splitter" in stages:
            splitter_backend = str(request.backend_config.get("splitter", {}).get("backend") or "demucs")
            for module_name in get_splitter_requirements(splitter_backend):
                required_modules[module_name] = f"splitter backend {splitter_backend}"

        if "filter" in stages:
            for filter_name in request.backend_config.get("filter", {}).get("chain", []) or []:
                for module_name in get_filter_requirements(str(filter_name)):
                    required_modules[module_name] = f"filter backend {filter_name}"

        if "transcriber" in stages:
            _, transcriber_backend = resolve_transcriber_backend(
                effective_language,
                str(request.backend_config.get("transcriber", {}).get("backend"))
                if request.backend_config.get("transcriber", {}).get("backend")
                else None,
            )
            for module_name in get_transcriber_requirements(transcriber_backend):
                required_modules[module_name] = f"transcriber backend {transcriber_backend}"

        if "parser" in stages:
            for module_name in get_parser_requirements(effective_language):
                required_modules[module_name] = f"parser language {effective_language}"

        missing: list[str] = []
        for module_name, reason in required_modules.items():
            if importlib.util.find_spec(module_name) is None:
                missing.append(f"{module_name} ({reason})")

        if missing:
            message = _("Preflight dependency check failed. Missing module(s): {}").format(", ".join(sorted(missing)))
            logger.error(message)
            raise RuntimeError(message)

        if "transcriber" in stages:
            transcriber_cfg = dict(request.backend_config.get("transcriber", {}))
            _, transcriber_backend = resolve_transcriber_backend(
                effective_language,
                str(transcriber_cfg.get("backend")) if transcriber_cfg.get("backend") else None,
            )
            transcriber = self._get_transcriber(effective_language, transcriber_cfg)
            logger.info(_("Running transcriber preflight for backend=%s language=%s"), transcriber_backend, effective_language)
            preflight_stage = self.progress_reporter.stage(
                "preflight",
                total=1 + transcriber.preflight_phase_total(effective_language),
                unit="phase",
            )
            preflight_failed = True
            preflight_failure_message = _("preflight failed")
            try:
                preflight_stage.phase(_("loading transcriber backend"))
                preflight_report = transcriber.preflight(effective_language, stage=preflight_stage)
                logger.info(_("Transcriber preflight passed: %s"), json.dumps(preflight_report, ensure_ascii=False))
                preflight_failed = False
            except Exception as exc:
                preflight_failure_message = _("preflight failed: {}: {}").format(exc.__class__.__name__, exc)
                raise
            finally:
                if preflight_failed:
                    preflight_stage.fail(preflight_failure_message)
                else:
                    preflight_stage.close(_("preflight complete"))

    def _ensure_intermediate_dir_ownership(self, request: PipelineRequest, stages: list[str]) -> None:
        request.intermediate_dir.mkdir(parents=True, exist_ok=True)
        marker_path = request.intermediate_dir / _INTERMEDIATE_OWNER_MARKER
        marker = {
            "owned_by": "py-roller",
            "intermediate_dir": str(request.intermediate_dir.resolve()),
            "stages": stages,
        }
        marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")

    def _require_registry_item(self, registry: dict[str, Any], key: str, stage: str):
        if key not in registry:
            raise ValueError(_("Stage '{}' requires artifact '{}', but it is not available.").format(stage, key))
        return registry[key]

    def _require_audio_role(self, registry: dict[str, Any], role: str, stage: str) -> AudioArtifact:
        item = self._require_registry_item(registry, role, stage)
        if not isinstance(item, AudioArtifact):
            raise TypeError(_("Artifact '{}' is not an AudioArtifact.").format(role))
        return item

    def _require_audio_input(self, registry: dict[str, Any], stage: str) -> AudioArtifact:
        for role in ("filtered_vocal_audio", "vocal_audio", "mixed_audio"):
            item = registry.get(role)
            if isinstance(item, AudioArtifact):
                return item
        raise ValueError(_("Stage '{}' requires an audio artifact, but none is available.").format(stage))

    def _load_lyrics_document(self, lyrics_path: Path, language: str, requested_encoding: str | None) -> LyricsDocument:
        normalized_request = self._normalize_lyrics_encoding(requested_encoding)
        raw_text, used_encoding = self._read_lyrics_text(lyrics_path, normalized_request)

        lines: list[LyricLine] = []
        last_was_spacing = False
        for raw_line in raw_text.splitlines():
            stripped = raw_line.strip()
            if stripped:
                lines.append(LyricLine(line_index=len(lines), raw_text=stripped))
                last_was_spacing = False
                continue
            if not last_was_spacing:
                lines.append(
                    LyricLine(
                        line_index=len(lines),
                        raw_text="",
                        metadata={"is_spacing": True},
                    )
                )
                last_was_spacing = True

        logger.info(
            _("Loaded lyrics: %d lines from %s (spacing lines preserved, requested_encoding=%s, used_encoding=%s)"),
            len(lines),
            lyrics_path,
            normalized_request,
            used_encoding,
        )
        return LyricsDocument(
            source_path=lyrics_path,
            raw_text=raw_text,
            encoding=used_encoding,
            lines=lines,
            language=language,
            metadata={"spacing_lines_preserved": True, "requested_encoding": normalized_request},
        )

    def _normalize_lyrics_encoding(self, requested_encoding: str | None) -> str:
        normalized = _LYRICS_ENCODING_ALIASES.get((requested_encoding or "auto").strip().lower())
        if normalized is None:
            raise ValueError(
                _("Unsupported parser lyrics encoding {}. Supported values: auto, utf-8, utf-8-sig, utf-16, gbk, gb18030, shift-jis.").format(repr(requested_encoding))
            )
        return normalized

    def _read_lyrics_text(self, lyrics_path: Path, requested_encoding: str) -> tuple[str, str]:
        if requested_encoding != "auto":
            try:
                return lyrics_path.read_text(encoding=requested_encoding), requested_encoding
            except UnicodeDecodeError as exc:
                logger.error(_("Lyrics file decode failed with encoding=%s: %s"), requested_encoding, lyrics_path)
                raise ValueError(_("Lyrics file could not be decoded as {}: {}").format(requested_encoding, lyrics_path)) from exc

        for encoding in _AUTO_LYRICS_ENCODINGS:
            try:
                return lyrics_path.read_text(encoding=encoding), encoding
            except UnicodeDecodeError:
                continue
        logger.error(_("Lyrics file could not be decoded with auto encodings: %s"), lyrics_path)
        raise ValueError(
            _("Lyrics file could not be decoded with auto mode. Tried utf-8-sig, utf-16, gb18030, shift-jis: {}").format(lyrics_path)
        )

    def _materialize_audio_artifact(self, artifact: AudioArtifact, destination: Path, stage_name: str, role: str) -> AudioArtifact:
        if artifact.path is None:
            raise ValueError(_("Cannot materialize audio artifact from stage '{}' without a source path.").format(stage_name))
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact.path, destination)
        metadata = dict(artifact.metadata)
        metadata["materialized_from"] = str(artifact.path)
        logger.info(_("Materialized %s output to %s"), stage_name, destination)
        return AudioArtifact(
            artifact_id=make_id("artifact"),
            stage=stage_name,
            kind="audio",
            path=destination,
            sample_rate=artifact.sample_rate,
            channels=artifact.channels,
            duration=artifact.duration,
            role=role,
            metadata=metadata,
        )

    def _log_inputs(self, request: PipelineRequest, stages: list[str]) -> None:
        if request.audio_path is not None:
            inferred_role = self._infer_input_audio_role(stages)
            logger.info(_("Input audio: %s (%s)"), request.audio_path, inferred_role)
        if request.lyrics_path is not None:
            logger.info(_("Input lyrics: %s"), request.lyrics_path)
        if request.parser_lyrics_encoding is not None:
            logger.info(_("Parser lyrics encoding request: %s"), request.parser_lyrics_encoding)
        if request.timed_units_path is not None:
            logger.info(_("Input timed_units: %s"), request.timed_units_path)
        if request.parsed_lyrics_path is not None:
            logger.info(_("Input parsed_lyrics: %s"), request.parsed_lyrics_path)
        if request.alignment_result_path is not None:
            logger.info(_("Input alignment_result: %s"), request.alignment_result_path)
