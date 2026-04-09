from __future__ import annotations

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
from pyroller.progress import NullProgressReporter, ProgressReporter
from pyroller.splitter import DemucsSplitter
from pyroller.transcriber import build_transcriber
from pyroller.transcriber.registry import get_transcriber_requirements, resolve_transcriber_backend
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
    def __init__(self, progress_reporter: ProgressReporter | None = None) -> None:
        self.progress_reporter = progress_reporter or NullProgressReporter()

    def run(self, request: PipelineRequest) -> RunPipelineResult:
        stages = self._resolve_execution_plan(request)
        effective_language = self._resolve_language(request.language)
        self._validate_request(request, stages)
        self._preflight_environment(request, stages, effective_language)
        self._ensure_intermediate_dir_ownership(request, stages)

        registry: dict[str, Any] = {}
        result = RunPipelineResult(executed_stages=[])
        success = False
        try:
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

            logger.info("Execution plan resolved: %s", " -> ".join(stages))
            if effective_language != request.language:
                logger.info("Effective pipeline language resolved to %s (requested=%s)", effective_language, request.language)
            logger.info("Intermediate task dir: %s", request.intermediate_dir)
            self._log_inputs(request, stages)

            if "splitter" in stages:
                input_audio = self._require_audio_role(registry, "mixed_audio", "splitter")
                splitter = DemucsSplitter(
                    output_dir=request.intermediate_dir / "splitter",
                    model=str(splitter_cfg.get("model", "htdemucs")),
                    two_stems=str(splitter_cfg.get("two_stems", "vocals")),
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
                logger.info("Stage splitter complete -> %s", split_artifact.path)

            if "filter" in stages:
                input_audio = self._require_audio_input(registry, "filter")
                chain_names = [str(item) for item in filter_cfg.get("chain", [])]
                filter_chain = build_filter_chain(
                    chain_names=chain_names,
                    output_dir=request.intermediate_dir / "filter",
                    config=filter_cfg,
                )
                logger.info("Using filter chain: %s", ", ".join(chain_names) if chain_names else "<empty>")
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
                logger.info("Stage filter complete -> %s", filtered_artifact.path)

            if "transcriber" in stages:
                input_audio = self._require_audio_input(registry, "transcriber")
                transcriber = build_transcriber(
                    language=effective_language,
                    backend_name=str(transcriber_cfg.get("backend")) if transcriber_cfg.get("backend") else None,
                    config=transcriber_cfg,
                )
                transcription = transcriber.transcribe(input_audio, language=effective_language, tone_mode="ignore", progress=self.progress_reporter)
                registry["timed_units"] = transcription
                result.transcription = transcription
                if request.output_timed_units_path is not None:
                    transcription.save(request.output_timed_units_path)
                    logger.info("Wrote timed_units artifact to %s", request.output_timed_units_path)
                result.executed_stages.append("transcriber")
                logger.info("Stage transcriber complete -> %d timed units", len(transcription.units))

            if "parser" in stages:
                lyrics_document = self._require_registry_item(registry, "lyrics_text", "parser")
                parser = get_lyrics_parser(effective_language, config=parser_cfg)
                parsed_lyrics = parser.parse(lyrics_document, language=effective_language, tone_mode="ignore")
                registry["parsed_lyrics"] = parsed_lyrics
                result.parsed_lyrics = parsed_lyrics
                if request.output_parsed_lyrics_path is not None:
                    parsed_lyrics.save(request.output_parsed_lyrics_path)
                    logger.info("Wrote parsed_lyrics artifact to %s", request.output_parsed_lyrics_path)
                result.executed_stages.append("parser")
                if (
                    effective_language == "zh"
                    and result.transcription is not None
                    and getattr(result.transcription, "backend", "") == "whisperx"
                    and int(parsed_lyrics.metadata.get("foreign_segment_count", 0)) > 0
                ):
                    logger.warning(
                        "zh_router_pinyin detected %d foreign/digit segments, but transcription backend=whisperx may drop non-Chinese text. Consider backend=mms_phonetic for mixed zh lyrics.",
                        int(parsed_lyrics.metadata.get("foreign_segment_count", 0)),
                    )
                logger.info("Stage parser complete -> %d lines", len(parsed_lyrics.lines))

            if "aligner" in stages:
                transcription = self._require_registry_item(registry, "timed_units", "aligner")
                parsed_lyrics = self._require_registry_item(registry, "parsed_lyrics", "aligner")
                aligner = build_aligner(
                    backend_name=str(aligner_cfg.get("backend")) if aligner_cfg.get("backend") else None,
                    config=aligner_cfg,
                )
                logger.info("Using aligner backend: %s", getattr(aligner, "name", aligner.__class__.__name__))
                alignment = aligner.align(transcription, parsed_lyrics, progress=self.progress_reporter)
                registry["alignment_result"] = alignment
                result.alignment = alignment
                if request.output_alignment_result_path is not None:
                    alignment.save(request.output_alignment_result_path)
                    logger.info("Wrote alignment_result artifact to %s", request.output_alignment_result_path)
                result.executed_stages.append("aligner")
                logger.info("Stage aligner complete -> %d aligned lines", len(alignment.lines))

            if "writer" in stages:
                alignment = self._require_registry_item(registry, "alignment_result", "writer")
                output_path = request.output_roller_path
                if output_path is None:
                    raise ValueError("Writer stage requires --output-roller.")
                writer = build_writer(
                    backend_name=str(writer_cfg.get("backend")) if writer_cfg.get("backend") else None,
                    config=writer_cfg,
                )
                logger.info("Using writer backend: %s", getattr(writer, "backend_name", getattr(writer, "name", writer.__class__.__name__)))
                result.write_result = writer.write(alignment, output_path)
                result.executed_stages.append("writer")
                logger.info("Stage writer complete -> %s", result.write_result.output_path)

            success = True
            return result
        finally:
            if success:
                self._cleanup_intermediate_dir(request)

    def _cleanup_intermediate_dir(self, request: PipelineRequest) -> None:
        if request.cleanup != "on-success":
            return
        if not request.intermediate_dir.exists():
            return
        marker_path = request.intermediate_dir / _INTERMEDIATE_OWNER_MARKER
        if not marker_path.exists():
            logger.warning("Refusing to remove intermediate dir without ownership marker: %s", request.intermediate_dir)
            return
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Refusing to remove intermediate dir with unreadable ownership marker: %s", request.intermediate_dir)
            return
        if not marker.get("owned_by") == "py-roller" or marker.get("intermediate_dir") != str(request.intermediate_dir.resolve()):
            logger.warning("Refusing to remove intermediate dir with mismatched ownership marker: %s", request.intermediate_dir)
            return
        shutil.rmtree(request.intermediate_dir, ignore_errors=True)

    def _resolve_execution_plan(self, request: PipelineRequest) -> list[str]:
        canonical: list[str] = []
        seen: set[str] = set()
        for raw in request.stages:
            try:
                stage = _STAGE_ALIASES[raw.lower()]
            except KeyError as exc:
                raise ValueError(f"Unknown stage: {raw}") from exc
            if stage not in seen:
                seen.add(stage)
                canonical.append(stage)
        return [stage for stage in _CANONICAL_STAGE_ORDER if stage in canonical]

    def _resolve_language(self, requested_language: str) -> str:
        normalized = (requested_language or "").strip().lower()
        if normalized in {"zh", "en", "mul"}:
            return normalized
        logger.error("Unsupported language=%s. Falling back to language=mul.", requested_language)
        return "mul"

    def _validate_request(self, request: PipelineRequest, stages: list[str]) -> None:
        if not stages:
            raise ValueError("At least one stage is required. Use --stages s,f,t,p,a,w or a subset.")
        self._validate_contiguous_stage_chain(stages)

        first_stage = stages[0]
        explicit_aligner_inputs: list[str] = []
        if request.timed_units_path is not None:
            explicit_aligner_inputs.append("--timed-units")
        if request.parsed_lyrics_path is not None:
            explicit_aligner_inputs.append("--parsed-lyrics")
        if first_stage != "aligner" and explicit_aligner_inputs:
            joined = " and ".join(explicit_aligner_inputs)
            raise ValueError(f"{joined} {'is' if len(explicit_aligner_inputs) == 1 else 'are'} only allowed when the selected stage chain starts with 'a'/'aligner'.")
        if first_stage != "writer" and request.alignment_result_path is not None:
            raise ValueError("--alignment-result is only allowed when the selected stage chain starts with 'w'/'writer'.")
        if "parser" not in stages and request.lyrics_path is not None:
            raise ValueError("--lyrics is only allowed when the selected stage chain includes 'p'/'parser'.")
        if "parser" not in stages and request.parser_lyrics_encoding is not None:
            raise ValueError("--parser-lyrics-encoding is only allowed when the selected stage chain includes 'p'/'parser'.")
        if first_stage not in {"splitter", "filter", "transcriber"} and request.audio_path is not None:
            raise ValueError("--audio is only allowed when the selected stage chain starts with 's'/'splitter', 'f'/'filter', or 't'/'transcriber'.")

        if request.output_vocal_audio_path is not None and "splitter" not in stages:
            raise ValueError("--output-vocal-audio requires stage 's'/'splitter'.")
        if request.output_filtered_audio_path is not None and "filter" not in stages:
            raise ValueError("--output-filtered-audio requires stage 'f'/'filter'.")
        if request.output_timed_units_path is not None and "transcriber" not in stages:
            raise ValueError("--output-timed-units requires stage 't'/'transcriber'.")
        if request.output_parsed_lyrics_path is not None and "parser" not in stages:
            raise ValueError("--output-parsed-lyrics requires stage 'p'/'parser'.")
        if request.output_alignment_result_path is not None and "aligner" not in stages:
            raise ValueError("--output-alignment-result requires stage 'a'/'aligner'.")
        if request.output_roller_path is not None and "writer" not in stages:
            raise ValueError("--output-roller requires stage 'w'/'writer'.")
        if "writer" in stages and request.output_roller_path is None:
            raise ValueError("Stage 'writer' requires --output-roller.")

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

        if "splitter" not in stages and splitter_cfg.get("model") is not None:
            raise ValueError("--splitter-demucs-model is only allowed when the selected stage chain includes 's'/'splitter'.")
        if "filter" not in stages and "chain" in filter_cfg:
            raise ValueError("--filter-chain is only allowed when the selected stage chain includes 'f'/'filter'.")
        if "transcriber" not in stages:
            transcriber_flags = {
                "backend": "--transcriber-backend",
                "device": "--transcriber-device",
                "model_name": "--transcriber-model-name",
                "compute_type": "--transcriber-compute-type",
                "batch_size": "--transcriber-batch-size",
                "align_words": "--transcriber-no-align-words",
            }
            used = [flag for key, flag in transcriber_flags.items() if key in transcriber_cfg]
            if used:
                joined = ", ".join(used)
                raise ValueError(f"{joined} {'is' if len(used) == 1 else 'are'} only allowed when the selected stage chain includes 't'/'transcriber'.")
        if "aligner" not in stages:
            aligner_flags = {
                "backend": "--aligner-backend",
                "min_gap": "--aligner-min-gap",
            }
            used = [flag for key, flag in aligner_flags.items() if key in aligner_cfg]
            if used:
                joined = ", ".join(used)
                raise ValueError(f"{joined} {'is' if len(used) == 1 else 'are'} only allowed when the selected stage chain includes 'a'/'aligner'.")
        if "writer" not in stages:
            writer_flags = {
                "backend": "--writer-backend",
                "by_tag": "--writer-by-tag",
                "tag_type": "--writer-ass-karaoke-tag-type",
            }
            used = [flag for key, flag in writer_flags.items() if key in writer_cfg]
            if used:
                joined = ", ".join(used)
                raise ValueError(f"{joined} {'is' if len(used) == 1 else 'are'} only allowed when the selected stage chain includes 'w'/'writer'.")
        if "writer" in stages:
            chosen_writer_backend = str(writer_cfg.get("backend") or "lrc_ms")
            if writer_cfg.get("tag_type") is not None and chosen_writer_backend != "ass_karaoke":
                raise ValueError("--writer-ass-karaoke-tag-type is only allowed when --writer-backend is 'ass_karaoke'.")

    def _validate_contiguous_stage_chain(self, stages: list[str]) -> None:
        indices = [_CANONICAL_STAGE_ORDER.index(stage) for stage in stages]
        expected = list(range(indices[0], indices[0] + len(indices)))
        if indices != expected:
            normalized = ",".join(stage[0] for stage in stages)
            raise ValueError(
                "Selected stages must form a contiguous chain in canonical order s,f,t,p,a,w. "
                f"Got normalized stages: {normalized}. For example, use s,f,t or t,p,a,w, but not s,t,w."
            )

    def _infer_input_audio_role(self, stages: list[str]) -> str:
        if not stages:
            raise ValueError("Cannot infer audio role without any stages.")
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
            return [] if {"mixed_audio", "vocal_audio", "filtered_vocal_audio"}.intersection(available) else ["audio_input"]
        if stage == "transcriber":
            return [] if {"mixed_audio", "vocal_audio", "filtered_vocal_audio"}.intersection(available) else ["audio_input"]
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
        return f"Stage '{stage}' is missing required input artifact(s): {joined}. Provide them explicitly or add the producing stage(s)."

    def _preflight_environment(self, request: PipelineRequest, stages: list[str], effective_language: str) -> None:
        required_modules: dict[str, str] = {}

        if "splitter" in stages:
            required_modules["demucs"] = "splitter backend demucs"

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
            message = "Preflight dependency check failed. Missing module(s): " + ", ".join(sorted(missing))
            logger.error(message)
            raise RuntimeError(message)

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
            raise ValueError(f"Stage '{stage}' requires artifact '{key}', but it is not available.")
        return registry[key]

    def _require_audio_role(self, registry: dict[str, Any], role: str, stage: str) -> AudioArtifact:
        item = self._require_registry_item(registry, role, stage)
        if not isinstance(item, AudioArtifact):
            raise TypeError(f"Artifact '{role}' is not an AudioArtifact.")
        return item

    def _require_audio_input(self, registry: dict[str, Any], stage: str) -> AudioArtifact:
        for role in ("filtered_vocal_audio", "vocal_audio", "mixed_audio"):
            item = registry.get(role)
            if isinstance(item, AudioArtifact):
                return item
        raise ValueError(f"Stage '{stage}' requires an audio artifact, but none is available.")

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
            "Loaded lyrics: %d lines from %s (spacing lines preserved, requested_encoding=%s, used_encoding=%s)",
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
                "Unsupported parser lyrics encoding "
                f"{requested_encoding!r}. Supported values: auto, utf-8, utf-8-sig, utf-16, gbk, gb18030, shift-jis."
            )
        return normalized

    def _read_lyrics_text(self, lyrics_path: Path, requested_encoding: str) -> tuple[str, str]:
        if requested_encoding != "auto":
            try:
                return lyrics_path.read_text(encoding=requested_encoding), requested_encoding
            except UnicodeDecodeError as exc:
                logger.error("Lyrics file decode failed with encoding=%s: %s", requested_encoding, lyrics_path)
                raise ValueError(f"Lyrics file could not be decoded as {requested_encoding}: {lyrics_path}") from exc

        for encoding in _AUTO_LYRICS_ENCODINGS:
            try:
                return lyrics_path.read_text(encoding=encoding), encoding
            except UnicodeDecodeError:
                continue
        logger.error("Lyrics file could not be decoded with auto encodings: %s", lyrics_path)
        raise ValueError(
            "Lyrics file could not be decoded with auto mode. Tried utf-8-sig, utf-16, gb18030, shift-jis: "
            f"{lyrics_path}"
        )

    def _materialize_audio_artifact(self, artifact: AudioArtifact, destination: Path, stage_name: str, role: str) -> AudioArtifact:
        if artifact.path is None:
            raise ValueError(f"Cannot materialize audio artifact from stage '{stage_name}' without a source path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact.path, destination)
        metadata = dict(artifact.metadata)
        metadata["materialized_from"] = str(artifact.path)
        logger.info("Materialized %s output to %s", stage_name, destination)
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
            logger.info("Input audio: %s (%s)", request.audio_path, inferred_role)
        if request.lyrics_path is not None:
            logger.info("Input lyrics: %s", request.lyrics_path)
        if request.parser_lyrics_encoding is not None:
            logger.info("Parser lyrics encoding request: %s", request.parser_lyrics_encoding)
        if request.timed_units_path is not None:
            logger.info("Input timed_units: %s", request.timed_units_path)
        if request.parsed_lyrics_path is not None:
            logger.info("Input parsed_lyrics: %s", request.parsed_lyrics_path)
        if request.alignment_result_path is not None:
            logger.info("Input alignment_result: %s", request.alignment_result_path)
