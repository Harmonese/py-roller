# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows Keep a Changelog and this project uses Semantic Versioning.

## [0.8.3] - 2026-06-16

### Fixed
- Fixed `py-roller install --profile cpu` on macOS and Windows by installing the CPU Torch/TorchAudio stack from the default PyPI index. Linux CPU installs continue to use the official PyTorch CPU wheel index, and CUDA installs continue to use the CUDA 12.4 wheel index.

## [0.8.2] - 2026-06-15

### Changed
- Split CLI run-like execution into `pyroller.cli.runlike` so `pyroller.cli.main` stays focused on parser construction and command dispatch.
- Moved reusable doctor report collection to `pyroller.doctor`; the CLI doctor module now only renders and exits.
- Split batch internals into `batch_models`, `batch_builder`, and `batch_runner` while keeping `pyroller.batch` as the stable public facade for existing imports.
- Added public pipeline stage resolution and request validation helpers in `pyroller.pipeline.stages` and `pyroller.pipeline.validation`.
- Reused shared artifact-path helpers for both batch results and protocol run reports.
- Updated CLI help and README wording to describe JSON/YAML batch manifests consistently.

### Fixed
- Fixed duplicate `continue_on_error` argument wiring in batch execution.
- Invalid pipeline languages now fail explicitly instead of logging and falling back to `mul`; users can still choose `mul` intentionally.

## [0.8.1] - 2026-06-04

### Added
- Added `pyroller.engine` as the internal protocol execution boundary for run, batch, cache-model, and doctor flows.
- Added unified protocol v1 final report envelopes for machine outputs, including `schema_version`, `engine`, `engine_version`, `protocol_version`, `type`, `status`, `artifact_paths`, and structured `error` data when present.
- Added batch task protocol results with stable `task_id`, `status`, `artifact_paths`, `log_file`, and `error` fields.
- Added artifact schema validation errors for loading timed units, parsed lyrics, and alignment artifacts.

### Changed
- Moved machine execution out of the CLI layer so `pyroller.cli.main` mainly parses arguments, reads requests, and prints reports.
- Improved `cache-model`, `doctor`, and `install` JSON reports to share the same protocol envelope shape as run and batch.
- Improved batch progress events with batch-level and task-level protocol event data.

## [0.8.0] - 2026-06-04

### Added
- Added protocol v1 for GUI/frontends: `py-roller capabilities --output-format json`, `run --request request.json --progress-format jsonl --output-format json`, and `batch --request request.json --progress-format jsonl --output-format json`.
- Added protocol v1 final JSON reports for run, batch, cache-model, doctor, and install flows.
- Added protocol v1 JSONL event fields (`schema_version`, `stage`, `message`, `progress`, `timestamp`) to pipeline and install progress streams.
- Added JSON batch manifest loading alongside existing YAML manifests.

### Changed
- Bumped package version to `0.8.0` for the new machine protocol boundary.
- Made `--stages` optional when `run` or `batch` receives a protocol `--request` file.
- `doctor --output-format json` and `install --output-format json` now include protocol metadata (`schema_version`, `engine`, `engine_version`, `protocol_version`, `type`, `status`).

## [0.6.4] - 2026-05-21

### Changed
- Made `py-roller install` the sole supported audio runtime installation path and removed the legacy `audio-core` extra from package metadata.
- Switched SOCKS proxy support from `httpx[socks]`/`socksio` to `requests[socks]`/`PySocks` to match Hugging Face's download stack.
- Removed `torchvision` from the official install profiles because py-roller does not use vision workloads.

### Fixed
- Fixed Hugging Face large-file download failure hints so interrupted partial streams recommend retrying, lowering `--transcriber-hf-max-workers`, increasing HF timeouts, and checking the configured proxy.
- Removed the unused `g2p_en` probing path so English ARPABET conversion consistently uses the declared `pronouncing` dependency and grapheme fallback.

## [0.6.3] - 2026-05-20

### Added
- Added `scripts/check_i18n.py` to validate locale coverage against code message keys, including missing, extra, and untranslated entries.

### Changed
- Made argparse i18n installation explicit during CLI parser construction instead of hiding the private argparse gettext patch inside `_()`.
- Normalized locale resources to the current message key set and completed remaining non-English fallback translations across supported locales.
- Improved locale detection for unsupported `PYROLLER_LANG` values and colon-separated `LANGUAGE` candidates.
- Documented that the faster-whisper `turbo` alias resolves to a CTranslate2-compatible turbo snapshot rather than a Systran snapshot.

### Fixed
- Fixed the faster-whisper `turbo` and `large-v3-turbo` aliases resolving to the nonexistent `Systran/faster-whisper-turbo` repository.
- Fixed manifest batch building only retaining the final manifest task instead of all tasks.
- Fixed manifest duplicate task/output conflict validation only applying to the final task.
- Fixed `--transcriber-vad-filter` and `--writer-spacing` being accepted when their corresponding stages were not selected.
- Fixed duplicated run-section config validation assignment.
- Added an explicit log message when requested stages are normalized to canonical execution order.

## [0.6.2] - 2026-05-17

### Fixed
- Fixed an error in i18n of `transcriber`.
- Fixed an error causing failure pipeline run.


## [0.6.1] - 2026-05-17

### Fixed
- Fixed `requires-python` upper bound: changed from `>=3.10` to `>=3.10,<3.13` to prevent installation on Python 3.13+ which is not yet supported.

## [0.6.0] - 2026-05-17

### Added
- Added multi-language i18n support across 8 locales: Simplified Chinese (zh), Traditional Chinese Taiwan (zh_Hant), Traditional Chinese Hong Kong (zh_Hant_HK), Japanese (ja), Korean (ko), Polish (pl), Portuguese (pt), and Slovak (sk).
- Display language auto-detects from `LANG`/`LC_ALL`/`LANGUAGE` environment variables; override with `PYROLLER_LANG` (supports both POSIX `zh_TW` and BCP 47 `zh-Hant` formats).
- `zh_Hant_HK` automatically falls back to `zh_Hant` if the HK-specific locale file is missing.
- All CLI help text, argparse built-in strings, pipeline output summaries, error messages, logger messages, doctor reports, install progress, and download hints are now translated.
- `pyroller/i18n.py` normalizes locale detection with script-aware zh handling (`zh-Hant` → Traditional, `zh-HK` → Traditional HK, `zh-TW` → Traditional, `zh-CN` → Simplified).

### Changed
- **Format specifier alignment**: unified `{}`/`{!r}` (str.format) and `%s`/`%d`/`%r` (printf) style across all locale files so both format families match code usage. Each locale file now contains 664 translation entries with both format variants.
- `--language` help now documents the distinction between pipeline language (transcription/parsing) and display language (i18n via `PYROLLER_LANG`).
- Updated `--help` common commands section with `PYROLLER_LANG=zh` example.
- `pyroller/resources/locales/` now contains 8 JSON translation files.

### Fixed
- Fixed ~80 silent translation fallbacks where `{}`-style `_()` calls could not find `%s`-style zh.json keys, causing untranslated English output even when locale was set.
- Fixed 12 hardcoded English strings in `filter/registry.py`, `writer/registry.py`, `transcriber/composed.py`, `transcriber/hf_download_config.py`, and `utils/text.py` that bypassed `_()` entirely.
- Fixed `model_resolver.py` splitting a single translatable error message across three `_()` calls, none of which matched any locale key.
- Fixed `cli/doctor.py` using named format args (`{py_ver}`) that mismatched zh.json positional `{}` keys.
- Fixed `JsonlStageProgress` using `{}` format while zh.json only had `%s` format for stage lifecycle messages.
- Fixed leading-newline mismatches in doctor human output.
- Fixed `pyroller.progress` logger output consistency: `LoggingStageProgress` and `JsonlStageProgress` now share the same format keys.

## [0.5.10] - 2026-05-16

### Added
- Added Chinese (zh) i18n support: all CLI help text, pipeline output summaries, error messages, logger messages, doctor reports, install progress, and argparse built-in strings are now translated when the system locale is Chinese or `PYROLLER_LANG=zh` is set.
- Added `pyroller/i18n.py` with locale auto-detection from `LANG`/`LC_ALL`/`LANGUAGE` and `PYROLLER_LANG` override.
- Added `pyroller/resources/locales/zh.json` with 487 Chinese translation entries covering every user-facing string.

### Changed
- `pyroller/__init__.py` now exports `_()` for translation.
- `pyproject.toml` includes `resources/locales/*.json` in package data.

## [0.5.9] - 2026-05-16

### Fixed
- Fixed `use_batched` incorrectly coupling VAD setting with batching availability, and metadata `batched_inference` now reflects the actual inference path taken.
- Fixed LRC timestamp formatting where `59.999999` seconds could round to invalid `[00:60.00]` by switching to integer-tick arithmetic.
- Fixed `run:` YAML config section silently rejecting all keys; keys under `run:` are now merged into `shared:`.
- Fixed `lrc_compressed` writer silently ignoring `writer_spacing=keep`; the combination now raises a clear error.
- Fixed ASS karaoke writer missing `by_tag` in output header and not escaping ASS control characters (`{`, `}`, `\`) in lyric text.
- Fixed `--transcriber-local-files-only` using `store_true` instead of `BooleanOptionalAction`, preventing CLI negation when a config file enabled it.
- Fixed progress phase counting for `--aligner-repetition full` where the total was over-counted by `len(lyric_units)`.
- Fixed Demucs output path fragility: added glob-based fallback when the expected output path does not match.
- Fixed parser and aligner registries silently discarding all backend configuration when constructors accept `**kwargs`.
- Fixed `ComposablePipelineRunner` lifecycle in batch validation: `runner.close()` is now always called via `try/finally`.
- Fixed gap-filling in alignment line end times: the gap to the next line is now proportionally redistributed across all syllables instead of stretching only the last one.
- Fixed `--transcriber-vad-filter` bypassing backend compatibility validation.

### Changed
- `__version__` in `pyroller/__init__.py` now reads from package metadata (`importlib.metadata.version`), leaving `pyproject.toml` as the single source of truth.

## [0.5.8] - 2026-05-15

### Fixed
- Fixed `BatchedInferencePipeline` crash when `vad_filter` is disabled; the engine now falls back to non-batched transcription.

## [0.5.7] - 2026-05-15

### Added
- Added `--transcriber-vad-filter` / `--no-transcriber-vad-filter` (default: enabled) to skip silence during faster-whisper transcription, reducing transcription time by 20–40% for typical songs.
- Added `py-roller cache-model` command to pre-download transcriber models into the local model store before running pipelines.
- Added automatic GPU detection: when no `--transcriber-device` is specified and CUDA is available, the transcriber automatically uses `cuda` with `float16` compute type.

### Changed
- faster-whisper `vad_filter` now defaults to `True` (was hardcoded to `False`).

## [0.5.6] - 2026-05-13

### Added
- Added `py-roller doctor --output-format json` for machine-readable runtime health reports.
- Added `py-roller install --progress-format {human,jsonl,both}` so GUI frontends can follow install lifecycle, selected profiles, subprocess steps, validation, doctor, completion, failure, and heartbeat events via `PYROLLER_EVENT` JSONL lines.
- Added `py-roller install --output-format {human,json}` for final machine-readable install reports containing requested/selected profiles, step results, validation results, and doctor summaries.
- Added install subprocess heartbeat events so frontends can distinguish active long-running pip operations from stalled jobs.

### Changed
- Reworked `doctor` internals around reusable report collection, human rendering, and JSON serialization without changing the default terminal checklist.
- Reworked `install` subprocess execution from `subprocess.run` to streamed `Popen` execution so output and machine-readable events can be emitted while commands are running.
- Updated CLI help and README coverage for doctor JSON reports and install JSONL/JSON integration modes.

## [0.5.5] - 2026-05-12

### Fixed
- Fixed a major runtime issue that causes failure during the `transcriber` stage.

## [0.5.4] - 2026-05-12

### Added
- Formalized `PYROLLER_EVENT` JSONL progress events with canonical `progress` values from `0.0` to `1.0`, timestamp fields, and compatibility `percent` aliases for early GUI consumers.
- Added normalized progress stages for GUI frontends: `preflight`, `model_download`, `splitter`, `filter`, `transcriber`, `parser`, `aligner`, and `writer`.
- Added explicit run lifecycle events (`run_started`, `run_completed`, `run_failed`) and `artifact_written` events for saved pipeline artifacts.
- Added heartbeat events for long-running model download and faster-whisper transcription periods so frontends can distinguish active work from stalled jobs.
- Added Demucs splitter progress events derived from native seconds-based progress output.

### Changed
- Improved Hugging Face model download progress events with file count, largest file name, byte totals, cache directory, proxy/XET settings, and transfer speed.
- Improved faster-whisper transcription progress events with segment count, last processed audio time, duration hints, and text previews when available.
- Added parser and writer stage events so short final stages still appear in GUI progress timelines.
- Updated README and CLI help coverage for `--progress-format`, restricted-network downloads, local-only model reuse, and progress troubleshooting.

### Fixed
- Reduced ambiguity between model download progress and Demucs progress by giving model downloads the `model_download` stage and Demucs separation the `splitter` stage.
- Fixed a regression where faster-whisper transcription could fail with `NameError: progress_heartbeat is not defined` when structured heartbeat progress was enabled.

## [0.5.3] - 2026-05-12

### Added
- Added `--progress-format {human,jsonl,both}` for structured progress output without breaking the default terminal-oriented CLI experience.
- Added `PYROLLER_EVENT` JSONL progress events for GUI frontends, covering stage lifecycle, Hugging Face model download progress, artifact writes, and failures.
- Added byte-level model-cache progress reporting for large Hugging Face downloads so frontends can distinguish active downloads from stalled jobs.
- Added live faster-whisper transcription progress events based on yielded segment end times, so GUI frontends do not appear stuck at the final transcriber phase.

### Fixed
- Fixed timed_units artifact serialization when transcriber metadata contains pathlib Path values, such as --transcriber-model-path from the CLI.
- Normalized transcriber engine metadata model_path values to strings for faster-whisper and wav2vec2/CTC backends.


## [0.5.2] - 2026-05-12

### Fixed
- Fixed Hugging Face timeout environment overrides so `--transcriber-hf-etag-timeout` and `--transcriber-hf-download-timeout` are exported as integer seconds instead of float strings such as `120.0`, which could make `huggingface_hub` fail during transcriber preflight.
- Normalized Hugging Face timeout CLI/config values to positive integer seconds while still accepting existing numeric inputs like `120.0` from integrations.

## [0.5.1] - 2026-05-12

### Added
- Renewed `install --help` and `doctor --help`.

### Changed
- Rewritten README and `--help` information.

### Fixed
- `constraints/*.txt` outdated on still using `Torch 2.5.1` instead of `Torch 2.6.0`.
- `v0.5.0` can't be downloaded on PyPI due to GitHub Action failure, this version resolved this problem.

## [0.5.0] - 2026-05-12

### Added
- Added `py-roller install` for installing a validated local audio/transcriber runtime from built-in profiles.
- Added official install profiles for CPU and CUDA 12.4, backed by bundled Torch/Torchaudio/TorchVision constraints and `audio-core` requirements.
- Added `py-roller doctor` for checking Python, Torch, Torchaudio, faster-whisper, CTranslate2, Transformers, Demucs, librosa, and SOCKS proxy support.
- Added a local transcriber model store rooted at `~/.cache/py-roller/models/transcriber` by default.
- Added `--transcriber-model-path` and `--transcriber-local-files-only` for explicit model-store and offline workflows.
- Added Hugging Face download reliability controls for transcriber models: `--transcriber-hf-xet`, `--transcriber-hf-proxy`, `--transcriber-hf-etag-timeout`, `--transcriber-hf-download-timeout`, and `--transcriber-hf-max-workers`.
- Added structured model download logging, including repository id, cache path, XET mode, proxy, timeout, and worker settings.
- Added early SOCKS proxy validation with actionable repair guidance when `socksio` is unavailable.
- Added repetition-aware aligner modes via `--aligner-repetition {none,few,full}` for repeated or partially omitted lyric sections.
- Added faster-whisper based transcriber coverage for `zh`, `en`, and `mul` language modes with text-derived pinyin, ARPABET, and IPA unitizers.

### Changed
- Reworked the transcriber subsystem into a composable engine plus unitizer design.
- Changed the default transcriber backend for `zh`, `en`, and `mul` to `faster_whisper` while keeping CTC-based phoneme backends available as optional fallbacks.
- Replaced direct backend-specific model loading with a shared model resolver that can materialize Hugging Face snapshots into the py-roller model store.
- Improved transcriber preflight so backend imports, model resolution, and model materialization report progress and fail with clearer diagnostics.
- Applied Hugging Face network environment overrides before importing faster-whisper, Transformers, or huggingface_hub so XET/proxy/timeout settings take effect reliably.
- Updated the dependency policy to prefer the validated Torch 2.6.0 family and to install SOCKS proxy support through `httpx[socks]` by default.
- Updated ASS writer spacing documentation and defaults to follow the shared `writer_spacing` behavior.
- Expanded README coverage for installation, model-store behavior, offline runs, restricted-network downloads, dependency policy, and repetition alignment.

### Fixed
- Fixed a poor preflight experience where Hugging Face model downloads could hang or fail without enough user-controllable download settings.
- Improved error hints for XET/CAS failures, proxy failures, SOCKS dependency failures, and download timeouts.
- Fixed configuration propagation so Hugging Face download options work from both CLI arguments and YAML config defaults.

## [0.4.9-p]

### Changed
- Published a preflight release for the upcoming `py-roller` v0.5.0 line.
- Held the full v0.5.0 release for another round of tests.

## [0.4.1]

### Added
- Added configurable splitter backend selection via `--splitter-backend`.
- Added Demucs-specific splitter configuration flags: `--splitter-demucs-model`, `--splitter-demucs-device`, `--splitter-demucs-jobs`, `--splitter-demucs-overlap`, and `--splitter-demucs-segment`.

### Changed
- Modularized the splitter design so Demucs settings can be routed through backend-specific configuration instead of fixed pipeline defaults.

## [0.4.0]

### Added
- Initial public release of `py-roller`.
- Added Windows process cleanup handling.

### Changed
- Confirmed the project name as `py-roller`.
- Unified package and import naming under `pyroller`.
- Improved README and packaging metadata for the first public release.
- Updated ASS writer behavior.
