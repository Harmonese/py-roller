# Changelog

All notable changes to this project will be documented in this file.

The format loosely follows Keep a Changelog and this project uses Semantic Versioning.

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

## [0.4.9-p] - 2026-05-11

### Changed
- Published a preflight release for the upcoming `py-roller` v0.5.0 line.
- Held the full v0.5.0 release for another round of tests.

## [0.4.1] - 2026-04-10

### Added
- Added configurable splitter backend selection via `--splitter-backend`.
- Added Demucs-specific splitter configuration flags: `--splitter-demucs-model`, `--splitter-demucs-device`, `--splitter-demucs-jobs`, `--splitter-demucs-overlap`, and `--splitter-demucs-segment`.

### Changed
- Modularized the splitter design so Demucs settings can be routed through backend-specific configuration instead of fixed pipeline defaults.

## [0.4.0] - 2026-04-09

### Added
- Initial public release of `py-roller`.
- Added Windows process cleanup handling.

### Changed
- Confirmed the project name as `py-roller`.
- Unified package and import naming under `pyroller`.
- Improved README and packaging metadata for the first public release.
- Updated ASS writer behavior.
