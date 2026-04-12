# py-roller

`py-roller` is a CLI Solution for automatic rolling lyrics generating.

To be specific, `py-roller` is a composable lyric-audio alignment pipeline CLI for staged execution, batch processing, and LRC/ASS export. Designed to support multiple transcriber back-ends including WhisperX and wav2vec2.

- Package name: `py-roller`
- CLI command: `py-roller`
- Python import package: `pyroller`

## Quick overview

`py-roller` treats alignment as a contiguous stage chain:

```text
s -> f -> t -> p -> a -> w
splitter -> filter -> transcriber -> parser -> aligner -> writer
```

Core modes:

- `run`: execute one contiguous stage chain for one task
- `batch`: execute the same contiguous stage chain across many tasks

Core artifact types:

- input audio / lyrics
- intermediate vocal and filtered audio
- `timed_units`
- `parsed_lyrics`
- `alignment_result`
- roller outputs such as LRC or ASS

## Installation

From source, first install the lightweight base package:

```bash
pip install -e .
```

Then let `py-roller` install the validated audio/transcriber runtime for your machine:

```bash
py-roller install
```

Profiles:

- `auto` (default): choose the best validated profile for this machine, then automatically fall back to CPU if validation fails
- `cpu`: official CPU-only stable profile
- `cu124`: official CUDA 12.4 profile

Useful variants:

```bash
py-roller install --profile cpu
py-roller install --profile cu124
py-roller install --dry-run
py-roller doctor
```

Notes:

- `py-roller install` still uses `pip` underneath, but it does **not** rely on pip to guess the correct Torch/Torchaudio flavor.
- The command first installs the selected Torch profile, then installs the bundled `audio-core` runtime requirements with the matching constraints file, validates the resulting environment, and finally runs `py-roller doctor` unless you pass `--skip-doctor`.

After installation, the CLI command is:

```bash
py-roller
```

## Quick start

### Full pipeline: audio + lyrics -> LRC

```bash
py-roller run \
  --stages s,f,t,p,a,w \
  --audio ./song.mp3 \
  --lyrics ./song.txt \
  --filter-chain noise_gate,dereverb \
  --output-roller ./song.lrc
  --language zh # Choose as you like
```

### Start from a prepared vocal track

```bash
py-roller run \
  --stages t,p,a,w \
  --audio ./vocals.wav \
  --lyrics ./song.txt \
  --writer-backend ass_karaoke \
  --output-roller ./song.ass
  --language zh # Choose as you like
```

### Batch processing by stem

```bash
py-roller batch \
  --stages t,p,a,w \
  --audio ./audio_dir \
  --lyrics ./lyrics_dir \
  --output-roller ./out_dir
  --language zh # Choose as you like
```

## Core execution model

### Contiguous stage chains only

The CLI only accepts contiguous subchains of the canonical order.

Valid examples:

- `s,f,t,p,a,w`
- `t,p,a,w`
- `a,w`
- `w`

Invalid examples:

- `s,t,w`
- `s,p,a`

### Legal chain starts

Explicit artifact inputs are only valid at the correct chain start:

- `--audio` is valid when the chain starts at `s`, `f`, or `t`
- `--lyrics` is valid when the chain includes `p`
- `--timed-units` and `--parsed-lyrics` are only valid when the chain starts at `a`
- `--alignment-result` is only valid when the chain starts at `w`

### Final outputs vs intermediate artifacts

Final user-requested outputs are only the explicit `--output-*` paths:

- `--output-vocal-audio`
- `--output-filtered-audio`
- `--output-timed-units`
- `--output-parsed-lyrics`
- `--output-alignment-result`
- `--output-roller`

Everything else created under `--intermediate` is treated as intermediate state.

## Common workflows

### Start from raw audio

Use the full chain when you want splitting, filtering, transcription, alignment, and final writing in one command.

```bash
py-roller run \
  --stages s,f,t,p,a,w \
  --audio ./song.mp3 \
  --lyrics ./song.txt \
  --output-roller ./song.lrc
```

### Start from filtered or vocal audio

Skip splitter/filter when you already have a suitable track for transcription.

```bash
py-roller run \
  --stages t,p,a,w \
  --audio ./vocals.wav \
  --lyrics ./song.txt \
  --output-roller ./song.lrc
```

### Start from aligner artifacts

```bash
py-roller run \
  --stages a,w \
  --timed-units ./song.timed_units.json \
  --parsed-lyrics ./song.parsed_lyrics.json \
  --output-roller ./song.lrc
```

### Rewrite only from an existing alignment result

```bash
py-roller run \
  --stages w \
  --alignment-result ./song.alignment.json \
  --writer-backend ass_karaoke \
  --output-roller ./song.ass
```

## Backend defaults

Default backend selection is language-aware. Please note that the default selection of language is "mul" which works poorly when tested on both Chinese and English, please use the `--language` flag to specify the desired language if the language is directly supported.

### Transcriber defaults

- `zh` -> `mms_phonetic`
- `en` -> `whisperx` (validated through the pinned `.[audio]` bundle)
- `mul` -> `wav2vec2_phoneme`

### Parser defaults

- `zh` -> `zh_router_pinyin`
- `en` -> `en_arpabet`
- `mul` -> `mul_ipa`

### Other defaults

- aligner backend -> `global_dp_v1`
- writer backend -> `lrc_ms`
- language -> `mul`
- `writer_spacing` -> keep
- `cleanup` -> `on-success`
- transcriber model store -> `~/.cache/py-roller/models/transcriber`

## Transcriber model store and offline behavior

Transcriber execution is local-only. `py-roller` does not send audio to a cloud transcription API.

Model resolution follows this order:

1. resolve `transcriber_model_name` (or the backend default model name)
2. look for the model in the py-roller transcriber model store
3. if not found and local-only mode is disabled, download or materialize it into the model store
4. load the resolved local model path for inference

Useful options:

- `--transcriber-model-path`: choose the py-roller transcriber model store root
- `--transcriber-model-name`: choose a model alias, model repo id, or an explicit local model path
- `--transcriber-local-files-only`: refuse network access and read only from local files/cache

`.[audio]` installs the project's official audio feature set. Its declared dependency chain is intentionally simple and follows the package-level dependencies from the reference v0.4.0 setup rather than a WhisperX-specific lock bundle.

If WhisperX fails in Pyannote VAD checkpoint loading, treat that as an environment compatibility problem inside the installed audio stack rather than a model-store bug.

Examples:

```bash
py-roller run   --stages t,p,a,w   --audio ./vocals.wav   --lyrics ./song.txt   --transcriber-model-path ./models/transcriber   --output-roller ./song.lrc
```

```bash
py-roller run   --stages t,p,a,w   --audio ./vocals.wav   --lyrics ./song.txt   --transcriber-model-path ./models/transcriber   --transcriber-local-files-only   --output-roller ./song.lrc
```

If you are on a restricted network, pre-populate the model store and then rerun with `--transcriber-local-files-only`.

## Writer behavior

### LRC

The default writer is `lrc_ms` which writes LRC lines with millisecond precision. Other supported writer backends are:

- `lrc_cs`: writes LRC lines with centiscond precision
- `lrc_compressed`: writes LRC lines with millisecond precision, but compresses consecutive lines with the same timestamp
- `ass_karaoke`: see below

### ASS karaoke

`ass_karaoke` writes ASS dialogue lines with karaoke timing tags.

Current defaults:

- structural / spacing line output follows `writer_spacing` (`keep` by default)
- display end time prefers matched unit timing instead of blindly extending to the next line
- unmatched lines receive a short visible duration fallback

Example:

```bash
py-roller run \
  --stages w \
  --alignment-result ./song.alignment.json \
  --writer-backend ass_karaoke \
  --output-roller ./song.ass
```

## Progress reporting

The project exposes a reusable progress-reporting interface so CLI and future GUI frontends can share the same stage updates.

Current behavior:

- splitter: Demucs progress plus wrapper stage progress
- filter: phase progress
- transcriber: phase progress
- aligner: phase progress plus DP row progress

In single-task `run`, progress is shown as CLI progress bars when the terminal supports it. In `batch`, per-task progress is logged to avoid multiple workers fighting for one terminal.

## Intermediate files and cleanup

Intermediate files live under:

```text
--intermediate/<task-id>/splitter
--intermediate/<task-id>/filter
--intermediate/<task-id>/logs
```

Default intermediate root:

```text
<system temp>/py-roller-artifacts
```

Cleanup policy:

- `--cleanup on-success` keeps successful runs tidy by removing per-task intermediate directories
- `--cleanup never` keeps intermediate audio and logs for inspection

## Batch mode

`batch` uses the same stage semantics as `run`, but applies them to many tasks.

### Directory pairing

Directory mode currently supports:

```text
--pair-by stem
```

Default candidate globs:

- `--audio-glob "*.mp3"`
- `--lyrics-glob "*.txt"`

Matching is non-recursive.

Example:

```bash
py-roller batch \
  --stages t,p,a,w \
  --audio ./audio_dir \
  --lyrics ./lyrics_dir \
  --output-roller ./out_dir
```

### Batch controls

- `--jobs N`: maximum number of parallel workers
- `--continue-on-error`: keep processing remaining tasks after failures
- `--skip-existing`: skip tasks whose declared final outputs already exist
- `--manifest jobs.yaml`: load explicit per-task paths from YAML instead of pairing by stem

### Parallelism guidance

`--jobs` controls how many tasks run at the same time. This is separate from any model-level batch size.

Recommended starting point:

- CPU-only: `--jobs 1` or `--jobs 2`
- single GPU: usually `--jobs 1`

## YAML manifest format

Manifest mode is useful when filenames do not match cleanly by stem.

The manifest defines per-task input and output paths only. It does not override stage selection, language, backend choice, filter settings, jobs, or other batch-level options.

Supported top-level forms:

```yaml
tasks:
  - id: song01
    audio: ./audio/song01_master.mp3
    lyrics: ./lyrics/song01_final.txt
    output_roller: ./out/song01.lrc
```

or:

```yaml
- id: song01
  audio: ./audio/song01_master.mp3
  lyrics: ./lyrics/song01_final.txt
  output_roller: ./out/song01.lrc
```

Allowed manifest input keys:

- `audio`
- `lyrics`
- `timed_units`
- `parsed_lyrics`
- `alignment_result`

Allowed manifest output keys:

- `output_vocal_audio`
- `output_filtered_audio`
- `output_timed_units`
- `output_parsed_lyrics`
- `output_alignment_result`
- `output_roller`

Optional helper key:

- `id`

Validation rules:

- each task must be a mapping
- unknown keys are rejected
- inputs must match the selected chain start
- outputs must be valid final outputs for the selected chain
- task ids / stems must be unique
- final output paths must not conflict across tasks
- relative paths are resolved relative to the manifest file location

## YAML config for default CLI options

Use `--config` to load YAML defaults.

Priority order:

```text
built-in defaults < config YAML < explicit CLI arguments
```

Section model:

- `shared`: defaults applied to both `run` and `batch`
- `run`: currently no extra keys beyond `shared`
- `batch`: defaults for batch-only options such as `jobs` and `skip_existing`

Example:

```yaml
shared:
  language: mul
  writer_spacing: keep
  writer_backend: lrc_ms
  intermediate: ./tmp/py-roller-artifacts
  cleanup: on-success
  transcriber_device: cpu
  transcriber_model_path: ~/.cache/py-roller/models/transcriber
  transcriber_local_files_only: false
  splitter_backend: demucs
  splitter_demucs_model: htdemucs
  splitter_demucs_device: cpu
  splitter_demucs_jobs: 0
  splitter_demucs_overlap: 0.25
  splitter_demucs_segment: 8
  filter_chain:
    - noise_gate
    - dereverb

batch:
  jobs: 2
  audio_glob: "*.mp3"
  lyrics_glob: "*.txt"
  timed_units_glob: "*.json"
  parsed_lyrics_glob: "*.json"
  alignment_result_glob: "*.json"
  continue_on_error: true
```

`filter_chain` can be written either as a comma-separated string or as a YAML list.

## Troubleshooting

### Interruption and child process cleanup

Batch fail-fast actively stops launching further work after the first task failure when `--continue-on-error` is not set, and worker cleanup includes a Windows-specific process-tree branch.

For older runs or already orphaned processes, Linux/macOS cleanup examples are still useful:

```bash
pkill -TERM -f 'python .*pyroller'
pkill -TERM -f 'demucs.separate|demucs'
```

If anything still survives:

```bash
pkill -KILL -f 'python .*pyroller'
pkill -KILL -f 'demucs.separate|demucs'
```

Inspect candidates first with:

```bash
ps -ef | grep -E 'pyroller|demucs'
```

## Dependency policy

`py-roller install` now prefers the newest validated dependency line instead of the older pre-2.6 Torch family. In practice this means:

- Torch/TorchAudio/TorchVision are installed from the official 2.6.0 family for every built-in profile.
- WhisperX environments pin `pyannote.audio==3.4.0`, because WhisperX users have reported that `pyannote.audio 4.x` breaks compatibility.
- SOCKS-proxy support is installed by default through `httpx[socks]`, so Hugging Face downloads do not fail just because `socksio` is missing.

If you upgrade or override these packages manually, run `py-roller doctor` before using WhisperX-heavy pipelines.
