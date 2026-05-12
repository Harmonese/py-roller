# py-roller

`py-roller` is a local command-line pipeline for generating rolling lyric files from audio and plain-text lyrics.

It can split vocals, filter audio, transcribe local audio with faster-whisper or wav2vec2-style backends, parse lyrics, align lyric lines, and export LRC or ASS karaoke output.

- Package name: `py-roller`
- CLI command: `py-roller`
- Python import package: `pyroller`

## Install

Use a fresh virtual environment when possible. First install the lightweight base package from source:

```bash
pip install -e .
```

Then install the validated audio/transcriber runtime:

```bash
py-roller install
py-roller doctor
```

`py-roller install` installs a pinned Torch/Torchaudio/TorchVision profile first, then installs the bundled `audio-core` requirements with matching constraints, validates the environment, and runs `py-roller doctor` unless `--skip-doctor` is passed.

Install profiles:

- `auto` default: try the best validated profile for this machine, then fall back to CPU if validation fails.
- `cpu`: force the CPU-only profile.
- `cu124`: force the CUDA 12.4 profile.

Useful install commands:

```bash
py-roller install --profile cpu
py-roller install --profile cu124
py-roller install --dry-run
py-roller install --no-reset-torch
```

## Quick start

Set `--language` explicitly whenever the song language is known. Use `zh` for Chinese, `en` for English, and `mul` only when you need the multilingual fallback.

### Raw audio + lyrics -> LRC

```bash
py-roller run \
  --stages s,f,t,p,a,w \
  --audio ./song.mp3 \
  --lyrics ./song.txt \
  --language zh \
  --filter-chain noise_gate,dereverb \
  --output-roller ./song.lrc
```

### Prepared vocal track + lyrics -> ASS karaoke

```bash
py-roller run \
  --stages t,p,a,w \
  --audio ./vocals.wav \
  --lyrics ./song.txt \
  --language zh \
  --writer-backend ass_karaoke \
  --output-roller ./song.ass
```

### Batch process directories by filename stem

```bash
py-roller batch \
  --stages t,p,a,w \
  --audio ./audio_dir \
  --lyrics ./lyrics_dir \
  --language zh \
  --output-roller ./out_dir
```

## Pipeline model

`py-roller` runs a contiguous chain of stages in this fixed order:

```text
s -> f -> t -> p -> a -> w
splitter -> filter -> transcriber -> parser -> aligner -> writer
```

Valid examples:

- `s,f,t,p,a,w`: full pipeline from raw audio and lyrics.
- `t,p,a,w`: start from prepared vocal/filtered audio.
- `a,w`: start from existing `timed_units` and `parsed_lyrics` artifacts.
- `w`: rewrite from an existing `alignment_result` artifact.

Invalid examples:

- `s,t,w`: skips required intermediate stages.
- `s,p,a`: skips required intermediate stages.

Legal chain-start inputs:

- `--audio`: valid when the chain starts at `s`, `f`, or `t`.
- `--lyrics`: required when the chain includes `p`.
- `--timed-units` and `--parsed-lyrics`: valid when the chain starts at `a`.
- `--alignment-result`: valid when the chain starts at `w`.

Final outputs are only the explicit `--output-*` paths:

- `--output-vocal-audio`
- `--output-filtered-audio`
- `--output-timed-units`
- `--output-parsed-lyrics`
- `--output-alignment-result`
- `--output-roller`

Intermediate files under `--intermediate` are temporary working state unless `--cleanup never` is used.

## Common workflows

### Start from raw audio

```bash
py-roller run \
  --stages s,f,t,p,a,w \
  --audio ./song.mp3 \
  --lyrics ./song.txt \
  --language zh \
  --output-roller ./song.lrc
```

### Start from already-separated vocals

```bash
py-roller run \
  --stages t,p,a,w \
  --audio ./vocals.wav \
  --lyrics ./song.txt \
  --language zh \
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

For repeated or partially omitted lyrics, choose a repetition mode explicitly:

```bash
py-roller run \
  --stages a,w \
  --timed-units ./song.timed_units.json \
  --parsed-lyrics ./song.parsed_lyrics.json \
  --aligner-repetition few \
  --output-roller ./song.lrc
```

`--aligner-repetition` accepts:

- `none`: default standard `global_dp_v1` behavior; best when repeated lyric lines are written out in full.
- `few`: uses global DP as a proposal, then repairs sparse repeated or omitted regions between trusted anchors.
- `full`: uses per-line candidate generation plus beam search for highly repetitive or anchorless songs.

### Rewrite only from an existing alignment result

```bash
py-roller run \
  --stages w \
  --alignment-result ./song.alignment.json \
  --writer-backend ass_karaoke \
  --output-roller ./song.ass
```

## Backends and defaults

Backend selection is language-aware. The default language is `mul` for compatibility, but `zh` or `en` gives clearer transcriber/parser defaults when the song language is known.

### Transcriber defaults

- `zh` -> `faster_whisper`
- `en` -> `faster_whisper`
- `mul` -> `faster_whisper`

Additional transcriber backends:

- `zh` also supports `--transcriber-backend mms_phonetic` for the Chinese phonetic CTC path.
- `mul` also supports `--transcriber-backend wav2vec2_phoneme` for the multilingual phoneme CTC fallback.

### Parser defaults

- `zh` -> `zh_router_pinyin`
- `en` -> `en_arpabet`
- `mul` -> `mul_ipa`

### Other defaults

- aligner backend -> `global_dp_v1`
- aligner repetition mode -> `none`
- writer backend -> `lrc_ms`
- writer spacing -> `keep`
- cleanup policy -> `on-success`
- transcriber model store -> `~/.cache/py-roller/models/transcriber`

## Transcriber models and Hugging Face downloads

Transcriber execution is local. `py-roller` does not send audio to a cloud transcription API.

Model resolution order:

1. Resolve `--transcriber-model-name`, or use the backend default model name.
2. Look for the model in the py-roller transcriber model store.
3. If not found and offline mode is not enabled, materialize/download the model into the model store.
4. Load the resolved local model path for inference.

Useful model options:

- `--transcriber-model-path`: local model store root.
- `--transcriber-model-name`: model alias, Hugging Face repo id, or explicit local path.
- `--transcriber-local-files-only`: refuse network access and use only local files/cache.

For `faster_whisper`, aliases such as `large-v2`, `large-v3`, and `turbo` resolve to the corresponding `Systran/faster-whisper-*` snapshots.

Example with a custom model store:

```bash
py-roller run \
  --stages t,p,a,w \
  --audio ./vocals.wav \
  --lyrics ./song.txt \
  --language zh \
  --transcriber-model-path ./models/transcriber \
  --output-roller ./song.lrc
```

Offline run after the model already exists locally:

```bash
py-roller run \
  --stages t,p,a,w \
  --audio ./vocals.wav \
  --lyrics ./song.txt \
  --language zh \
  --transcriber-model-path ./models/transcriber \
  --transcriber-local-files-only \
  --output-roller ./song.lrc
```

### Restricted or unstable networks

Hugging Face model downloads can be affected by proxies, timeouts, and XET/CAS behavior. `py-roller` exposes the common controls directly:

- `--transcriber-hf-xet {auto,on,off}`: use `off` when XET/CAS hangs or fails on your network.
- `--transcriber-hf-proxy URL`: use one HTTP or SOCKS proxy for model downloads.
- `--transcriber-hf-etag-timeout SECONDS`: metadata/etag timeout.
- `--transcriber-hf-download-timeout SECONDS`: large file download timeout.
- `--transcriber-hf-max-workers INT`: snapshot download parallelism; lower values such as `1` or `2` are often better for fragile proxies.

Avoid XET/CAS when it is unreliable:

```bash
py-roller run \
  --stages t,p,a,w \
  --audio ./vocals.wav \
  --lyrics ./song.txt \
  --language zh \
  --transcriber-hf-xet off \
  --output-roller ./song.lrc
```

Use a local SOCKS proxy and conservative download settings:

```bash
py-roller run \
  --stages t,p,a,w \
  --audio ./vocals.wav \
  --lyrics ./song.txt \
  --language zh \
  --transcriber-hf-proxy socks5://127.0.0.1:7890 \
  --transcriber-hf-download-timeout 120 \
  --transcriber-hf-etag-timeout 30 \
  --transcriber-hf-max-workers 2 \
  --output-roller ./song.lrc
```

`audio-core` installs SOCKS support through `httpx[socks]`. If the environment was installed manually and SOCKS support is missing, run:

```bash
py-roller install
```

or install the missing dependency directly:

```bash
pip install "httpx[socks]"
```

## Writer behavior

### LRC

The default writer is `lrc_ms`, which writes LRC lines with millisecond precision.

Supported writer backends:

- `lrc_ms`: millisecond precision.
- `lrc_cs`: centisecond precision.
- `lrc_compressed`: millisecond precision, with consecutive identical timestamps compressed.
- `ass_karaoke`: ASS dialogue output with karaoke timing tags.

### ASS karaoke

`ass_karaoke` writes ASS dialogue lines with karaoke timing tags.

Current behavior:

- structural/spacing line output follows `--writer-spacing` (`keep` by default).
- display end time prefers matched unit timing instead of blindly extending to the next line.
- unmatched lines receive a short visible-duration fallback.

Example:

```bash
py-roller run \
  --stages w \
  --alignment-result ./song.alignment.json \
  --writer-backend ass_karaoke \
  --output-roller ./song.ass
```

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
  --language zh \
  --output-roller ./out_dir
```

### Batch controls

- `--jobs N`: maximum number of parallel workers.
- `--continue-on-error`: keep processing remaining tasks after failures.
- `--skip-existing`: skip tasks whose declared final outputs already exist.
- `--manifest jobs.yaml`: load explicit per-task paths from YAML instead of pairing by stem.

Parallelism guidance:

- CPU-only: start with `--jobs 1` or `--jobs 2`.
- Single GPU: usually start with `--jobs 1`.

### YAML manifest format

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

- each task must be a mapping.
- unknown keys are rejected.
- inputs must match the selected chain start.
- outputs must be valid final outputs for the selected chain.
- task ids/stems must be unique.
- final output paths must not conflict across tasks.
- relative paths are resolved relative to the manifest file location.

## YAML config for CLI defaults

Use `--config` to load YAML defaults.

Priority order:

```text
built-in defaults < config YAML < explicit CLI arguments
```

Section model:

- `shared`: defaults applied to both `run` and `batch`.
- `run`: currently no extra keys beyond `shared`.
- `batch`: defaults for batch-only options such as `jobs` and `skip_existing`.

Example:

```yaml
shared:
  language: zh
  writer_spacing: keep
  writer_backend: lrc_ms
  intermediate: ./tmp/py-roller-artifacts
  cleanup: on-success
  transcriber_device: cpu
  transcriber_model_path: ~/.cache/py-roller/models/transcriber
  transcriber_local_files_only: false
  transcriber_hf_xet: auto
  transcriber_hf_proxy: null
  transcriber_hf_download_timeout: 120
  transcriber_hf_etag_timeout: 30
  transcriber_hf_max_workers: 2
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

`filter_chain` can be written either as a comma-separated string or as a YAML list. Quote `on` or `off` for `transcriber_hf_xet` if your YAML parser treats them as booleans; py-roller also accepts boolean `true`/`false` there as `on`/`off` for convenience.

## Progress, logs, and cleanup

The project exposes progress in two layers:

- human-readable logs for normal terminal use;
- optional machine-readable JSONL events for GUI frontends such as lrc-roller.

Use `--progress-format` to choose the progress output mode:

```bash
py-roller run ... --progress-format human   # default, terminal-friendly logs
py-roller run ... --progress-format jsonl   # structured PYROLLER_EVENT lines
py-roller run ... --progress-format both    # logs plus structured events
```

`jsonl` emits one parseable event per line with the `PYROLLER_EVENT ` prefix, for example:

```text
PYROLLER_EVENT {"type":"download_progress","stage":"model-download","repo_id":"Systran/faster-whisper-large-v2","bytes_downloaded":1534203904,"bytes_total":3086912962,"percent":49.7}
```

This is intended for frontends that need reliable stage and download progress instead of parsing mixed logs from `tqdm`, Demucs, and `huggingface_hub`. Human-readable mode remains the default so existing CLI workflows are unchanged.

Current progress coverage:

- model preflight and Hugging Face model download events, including cache path, proxy/XET settings, bytes downloaded, total bytes when known, and estimated speed;
- splitter/Demucs wrapper progress;
- filter phase progress;
- transcriber phase progress, including faster-whisper segment-based inference updates when available;
- parser, aligner, and writer stage events;
- artifact write events and failure events.

In single-task `run`, human progress is shown as terminal logs/progress bars when supported. In `batch`, per-task progress is logged to avoid multiple workers fighting for one terminal. GUI frontends should prefer `--progress-format jsonl`.

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

- `--cleanup on-success`: remove per-task intermediate directories after successful tasks.
- `--cleanup never`: keep intermediate audio and logs for inspection.

## Troubleshooting

### Check the environment

```bash
py-roller doctor
```

`doctor` checks Python, Torch, Torchaudio, faster-whisper, CTranslate2, transformers, SOCKS proxy support, Demucs, and librosa.

If it reports a broken audio/transcriber environment, start with:

```bash
py-roller install
```

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

`py-roller install` prefers the newest validated dependency line for this release:

- Torch/TorchAudio/TorchVision are installed from the official 2.6.0 family for every built-in profile.
- SOCKS proxy support is installed by default through `httpx[socks]`, so Hugging Face downloads do not fail merely because `socksio` is missing.

If you upgrade or override audio/transcriber packages manually, run `py-roller doctor` before using transcription-heavy pipelines.
