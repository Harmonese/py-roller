# v1 engine migration and validation

Date: 2026-10-14. Machine: Apple M2, 8 GB. This is an intentionally breaking
replacement of the main engine, authorized without Rolling Pebble compatibility.
No files in the sibling Rolling Pebble project were changed.

## Shipped architecture

`py-roller`, `python -m pyroller` and `pyroller.engine.align` now use the standalone
`pyroller.hybrid` package. Production inference does not import `src-reconstruct`
or the historical pipeline. Model resolution, audio decoding, cached inference,
instance discovery, local alignment, previews and artifacts work from audio files.
The wheel contains only the new package and its C++ alignment extension.

The default is original mix audio. Optional Demucs is explicit, FFmpeg-decoded,
seeded and thread-bounded. Whisper large-v2 CPU/int8 discovers words; wav2vec2
large-960h-lv60-self supplies local timing evidence. Models load sequentially.
No TTS, IPA similarity, whole-song CTC lyric graph, Qwen, recursive repair or
Rolling Pebble protocol adapter participates in the new main path.

CTC and Whisper caches include audio/checkpoint hashes and backend settings.
Failures preserve completed evidence; output directories cannot be overwritten.
Events are full revision snapshots, not the old suffix-patch protocol. Final
means computation ended; all lyric instances remain unverified. The engine is
file-oriented: CTC preprocessing precedes Whisper-window previews. It does not
yet provide low-latency live streaming.

## Algorithm changes and their limits

Onsets and ends are decided separately. A bad end or a neighboring coarse end
can no longer automatically roll back a valid onset. Ordered onset alternatives
are selected together. Ordinary corrections retain the prototype's 2-second onset
prior; larger shifts require matching word anchors and first-word CTC support.
Display ends are bounded separately; raw acoustic ends and character paths remain
in diagnostics.

An initial attempt required strong first-word support even for ordinary shifts.
Six-view testing showed that this rejected many correct timings: CTC blank can
dominate where a complete known-text path nevertheless has a useful boundary.
That universal gate was removed. This was a shared rule correction, not per-song
threshold tuning. Both the initial and final outputs remain in local artifacts.

Short/long prefix variants are independently scored on the same local evidence.
However, automatic shortening was not retained: at Fly for Your Life 53.07s it
deleted the valid `eyes`, while at 54.56s the same rule would correctly remove it.
Weak CTC suffix evidence does not establish absence. The engine therefore preserves
the Whisper-selected identity and records `alternative_text`, scores and word
support under `variant_decisions`. Reliable automatic identity correction remains
unfinished; the report does not count proposals as recovered lyrics.

## Fixed six-view replay

No reference times or ordering enter inference. Evaluation pairs normalized text
one-to-one within 2 seconds; the primary count requires onset error <=0.5 seconds.
Existing LRC is not newly audited ground truth. All six views use the same settings.

| Song | Audio | Previous hybrid | New main engine | Reference instances |
|---|---|---:|---:|---:|
| Blossom | mix | 17 | 16 | 23 |
| Blossom | vocals | 17 | 19 | 23 |
| Within | mix | 27 | 27 | 34 |
| Within | vocals | 1 | 2 | 34 |
| Fly for Your Life | mix | 19 | 16 | 42 |
| Fly for Your Life | vocals | 28 | 29 | 42 |

The six-view total is unchanged: 109 -> 109. Mix-only counts fall from 63 to 59;
vocals-only counts rise from 46 to 50. These are repeated views, not independent
test songs. There is no demonstrated overall accuracy gain or regression-free
claim. The architecture is the new default because this migration was explicitly
requested, not because it won every accuracy comparison.

Useful recoveries in Blossom vocals: `I had lost my sight` at 126.00s and
`You're spending it in your head...` at 174.59s. Within original mix retains all
27 previously passing instances. Separated Within remains unsuitable because
Demucs removes robotic singing, as established by the user's listening.

Remaining regressions primarily show the cost of removing the old end-time veto:

- Blossom mix, 62.33s: onset moved from 62.00 to 60.0125. The old end guard had
  incidentally rejected that bad path. The onset-only 2-second prior allows it.
- Fly mix, 54.56s, 69.48s and 196.64s: previously retained coarse times become
  incorrect local CTC onsets. The old end guard also incidentally protected them.
- Fly vocals, 208.51s: local CTC chooses 206.5325, confusing a nearby repeated
  phrase. The new ordinary-onset rule accepts it without strong anchors.
- Fly mix, 94.64s: a valid 94.9125 CTC proposal has no argmax character support;
  its rank ties coarse, so conservative tie-breaking keeps 95.24s, just outside
  the 0.5-second acceptance threshold.

This identifies the next algorithm problem as comparing local boundary proposals
and neighboring repetitions, rather than adding another whole-song recovery pass.
The current argmax support and onset prior are heuristics, not calibrated confidence.

Detailed local outputs:

- [Per-song summary and LRC links](../../py-roller-artifacts/v1-release-validation/summary.md)
- [Initial conservative-gate result](../../py-roller-artifacts/v1-validation/summary.md)
- Each song directory contains `result.json`, `evaluation.json`, `coarse.lrc` and
  `result.lrc`; diagnostics include all variant proposals and onset decisions.

## Runtime and tests

Full-song cached algorithm work takes approximately 0.06-0.40 seconds per view.
This includes stitching, text search and local CTC alignment, but excludes cache
reading, model-weight hashing and neural inference. It is not end-to-end latency.

An actual 32-second Within mix clip was decoded and run through both neural models
with empty evidence caches, producing a preview followed by a final 9-line LRC:

| Component | Seconds |
|---|---:|
| CTC inference | 3.34 |
| Whisper inference | 39.82 |
| CTC model load | 5.07 |
| Whisper model load | 3.66 |
| Alignment across both revisions | 0.057 |
| Total, including imports, hashing and I/O | 70.59 |

The immediately repeated cached run took 15.64 seconds total, with 8 CTC and 2
Whisper cache hits, zero model loads and zero neural inference. Hashing/imports
remain a significant cached-startup cost. These measurements precede the added
vocabulary-file fingerprint; that stricter key correctly invalidates older
Whisper entries. Later simultaneous smoke checks are functional checks, not
comparable performance benchmarks.

Validation: 27 new engine tests and 75 independent experimental regression tests
passed. The new suite covers text ordering/repeats, overlap stitching, unknown
audio, exact small CTC paths, repeated labels, identity ambiguity, onset/end
independence, anchored large shifts, legal onset chains, unsupported characters,
cache identity, cancellation, snapshot reconstruction and right-context clocks.
Fresh Whisper numpy timestamps are converted at ingestion, with a regression
test ensuring that cold inference and JSON-cache replay have identical serializable
results. This fixed a serialization failure exposed by the final cold smoke run.
The final code then passed a fresh 12-second neural run (zero cache hits; CTC
1.15s, Whisper 10.84s, reported total 34.10s), and the 32-second mix rerun produced
all 9 output lines with a preview. Optional Demucs -> CTC -> Whisper also completed
on that 32-second clip, producing zero matched lines after the robotic voice was
removed. That is a successful runtime check, not a separation-quality success.
All six song structural checks pass. The historical `tests/` directory covers
the removed 0.9 API and is intentionally not the new default suite.

Both wheel and source archive were built and checked to exclude the legacy
pipeline. An isolated wheel install outside the checkout imports the new API,
runs the C++ kernel and exposes the new CLI. The environment also installs the
new editable CLI. Its Python 3.12 SSL extension currently cannot load a missing
Homebrew OpenSSL library. Existing local models and offline inference work;
fresh HTTPS model downloads were not verified. Repair/reinstall that Python and
its OpenSSL dependency before relying on network downloads. Global Homebrew
state was not modified by this change.
