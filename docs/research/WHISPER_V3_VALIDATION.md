# Whisper large-v3 validation (2026-10-15)

Default remains large-v2. Large-v3 improved Within mix, but regressed Fly,
including lyrics absent from its raw transcription. This is not evidence for
a universal upgrade. No matching or timing parameters were changed.

## Setup

- Systran/faster-whisper-large-v3 revision edaa852ec7e145841d8ffdb056a99866b5f0a478.
- model.bin SHA256: 69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1.
- CPU int8, four threads, beam 5, English, no prompts, no previous-text conditioning,
  no VAD; same 26-second cores with two-second context.
- Replayed cached v2 windows and reused identical CTC evidence for both models.
- Default weighted matching, evidence stitching, legacy timing.
- Reference timestamps used only for evaluation.

## Results

Numbers are v2 -> v3. Missing counts compare normalized whole-line identities,
including repeat counts. Pairing maximizes identity matches then minimizes time
error; distant repetitions can mask omissions. The <=2s column is an additional
temporal check, not manually verified recall.

| Song/audio | Missing | Paired error >0.5s | Paired error <=2s | Extra |
|---|---:|---:|---:|---:|
| Within/mix | 2 -> 0 | 5 -> 6 | 29 -> 32 | 0 -> 0 |
| Within/vocals | 32 -> 32 | 0 -> 0 | 2 -> 2 | 0 -> 0 |
| Fly/mix | 5 -> 7 | 18 -> 19 | 32 -> 33 | 1 -> 0 |
| Fly/vocals | 3 -> 5 | 5 -> 6 | 38 -> 35 | 1 -> 1 |

Within mix gains three <=2s instances at 139.14, 147.88 and 153.32 seconds,
with no losses by that check. Its separated vocals remain severely incomplete.
Fly mix gains six <=2s instances and loses five. The first 28-second v3 raw
window says "Music Outro", losing the opening lyrics before matching begins.
Fly vocals gains one <=2s instance and loses four, including both instances
of "Oh it's a miracle" at 82.75 and 224.83 seconds.
Pure-music hallucinations also remain (e.g. Within's "Thank you").

V3 inference seconds: Within mix 261.2, vocals 181.4; Fly mix 283.1,
vocals 295.0. Loading is separately recorded in each run.json. V2 reused
historical evidence, so these are not a controlled v2/v3 speed comparison.

## Incomplete arm

Lily (E18) could not run: its original audio, separated vocals and reference LRC
are on the currently unmounted /Volumes/StudioVantablack volume. The script
stopped at E18 reference loading after saving the v2 replay. No Lily v3 result
is claimed. Re-running scripts/validate_whisper_v3.py with the same model and
output arguments resumes saved arms after that volume becomes available.

Detailed instance deltas: ../../py-roller-artifacts/whisper-v3-validation/summary.md.
Each completed arm contains raw-windows.json, raw-transcript.txt, result.json,
result.lrc and run.json. These results have not been manually listened through.

## Model cleanup

Removed abandoned Qwen3-ASR-1.7B-8bit, Qwen3-ForcedAligner-0.6B-8bit,
SOFA-GTSinger, parakeet-ctc-0.6b, wav2vec2-lv-60-espeak-cv-ft and
faster-whisper-small model directories, reclaiming approximately 8.5 GiB.
Kept large-v2/v3, default CTC, Demucs, audio, references and experiment evidence.
Immediately after cleanup free disk space increased from 5.4 GiB to about
14 GiB; subsequent inference changes free space through system memory/swap.
