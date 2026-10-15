# POWER kernel integration

Experimental CLI: `--scoring-mode power`. Default remains `weighted`.

## Scope and provenance

Uses unmodified `levenshtein.py` and `phonemes.py` from NickRuiz/power-asr
commit bbb42f1c20a2a78303c427b0f0361fbd7499e428, with its MIT license.
This is a POWER-kernel adaptation, NOT the complete upstream PowerAligner
or its POWER word-error-rate metric. No new acoustic model is involved.

The adapter retains existing matching as a fallback and adds 1..3-word
phonetic transitions. CMUdict first pronunciations supply lowercase phones
and word boundaries; no syllable annotation or upstream OOV heuristics are used.
Unknown words fall back to the original costs. POWER's reserved boundaries,
phoneme classes and 4/3/3 edit costs determine the path. The adapter divides
path cost by four times the larger phone count, caps the ratio at one and
multiplies by the larger word count. Existing function-word weights then apply.
This normalization is our experimental choice, not a calibrated POWER score.

Current candidate length limits, 0.72 gate, global selection and legacy CTC
remain in use. The new costs also affect evidence stitching. This experiment
therefore measures an end-to-end adaptation, not an isolated upstream algorithm.
Multiword operations expose coarse Cartesian word mappings, not precise
phone-to-word timing anchors. Do not treat these as validated acoustic anchors.

## Fixed cached validation

Run `scripts/validate_power.py py-roller-artifacts/power-kernel-validation`.
Four songs, original mix and separated vocals, both baseline and experimental
arms. No new model inference, prompts, reference-time inference or parameter scan.
Results and LRCs are in `py-roller-artifacts/power-kernel-validation`.

| Song/view | Missing baseline -> power | Matched >0.5s | Extra | Processing seconds |
|---|---:|---:|---:|---:|
| Fly/mix | 5 -> 5 | 18 -> 18 | 1 -> 1 | 3.14 -> 21.81 |
| Fly/vocals | 3 -> 3 | 5 -> 5 | 1 -> 1 | 2.72 -> 23.98 |
| All My Friends/mix | 23 -> 17 | 10 -> 17 | 7 -> 8 | 3.26 -> 52.78 |
| All My Friends/vocals | 21 -> 18 | 6 -> 4 | 5 -> 5 | 3.00 -> 52.54 |
| Something Comforting/mix | 12 -> 12 | 1 -> 2 | 1 -> 7 | 1.69 -> 16.96 |
| Something Comforting/vocals | 13 -> 8 | 1 -> 0 | 2 -> 4 | 2.01 -> 20.42 |
| Lily/mix | 8 -> 5 | 5 -> 4 | 0 -> 0 | 6.30 -> 150.45 |
| Lily/vocals | 8 -> 5 | 5 -> 3 | 0 -> 0 | 6.78 -> 150.43 |

Missing counts pair exact normalized identities without a time gate.
Consequently distant wrong repetitions can mask missing instances. The report
also tracks transitions into/out of <=2s pairs: 26 gains and 9 losses across
the eight views. These are reference-based results, not listening confirmation.
The two views of a song are not independent samples.

The baseline second-category cohort of 54 has NOT been reevaluated in full:
this experiment covers four songs only. No claim of resolving that cohort.

## Conclusion

Keep experimental. Lily shows useful recovery; Fly is unchanged. All My Friends
mix and Something Comforting expose false selection and repeat-position risks.
Total cached processing rises from about 29 to 589 seconds. This is not a
production-ready replacement. A phonetic candidate can be discovered without
being safely accepted; retaining the old gate does not retain its calibration.
Future evaluation should separate candidate recall from selection precision,
and examine actual phone mappings before changing timing.

Validation: 97 tests passed; CLI exposes the experimental flag. No dependencies
were installed. A Pyphen installation attempt failed due to the local Python
SSL configuration; the vendored kernels do not need Pyphen.
