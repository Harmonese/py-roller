# v1: complete comparison with reference LRC

Comparison of the released six-view replay, not the previous prototype. No inference was rerun or changed. Existing reference LRC is used as supplied; this is a textual/path audit, not a fresh listening audit.

Positive error = late; negative error = early. Official pairing is one-to-one, exact normalized text and at most 2 seconds apart. OK means absolute onset error <=0.5 seconds. A missing official pair is not automatically missing content. For an unpaired reference, an unused exact-text output within 12 seconds is shown separately as an unpaired timing discrepancy, never counted as a recovery. Long-distance repeats are not reassigned.

| Song | Audio | Reference | Output | OK | 0.5-2s error | Unpaired reference | Unpaired output |
|---|---|---:|---:|---:|---:|---:|---:|
| Blossom | mix | 23 | 22 | 16 | 4 | 3 | 2 |
| Blossom | vocals | 23 | 22 | 19 | 3 | 1 | 0 |
| Within | mix | 34 | 32 | 27 | 2 | 5 | 3 |
| Within | vocals | 34 | 2 | 2 | 0 | 32 | 0 |
| Fly for Your Life | mix | 42 | 33 | 16 | 14 | 12 | 3 |
| Fly for Your Life | vocals | 42 | 35 | 29 | 5 | 8 | 1 |

## Interpretation

- Blossom vocals: one text omission at 35.67s, and three remaining onset errors. Mix has one text omission at 86.56s and six onset errors, including two larger than 2 seconds.
- Within mix: two missing lyric instances at 147.88s and 153.32s; four onset errors at 130.48s, 139.14s, 156.55s and 161.76s. The first line is a separate 9.15-second reference/output disagreement needing listening. Separated vocals omit 32 of 34 references.
- Fly vocals: eight references without exact paired output, including the short/long identity error at 54.56s, two clear contraction-matching failures at 88.13s and 167.22s, and five onset errors. Short text at 147.41s and late backing responses are absent from the stitched transcript.
- Fly mix: 14 matched lines have 0.5-2-second onset error and two further lines have >2-second error (53.07s and 122.30s). Other unpaired references mix missing content, prefix-version competition and response/repetition loss. See every row below.

## Evidence examples

- Blossom vocals 35.67s: reference `I'm well-wishing for a life, I picture you happy, oh`; stitched ASR `i'm now wishing for a life a picture you`, then the next line. No whole-line candidate.
- Blossom mix 86.56s: reference `We'll stay together`; ASR `if we'll all stay together`. No candidate. An inserted `all` is too costly for the short-line matching rule.
- Blossom vocals 182.33s: coarse `I'm` starts at 180.4s but `sorry` at 182.7s. CTC proposes a better boundary but is rejected by the large-shift/first-word support rule.
- Within 147.88-153.32s: ASR instead says `i am lost i can't find my way back to the way i used to be`. The target `Many rooms...` and `Where are the locks...` are not discovered.
- Fly vocals 88.13s and 167.22s: ASR clearly gives `i've got spirit i've got wings`; target is `I got spirit, I got wings`. Two whole-word substitutions reduce direct six-word similarity to 4/6, below 0.72. Neither instance enters the candidate set.
- Fly vocals 23.07s: response candidate exists at 21.86-24.98s, but consumes word tokens also used by the preceding lead line and loses joint text selection. The transcript contains only one standalone response after the lead-line ending, not all reference repetitions.
- Fly vocals 54.56s: output at 54.9525s is `You've got those dangerous eyes` instead of the short `You've got those dangerous`. The short candidate exists and CTC proposes it, but automatic identity rewriting is disabled.
- Fly vocals 208.51s: coarse start 208.50s was accurate; CTC shifts it to 206.5325s despite zero supporting anchors. This is an algorithmic timing regression.
- Within first line: supplied reference 51.22s, Whisper 59.74s and CTC 60.3725s. Agreement between the models cannot establish which timestamp is correct; retain as an unresolved reference/audio discrepancy.

## Blossom

Reference: `src-reconstruct/artifacts/whisper-extended/references/E01.lrc`.

| # | Reference | Text | Mix output / error | Vocals output / error |
|---:|---|---|---|---|
| 1 | 00:26.84 | I close my eyes, for once I don't have to try | 00:26.93 / +0.093s OK | 00:26.97 / +0.133s OK |
| 2 | 00:35.67 | I'm well-wishing for a life, I picture you happy, oh | 00:34.23 / -1.438s ERROR | NO EXACT PAIRED OUTPUT |
| 3 | 00:45.07 | In the weeds we walked, your hand running through the moss | 00:43.79 / -1.277s ERROR | 00:45.27 / +0.203s OK |
| 4 | 00:54.28 | I was so lost in thought, but you were there living | 00:54.39 / +0.113s OK | 00:54.39 / +0.113s OK |
| 5 | 01:02.33 | You've taken me with you | 01:00.01 / -2.317s UNPAIRED | 01:02.39 / +0.063s OK |
| 6 | 01:09.43 | Can I slow the days? I wish that I'll find the faith | 01:09.51 / +0.082s OK | 01:09.51 / +0.082s OK |
| 7 | 01:18.28 | I'll die and I'll save your place, and you'll never suffer | 01:18.47 / +0.193s OK | 01:18.87 / +0.593s ERROR |
| 8 | 01:26.56 | We'll stay together | NO EXACT PAIRED OUTPUT | 01:26.63 / +0.073s OK |
| 9 | 01:34.82 | Oh, it's not alright, that one day we're all out of time | 01:34.91 / +0.093s OK | 01:34.95 / +0.133s OK |
| 10 | 01:43.97 | I'll write you another life, I'm sorry for crying | 01:44.15 / +0.183s OK | 01:43.31 / -0.657s ERROR |
| 11 | 01:51.82 | It's just that I love you | 01:51.99 / +0.173s OK | 01:51.99 / +0.173s OK |
| 12 | 01:57.06 | One day can't be far enough | 01:57.25 / +0.193s OK | 01:57.25 / +0.193s OK |
| 13 | 02:06.00 | I had lost my sight | 02:03.55 / -2.447s UNPAIRED | 02:06.19 / +0.193s OK |
| 14 | 02:13.87 | So you sang to me the beauty you'd been seeing | 02:14.09 / +0.222s OK | 02:14.09 / +0.222s OK |
| 15 | 02:22.03 | If I can't stop time | 02:22.21 / +0.183s OK | 02:22.25 / +0.222s OK |
| 16 | 02:29.66 | I'll build a world where God cannot take us | 02:28.77 / -0.888s ERROR | 02:29.85 / +0.192s OK |
| 17 | 02:38.06 | There's no need to think of time | 02:38.15 / +0.093s OK | 02:38.13 / +0.072s OK |
| 18 | 02:46.41 | But then you said, "Don't think of the time that's left | 02:46.61 / +0.202s OK | 02:46.59 / +0.183s OK |
| 19 | 02:54.59 | You're spending it in your head, I just want you laughing" | 02:54.65 / +0.062s OK | 02:54.63 / +0.042s OK |
| 20 | 03:02.33 | I'm sorry for crying, I just want you happy | 03:01.13 / -1.198s ERROR | 03:00.40 / -1.930s ERROR |
| 21 | 03:14.42 | So at last, goodbye | 03:14.65 / +0.233s OK | 03:14.67 / +0.252s OK |
| 22 | 03:22.87 | I'll be happy just to have known you | 03:23.03 / +0.162s OK | 03:23.05 / +0.182s OK |
| 23 | 03:31.32 | And we wished each other well | 03:31.59 / +0.273s OK | 03:31.57 / +0.252s OK |

### mix: officially unpaired outputs

- 01:00.01: You've taken me with you (same-text timing discrepancy listed above)
- 02:03.55: I had lost my sight (same-text timing discrepancy listed above)

[LRC](../../py-roller-artifacts/v1-release-validation/mix/E01/result.lrc) / [Detailed engine diagnostics](../../py-roller-artifacts/v1-release-validation/mix/E01/result.json)

### vocals: officially unpaired outputs

None.

[LRC](../../py-roller-artifacts/v1-release-validation/vocals/E01/result.lrc) / [Detailed engine diagnostics](../../py-roller-artifacts/v1-release-validation/vocals/E01/result.json)

## Within

Reference: `src-reconstruct/artifacts/whisper-extended/references/E06.lrc`.

| # | Reference | Text | Mix output / error | Vocals output / error |
|---:|---|---|---|---|
| 1 | 00:51.22 | There are so many things that I don't understand | 01:00.37 / +9.153s UNPAIRED | NO EXACT PAIRED OUTPUT |
| 2 | 01:09.31 | There's a world within me that I cannot explain | 01:09.07 / -0.237s OK | 01:09.03 / -0.278s OK |
| 3 | 01:18.12 | Many rooms to explore, but the doors look the same | 01:17.79 / -0.328s OK | NO EXACT PAIRED OUTPUT |
| 4 | 01:26.79 | I am lost I can't even remember my name | 01:26.63 / -0.157s OK | 01:26.63 / -0.157s OK |
| 5 | 01:35.17 | I've been, for some time | 01:35.07 / -0.097s OK | NO EXACT PAIRED OUTPUT |
| 6 | 01:37.65 | Looking for someone | 01:37.19 / -0.457s OK | NO EXACT PAIRED OUTPUT |
| 7 | 01:39.64 | I need to know now | 01:39.49 / -0.147s OK | NO EXACT PAIRED OUTPUT |
| 8 | 01:41.71 | Please tell me who I am | 01:41.53 / -0.178s OK | NO EXACT PAIRED OUTPUT |
| 9 | 01:44.15 | I've been, for some time | 01:43.83 / -0.317s OK | NO EXACT PAIRED OUTPUT |
| 10 | 01:46.10 | Looking for someone | 01:45.95 / -0.147s OK | NO EXACT PAIRED OUTPUT |
| 11 | 01:48.28 | I need to know now | 01:48.21 / -0.067s OK | NO EXACT PAIRED OUTPUT |
| 12 | 01:50.46 | Please tell me who I am | 01:50.25 / -0.207s OK | NO EXACT PAIRED OUTPUT |
| 13 | 02:10.48 | There are so many things that I don't understand | 02:08.27 / -2.208s UNPAIRED | NO EXACT PAIRED OUTPUT |
| 14 | 02:19.14 | There's a world within me that I cannot explain | 02:15.93 / -3.207s UNPAIRED | NO EXACT PAIRED OUTPUT |
| 15 | 02:27.88 | Many rooms to explore, but the doors look the same | NO EXACT PAIRED OUTPUT | NO EXACT PAIRED OUTPUT |
| 16 | 02:33.32 | (Where are the locks to try the key) | NO EXACT PAIRED OUTPUT | NO EXACT PAIRED OUTPUT |
| 17 | 02:36.55 | I am lost I can't even remember my name | 02:35.61 / -0.938s ERROR | NO EXACT PAIRED OUTPUT |
| 18 | 02:41.76 | (And I wonder why) | 02:40.79 / -0.968s ERROR | NO EXACT PAIRED OUTPUT |
| 19 | 02:45.01 | I've been, for some time | 02:44.89 / -0.118s OK | NO EXACT PAIRED OUTPUT |
| 20 | 02:47.15 | Looking for someone | 02:46.87 / -0.278s OK | NO EXACT PAIRED OUTPUT |
| 21 | 02:49.45 | I need to know now | 02:49.27 / -0.178s OK | NO EXACT PAIRED OUTPUT |
| 22 | 02:51.43 | Please tell me who I am | 02:51.33 / -0.098s OK | NO EXACT PAIRED OUTPUT |
| 23 | 02:53.88 | I've been, for some time | 02:53.61 / -0.268s OK | NO EXACT PAIRED OUTPUT |
| 24 | 02:55.84 | Looking for someone | 02:55.69 / -0.148s OK | NO EXACT PAIRED OUTPUT |
| 25 | 02:58.09 | I need to know now | 02:57.97 / -0.118s OK | NO EXACT PAIRED OUTPUT |
| 26 | 03:00.12 | Please tell me who I am | 03:00.07 / -0.048s OK | NO EXACT PAIRED OUTPUT |
| 27 | 03:02.64 | I've been, for some time | 03:02.33 / -0.308s OK | NO EXACT PAIRED OUTPUT |
| 28 | 03:04.58 | Looking for someone | 03:04.12 / -0.460s OK | NO EXACT PAIRED OUTPUT |
| 29 | 03:06.75 | I need to know now | 03:06.73 / -0.018s OK | NO EXACT PAIRED OUTPUT |
| 30 | 03:08.83 | Please tell me who I am | 03:08.48 / -0.350s OK | NO EXACT PAIRED OUTPUT |
| 31 | 03:11.23 | I've been, for some time | 03:11.05 / -0.178s OK | NO EXACT PAIRED OUTPUT |
| 32 | 03:13.38 | Looking for someone | 03:13.15 / -0.227s OK | NO EXACT PAIRED OUTPUT |
| 33 | 03:15.45 | I need to know now | 03:15.41 / -0.037s OK | NO EXACT PAIRED OUTPUT |
| 34 | 03:17.75 | Please tell me who I am | 03:17.51 / -0.238s OK | NO EXACT PAIRED OUTPUT |

### mix: officially unpaired outputs

- 01:00.37: There are so many things that I don't understand (same-text timing discrepancy listed above)
- 02:08.27: There are so many things that I don't understand (same-text timing discrepancy listed above)
- 02:15.93: There's a world within me that I cannot explain (same-text timing discrepancy listed above)

[LRC](../../py-roller-artifacts/v1-release-validation/mix/E06/result.lrc) / [Detailed engine diagnostics](../../py-roller-artifacts/v1-release-validation/mix/E06/result.json)

### vocals: officially unpaired outputs

None.

[LRC](../../py-roller-artifacts/v1-release-validation/vocals/E06/result.lrc) / [Detailed engine diagnostics](../../py-roller-artifacts/v1-release-validation/vocals/E06/result.json)

## Fly for Your Life

Reference: `src-reconstruct/artifacts/extended-vocal-validation/E08/reference.lrc`.

| # | Reference | Text | Mix output / error | Vocals output / error |
|---:|---|---|---|---|
| 1 | 00:04.59 | We own the sky, you and I | 00:05.03 / +0.443s OK | 00:05.03 / +0.443s OK |
| 2 | 00:09.96 | (We own the sky, you and I) | 00:10.04 / +0.080s OK | 00:09.90 / -0.060s OK |
| 3 | 00:18.18 | Take it tonight, fly for your life | 00:19.07 / +0.892s ERROR | 00:18.25 / +0.073s OK |
| 4 | 00:23.07 | (Fly for your life, fly for your life) | NO EXACT PAIRED OUTPUT | NO EXACT PAIRED OUTPUT |
| 5 | 00:43.12 | There's one thing I know for sure | NO EXACT PAIRED OUTPUT | 00:43.25 / +0.133s OK |
| 6 | 00:47.22 | You gotta fly for your life | 00:45.81 / -1.407s ERROR | 00:47.45 / +0.233s OK |
| 7 | 00:53.07 | You've got those dangerous eyes | 00:49.25 / -3.817s UNPAIRED | 00:53.21 / +0.143s OK |
| 8 | 00:54.56 | You've got those dangerous | 00:55.55 / +0.992s ERROR | NO EXACT PAIRED OUTPUT |
| 9 | 00:56.74 | A world I've never seen before | NO EXACT PAIRED OUTPUT | 00:57.01 / +0.273s OK |
| 10 | 01:00.46 | A silhouette into the sun | NO EXACT PAIRED OUTPUT | 01:00.67 / +0.213s OK |
| 11 | 01:06.51 | Are you ready to burn | 01:06.75 / +0.242s OK | 01:06.75 / +0.242s OK |
| 12 | 01:08.06 | Are you ready | 01:08.05 / -0.007s OK | 01:08.43 / +0.373s OK |
| 13 | 01:09.48 | We'll keep falling in love | 01:08.79 / -0.688s ERROR | 01:09.59 / +0.112s OK |
| 14 | 01:16.26 | We'll keep falling in love | 01:16.77 / +0.513s ERROR | 01:16.51 / +0.252s OK |
| 15 | 01:22.75 | Oh it's a miracle | 01:22.30 / -0.450s OK | 01:23.75 / +1.002s ERROR |
| 16 | 01:28.13 | I got spirit, I got wings | NO EXACT PAIRED OUTPUT | NO EXACT PAIRED OUTPUT |
| 17 | 01:34.64 | I got fire in my lungs | 01:35.24 / +0.600s ERROR | 01:34.91 / +0.273s OK |
| 18 | 02:02.30 | I don't know what you're fighting for | 01:59.11 / -3.187s UNPAIRED | 02:02.75 / +0.453s OK |
| 19 | 02:06.68 | You better fly for your life | 02:06.89 / +0.213s OK | 02:06.89 / +0.213s OK |
| 20 | 02:12.34 | It's just another disguise | 02:11.91 / -0.428s OK | 02:12.69 / +0.352s OK |
| 21 | 02:16.29 | Do you ever think of me | 02:16.63 / +0.343s OK | 02:16.61 / +0.322s OK |
| 22 | 02:19.93 | Cause you're the ghost in my heart | 02:20.11 / +0.182s OK | 02:20.11 / +0.182s OK |
| 23 | 02:25.68 | You've got those dangerous eyes | 02:25.93 / +0.252s OK | 02:25.89 / +0.212s OK |
| 24 | 02:27.41 | You've got those dangerous | NO EXACT PAIRED OUTPUT | NO EXACT PAIRED OUTPUT |
| 25 | 02:28.96 | We'll keep falling in love | 02:28.92 / -0.040s OK | 02:27.61 / -1.348s ERROR |
| 26 | 02:35.55 | We'll Keep falling in love | 02:35.02 / -0.530s ERROR | 02:35.74 / +0.190s OK |
| 27 | 02:42.06 | Oh it's a miracle | 02:41.60 / -0.460s OK | 02:42.43 / +0.373s OK |
| 28 | 02:47.22 | I got spirit, I got wings | 02:46.00 / -1.220s ERROR | NO EXACT PAIRED OUTPUT |
| 29 | 02:54.05 | I got fire in my lungs | 02:53.30 / -0.750s ERROR | 02:52.79 / -1.258s ERROR |
| 30 | 03:03.13 | We own the sky, you and I | 03:03.79 / +0.662s ERROR | 03:02.44 / -0.690s ERROR |
| 31 | 03:08.52 | (We own the sky, you and I) | 03:08.88 / +0.360s OK | 03:08.76 / +0.240s OK |
| 32 | 03:16.64 | Take it tonight, fly for your life | 03:17.37 / +0.733s ERROR | 03:17.01 / +0.373s OK |
| 33 | 03:21.81 | (Fly for your life, fly for your life) | NO EXACT PAIRED OUTPUT | NO EXACT PAIRED OUTPUT |
| 34 | 03:26.10 | Just fly for your life | NO EXACT PAIRED OUTPUT | NO EXACT PAIRED OUTPUT |
| 35 | 03:28.51 | We'll keep falling in love | 03:27.88 / -0.630s ERROR | 03:26.53 / -1.977s ERROR |
| 36 | 03:34.93 | We'll keep falling in love | NO EXACT PAIRED OUTPUT | 03:35.00 / +0.070s OK |
| 37 | 03:44.83 | Oh it's a miracle | 03:45.00 / +0.170s OK | 03:45.00 / +0.170s OK |
| 38 | 03:50.19 | I got spirit, I got wings | NO EXACT PAIRED OUTPUT | NO EXACT PAIRED OUTPUT |
| 39 | 03:58.38 | We'll keep falling in love | 03:57.62 / -0.760s ERROR | 03:58.43 / +0.053s OK |
| 40 | 04:04.97 | We'll keep falling in love | 04:04.26 / -0.710s ERROR | 04:04.99 / +0.023s OK |
| 41 | 04:11.45 | We'll keep falling in love | 04:11.00 / -0.450s OK | 04:11.28 / -0.170s OK |
| 42 | 04:17.80 | We'll keep falling in love | 04:17.64 / -0.160s OK | 04:18.27 / +0.472s OK |

### mix: officially unpaired outputs

- 00:49.25: You've got those dangerous eyes (same-text timing discrepancy listed above)
- 01:59.11: I don't know what you're fighting for (same-text timing discrepancy listed above)
- 02:22.58: You've got those dangerous eyes

[LRC](../../py-roller-artifacts/v1-release-validation/mix/E08/result.lrc) / [Detailed engine diagnostics](../../py-roller-artifacts/v1-release-validation/mix/E08/result.json)

### vocals: officially unpaired outputs

- 00:54.95: You've got those dangerous eyes

[LRC](../../py-roller-artifacts/v1-release-validation/vocals/E08/result.lrc) / [Detailed engine diagnostics](../../py-roller-artifacts/v1-release-validation/vocals/E08/result.json)
