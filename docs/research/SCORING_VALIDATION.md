# Matching score consistency and function-word costs

## Shipped behavior

`weighted` is now the default robust-matcher scoring policy, promoted at the
user's request after reviewing the results below. It includes consistency and
function-word discounts. `consistent` retains consistency without discounts;
`original` reproduces the previous policy. Evidence stitching and legacy CTC timing remain default.
The legacy matcher does not use these policies. No model, search stage, global
similarity threshold, lyric-order assumption or timing algorithm changed.

For lyric length n, observed span length m, and minimum edit cost D:

- Original: similarity S = 1 - D / max(n,m); path gain = n - 2D - 0.5.
- Consistent/weighted: M = max(n,m); S = 1 - D/M;
  path gain = M - 2D - 0.5 = M(2S - 1) - 0.5.
- Admission still requires S >= 0.72. Path selection sums gains, not normalized
  similarities: explaining more supported words remains valuable, while each
  new instance costs 0.5. Unknown words still have zero gain.
- An edge substitution is now charged like an interior substitution instead of
  automatically rejecting the whole candidate. Unsupported edge insertions are
  still rejected. At least one supported correspondence is required.
- Weighted errors use weight 0.35 for the fixed FUNCTION_WORDS set, otherwise 1.
  Deletion/insertion uses the affected word's weight; substitution uses the
  larger of the two weights. Phrase operations use the larger mean weight.
  Correct word coverage and normalization are NOT discounted. Negation,
  quantifiers and personal pronouns retain full weight; possessive determiners
  are included in the function-word set. No per-song weight tuning was done.
- Existing spelling, CMUdict pronunciation, homophone and contraction costs are
  unchanged. This is dictionary pronunciation similarity, not acoustic scoring.
- `operations.penalty` stores the weighted edit charge. `operations.cost` and
  `matched_words.cost` retain unweighted correspondence costs, so an unrelated
  pair does not become a reliable CTC timing anchor merely through discounting.

Both splice selection and final lyric selection use this policy. The experiment
therefore measures their combined effect, not just final candidate admission.

## Validation

Final artifacts: `py-roller-artifacts/scoring-final/summary.md` and `summary.json`;
each policy/view/song directory includes result JSON, final LRC and coarse LRC.
Reproduce with `python scripts/validate_scoring.py NEW_OUTPUT_DIRECTORY`.

Seven songs, eleven audio views, three policies. All runs reuse identical
unprompted Whisper and matching CTC caches. Gloria, Teenage Anarchist and After
Dark have only vocals CTC caches available for this comparison. References are
loaded only after inference and were not newly audited by listening.

The counts below compare original with weighted. Consistent has unchanged
missing/extra/>2s/<=0.5s counts in every view.

| Song / audio | Missing | Extra | Matched but >2s |
|---|---:|---:|---:|
| Blossom / mix | 0 -> 0 | 0 -> 1 | 2 -> 2 |
| Blossom / vocals | 1 -> 1 | 1 -> 2 | 0 -> 1 |
| Within / mix | 2 -> 2 | 0 -> 0 | 3 -> 3 |
| Within / vocals | 32 -> 32 | 0 -> 0 | 0 -> 0 |
| Fly / mix | 7 -> 5 | 1 -> 1 | 4 -> 5 |
| Fly / vocals | 3 -> 3 | 1 -> 1 | 1 -> 1 |
| Gloria / vocals | 7 -> 7 | 0 -> 0 | 2 -> 2 |
| Teenage Anarchist / vocals | 5 -> 3 | 0 -> 0 | 1 -> 0 |
| The Veldt / mix | 3 -> 3 | 1 -> 1 | 8 -> 8 |
| The Veldt / vocals | 5 -> 5 | 2 -> 2 | 11 -> 11 |
| After Dark / vocals | 3 -> 3 | 0 -> 0 | 2 -> 2 |

Weighted recovered four reference identities at the following locations:

| Instance | Reference | Output | Error |
|---|---:|---:|---:|
| Fly mix: A world I've never seen before | 56.74 | 57.2325 | +0.4925 |
| Fly mix: A silhouette into the sun | 60.46 | 59.1125 | -1.3475 |
| Teenage Anarchist: All I saw was bloodless ideology | 65.43 | 65.4925 | +0.0625 |
| Teenage Anarchist: But the politics were too convenient | 77.72 | 77.9725 | +0.2525 |

No reference identity was newly missing under weighted in this fixed set, but
Blossom gained a duplicate `I had lost my sight` in both views. Whisper's
`i had lost` fragment now has D=1.35 (missing `my` costs 0.35 and `sight` costs 1),
S=0.73, and positive path gain. It is admitted immediately before a complete
instance. This is insufficient evidence for a separate full instance, not a
CTC model failure. The vocals extra starts at 120.7125, while the subsequent
complete instance starts at 123.18 versus reference 126.00. More permissive
matching can also change CTC grouping and neighbouring onsets.

Per-identity occurrence pairing maximizes match count before minimizing timing
error, without a time gate. Thus these are diagnostic counts, not verified recall;
hallucinated repeats can mask omissions. Recovered rows above were additionally
checked for proximity to their reference instances. Existing Blossom false
repeat and Within's separated-vocal loss remain visible rather than excluded.

## Cost and decision

Summed cached algorithm time over eleven views: original 32.280s, consistent
33.354s (+3.3%), weighted 37.875s (+17.3%). Model inference time added: zero.
These are single-run wall times including stitching, matching and CTC timing,
excluding cache/model-file loading. Per-stage times are in each result and the
summary JSON. This is not an end-to-end cold-model benchmark.

168 tests pass, including CLI/runtime policy propagation, shared-score formula,
inserted words, function-word errors, preserved timing-anchor costs, negation,
unknown words and repeated nonoverlapping occurrences. All 33 final replays
pass timestamp/order/no-word-reuse assertions.

The initial validation promoted only consistency because weighted costs also
added Blossom false positives. The user subsequently chose weighted as the
default for broader testing, accepting this recorded tradeoff. This promotion
does not change the validation results or imply those false positives were fixed.
No extra exception, recursive repair stage or song-specific threshold was added.

## Implementation corrections retained for audit

`scoring-fixed-v1` initially normalized by lyric-side mass, incorrectly rejecting
`we'll all stay together`; this was corrected to the original max-length basis.
`scoring-fixed-v2` also discounted correct function-word coverage, unintentionally
rejecting some unchanged content-word errors. The final implementation discounts
only edit penalties. Those preliminary artifacts remain available; neither is
the shipped policy. These were design corrections, not a parameter grid search,
but this set has informed development and is not a held-out accuracy estimate.
