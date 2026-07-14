# Probe ceiling + col(K_l) residency

Follow-up to the ghost-token lens's L20→L26 collapse
(`out/README-ghost.md`, `out/ghost_results.json`): the ghost lens recovers the
input token well through L20, then falls to 6% (L23) and 0% (L26). Two
explanations were on the table:

- **(A) erasure** — input-token identity is truly gone from the residual
  stream by L23–26.
- **(B) lens blindness** — identity is still linearly present but has rotated
  out of the column space of `K_l`, so a readout that can only score tokens
  through `K_l e_v` is blind to it even though the information is there.

Implementation: `out/probe_residency.py`; results: `out/probe_residency.json`.

## Setup

- Model: Qwen/Qwen3-0.6B (28 layers, d=1024), fp16 on MPS for the forward
  passes; all linear algebra (ridge solves, decodes, SVDs) in fp32 on CPU.
- Same 24 fit / 8 held-out WikiText-103 prompts, same `skip_first=16`
  position convention, same 9 layers {2,5,8,11,14,17,20,23,26} as
  `out/fit_ghost.py` / `out/eval_ghost.py` — one forward pass per prompt,
  all layers hooked at once, activations cached fp32 CPU
  (`out/probe_activations.pt`).
- **Experiment 1 (probe ceiling):** per layer, ridge regression
  `h_l -> e_actual_token` (the model's input-embedding vector of the token
  actually at that position), ~2,220 training positions (20 of the 24 fit
  prompts; the other 4 held out to pick the ridge strength by cosine on a
  validation split), decoded on the 888 held-out positions by nearest
  neighbor over the full 151,669-token input-embedding matrix. Both a cosine
  decode and the ghost-style score `<Wh,e_v> - 0.5||e_v||^2` are reported.
  Controls: a probe fit on shuffled labels, and the ghost lens's own top-1
  for reference.
- **Confound control (predictability stratification):** a late `h_p` also
  carries the model's *prediction* of token `p+1`; in ordinary text token `p`
  is often guessable from token `p-1`'s prediction plus topical continuity,
  which would let a probe "recover" input identity by inverting output
  statistics rather than reading stored identity — faking a world-B verdict.
  So every evaluated position also carries the model's own rank of the
  actual token under its final-layer prediction at position `p-1`. Buckets:
  **high-predictability** (rank ≤ 10, n=635/888) and **surprising** (rank >
  100, n=127/888). The decisive test is probe accuracy on the *surprising*
  bucket at L23/L26: if it holds up there, that's genuine stored identity
  (world B); if late-layer recovery is concentrated entirely in the
  predictable bucket, it's statistics inversion (world A in disguise).
- **Experiment 2 (col(K_l) residency):** `K_l` is square and numerically full
  rank, so raw column-space membership is vacuous. Instead: SVD
  `K_l = U S V^T`, residency `r(k) = ||P_k h||^2 / ||h||^2` for `P_k` =
  projection onto the top-`k` left singular vectors of `K_l`, `k` in
  {8,16,32,64,128,256,512,1024}, averaged over the 888 held-out positions.
  Baseline: the same curve for `k` random orthonormal directions (a single
  QR-orthonormalized Gaussian draw per k, expected `r(k) = k/1024`; reported
  as a 5-draw average). Effective rank = min `k` capturing 90%/99% of the
  squared singular-value mass.
- **Sanity check:** before trusting any of the above, layer-2 ghost top-1 was
  reproduced directly from the cached activations against `out/backward_lens.pt`'s
  `K_2` and compared to `out/ghost_results.json`. Reproduced 93.13% against a
  reference of 93.13% — exact match, confirming the activation/position
  extraction matches `eval_ghost.py`'s conventions.

## Results

**Ghost lens vs. probe ceiling** (all % on the 888 held-out positions; probe
uses cosine-nearest-neighbor decode unless noted):

| layer | ghost top-1 | ghost top-10 | probe top-1 (cos) | probe top-1 (ghost-style) | probe top-10 (cos) | shuffled control top-1 |
|------:|------------:|-------------:|-------------------:|--------------------------:|--------------------:|------------------------:|
| 2     | 93.1 | 98.3 | 44.9 | 29.7 | 95.4 | 2.5 |
| 5     | 61.0 | 87.4 | 37.0 | 28.6 | 85.2 | 2.9 |
| 8     | 48.9 | 74.0 | 32.0 | 25.3 | 77.8 | 2.1 |
| 11    | 65.4 | 94.3 | 32.9 | 21.5 | 73.5 | 2.6 |
| 14    | 56.2 | 88.6 | 31.3 | 23.0 | 73.5 | 1.9 |
| 17    | 73.5 | 95.2 | 29.8 | 24.4 | 74.0 | 2.1 |
| 20    | 37.5 | 64.2 | 33.8 | 24.8 | 78.5 | 2.4 |
| **23**| **6.0** | 16.9 | **32.2** | 22.5 | 72.7 | 2.9 |
| **26**| **0.0** | 0.2 | **27.9** | 19.5 | 65.3 | 2.0 |

**Predictability-stratified probe top-1 (cosine decode)**, n=635
high-predictability / n=127 surprising out of 888 total:

| layer | overall | high-pred. (rank≤10) | surprising (rank>100) | surprising, shuffled control |
|------:|--------:|----------------------:|------------------------:|------------------------------:|
| 2     | 44.9 | 46.9 | 40.2 | 0.0 |
| 5     | 37.0 | 40.3 | 28.3 | 0.8 |
| 8     | 32.0 | 35.4 | 23.6 | 0.0 |
| 11    | 32.9 | 37.2 | 19.7 | 0.8 |
| 14    | 31.3 | 35.4 | 18.1 | 0.0 |
| 17    | 29.8 | 33.7 | 17.3 | 0.0 |
| 20    | 33.8 | 37.6 | 18.9 | 0.0 |
| **23**| 32.2 | 36.2 | **18.9** | 0.0 |
| **26**| 27.9 | 33.4 | **8.7** | 0.0 |

**col(K_l) residency at the effective rank**, held-out positions:

| layer | eff. rank 90% | eff. rank 99% | r(k=eff99) signal | r(k=eff99) random baseline |
|------:|---------------:|---------------:|--------------------:|-----------------------------:|
| 2     | 287 | 501 | 0.715 | 0.495 |
| 5     | 118 | 354 | 0.721 | 0.346 |
| 8     | 90  | 261 | 0.678 | 0.254 |
| 11    | 76  | 164 | 0.683 | 0.159 |
| 14    | 77  | 157 | 0.666 | 0.150 |
| 17    | 70  | 136 | 0.562 | 0.130 |
| 20    | 73  | 150 | 0.506 | 0.150 |
| **23**| 79  | 187 | **0.638** | 0.188 |
| **26**| 79  | 214 | **0.762** | 0.214 |

(Full residency curves at k=8..1024 and the eff-rank-90 numbers are in
`out/probe_residency.json`.)

## Verdict: (B), lens blindness — with a real but partial decline, not pure erasure

Three independent pieces of evidence point the same way:

1. **The probe does not collapse where the ghost lens does.** Ghost top-1
   falls off a cliff at L23 (6.0%) and flatlines at L26 (0.0%). The
   ridge-probe ceiling on the exact same cached activations stays at
   32.2% / 27.9% top-1 (72.7% / 65.3% top-10) — close to its L8–L20 plateau
   (30–38%) — and 8–14x above the shuffled-label control (2–3%) at every
   layer including L26. If identity were truly gone from the residual stream,
   no linear readout, however well-fit, could do this.
2. **The surprising-token bucket rules out the statistics-inversion
   confound.** If deep-layer "recovery" were really the probe inverting
   next-token predictions rather than reading stored identity, it should
   vanish exactly on tokens the model couldn't have predicted from context
   (rank > 100 at p-1). It doesn't: 18.9% at L23 (vs. 0.0% shuffled, n=127) —
   indistinguishable from L20's 18.9%. L26 does show real decay here (8.7%,
   down from 17–40% at every earlier layer) — still ~9 points above the
   shuffled floor, but the smallest margin in the table, and the one place
   the data is consistent with partial erosion on top of lens blindness.
3. **h never leaves col(K_l)'s reachable subspace.** Residency at the
   effective rank (99% singular-value mass) is 0.5–0.76 at every fitted
   layer, 2.5–4x its random-direction baseline, with *no dip* at L23–26 —
   if anything L26's 0.762 is the highest value in the table. So the ghost
   collapse is not h rotating out of `K_l`'s column space in the bulk-energy
   sense: most of `h`'s norm is still expressible from that same ~150–500
   dimensional subspace at L23/L26 as at L11/L14. What breaks is something
   sharper than subspace membership — the residual's component along the
   *specific* per-vocab-token direction `K_l e_actual` that the ghost score
   needs, not its presence in the general subspace.

Put together: this is **world B**, with one caveat the data itself surfaces.
Identity is linearly decodable throughout — the ghost lens's 0% at L26 is a
property of *that particular readout* (anchored to vocabulary directions
`K_l e_v` fit from early-layer dynamics), not of the residual stream. But the
surprising-bucket numbers show a genuine, if modest, decline by L26 (8.7%),
so "identity is perfectly preserved forever" would overstate it — the
honest picture is lens blindness dominating, with a small additional
erosion component stacking on top of it by the last fitted layer.

## Caveats

- The probe ceiling (~30–45% top-1) is a different quantity from the ghost
  lens's raw score — it is *trained on the actual labels* with per-layer
  ridge tuning, so it's an upper bound a hindsight-fit linear readout can
  reach, not a claim that the model or any zero-shot lens actually uses this
  readout. The comparison to the ghost lens is about whether the *ceiling*
  moves with layer depth (it barely does) even though the ghost lens's score
  (an untrained, `K_l`-anchored readout) does.
- The surprising bucket is small (n=127 of 888); L26's 8.7% is ~11 positions
  — real signal against a near-zero shuffled floor, but not enough positions
  to further subdivide with confidence.
- Ridge strength was tuned per layer by validation cosine on 4 held-out fit
  prompts, then the final `W` was refit on all 24 fit-prompt positions at
  that lambda; the shuffled control reuses the same lambda (no separate
  tuning), which is conservative (a shuffled-label fit tuned for shuffled
  labels would do even worse, not better).
- Residency's random baseline is a 5-draw average of one random rotation's
  leading k columns per draw (nested by construction via QR), not a
  fully independent-subspace-per-k Monte Carlo; variance across draws was
  not separately reported but the signal-vs-baseline gaps (2.5–4x) are far
  larger than plausible draw-to-draw noise at n=888 positions.

## Per-position: displacement vs coexistence

Follow-up question (`out/quadrant_analysis.py`, `out/quadrant_results.json`):
at deep layers, are the positions where the ghost lens *still* works the ones
where the output basis hasn't been written yet — i.e. is ghost survival just
"pseudo-junk on the forward side"? Per held-out position at layers
{14,17,20,23,26}: ghost rank of the actual token ("ghost-survives" = rank <
10, the README-ghost result-5 threshold), forward-lens top-1 == model final
top-1 ("output-converged", the eval_ghost crossover convention, plus the
graded rank), probe rank of the actual token (cosine decode, same ridge
lambdas as above), ‖h‖, and the ghost score decomposition s_actual =
⟨h,K e_a⟩−½‖K e_a‖² vs s_top1 = the best *competing* ghost score. Everything
reuses the cached activations; the only new model work was one forward pass
per eval prompt for final-layer logits.

**Quadrant tables** (per-cell: n / probe top-1 / mean ‖h‖ / mean margin =
s_actual − s_top1):

L20 — 568/888 ghost-survivors, P(converged) overall = 0.13:

| | converged | not converged |
|---|---|---|
| ghost survives | 57 / 22.8% / 185 / −30 | 511 / 28.4% / 189 / +108 |
| ghost dead     | 58 / 51.7% / 181 / −379 | 262 / 42.7% / 182 / −380 |

L23 — 149/888 survivors, P(converged) overall = 0.36:

| | converged | not converged |
|---|---|---|
| ghost survives | 41 / 17.1% / 333 / −94 | 108 / 15.7% / 341 / −16 |
| ghost dead     | 275 / 35.6% / 340 / −2237 | 464 / 35.3% / 357 / −2477 |

L26 — 2/888 survivors (uninformative; both quadrant rows are the ghost-dead
row: 625 converged / 261 not).

The decisive cell: **P(converged | ghost-survives) = 0.28 at L23 vs 0.36
overall** (0.10 vs 0.13 at L20). Ghost-survivors are only *mildly* less
output-converged than average — a weak displacement tendency (ratio ~0.77),
nowhere near "survivors = positions the output hasn't been written to."
28% of L23 ghost-survivors carry a fully converged forward readout at the
same position: input-identity and output-prediction signals demonstrably
coexist. A genuinely surprising inversion: at L23 the *probe* is roughly
twice as accurate on ghost-dead positions (35%) as on ghost-survivors
(16–17%) — deep-layer ghost survival does not even mark the positions where
linear identity is strongest; it marks positions where K_l's specific
readout directions happen to still align.

**Shouted-over decomposition** (means over all 888 positions; "gfpo" = the
262 positions where ghost fails at L23 but the probe still decodes the
token):

| layer | s_actual | s_top1 (competitor) | ‖h‖ | s_actual/‖h‖ | s_top1/‖h‖ | gfpo s_actual |
|------:|---------:|--------------------:|-----:|--------------:|------------:|--------------:|
| 14 | 158 | 153 | 53 | 2.9 | 2.8 | 116 |
| 17 | 451 | 412 | 95 | 4.7 | 4.3 | 311 |
| 20 | 945 | 1021 | 186 | 4.9 | 5.4 | 458 |
| 23 | 978 | 2971 | 349 | 2.8 | 8.5 | 309 |
| 26 | 512 | 8481 | 602 | 0.8 | 14.0 | 546 |

The identity component does **not** collapse: in absolute terms s_actual
*rises* through L23 (158 → 978) and is still ~512 at L26 — even on the gfpo
subset it is essentially flat (116 → 546, no monotone decay). What kills the
ghost readout is the competitor blow-up: s_top1 grows 3x from L20 to L23 and
another ~3x to L26 (to ~8500, vs s_actual ~500). Per unit ‖h‖ both effects
are visible — identity alignment sags ~3–6x after L20 while the loudest
competitor direction grows ~3x — but the dominant term at the collapse point
is the shouting, not the shrinking.

**Coupling, probe-based** (Pearson, both sides as −log10(rank+1), so
positive = identity-strong positions are also more output-converged):
+0.088 at L20, +0.012 at L23. The same-layer, same-metric coupling computed
with the old *ghost*-based identity measure is −0.194 / −0.051 — i.e. the
sign flips once the identity measure is unconfounded from K_l. (Caution on
comparability: README-ghost result 3's "−0.25 at L20" was a
held-until/settles-at *timing* construction from `pivot_analysis.py`, not
this same-layer graded rank correlation; the honest statement is that with
the probe-based measure the identity↔convergence coupling is essentially
zero to weakly positive — no sign of the strong negative coupling
displacement would predict.)

**Verdict on the displacement hypothesis:** not supported. Ghost survival at
deep layers is only weakly associated with an unwritten forward readout
(0.28 vs 0.36 base rate), a quarter-to-third of survivors are fully
converged positions, the probe finds identity as well or *better* where the
ghost has died, and the score decomposition shows the ghost collapse is
competitors outgrowing a still-present identity component rather than that
component vanishing. Coexistence — the two signals stack in the same
residual vector, and the ghost lens loses the input token because other
directions in K_l's image get louder, not because the identity was displaced
to make room for the output. Strongest caveat: "output-converged" is itself
a mean-Jacobian (forward-lens) readout with a 13–36% base rate at L20–23 and
subject to the very lens-blindness this experiment diagnosed on the backward
side, and the decisive L23 cell holds only 41 positions; a
displacement-shaped effect hiding below that resolution, or invisible to the
forward lens, can't be ruled out.

## Ghost-death as a workspace marker

Third question (`out/workspace_marker.py`, `out/workspace_results.json`):
does ghost-death at a position mark the *arrival* of meaningful output-basis
("workspace") content there — not necessarily the final prediction, just
semantic information about where the text is going? Mechanical motivation:
the quadrant analysis showed ghost-death = the competing K-space score
exploding, and `ghost_patch.json` showed the loud junk is output-flavored
causally. Two informativeness measures on the forward-lens top-5 readout,
per (position, layer) at layers {8,…,26}: **future-overlap** (any top-5
token literally among the next 32 document tokens, excluding the immediate
next token to separate planning from next-token prediction) and **semantic
future-similarity** (mean over top-5 of max cosine, input-embedding space,
to the next-32 window; null floor = the identical statistic against the
matched-position window of a different eval prompt).

**Cross-sectional** (ghost-dead = rank ≥ 1000, ghost-alive = rank < 10;
sim / overlap-excl-next, null in parentheses):

| layer | ghost-dead | ghost-alive | probe-dead sim | probe-alive sim | null |
|------:|-----------:|------------:|---------------:|----------------:|-----:|
| 8  | n=13: .203 / .00 | n=648: .196 / .09 | .208 | .207 | .184 / .10 |
| 11 | n=1 | n=829: .199 / .10 | .216 | .196 | .170 / .06 |
| 14 | n=2 | n=774: .199 / .10 | .214 | .194 | .169 / .06 |
| 17 | n=1 | n=838: .208 / .09 | .223 | .204 | .169 / .06 |
| 20 | n=77: .248 / .26 | n=562: .235 / .19 | .235 | .242 | .184 / .11 |
| 23 | n=570: .301 / .36 | n=147: .241 / .24 | .270 | .294 | .218 / .20 |
| 26 | n=861: .365 / .56 | n=2 | .372 | .364 | .302 / .45 |

At face value the key cell delivers: L23 ghost-dead positions read out
future-informative content well above both the null (.301 vs .218) and the
ghost-alive positions at the same layer (.241). But the predictability
control deflates much of the dead/alive *gap*: every one of the 77
ghost-dead positions at L20 is a high-predictability position (the
surprising bucket is empty — ghost-death starts at predictable positions),
and within the high-predictability bucket the L20 gap vanishes entirely
(dead .248 vs alive .249). At L23 a within-bucket gap survives for the
semantic measure (dead .311 over a .228 null = +.083, vs alive .262 over
.222 = +.040) but not for literal overlap (.37 vs .36). The probe split
confirms the quadrant-analysis lesson: probe-dead vs probe-alive barely
differ (L23: .270 vs .294, probe-*alive* slightly higher) — losing linear
identity is not what marks informativeness.

**Event alignment** (the decisive test). Ghost-death layer per position by
`transition_analysis.py`'s convention (held rank < 10, then first sampled
layer with rank > 1000), on the full {2,…,26} grid: defined at 861/888
positions, but massively concentrated — 69 die at L20, 495 at L23, 293 at
L26. Workspace-onset layer = first layer whose semantic similarity exceeds
that layer's null mean + kσ:

- corr(death layer, onset layer) = **+0.02 (k=1, n=452), +0.02 (k=2,
  n=293), −0.22 (k=3, n=147)** — nothing stable, and what signal exists at
  the strictest threshold points the wrong way for the hypothesis.
- Event-aligned average of null-corrected similarity vs a shuffled-death
  control (death layers permuted across positions, 20 draws):

| rel. layer | −12 | −9 | −6 | −3 | 0 | +3 | +6 |
|---|---|---|---|---|---|---|---|
| true | +.028 | +.031 | +.037 | +.057 | +.068 | +.080 | +.004 |
| shuffled | +.029 | +.033 | +.045 | +.061 | +.067 | +.061 | +.063 |

No step at relative layer 0: the true curve is indistinguishable from the
shuffled one through the death event (.068 vs .067 at 0), i.e. the entire
apparent rise toward death is reproduced when each position is assigned a
random other position's death layer — it is the global depth profile, not a
per-position event. The only true-vs-shuffled excess is a small one *after*
death (+.080 vs +.061 at rel +3), and the rel +6 cell (n=73, dominated by
the early L20 deaths measured at L26) actually runs *below* shuffled.

**Verdict:** the loud content is real but the flag is not personal. At the
population level the user's intuition holds — by L23 the forward readout at
ghost-dead positions carries genuinely future-relevant semantic content,
above the cross-prompt null and (for the semantic measure, within the
predictability control) above still-alive positions. But ghost-death is not
a usable *per-position* marker for "workspace content has arrived here":
death layers barely vary (three adjacent sampled layers hold 99% of
events), their correlation with per-position workspace onset is ~0, and the
event-aligned curve shows no step that survives the shuffled-death control.
Ghost-death timing is a global phase-transition clock (consistent with
README-ghost result 3), and conditioning on it per position adds almost
nothing beyond knowing the layer index. Strongest caveat: with 95% of
deaths packed into {20, 23, 26} and onset measured on the same coarse
3-layer grid, the design has very little per-position variance to work
with — a fine-grained (every-layer) rank trajectory could still reveal
alignment that this 3-layer sampling cannot resolve, so the negative
event-alignment result is best read as "no evidence at this resolution",
not "proven absent".
