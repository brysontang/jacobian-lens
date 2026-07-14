# Ghost-token (backward) lens experiment

The inverse of the Jacobian lens: instead of asking what an activation is
disposed to make the model *say* (forward transport to the unembedding), ask
what input token the activation looks like it *came from* (transport of the
vocabulary through `K_l = E[∂h_l/∂emb]`, nearest forward-image readout).
Implementation: `jlens/backward.py`; tests: `tests/test_backward.py`.

The question this answers (2026-07-06 discussion): after self-attention mixes
tokens together, what is each residual-stream position *holding* — is the
model rewriting/rearranging its input in place? The forward lens can't answer
this because it projects everything onto "next token".

## Setup

- Model: Qwen/Qwen3-0.6B (dense, 28 layers, d=1024), fp16 on MPS (M1 Pro).
  Qwen3.5-2B was benchmarked and rejected: fp16 gradients NaN past layer 20
  on its hybrid linear-attention torch fallback, and >24 h fit time.
- Both lenses fitted at layers {2, 5, 8, 11, 14, 17, 20, 23, 26} on the first
  24 WikiText-103 prompts (seq 128, dim_batch 16). `out/fit_ghost.py`,
  checkpointed. Eval on the following 8 prompts (held out).
- Reverse mode cannot amortise across target layers (mirror-asymmetry with
  the forward fit) — the v1 fix is forward-mode JVP, which would give all
  layers in one pass-set.

## Files

- `fit_ghost.py` — fits backward + forward lenses (run from repo root).
- `eval_ghost.py` — retention/crossover curves on held-out WikiText, ghost
  grids for the ASCII face + a French prompt, cross-position (self/moved/
  novel) stats. Writes `ghost_results.json`.
- `pivot_analysis.py` — per-position controls: does hold duration predict
  settle timing? Writes/reads `pivot_data.json` (`--analyze` reuses cache).
- `ghost_patch.py` — causal test: substitute a position's L23 ghost top-1
  for its actual token, measure prediction preservation vs shuffled-ghost
  and random controls. Writes `ghost_patch.json`.
- `transition_analysis.py` — jump-vs-slide: per-position wall-to-wall
  transit widths and timing on both axes (cached data only).
- `ghost-lens-results.html` — the self-contained interactive results page
  (all data inlined; open locally).
- `fit_offset.py` — fits the offset-resolved K_Δ lens (see result 6).
- `eval_offset.py` — K_Δ distance-selectivity matrices, top-1 provenance,
  French-prompt profile grids. Writes `offset_results.json`.
- `offset_deflate.py` — explaining-away control: re-scores Δ>0 bins after
  subtracting the self component K₀e_self. Appends into `offset_results.json`.
- `backward_lens.pt`, `forward_lens.pt`, `offset_lens.pt` — fitted lenses
  (not committed, 18–88 MB; reproduce with `fit_ghost.py` / `fit_offset.py`).
- `probe_residency.py`, `quadrant_analysis.py`, `workspace_marker.py` —
  follow-up experiments (2026-07-14) reinterpreting the collapse; results in
  `probe_residency.json` / `quadrant_results.json` / `workspace_results.json`,
  written up in `README-probe.md`.

## Results (2026-07-07)

**1. Retention/crossover** (`eval_ghost.py`, 888 held-out positions): ghost
top-1 recovers the actual input token 93% at L2, 49–74% through L20, then 6%
(L23) and 0% (L26); the forward lens's match with the model's final top-1
rises in mirror image (13% → 36% → 70%). Crossover between L20 and L23
(~75–80% depth). The no-Jacobian baseline is ~0% at every layer. The curve is
non-monotonic: a dip at L8 before recovering through L17 (early
detokenization?).

**2. No rearrangement.** Ghost top-1 is a token from elsewhere in the prompt
at ≤4% of positions at every layer (both showcase prompts). Identity is lost
by dissolution into out-of-prompt tokens, never by permutation. The ASCII
face reconstructs verbatim through L20; a French prompt drifts
cross-lingually where it diverges (avec→WITH, marchands→merchants; at L23 a
`mercado` attractor appears at positions predicting market vocabulary).

**3. The handoff is a global phase transition, not a per-token trigger**
(`pivot_analysis.py`). Population-level sequencing is robust (mean identity
release ~L19, mean prediction lock-in ~L25), but per-position hold duration
does not predict per-position settle timing: partial correlation +0.06–0.09
controlling for continuation-vs-new-word and prediction confidence. At
matched layers the coupling is *negative* (−0.25 at L20): positions holding
their input token well are also closer to converged — consistent with a
common per-position "linearization quality" factor. Word-fragments whose
prediction is their own continuation hold identity longer (the spelling
effect), but modestly on English text.

**4. Deep-layer ghost junk is output-flavored, gradedly** (`ghost_patch.py`,
320 diverged positions × 3 conditions). Replacing a position's actual input
token with its L23 ghost top-1 — the edit is to the *prompt*, then a full
rerun — preserves the model's prediction 15.0% vs 9.4%
(shuffled ghost, vocabulary-matched) vs 6.9% (random); paired z = 2.7 / 3.8.
Stratified: mildly dissolved positions (actual's ghost rank < 100) preserve
at 33% with median rank 4 — those ghosts are functional synonyms. Deeply
dissolved junk (rank ≥ 100) barely beats control (9.4% vs 8.2%): by L23 the
state has left the input manifold and the nearest input token is a lossy
shadow, not a causal stand-in.

**5. Release is a snap; convergence is a climb** (`transition_analysis.py`).
97% of input-identity wall-to-wall trips (ghost rank <10 → >1000) complete
within a six-layer window (52% within three), departures synchronized at
L17–20. Output lock-in transits are 3–15 layers wide with departures
staggered from L2. Ordering at 749 positions with both transitions: output
locks first 50%, same layer 28%, input dissolves first 22% — the model
mostly doesn't release a token's identity until its prediction is settled.

Metric-robustness: deep-layer dissolution is identical under l2, dot, and
cosine readouts (single-prompt probe) — not an artifact of the norm term.

**6. The mean transport has no "from where"** (`jlens/offset.py`,
`fit_offset.py`, `eval_offset.py`). The fitted K sums over target-position
offsets, blind to *where* information came from; the offset-resolved family
`K_Δ = E[∂h_l[p+Δ]/∂emb[p]]` (cotangent seeded at a single target per
backward pass, gradients binned by lookback distance; 3 layers × 14 bands,
16 prompts, ~6.5 h) was meant to fix that. Findings, held-out eval:

- **Norms**: the Δ=0 self-path dominates everywhere but grows only 2.7×
  from L14→L23 while every cross-position band roughly triples — relative
  neighbor-transported mass rises into the handoff. Non-monotonic bump at
  Δ=12–15; uptick in the far band (96–127).
- **Selectivity (the null)**: scoring each band's readout against the token
  exactly d back, normalized by a shuffled-target control: the only strong
  column is d=0 — every band leaks the *self* token (L14 Δ=1 band: self 19×
  above chance, its own matched neighbor 1.4×). All K_Δ images share one
  token-content channel; the score matches whatever is loudest in h. Top-1
  of every Δ>0 band is ~100% out-of-prompt junk.
- **The one echo**: adjacent tokens at L20 — 2.9× (Δ=1), 2.0× (Δ=2) — peak
  exactly in the migration band, gone by L23.
- **Not masked, absent**: deflating h′ = h − K₀e_self (oracle explaining-away
  of the self channel) changes nothing (2.9× → 2.8×). The neighbor signal
  isn't hiding under self; it isn't linearly there in the mean.
- **Far-band norms are content-free**: ratios ≤1.0 at Δ≥32 — a sink-like
  pathway, loud in gradient, silent in token-specific signal (BOS itself was
  excluded by skip_first=16).

Verdict: *what — itself; from where — unattributable in the mean.* Attention
routing is content-dependent, so averaging over prompts/positions destroys
source attribution. This extends "dissolution, not permutation" to the
transport domain: even mid-migration a position never becomes readable as
its neighbors.

## Follow-up (2026-07-14): the collapse reinterpreted

The L23–26 collapse in result 1 turned out to be a property of the *readout*,
not the information: a trained linear probe on the same activations recovers
input identity 32%/28% top-1 at L23/L26 (surviving a predictability control),
the actual token's ghost score *rises* through L23 while the best competing
score grows ~50×, and per-position analysis rejects displacement in favor of
identity and output signals coexisting in the same residual vector.
Ghost-death is a global phase clock whose "junk" readouts are genuinely
future-informative at the population level, but not a per-position workspace
marker at this fit resolution. Full setup, tables, and caveats:
`README-probe.md`.

## Next: per-prompt Jacobians

Per-position pivot attribution needs the *unaveraged* `∂h_l[t]/∂emb[p]` on a
specific prompt — seed a handful of handoff-band targets (e.g. the mercado
positions) and read the true per-source decomposition. Expensive per
position, cheap in total: a dozen targets, not a corpus.
