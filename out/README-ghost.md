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
- `fit_offset.py` — fits the offset-resolved K_Δ lens (see below).
- `backward_lens.pt`, `forward_lens.pt` — fitted lenses (not committed,
  ~18 MB each; reproduce with `fit_ghost.py`).

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
320 diverged positions × 3 conditions). Substituting a position's L23 ghost
top-1 for its actual token preserves the model's prediction 15.0% vs 9.4%
(shuffled ghost, vocabulary-matched) vs 6.9% (random); paired z = 2.7 / 3.8.
Stratified: mildly dissolved positions (actual's ghost rank < 100) preserve
at 33% with median rank 4 — those ghosts are functional synonyms. Deeply
dissolved junk (rank ≥ 100) barely beats control (9.4% vs 8.2%): by L23 the
state has left the input manifold and the nearest input token is a lossy
shadow, not a causal stand-in.

Metric-robustness: deep-layer dissolution is identical under l2, dot, and
cosine readouts (single-prompt probe) — not an artifact of the norm term.

## Next: offset-resolved K_Δ ("what, and from where")

The fitted K sums over target-position offsets (grad at emb position p mixes
`∂h_l[p']/∂emb[p]` for all p' ≥ p), so every readout above is blind to
*where* information came from. `jlens/offset.py` fits one matrix per lookback
distance, `K_Δ = E[∂h_l[p+Δ]/∂emb[p]]`, by seeding the cotangent at a single
target position per backward pass and binning gradients by distance. Decoding
a position against the K_Δ family gives a ghost profile over distance — the
instrument for cross-position pivot detection that experiments 3–4 lacked.
