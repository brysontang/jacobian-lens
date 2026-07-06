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

- Model: Qwen/Qwen3.5-2B (hybrid linear-attention; d=2048, 24 layers, tied
  embeddings, vocab 248320), fp16 on MPS (M1 Pro 32GB).
- Both lenses fitted at layers {2, 6, 10, 14, 18, 22} on the first 24
  WikiText-103 prompts (seq 128). `out/fit_ghost.py`, checkpointed.
  dim_batch set from the MPS benchmark (scratchpad bench_backward.py).
- Reverse mode cannot amortise across target layers (mirror-asymmetry with
  the forward fit) — the v1 fix is forward-mode JVP, which would give all 24
  layers in one pass-set.

## Files

- `fit_ghost.py` — fits backward + forward lenses (run from repo root).
- `eval_ghost.py` — retention/crossover curves on held-out WikiText, ghost
  grids for the ASCII face + a French prompt, cross-position (self/moved/
  novel) stats. Writes `ghost_results.json`.
- `backward_lens.pt`, `forward_lens.pt` — fitted lenses (gitignored).

## Verified so far

Estimator validated exactly against analytic Jacobians on the attention-free
TinyDecoder (`tests/test_backward.py`, 6 tests; full suite 38 passing).
End-to-end sanity on Qwen (ghost readout of a single-prompt lens) is part of
the benchmark script and should be read from its output, not assumed.
