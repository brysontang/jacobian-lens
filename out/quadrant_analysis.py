"""Per-position displacement vs coexistence: do ghost-survivors at deep
layers sit exactly where the output basis hasn't been written yet?

Follow-up to out/probe_residency.py (world-B verdict: the ghost collapse at
L23-26 is lens blindness, not erasure). The remaining per-position question:
at deep layers, are the positions where the ghost lens STILL recovers the
input token the ones whose forward readout has NOT yet converged on the
model's final prediction ("the output hasn't been written here yet"), or do
input-identity and output-prediction signals coexist at the same positions?

Per held-out position (888) at layers {14, 17, 20, 23, 26}:
  ghost side   — rank of the actual input token under the ghost score
                 (K_l from out/backward_lens.pt, l2 metric, exactly
                 BackwardLens.ghost_logits); "ghost-survives" = rank < 10,
                 the same threshold as README-ghost result 5 /
                 pivot_analysis.held_until.
  forward side — forward-lens top-1 (out/forward_lens.pt, transport +
                 unembed, eval_ghost.py conventions) == the model's own
                 final-layer top-1 at that position ("output-converged"),
                 plus the graded version: rank of the final top-1 token in
                 the forward readout.
  probe side   — the ridge probe from probe_residency.py (refit at the
                 per-layer best lambda recorded in out/probe_residency.json),
                 cosine decode: top-1 correct + rank of the actual token.
  magnitudes   — ||h||; s_actual = <h, K e_a> - .5||K e_a||^2; s_top1 = the
                 best *competing* ghost score (max over v != actual);
                 margin = s_actual - s_top1 (positive iff ghost top-1 correct).

Outputs (out/quadrant_results.json):
  quadrants    — at L20/L23/L26, counts in {ghost-survives x output-converged}
                 with per-quadrant probe top-1, mean ||h||, mean margin; the
                 decisive cell: among ghost-survivors, the output-converged
                 fraction (high -> coexistence, low -> displacement).
  trajectory   — mean s_actual / s_top1 / ||h|| across L14->L26, overall and
                 for the "ghost fails at L23 but probe succeeds" subset: did
                 the identity component shrink, or was it shouted over?
  correlation  — at L20/L23: pearson corr of probe identity strength
                 (-log10(probe rank + 1)) vs forward convergence
                 (-log10(fwd rank of final top-1 + 1)); the probe-based,
                 confound-free replacement for README-ghost result 3's
                 ghost-based -0.25 coupling. Positive = identity-strong
                 positions are also MORE converged (coexistence).

Reuses out/probe_activations.pt (written by probe_residency.py); the only
model work is one cheap forward pass per eval prompt for final-layer logits
plus the forward-lens unembeds, all cached to out/quadrant_cache.pt.

Run from repo root: uv run python out/quadrant_analysis.py
Analysis-only re-run (no model): uv run python out/quadrant_analysis.py
(the cache short-circuits it); force recompute with --force.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SKIP = 16
N_FIT = 24
N_EVAL = 8
MAX_SEQ_LEN = 128
LAYERS_Q = [14, 17, 20, 23, 26]
QUADRANT_LAYERS = [20, 23, 26]
MODEL = "Qwen/Qwen3-0.6B"
VOCAB_CHUNK = 32768
GHOST_SURVIVES_RANK = 10  # README-ghost result 5 / pivot_analysis convention

ACT_CACHE = "out/probe_activations.pt"
CACHE = "out/quadrant_cache.pt"
RESULTS = "out/quadrant_results.json"
PROBE_RESULTS = "out/probe_residency.json"


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    d = X.shape[1]
    return torch.linalg.solve(
        X.T @ X + lam * torch.eye(d, dtype=X.dtype), X.T @ Y
    )


def ghost_stats(
    h: torch.Tensor,
    K: torch.Tensor,
    embed_weight: torch.Tensor,
    actual: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Chunked ghost readout (l2 metric, identical to
    BackwardLens.ghost_logits): per position, the actual token's rank and
    score, the best competing score/token, and the overall argmax."""
    g = h @ K  # [n, d]: <h, K e_v> == <g, e_v>
    Kgram = K.T @ K
    n = h.shape[0]

    actual_rows = embed_weight[actual]
    s_actual = (g * actual_rows).sum(-1) - 0.5 * (
        (actual_rows @ Kgram) * actual_rows
    ).sum(-1)

    rank = torch.zeros(n, dtype=torch.long)
    best_comp = torch.full((n,), -torch.inf)
    best_comp_tok = torch.zeros(n, dtype=torch.long)
    offset = 0
    for rows in embed_weight.split(VOCAB_CHUNK):
        sq = ((rows @ Kgram) * rows).sum(-1)  # ||K e_v||^2 for this chunk
        scores = g @ rows.T - 0.5 * sq[None, :]  # [n, chunk]
        rank += (scores > s_actual[:, None]).sum(-1)
        # best competitor: mask out the actual token where it lives here
        in_chunk = (actual >= offset) & (actual < offset + rows.shape[0])
        if in_chunk.any():
            idx = in_chunk.nonzero(as_tuple=True)[0]
            scores[idx, actual[idx] - offset] = -torch.inf
        chunk_best, chunk_arg = scores.max(-1)
        better = chunk_best > best_comp
        best_comp[better] = chunk_best[better]
        best_comp_tok[better] = chunk_arg[better] + offset
        offset += rows.shape[0]

    ghost_top1 = torch.where(s_actual > best_comp, actual, best_comp_tok)
    return {
        "rank": rank,
        "s_actual": s_actual,
        "s_top1_competitor": best_comp,
        "ghost_top1": ghost_top1,
    }


def probe_ranks_cosine(
    pred: torch.Tensor, embed_weight: torch.Tensor, actual: torch.Tensor
) -> torch.Tensor:
    """Rank (0 = top) of actual[i] under cosine of pred[i] vs every vocab row
    (same decode as probe_residency.py)."""
    qn = pred / pred.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    actual_rows = embed_weight[actual]
    actual_score = (qn * (actual_rows / actual_rows.norm(dim=-1, keepdim=True))).sum(-1)
    counts = torch.zeros(pred.shape[0], dtype=torch.long)
    for rows in embed_weight.split(VOCAB_CHUNK):
        rows_n = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        counts += ((qn @ rows_n.T) > actual_score[:, None]).sum(-1)
    return counts


# ------------------------------------------------------------- collection


def collect(device: str) -> dict:
    """Everything per-position the analysis needs, cached to CACHE."""
    data = torch.load(ACT_CACHE, map_location="cpu", weights_only=True)
    ev = data["eval"]
    fit = data["fit"]
    embed_weight = data["embed_weight"]
    actual = ev["actual"]
    prompt_idx = ev["prompt_idx"]

    from jlens.backward import BackwardLens

    import jlens

    backward = BackwardLens.load("out/backward_lens.pt")
    forward = jlens.JacobianLens.load("out/forward_lens.pt")
    best_lambda = {
        int(l): v["best_lambda"]
        for l, v in json.load(open(PROBE_RESULTS))["probe"].items()
    }

    # ---- model-dependent part: final-layer logits + forward-lens readout
    import transformers

    from jlens.examples import load_wikitext_prompts
    from jlens.hooks import ActivationRecorder

    jlens.configure_logging()
    print(f"loading {MODEL} on {device}...", flush=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=device
    )
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    final_layer = model.n_layers - 1

    eval_prompts = load_wikitext_prompts(N_FIT + N_EVAL)[N_FIT:]
    final_top1_chunks = []
    for i, prompt in enumerate(eval_prompts):
        t0 = time.perf_counter()
        ids = model.encode(prompt, max_length=MAX_SEQ_LEN)
        with torch.no_grad(), ActivationRecorder(
            model.layers, at=[final_layer]
        ) as rec:
            model.forward(ids)
            h_final = rec.activations[final_layer][0].detach()
        n = ids.shape[1]
        pos = torch.arange(SKIP, n - 1)
        ids_cpu = ids[0].cpu()
        # sanity: this prompt's tokens must match the cached actual slice
        cached = actual[prompt_idx == i]
        if not torch.equal(ids_cpu[pos], cached):
            raise RuntimeError(
                f"eval prompt {i}: tokenization mismatch vs {ACT_CACHE}"
            )
        model_logits = model.unembed(h_final).float().cpu()  # [seq, vocab]
        final_top1_chunks.append(model_logits[pos].argmax(-1))
        print(
            f"  [final] prompt {i+1}/{N_EVAL} ({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )
    final_top1 = torch.cat(final_top1_chunks)  # [888]

    per_layer: dict[int, dict[str, torch.Tensor]] = {}
    for layer in LAYERS_Q:
        t0 = time.perf_counter()
        h = ev["h"][layer]  # [888, d] fp32 cpu

        # forward-lens readout (eval_ghost conventions: transport + unembed)
        with torch.no_grad():
            transported = forward.transport(h.to(model.input_device), layer)
            fwd_logits = model.unembed(transported).float().cpu()  # [888, vocab]
        fwd_top1 = fwd_logits.argmax(-1)
        picked = fwd_logits.gather(1, final_top1[:, None])
        fwd_rank = (fwd_logits > picked).sum(-1)
        del fwd_logits

        # ghost readout
        gs = ghost_stats(h, backward.K[layer], embed_weight, actual)

        # probe readout (refit at recorded lambda on all 24 fit prompts,
        # exactly probe_residency.py's final refit)
        W = ridge_fit(fit["h"][layer], embed_weight[fit["actual"]], best_lambda[layer])
        probe_rank = probe_ranks_cosine(h @ W, embed_weight, actual)

        per_layer[layer] = {
            "ghost_rank": gs["rank"],
            "s_actual": gs["s_actual"],
            "s_top1_competitor": gs["s_top1_competitor"],
            "ghost_top1": gs["ghost_top1"],
            "h_norm": h.norm(dim=-1),
            "fwd_top1": fwd_top1,
            "fwd_rank_final_top1": fwd_rank,
            "probe_rank": probe_rank,
        }
        print(
            f"  [layer {layer}] ghost/fwd/probe done "
            f"({time.perf_counter()-t0:.0f}s)",
            flush=True,
        )

    return {
        "final_top1": final_top1,
        "actual": actual,
        "prompt_idx": prompt_idx,
        "per_layer": per_layer,
    }


# --------------------------------------------------------------- analysis


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="mps")
    ap.add_argument(
        "--force", action="store_true", help="recompute out/quadrant_cache.pt"
    )
    args = ap.parse_args()

    if not Path(ACT_CACHE).exists():
        raise SystemExit(f"{ACT_CACHE} missing; run out/probe_residency.py first")

    if not args.force and Path(CACHE).exists():
        print(f"loading cached per-position stats from {CACHE}", flush=True)
        cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    else:
        cache = collect(args.device)
        torch.save(cache, CACHE)
        print(f"wrote {CACHE}", flush=True)

    actual = cache["actual"]
    final_top1 = cache["final_top1"]
    per_layer = cache["per_layer"]
    n = len(actual)

    # ------------------------------------------------- quadrant tables
    print("\n=== quadrants: ghost-survives x output-converged ===", flush=True)
    quadrants = {}
    for layer in QUADRANT_LAYERS:
        d = per_layer[layer]
        survives = d["ghost_rank"] < GHOST_SURVIVES_RANK
        converged = d["fwd_top1"] == final_top1
        probe_correct = d["probe_rank"] == 0
        margin = d["s_actual"] - d["s_top1_competitor"]

        cells = {}
        for s_flag, s_name in ((True, "ghost_survives"), (False, "ghost_dead")):
            for c_flag, c_name in ((True, "converged"), (False, "not_converged")):
                m = (survives == s_flag) & (converged == c_flag)
                cnt = int(m.sum())
                cells[f"{s_name}/{c_name}"] = {
                    "n": cnt,
                    "probe_top1_acc": float(probe_correct[m].float().mean())
                    if cnt
                    else None,
                    "mean_h_norm": float(d["h_norm"][m].mean()) if cnt else None,
                    "mean_margin": float(margin[m].mean()) if cnt else None,
                }
        n_surv = int(survives.sum())
        frac_conv_given_surv = (
            float(converged[survives].float().mean()) if n_surv else None
        )
        quadrants[layer] = {
            "cells": cells,
            "n_ghost_survives": n_surv,
            "n_converged": int(converged.sum()),
            "frac_converged_overall": float(converged.float().mean()),
            "frac_converged_given_ghost_survives": frac_conv_given_surv,
        }
        print(f"  L{layer}: n_survives={n_surv}/{n}", flush=True)
        for name, c in cells.items():
            print(
                f"    {name:30s} n={c['n']:4d} probe_top1="
                f"{'--' if c['probe_top1_acc'] is None else f'{c['probe_top1_acc']*100:5.1f}%'} "
                f"mean||h||={'--' if c['mean_h_norm'] is None else f'{c['mean_h_norm']:6.1f}'} "
                f"mean_margin={'--' if c['mean_margin'] is None else f'{c['mean_margin']:8.1f}'}",
                flush=True,
            )
        print(
            f"    P(converged) overall = {quadrants[layer]['frac_converged_overall']:.2f}; "
            f"P(converged | ghost-survives) = "
            f"{'--' if frac_conv_given_surv is None else f'{frac_conv_given_surv:.2f}'}",
            flush=True,
        )

    # -------------------------------------- shouted-over decomposition
    print("\n=== s_actual vs s_top1 trajectory (L14 -> L26) ===", flush=True)
    d23 = per_layer[23]
    ghost_fails_probe_ok = (d23["ghost_rank"] >= GHOST_SURVIVES_RANK) & (
        d23["probe_rank"] == 0
    )
    n_gfpo = int(ghost_fails_probe_ok.sum())
    trajectory = {"n_ghost_fails_probe_ok_at_L23": n_gfpo, "layers": {}}
    for layer in LAYERS_Q:
        d = per_layer[layer]
        sub = ghost_fails_probe_ok
        trajectory["layers"][layer] = {
            "mean_s_actual": float(d["s_actual"].mean()),
            "mean_s_top1_competitor": float(d["s_top1_competitor"].mean()),
            "mean_margin": float((d["s_actual"] - d["s_top1_competitor"]).mean()),
            "mean_h_norm": float(d["h_norm"].mean()),
            "ghost_fails_probe_ok_at_L23": {
                "mean_s_actual": float(d["s_actual"][sub].mean()),
                "mean_s_top1_competitor": float(d["s_top1_competitor"][sub].mean()),
                "mean_h_norm": float(d["h_norm"][sub].mean()),
            },
        }
        t = trajectory["layers"][layer]
        print(
            f"  L{layer:2d}: s_actual={t['mean_s_actual']:9.1f} "
            f"s_top1={t['mean_s_top1_competitor']:9.1f} "
            f"margin={t['mean_margin']:8.1f} ||h||={t['mean_h_norm']:6.1f} | "
            f"gfpo(n={n_gfpo}): s_actual="
            f"{t['ghost_fails_probe_ok_at_L23']['mean_s_actual']:9.1f} "
            f"s_top1={t['ghost_fails_probe_ok_at_L23']['mean_s_top1_competitor']:9.1f}",
            flush=True,
        )

    # ---------------------------------------------------- correlation
    print("\n=== probe-identity vs forward-convergence coupling ===", flush=True)
    correlations = {}
    for layer in (20, 23):
        d = per_layer[layer]
        identity = -torch.log10(d["probe_rank"].float() + 1)
        convergence = -torch.log10(d["fwd_rank_final_top1"].float() + 1)
        r = corr(identity, convergence)
        # same measure with the old (confounded) ghost-based identity
        ghost_identity = -torch.log10(d["ghost_rank"].float() + 1)
        r_ghost = corr(ghost_identity, convergence)
        correlations[layer] = {
            "probe_identity_vs_convergence": r,
            "ghost_identity_vs_convergence": r_ghost,
            "convention": "-log10(rank+1) both sides; positive = "
            "identity-strong positions are more converged (coexistence)",
        }
        print(
            f"  L{layer}: corr(probe identity, convergence) = {r:+.3f}  "
            f"(ghost-based: {r_ghost:+.3f})",
            flush=True,
        )

    out = {
        "model": MODEL,
        "layers": LAYERS_Q,
        "n_positions": n,
        "ghost_survives_rank_lt": GHOST_SURVIVES_RANK,
        "output_converged_def": "forward-lens top-1 == model final-layer top-1",
        "quadrants": quadrants,
        "trajectory": trajectory,
        "correlations": correlations,
    }
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
