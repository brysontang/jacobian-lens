"""Linear-probe ceiling + col(K_l) residency for the ghost lens.

Two follow-ups to the ghost-token lens's L20->L26 collapse
(out/README-ghost.md, out/ghost_results.json):

  (A) input-token identity is truly erased from the residual stream by L23-26
  (B) identity is still linearly present but has rotated out of col(K_l), so
      the ghost readout (which can only score tokens through K_l) is blind

Experiment 1 (probe ceiling): cache h_l at every valid position of the 24
fit / 8 held-out eval WikiText prompts (same prompts, same skip_first
convention as out/fit_ghost.py / out/eval_ghost.py), then fit a ridge
regression h_l -> e_actual_token per layer. Decode by nearest neighbor over
the full input-embedding matrix (cosine and ghost-style <Wh,e_v>-.5||e_v||^2
scoring). This is the identity ceiling a *linear* readout can reach when it
is allowed to see the actual labels, unconstrained by K_l's column space.
Controls: shuffled-label probe (should be ~0%), ghost-lens top-1 for
reference.

Confound control (predictability stratification): a late h_p also encodes
the model's *prediction* of token p+1, and in natural text token p is often
inferable from p-1's prediction plus topical continuity (bigram inversion).
A probe that "recovers" input identity late by inverting output statistics
would fake a world-B verdict. So for every evaluated position we also record
the model's own rank of the actual token p under its final-layer prediction
at position p-1 ("predictability"), and report probe top-1 stratified into
high-predictability (rank <= 10) vs surprising (rank > 100) buckets. Identity
holding up on the *surprising* bucket at L23/L26 is the decisive evidence for
world B; recovery concentrated in the predictable bucket is world A wearing
a disguise.

Experiment 2 (col(K_l) residency): SVD K_l = U S V^T (K_l is square and
numerically full rank, so plain column-space membership is vacuous); the
residency curve r(k) = ||P_k h||^2 / ||h||^2 for P_k = projection onto the
top-k left singular vectors of K_l, k in {8,16,...,1024}, averaged over the
same held-out positions, vs. a k-random-orthonormal-direction baseline
(expected r(k) = k/1024). Effective rank = min k capturing 90%/99% of the
squared singular-value mass.

Run from repo root: uv run python out/probe_residency.py
Re-run analysis only (skip the model forward passes): add --no-forward (uses
the cached out/probe_activations.pt; fails if it doesn't exist yet).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

SKIP = 16  # match fit_ghost.py / eval_ghost.py's valid-position convention
N_FIT = 24
N_EVAL = 8
MAX_SEQ_LEN = 128
LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26]
MODEL = "Qwen/Qwen3-0.6B"
VOCAB_CHUNK = 32768

ACT_CACHE = "out/probe_activations.pt"
RESULTS = "out/probe_residency.json"
GHOST_RESULTS = "out/ghost_results.json"

RIDGE_GRID = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
N_VAL_PROMPTS = 4  # held out of the 24 fit prompts, for lambda selection only
RESIDENCY_KS = [8, 16, 32, 64, 128, 256, 512, 1024]
N_RANDOM_DRAWS = 5  # random-subspace baseline repeats, averaged down

HIGH_PRED_RANK = 10  # actual token rank <= this at p-1 -> "predictable"
SURPRISE_RANK = 100  # actual token rank > this at p-1 -> "surprising"


# --------------------------------------------------------------- collection


def collect_activations(device: str) -> dict:
    """One forward pass per prompt, all layers hooked at once. Returns a dict
    of CPU fp32 tensors: per split ('fit' / 'eval'), per layer h_l at every
    valid position, plus actual token ids, owning-prompt index, and the
    model's own predictability rank/logprob of the actual token (from its
    final-layer prediction at position p-1)."""
    import transformers

    import jlens
    from jlens.examples import load_wikitext_prompts
    from jlens.hooks import ActivationRecorder

    jlens.configure_logging()
    print(f"loading {MODEL} on {device}...", flush=True)
    t0 = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=device
    )
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    print(f"  loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    all_prompts = load_wikitext_prompts(N_FIT + N_EVAL)
    fit_prompts, eval_prompts = all_prompts[:N_FIT], all_prompts[N_FIT:]
    final_layer = model.n_layers - 1
    hook_layers = sorted(set(LAYERS) | {final_layer})

    def run_split(prompts: list[str], split_name: str) -> dict:
        h_by_layer = {l: [] for l in LAYERS}
        actual_chunks = []
        prompt_idx_chunks = []
        rank_pred_chunks = []
        logprob_pred_chunks = []
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            ids = model.encode(prompt, max_length=MAX_SEQ_LEN)
            with torch.no_grad(), ActivationRecorder(model.layers, at=hook_layers) as rec:
                model.forward(ids)
                acts = {l: rec.activations[l][0].detach() for l in hook_layers}
            n = ids.shape[1]
            pos = torch.arange(SKIP, n - 1)
            ids_cpu = ids[0].cpu()
            actual = ids_cpu[pos]

            final_h = acts[final_layer]  # [seq_len, d_model], on model device
            model_logits = model.unembed(final_h).float().cpu()  # [seq_len, vocab]
            predictor_logits = model_logits[pos - 1]  # predicts token at pos
            logprobs = torch.log_softmax(predictor_logits, dim=-1)
            actual_logprob = logprobs.gather(1, actual[:, None]).squeeze(1)
            picked = predictor_logits.gather(1, actual[:, None])
            rank_pred = (predictor_logits > picked).sum(-1)

            for l in LAYERS:
                h_by_layer[l].append(acts[l][pos].float().cpu())
            actual_chunks.append(actual)
            prompt_idx_chunks.append(torch.full((len(pos),), i, dtype=torch.long))
            rank_pred_chunks.append(rank_pred)
            logprob_pred_chunks.append(actual_logprob)
            print(
                f"  [{split_name}] prompt {i+1}/{len(prompts)} "
                f"seq_len={n} n_valid={len(pos)} ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

        return {
            "h": {l: torch.cat(v) for l, v in h_by_layer.items()},
            "actual": torch.cat(actual_chunks),
            "prompt_idx": torch.cat(prompt_idx_chunks),
            "rank_pred": torch.cat(rank_pred_chunks),
            "logprob_pred": torch.cat(logprob_pred_chunks),
        }

    fit_data = run_split(fit_prompts, "fit")
    eval_data = run_split(eval_prompts, "eval")
    embed_weight = model.embed_tokens.weight.detach().float().cpu()

    return {
        "fit": fit_data,
        "eval": eval_data,
        "embed_weight": embed_weight,
        "n_layers": model.n_layers,
        "d_model": model.d_model,
    }


def load_or_collect_activations(device: str, force: bool) -> dict:
    if not force and Path(ACT_CACHE).exists():
        print(f"loading cached activations from {ACT_CACHE}", flush=True)
        return torch.load(ACT_CACHE, map_location="cpu", weights_only=True)
    data = collect_activations(device)
    torch.save(data, ACT_CACHE)
    print(f"wrote {ACT_CACHE}", flush=True)
    return data


# -------------------------------------------------------------------- ridge


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    """W minimizing ||X W - Y||^2 + lam ||W||^2. Closed form: W = (X^T X + lam
    I)^-1 X^T Y. X: [n, d_in], Y: [n, d_out], W: [d_in, d_out]."""
    d = X.shape[1]
    XtX = X.T @ X
    XtY = X.T @ Y
    A = XtX + lam * torch.eye(d, dtype=X.dtype)
    return torch.linalg.solve(A, XtY)


def mean_cosine(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(pred, target, dim=-1).mean().item()


# ------------------------------------------------------------------- decode


def chunked_ranks(
    query: torch.Tensor,
    embed_weight: torch.Tensor,
    actual_ids: torch.Tensor,
    *,
    metric: str,
    sq_norms: torch.Tensor,
    chunk: int = VOCAB_CHUNK,
) -> torch.Tensor:
    """Rank (0 = top) of actual_ids[i] among all vocab rows scored against
    query[i], without materializing the full [n, vocab] score matrix.

    metric="cosine": score(v) = <query, e_v> / (||query|| ||e_v||)
    metric="ghost":  score(v) = <query, e_v> - 0.5 ||e_v||^2
    """
    n = query.shape[0]
    if metric == "cosine":
        qn = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        actual_rows = embed_weight[actual_ids]
        actual_rows_n = actual_rows / actual_rows.norm(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        actual_score = (qn * actual_rows_n).sum(-1)
    elif metric == "ghost":
        qn = query
        actual_rows = embed_weight[actual_ids]
        actual_sq = sq_norms[actual_ids]
        actual_score = (qn * actual_rows).sum(-1) - 0.5 * actual_sq
    else:
        raise ValueError(metric)

    counts = torch.zeros(n, dtype=torch.long)
    offset = 0
    for rows in embed_weight.split(chunk):
        this_sq = sq_norms[offset : offset + rows.shape[0]]
        offset += rows.shape[0]
        if metric == "cosine":
            rows_n = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores = qn @ rows_n.T
        else:
            scores = qn @ rows.T - 0.5 * this_sq[None, :]
        counts += (scores > actual_score[:, None]).sum(-1)
    return counts


def summarize_ranks(ranks: torch.Tensor) -> dict:
    r = ranks.float()
    return {
        "top1_acc": float((r == 0).float().mean()),
        "top10_acc": float((r < 10).float().mean()),
        "n": int(r.numel()),
    }


# -------------------------------------------------------------- residency


def residency_curve(
    K: torch.Tensor, H: torch.Tensor, ks: list[int], n_random_draws: int
) -> dict:
    d = K.shape[0]
    U, S, _Vt = torch.linalg.svd(K)
    h_norm2 = (H * H).sum(-1)

    s2 = S * S
    cum_s2 = torch.cumsum(s2, 0)
    total = cum_s2[-1]
    frac = cum_s2 / total
    eff90 = int((frac >= 0.90).nonzero()[0].item()) + 1
    eff99 = int((frac >= 0.99).nonzero()[0].item()) + 1

    # cum/cumr below hold the exact residency numerator for *every* k in
    # [1, d] (not just the reporting grid), so eff90/eff99 -- only known once
    # S is in hand -- can be read off exactly, no snapping to the grid.
    extract_ks = sorted(set(ks) | {eff90, eff99})

    proj = H @ U  # [n, d], columns ordered by descending singular value
    cum = torch.cumsum(proj * proj, dim=1)
    r_signal = {k: float((cum[:, k - 1] / h_norm2).mean()) for k in extract_ks}

    r_baseline_draws = {k: [] for k in extract_ks}
    for _ in range(n_random_draws):
        Q, _ = torch.linalg.qr(torch.randn(d, d))
        projr = H @ Q
        cumr = torch.cumsum(projr * projr, dim=1)
        for k in extract_ks:
            r_baseline_draws[k].append(float((cumr[:, k - 1] / h_norm2).mean()))
    r_baseline = {k: sum(v) / len(v) for k, v in r_baseline_draws.items()}

    return {
        "residency_signal": r_signal,
        "residency_baseline": r_baseline,
        "eff_rank_90": eff90,
        "eff_rank_99": eff99,
        "residency_at_eff_rank_99_signal": r_signal[eff99],
        "residency_at_eff_rank_99_baseline": r_baseline[eff99],
        "residency_at_eff_rank_90_signal": r_signal[eff90],
        "residency_at_eff_rank_90_baseline": r_baseline[eff90],
    }


# ------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--device", default="mps", help="device for the model forward passes"
    )
    ap.add_argument(
        "--no-forward",
        action="store_true",
        help="skip the model forward passes; require out/probe_activations.pt to exist",
    )
    ap.add_argument(
        "--force-forward",
        action="store_true",
        help="ignore out/probe_activations.pt and recompute it",
    )
    args = ap.parse_args()

    if args.no_forward and not Path(ACT_CACHE).exists():
        raise SystemExit(f"--no-forward given but {ACT_CACHE} does not exist")

    data = load_or_collect_activations(
        args.device, force=args.force_forward and not args.no_forward
    )
    embed_weight = data["embed_weight"]
    sq_norms = (embed_weight * embed_weight).sum(-1)

    fit = data["fit"]
    ev = data["eval"]
    fit_prompt_idx = fit["prompt_idx"]
    val_mask = fit_prompt_idx >= (N_FIT - N_VAL_PROMPTS)
    train_mask = ~val_mask

    from jlens.backward import BackwardLens

    backward_lens = BackwardLens.load("out/backward_lens.pt")

    ghost_ref = {}
    if Path(GHOST_RESULTS).exists():
        ghost_json = json.load(open(GHOST_RESULTS))
        ghost_ref = {
            int(l): v["top1_acc"] * 100
            for l, v in ghost_json["retention"]["backward"].items()
        }

    # ---------------------------------------------------- sanity check
    print("\n=== sanity check: reproduce ghost top-1 at layer 2 ===", flush=True)
    l2_h = ev["h"][2]
    l2_actual = ev["actual"]
    K2 = backward_lens.K[2]
    g2 = l2_h @ K2
    # Direct, dense (single layer, 888 positions -> fine in memory) using the
    # exact same formula as jlens.backward.BackwardLens.ghost_logits(metric="l2").
    Kgram = K2.T @ K2
    sq_K = ((embed_weight @ Kgram) * embed_weight).sum(-1)  # ||K e_v||^2 per token
    dot_K = g2 @ embed_weight.T
    ghost_logits_l2 = dot_K - 0.5 * sq_K[None, :]
    picked = ghost_logits_l2.gather(1, l2_actual[:, None])
    ranks_l2 = (ghost_logits_l2 > picked).sum(-1)
    reproduced_top1 = float((ranks_l2 == 0).float().mean()) * 100
    reference_top1 = ghost_ref.get(2)
    print(
        f"  reproduced layer-2 ghost top-1 = {reproduced_top1:.2f}% "
        f"(ghost_results.json: {reference_top1})",
        flush=True,
    )

    # ---------------------------------------------------- experiment 1
    print("\n=== experiment 1: linear-probe ceiling ===", flush=True)
    probe_results = {}
    for layer in LAYERS:
        t0 = time.perf_counter()
        X_all = fit["h"][layer]
        Y_all = embed_weight[fit["actual"]]
        X_tr, Y_tr = X_all[train_mask], Y_all[train_mask]
        X_val, Y_val = X_all[val_mask], Y_all[val_mask]

        best_lambda, best_cos = None, -2.0
        for lam in RIDGE_GRID:
            W = ridge_fit(X_tr, Y_tr, lam)
            cos = mean_cosine(X_val @ W, Y_val)
            if cos > best_cos:
                best_cos, best_lambda = cos, lam

        W_final = ridge_fit(X_all, Y_all, best_lambda)
        perm = torch.randperm(Y_all.shape[0])
        W_shuf = ridge_fit(X_all, Y_all[perm], best_lambda)

        X_ev = ev["h"][layer]
        actual_ev = ev["actual"]
        pred = X_ev @ W_final
        pred_shuf = X_ev @ W_shuf

        ranks_cos = chunked_ranks(pred, embed_weight, actual_ev, metric="cosine", sq_norms=sq_norms)
        ranks_ghost = chunked_ranks(pred, embed_weight, actual_ev, metric="ghost", sq_norms=sq_norms)
        ranks_cos_shuf = chunked_ranks(
            pred_shuf, embed_weight, actual_ev, metric="cosine", sq_norms=sq_norms
        )
        ranks_ghost_shuf = chunked_ranks(
            pred_shuf, embed_weight, actual_ev, metric="ghost", sq_norms=sq_norms
        )

        rank_pred_ev = ev["rank_pred"]
        high_pred_mask = rank_pred_ev <= HIGH_PRED_RANK
        surprising_mask = rank_pred_ev > SURPRISE_RANK

        def bucket_stats(ranks: torch.Tensor) -> dict:
            return {
                "overall": summarize_ranks(ranks),
                "high_predictability": summarize_ranks(ranks[high_pred_mask]),
                "surprising": summarize_ranks(ranks[surprising_mask]),
            }

        probe_results[layer] = {
            "best_lambda": best_lambda,
            "val_cosine_at_best_lambda": best_cos,
            "cosine_decode": bucket_stats(ranks_cos),
            "ghost_decode": bucket_stats(ranks_ghost),
            "shuffled_control_cosine": bucket_stats(ranks_cos_shuf),
            "shuffled_control_ghost": bucket_stats(ranks_ghost_shuf),
            "ghost_lens_top1_reference": ghost_ref.get(layer),
        }
        print(
            f"  layer {layer:2d}: lambda={best_lambda:g} "
            f"probe_top1_cos={probe_results[layer]['cosine_decode']['overall']['top1_acc']*100:5.1f}% "
            f"probe_top1_ghost={probe_results[layer]['ghost_decode']['overall']['top1_acc']*100:5.1f}% "
            f"surprising_n={probe_results[layer]['cosine_decode']['surprising']['n']} "
            f"surprising_top1_cos={probe_results[layer]['cosine_decode']['surprising']['top1_acc']*100:5.1f}% "
            f"shuffled_top1={probe_results[layer]['shuffled_control_cosine']['overall']['top1_acc']*100:5.1f}% "
            f"ghost_ref={ghost_ref.get(layer)} "
            f"({time.perf_counter()-t0:.0f}s)",
            flush=True,
        )

    print(
        f"  predictability buckets, eval set: n_total={len(rank_pred_ev)} "
        f"n_high_pred(rank<={HIGH_PRED_RANK})={int(high_pred_mask.sum())} "
        f"n_surprising(rank>{SURPRISE_RANK})={int(surprising_mask.sum())}",
        flush=True,
    )

    # ---------------------------------------------------- experiment 2
    print("\n=== experiment 2: col(K_l) residency ===", flush=True)
    residency_results = {}
    for layer in LAYERS:
        t0 = time.perf_counter()
        K = backward_lens.K[layer]
        H = ev["h"][layer]
        res = residency_curve(K, H, RESIDENCY_KS, N_RANDOM_DRAWS)
        residency_results[layer] = res
        eff99 = res["eff_rank_99"]
        r_signal_99 = res["residency_at_eff_rank_99_signal"]
        r_base_99 = res["residency_at_eff_rank_99_baseline"]
        grid_curve = {k: round(res["residency_signal"][k], 3) for k in RESIDENCY_KS}
        print(
            f"  layer {layer:2d}: eff_rank90={res['eff_rank_90']:4d} "
            f"eff_rank99={eff99:4d} "
            f"r(eff99) signal={r_signal_99:.3f} baseline={r_base_99:.3f} "
            f"r_curve={grid_curve} "
            f"({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )

    # -------------------------------------------------------------- write
    out = {
        "model": MODEL,
        "layers": LAYERS,
        "sanity_check": {
            "layer": 2,
            "reproduced_ghost_top1_pct": reproduced_top1,
            "reference_ghost_top1_pct": reference_top1,
        },
        "predictability_buckets": {
            "high_predictability_rank_le": HIGH_PRED_RANK,
            "surprising_rank_gt": SURPRISE_RANK,
            "n_total_eval_positions": int(len(rank_pred_ev)),
            "n_high_predictability": int(high_pred_mask.sum()),
            "n_surprising": int(surprising_mask.sum()),
        },
        "probe": probe_results,
        "residency": residency_results,
        "residency_ks": RESIDENCY_KS,
    }
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
