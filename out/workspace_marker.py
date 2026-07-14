"""Is ghost-death a per-position marker for the arrival of workspace content?

Third follow-up (after out/probe_residency.py and out/quadrant_analysis.py).
Hypothesis under test: ghost-death at a position marks the ARRIVAL of
meaningful output-basis ("workspace") content there — not necessarily the
final prediction, just semantic information about where the text is going.
Mechanically motivated: quadrant_analysis showed ghost-death = s_top1
exploding (something got loud in K-space), and ghost_patch.json showed the
loud junk is output-flavored causally. Question: is the loud content
MEANINGFUL, and does it arrive AT ghost-death, per position?

Forward-readout informativeness, per (position, layer) on the 888 held-out
positions, layers {8,11,14,17,20,23,26}, forward-lens top-5 readouts:

  1. future-overlap  — any forward top-5 token literally among the next 32
                       actual document tokens, EXCLUDING the immediate next
                       token (separates "planning" from plain next-token
                       prediction); the including-next version for reference.
  2. semantic sim    — mean over forward top-5 of max cosine (input-embedding
                       space) to the embeddings of the next-32 window (same
                       exclusion). Null floor: identical statistic against
                       the matched-position window of a DIFFERENT eval prompt
                       (prompt (i+1) mod 8).
  3. final-match     — rank of the model's final top-1 in the forward readout
                       (the strict convergence measure, as before).

Test A (cross-sectional): per layer, measures 1-2 for ghost-dead (ghost rank
>= 1000) vs ghost-alive (rank < 10) positions, the same split by probe rank
(probe-dead/alive; ghost-death != identity loss, so both conditionings), and
a predictability control (high-pred = model rank of actual <= 10 at p-1;
surprising = rank > 100; from probe_residency's stratification data).

Test B (event alignment, the decisive one): per position, ghost-death layer
= first sampled layer with ghost rank > 1000 after having been held (rank <
10) at an earlier sampled layer — exactly transition_analysis.py's
input_transition convention, on the full 9-layer grid {2,...,26}.
Workspace-onset layer = first layer whose semantic future-similarity exceeds
that layer's cross-prompt null mean + k sigma (k in {1,2,3}, sensitivity
reported). Deliverables: (i) corr(ghost-death layer, onset layer) across
positions; (ii) the event-aligned average — null-corrected semantic sim as a
function of (layer - death layer), against a shuffled-death-layer control
(same curve with death layers permuted across positions, 20 draws): a step
at relative layer 0 that the shuffle destroys = per-position marker; a
smooth co-rise the shuffle reproduces = global depth effect only.

Reuses out/probe_activations.pt; model work is one forward pass per eval
prompt (final-layer logits) plus forward-lens unembeds, cached to
out/workspace_cache.pt. Run from repo root:
  uv run python out/workspace_marker.py        (cache short-circuits reruns)
  uv run python out/workspace_marker.py --force  (recompute the cache)
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
ALL_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26]  # ghost-rank grid (death layer)
FWD_LAYERS = [8, 11, 14, 17, 20, 23, 26]  # forward-readout grid
MODEL = "Qwen/Qwen3-0.6B"
VOCAB_CHUNK = 32768
TOP_K = 5
FUTURE_WINDOW = 32
HELD = 10  # transition_analysis.py conventions
DISSOLVED = 1000
SIGMA_GRID = [1.0, 2.0, 3.0]
N_SHUFFLES = 20
REL_OFFSETS = [-12, -9, -6, -3, 0, 3, 6, 9, 12]

ACT_CACHE = "out/probe_activations.pt"
CACHE = "out/workspace_cache.pt"
RESULTS = "out/workspace_results.json"
PROBE_RESULTS = "out/probe_residency.json"

HIGH_PRED_RANK = 10
SURPRISE_RANK = 100


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float) -> torch.Tensor:
    d = X.shape[1]
    return torch.linalg.solve(X.T @ X + lam * torch.eye(d, dtype=X.dtype), X.T @ Y)


def ghost_rank_of_actual(
    h: torch.Tensor, K: torch.Tensor, embed_weight: torch.Tensor, actual: torch.Tensor
) -> torch.Tensor:
    """Rank (0 = top) of the actual token under the l2 ghost score (chunked;
    identical formula to BackwardLens.ghost_logits)."""
    g = h @ K
    Kgram = K.T @ K
    actual_rows = embed_weight[actual]
    s_actual = (g * actual_rows).sum(-1) - 0.5 * (
        (actual_rows @ Kgram) * actual_rows
    ).sum(-1)
    rank = torch.zeros(h.shape[0], dtype=torch.long)
    for rows in embed_weight.split(VOCAB_CHUNK):
        sq = ((rows @ Kgram) * rows).sum(-1)
        rank += ((g @ rows.T - 0.5 * sq[None, :]) > s_actual[:, None]).sum(-1)
    return rank


def probe_ranks_cosine(
    pred: torch.Tensor, embed_weight: torch.Tensor, actual: torch.Tensor
) -> torch.Tensor:
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
    data = torch.load(ACT_CACHE, map_location="cpu", weights_only=True)
    ev, fit = data["eval"], data["fit"]
    embed_weight = data["embed_weight"]
    actual, prompt_idx = ev["actual"], ev["prompt_idx"]

    import transformers

    import jlens
    from jlens.backward import BackwardLens
    from jlens.examples import load_wikitext_prompts
    from jlens.hooks import ActivationRecorder

    jlens.configure_logging()
    backward = BackwardLens.load("out/backward_lens.pt")
    forward = jlens.JacobianLens.load("out/forward_lens.pt")
    best_lambda = {
        int(l): v["best_lambda"]
        for l, v in json.load(open(PROBE_RESULTS))["probe"].items()
    }

    print(f"loading {MODEL} on {device}...", flush=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map=device
    )
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    final_layer = model.n_layers - 1

    eval_prompts = load_wikitext_prompts(N_FIT + N_EVAL)[N_FIT:]
    full_ids = torch.full((N_EVAL, MAX_SEQ_LEN), -1, dtype=torch.long)
    seq_lens = torch.zeros(N_EVAL, dtype=torch.long)
    final_top1_chunks = []
    for i, prompt in enumerate(eval_prompts):
        t0 = time.perf_counter()
        ids = model.encode(prompt, max_length=MAX_SEQ_LEN)
        with torch.no_grad(), ActivationRecorder(model.layers, at=[final_layer]) as rec:
            model.forward(ids)
            h_final = rec.activations[final_layer][0].detach()
        n = ids.shape[1]
        pos = torch.arange(SKIP, n - 1)
        ids_cpu = ids[0].cpu()
        if not torch.equal(ids_cpu[pos], actual[prompt_idx == i]):
            raise RuntimeError(f"eval prompt {i}: tokenization mismatch vs {ACT_CACHE}")
        full_ids[i, :n] = ids_cpu
        seq_lens[i] = n
        model_logits = model.unembed(h_final).float().cpu()
        final_top1_chunks.append(model_logits[pos].argmax(-1))
        print(f"  [final] prompt {i+1}/{N_EVAL} ({time.perf_counter()-t0:.1f}s)", flush=True)
    final_top1 = torch.cat(final_top1_chunks)

    fwd_top5: dict[int, torch.Tensor] = {}
    fwd_rank_final: dict[int, torch.Tensor] = {}
    probe_rank: dict[int, torch.Tensor] = {}
    for layer in FWD_LAYERS:
        t0 = time.perf_counter()
        h = ev["h"][layer]
        with torch.no_grad():
            transported = forward.transport(h.to(model.input_device), layer)
            logits = model.unembed(transported).float().cpu()  # [888, vocab]
        fwd_top5[layer] = logits.topk(TOP_K, dim=-1).indices
        picked = logits.gather(1, final_top1[:, None])
        fwd_rank_final[layer] = (logits > picked).sum(-1)
        del logits
        W = ridge_fit(fit["h"][layer], embed_weight[fit["actual"]], best_lambda[layer])
        probe_rank[layer] = probe_ranks_cosine(h @ W, embed_weight, actual)
        print(f"  [layer {layer}] fwd+probe done ({time.perf_counter()-t0:.0f}s)", flush=True)

    ghost_rank: dict[int, torch.Tensor] = {}
    for layer in ALL_LAYERS:
        t0 = time.perf_counter()
        ghost_rank[layer] = ghost_rank_of_actual(
            ev["h"][layer], backward.K[layer], embed_weight, actual
        )
        print(f"  [layer {layer}] ghost rank done ({time.perf_counter()-t0:.0f}s)", flush=True)

    return {
        "full_ids": full_ids,
        "seq_lens": seq_lens,
        "final_top1": final_top1,
        "actual": actual,
        "prompt_idx": prompt_idx,
        "rank_pred": ev["rank_pred"],
        "fwd_top5": fwd_top5,
        "fwd_rank_final": fwd_rank_final,
        "probe_rank": probe_rank,
        "ghost_rank": ghost_rank,
    }


# --------------------------------------------------------------- analysis


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--force", action="store_true", help="recompute the cache")
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

    data = torch.load(ACT_CACHE, map_location="cpu", weights_only=True)
    embed_weight = data["embed_weight"]
    emb_n = embed_weight / embed_weight.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    prompt_idx = cache["prompt_idx"]
    rank_pred = cache["rank_pred"]
    full_ids = cache["full_ids"]
    seq_lens = cache["seq_lens"]
    n = len(prompt_idx)

    # position of each row inside its prompt (rows are contiguous, in order)
    pos_in_prompt = torch.zeros(n, dtype=torch.long)
    for i in range(N_EVAL):
        m = prompt_idx == i
        pos_in_prompt[m] = torch.arange(int(m.sum())) + SKIP

    # future windows: real (own prompt, p+2..p+33) and null (next prompt,
    # same window) + the including-next variant (p+1..p+32), padded with -1
    def window_tokens(src_prompt: torch.Tensor, start: int) -> torch.Tensor:
        """[n, FUTURE_WINDOW] token ids, -1 past the prompt end."""
        out = torch.full((n, FUTURE_WINDOW), -1, dtype=torch.long)
        for row in range(n):
            i = int(src_prompt[row])
            p = int(pos_in_prompt[row])
            lo, hi = p + start, min(p + start + FUTURE_WINDOW, int(seq_lens[i]))
            if hi > lo:
                out[row, : hi - lo] = full_ids[i, lo:hi]
        return out

    next_prompt = (prompt_idx + 1) % N_EVAL
    win_excl = window_tokens(prompt_idx, 2)  # excludes immediate next
    win_incl = window_tokens(prompt_idx, 1)
    win_null = window_tokens(next_prompt, 2)

    def sem_sim(top5: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        """mean over top-5 of max cosine to the window tokens; NaN if the
        window is empty."""
        w = window.clamp_min(0)
        wemb = emb_n[w]  # [n, W, d]
        temb = emb_n[top5]  # [n, 5, d]
        sims = torch.einsum("nkd,nwd->nkw", temb, wemb)
        sims = sims.masked_fill((window < 0)[:, None, :], -torch.inf)
        best = sims.max(-1).values  # [n, 5]
        out = best.mean(-1)
        out[(window >= 0).sum(-1) == 0] = torch.nan
        return out

    def overlap(top5: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
        return (top5[:, :, None] == window[:, None, :]).any(-1).any(-1)

    valid = (win_excl >= 0).any(-1)  # window non-empty (drops last positions)
    sim_real = {l: sem_sim(cache["fwd_top5"][l], win_excl) for l in FWD_LAYERS}
    sim_null = {l: sem_sim(cache["fwd_top5"][l], win_null) for l in FWD_LAYERS}
    ov_excl = {l: overlap(cache["fwd_top5"][l], win_excl) for l in FWD_LAYERS}
    ov_incl = {l: overlap(cache["fwd_top5"][l], win_incl) for l in FWD_LAYERS}
    ov_null = {l: overlap(cache["fwd_top5"][l], win_null) for l in FWD_LAYERS}

    high_pred = rank_pred <= HIGH_PRED_RANK
    surprising = rank_pred > SURPRISE_RANK

    # ------------------------------------------------ test A: cross-section
    print("\n=== Test A: cross-sectional informativeness ===", flush=True)

    def group_stats(l: int, mask: torch.Tensor) -> dict:
        m = mask & valid
        cnt = int(m.sum())
        if cnt == 0:
            return {"n": 0}
        return {
            "n": cnt,
            "future_overlap_excl_next": float(ov_excl[l][m].float().mean()),
            "future_overlap_incl_next": float(ov_incl[l][m].float().mean()),
            "overlap_null": float(ov_null[l][m].float().mean()),
            "sem_sim": float(sim_real[l][m].nanmean()),
            "sem_sim_null": float(sim_null[l][m].nanmean()),
            "median_fwd_rank_final_top1": float(
                cache["fwd_rank_final"][l][m].float().median()
            ),
        }

    cross = {}
    for l in FWD_LAYERS:
        gdead = cache["ghost_rank"][l] >= DISSOLVED
        galive = cache["ghost_rank"][l] < HELD
        pdead = cache["probe_rank"][l] >= DISSOLVED
        palive = cache["probe_rank"][l] < HELD
        cross[l] = {
            "all": group_stats(l, torch.ones(n, dtype=torch.bool)),
            "ghost_dead": group_stats(l, gdead),
            "ghost_alive": group_stats(l, galive),
            "probe_dead": group_stats(l, pdead),
            "probe_alive": group_stats(l, palive),
            "ghost_dead_high_pred": group_stats(l, gdead & high_pred),
            "ghost_dead_surprising": group_stats(l, gdead & surprising),
            "ghost_alive_high_pred": group_stats(l, galive & high_pred),
            "ghost_alive_surprising": group_stats(l, galive & surprising),
        }
        a, d_, al = cross[l]["all"], cross[l]["ghost_dead"], cross[l]["ghost_alive"]
        print(
            f"  L{l:2d}: dead n={d_.get('n',0):3d} sim={d_.get('sem_sim',float('nan')):.3f} "
            f"ov={d_.get('future_overlap_excl_next',float('nan')):.2f} | "
            f"alive n={al.get('n',0):3d} sim={al.get('sem_sim',float('nan')):.3f} "
            f"ov={al.get('future_overlap_excl_next',float('nan')):.2f} | "
            f"null sim={a['sem_sim_null']:.3f} ov={a['overlap_null']:.2f}",
            flush=True,
        )

    # ---------------------------------------------- test B: event alignment
    print("\n=== Test B: event alignment ===", flush=True)
    L = ALL_LAYERS
    gr_grid = torch.stack([cache["ghost_rank"][l] for l in L])  # [9, n]

    death_layer = torch.full((n,), -1, dtype=torch.long)
    for row in range(n):
        gr = gr_grid[:, row]
        held_idx = (gr < HELD).nonzero(as_tuple=True)[0]
        if len(held_idx) == 0:
            continue
        rel = int(held_idx.max())
        for i in range(rel + 1, len(L)):
            if gr[i] > DISSOLVED:
                death_layer[row] = L[i]
                break
    has_death = death_layer >= 0
    print(
        f"  ghost-death (transition_analysis conventions): defined at "
        f"{int(has_death.sum())}/{n} positions; layer histogram "
        f"{ {l: int((death_layer == l).sum()) for l in L} }",
        flush=True,
    )

    # per-layer null thresholds from the null distribution across positions
    null_mean = {l: float(sim_null[l][valid].nanmean()) for l in FWD_LAYERS}
    null_std = {
        l: float(sim_null[l][valid][~sim_null[l][valid].isnan()].std())
        for l in FWD_LAYERS
    }

    event = {"null_mean": null_mean, "null_std": null_std, "onset": {}}
    for k in SIGMA_GRID:
        onset_layer = torch.full((n,), -1, dtype=torch.long)
        for row in range(n):
            if not valid[row]:
                continue
            for l in FWD_LAYERS:
                s = float(sim_real[l][row])
                if s == s and s > null_mean[l] + k * null_std[l]:
                    onset_layer[row] = l
                    break
        both = has_death & (onset_layer >= 0)
        n_both = int(both.sum())
        r = corr(death_layer[both], onset_layer[both]) if n_both >= 3 else float("nan")
        event["onset"][f"k={k:g}"] = {
            "n_onset_defined": int((onset_layer >= 0).sum()),
            "n_both_defined": n_both,
            "corr_death_vs_onset": r,
            "onset_layer_hist": {
                l: int((onset_layer == l).sum()) for l in FWD_LAYERS
            },
        }
        print(
            f"  k={k:g}: onset defined at {int((onset_layer >= 0).sum())} positions, "
            f"corr(death, onset) = {r:+.3f} (n={n_both})",
            flush=True,
        )

    # event-aligned average of null-corrected semantic similarity
    sim_corr_grid = {l: sim_real[l] - null_mean[l] for l in FWD_LAYERS}

    def aligned_curve(death: torch.Tensor) -> dict[int, tuple[float, int]]:
        sums = {o: 0.0 for o in REL_OFFSETS}
        cnts = {o: 0 for o in REL_OFFSETS}
        for l in FWD_LAYERS:
            s = sim_corr_grid[l]
            for o in REL_OFFSETS:
                m = has_death & valid & (l - death == o) & ~s.isnan()
                c = int(m.sum())
                if c:
                    sums[o] += float(s[m].sum())
                    cnts[o] += c
        return {
            o: ((sums[o] / cnts[o]) if cnts[o] else float("nan"), cnts[o])
            for o in REL_OFFSETS
        }

    true_curve = aligned_curve(death_layer)
    g = torch.Generator().manual_seed(0)
    shuf_acc = {o: [] for o in REL_OFFSETS}
    idx_death = has_death.nonzero(as_tuple=True)[0]
    for _ in range(N_SHUFFLES):
        d_sh = death_layer.clone()
        perm = idx_death[torch.randperm(len(idx_death), generator=g)]
        d_sh[idx_death] = death_layer[perm]
        for o, (v, c) in aligned_curve(d_sh).items():
            if c:
                shuf_acc[o].append(v)
    shuf_curve = {
        o: (sum(v) / len(v) if v else float("nan")) for o, v in shuf_acc.items()
    }

    print("\n  event-aligned null-corrected sem-sim (true vs shuffled-death):")
    for o in REL_OFFSETS:
        v, c = true_curve[o]
        print(
            f"    rel {o:+3d}: true={v:+.4f} (n={c})  shuffled={shuf_curve[o]:+.4f}",
            flush=True,
        )

    out = {
        "model": MODEL,
        "fwd_layers": FWD_LAYERS,
        "ghost_layers": ALL_LAYERS,
        "n_positions": n,
        "n_valid_windows": int(valid.sum()),
        "conventions": {
            "ghost_dead": f"ghost rank >= {DISSOLVED}",
            "ghost_alive": f"ghost rank < {HELD}",
            "death_layer": "transition_analysis input_transition: first "
            f"sampled layer with rank > {DISSOLVED} after last held (< {HELD})",
            "future_window": f"{FUTURE_WINDOW} tokens, primary measures "
            "exclude the immediate next token",
            "null": "matched-position window from eval prompt (i+1) mod 8",
        },
        "cross_sectional": cross,
        "event_alignment": {
            **event,
            "death_layer_hist": {l: int((death_layer == l).sum()) for l in L},
            "n_death_defined": int(has_death.sum()),
            "aligned_curve_true": {
                str(o): {"mean": v, "n": c} for o, (v, c) in true_curve.items()
            },
            "aligned_curve_shuffled": {str(o): v for o, v in shuf_curve.items()},
            "n_shuffles": N_SHUFFLES,
        },
    }
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
