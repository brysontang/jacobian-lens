"""Is ghost-identity retention load-bearing for the next-token handoff?

Per position on held-out WikiText: how deep the ghost lens still recovers the
input token ("held-until"), vs how early the forward lens locks onto the
model's final prediction ("settles-at"), with two confounds measured:

  - is_cont: the final prediction is a word-continuation piece (the position's
    job is to finish spelling its own token) — the French-grid hypothesis is
    that this flag explains most of the holding.
  - conf: model's final top-1 log-prob at the position (prediction ease).

Writes raw per-position rows to out/pivot_data.json, prints the analysis.
Re-run analysis without the model: .venv/bin/python out/pivot_analysis.py --analyze

Run from repo root: .venv/bin/python out/pivot_analysis.py
"""

import json
import sys
import time

import numpy as np
import torch

SKIP = 16
N_FIT = 24
N_EVAL = 8
DATA = "out/pivot_data.json"


def collect() -> list[dict]:
    import transformers

    import jlens
    from jlens.backward import BackwardLens
    from jlens.examples import load_wikitext_prompts

    jlens.configure_logging()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float16, device_map="mps"
    )
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = jlens.from_hf(hf, tok)
    backward = BackwardLens.load("out/backward_lens.pt")
    forward = jlens.JacobianLens.load("out/forward_lens.pt")
    layers = backward.target_layers
    assert layers == forward.source_layers

    rows = []
    prompts = load_wikitext_prompts(N_FIT + N_EVAL)[N_FIT:]
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        ghost, ids = backward.apply(model, prompt, max_seq_len=128)
        fwd, model_logits, _ = forward.apply(model, prompt, max_seq_len=128)
        ids = ids[0].cpu()
        n = len(ids)
        logprobs = model_logits.float().log_softmax(-1)
        final_pred = model_logits.argmax(-1)

        def rank_of(logits_row, t):
            return int((logits_row > logits_row[t]).sum())

        for p in range(SKIP, n - 1):
            fp = int(final_pred[p])
            pred_str = tok.decode([fp])
            rows.append({
                "prompt": i,
                "pos": p,
                "input_tok": tok.decode([int(ids[p])]),
                "ghost_rank": [rank_of(ghost[l][p], int(ids[p])) for l in layers],
                "fwdpred_rank": [rank_of(fwd[l][p], fp) for l in layers],
                "final_pred": pred_str,
                "actual_next": tok.decode([int(ids[p + 1])]),
                # continuation piece: glued to the previous token and wordlike
                "is_cont": bool(pred_str[:1].isalnum()),
                "conf": float(logprobs[p, fp]),
            })
        print(f"prompt {i + 1}/{N_EVAL} ({time.perf_counter() - t0:.0f}s)", flush=True)

    with open(DATA, "w") as f:
        json.dump({"layers": layers, "rows": rows}, f, ensure_ascii=False)
    print(f"wrote {DATA} ({len(rows)} positions)", flush=True)
    return rows


def analyze() -> None:
    d = json.load(open(DATA))
    layers = d["layers"]
    rows = d["rows"]
    L = np.array(layers)
    CENSOR = 28  # model has 28 layers; "not settled by L26" is censored here

    def held_until(gr, thresh):
        """Deepest sampled layer where the input token still ranks < thresh.
        0 if never held (released before the first sampled layer)."""
        idx = [i for i, r in enumerate(gr) if r < thresh]
        return int(L[max(idx)]) if idx else 0

    def settles_at(fr, thresh):
        """Shallowest sampled layer from which the final prediction ranks
        < thresh at every deeper sampled layer; CENSOR if never."""
        for i in range(len(fr)):
            if all(r < thresh for r in fr[i:]):
                return int(L[i])
        return CENSOR

    held = np.array([held_until(r["ghost_rank"], 10) for r in rows], float)
    settle = np.array([settles_at(r["fwdpred_rank"], 1) for r in rows], float)
    settle10 = np.array([settles_at(r["fwdpred_rank"], 10) for r in rows], float)
    cont = np.array([r["is_cont"] for r in rows], float)
    conf = np.array([r["conf"] for r in rows], float)
    n = len(rows)

    def corr(a, b):
        if len(a) < 3 or a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    print(f"n={n} positions, layers {layers}, settle censored at {CENSOR}")
    print(f"continuation predictions: {int(cont.sum())} ({cont.mean():.0%})")
    print(f"\nmeans: held-until {held.mean():.1f}, settles-at(rank0) {settle.mean():.1f}, "
          f"settles-at(rank<10) {settle10.mean():.1f}")
    print(f"censored (never settle, rank0): {(settle == CENSOR).mean():.0%}; "
          f"rank<10: {(settle10 == CENSOR).mean():.0%}")

    print("\n-- does spelling explain holding? --")
    print(f"mean held-until: continuation {held[cont == 1].mean():.1f} vs "
          f"new-word {held[cont == 0].mean():.1f}")
    print(f"corr(held, is_cont) = {corr(held, cont):+.2f}")
    print(f"corr(held, conf)    = {corr(held, conf):+.2f}")

    for name, s in (("rank0", settle), ("rank<10", settle10)):
        print(f"\n-- held vs settle ({name}) --")
        print(f"raw corr(held, settle) = {corr(held, s):+.2f}")
        m = s < CENSOR
        print(f"uncensored only (n={int(m.sum())}): {corr(held[m], s[m]):+.2f}")
        for flag, label in ((1, "continuation"), (0, "new-word")):
            g = cont == flag
            print(f"  within {label:12s} (n={int(g.sum())}): "
                  f"corr = {corr(held[g], s[g]):+.2f}")
        # partial: residualize both on [is_cont, conf], then correlate
        X = np.column_stack([np.ones(n), cont, conf])
        res_h = held - X @ np.linalg.lstsq(X, held, rcond=None)[0]
        res_s = s - X @ np.linalg.lstsq(X, s, rcond=None)[0]
        print(f"partial corr(held, settle | is_cont, conf) = {corr(res_h, res_s):+.2f}")


if __name__ == "__main__":
    if "--analyze" not in sys.argv:
        collect()
    analyze()
