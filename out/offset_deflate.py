"""Does neighbor signal emerge once the self channel is explained away?

eval_offset.py showed the Δ>0 readouts' strongest column is d=0 (self
leakage): all K_Δ images share a token-content channel and the readout
matches whatever is loudest in h. Under the additive linear model

    h[t] ≈ K_0 e_{x_t} + Σ_Δ K_Δ e_{x_{t-Δ}} + ...

deflating h' = h − K_0 e_{x_t} should unmask the neighbor terms if they
exist. Reports the baseline/rank signal ratio for bins Δ=1..3 at distances
1..3, raw vs deflated. Appends {"deflate": ...} to out/offset_results.json.

Run from repo root: .venv/bin/python out/offset_deflate.py
"""

import json
import time

import torch
import transformers

import jlens
from jlens.examples import load_wikitext_prompts
from jlens.hooks import ActivationRecorder
from jlens.offset import OffsetLens

jlens.configure_logging()

MODEL = "Qwen/Qwen3-0.6B"
SKIP = 16
N_FIT = 24
N_EVAL = 8
MAX_SEQ = 128
PROBE_BINS = [1, 2, 3]  # exact-distance bins
DISTS = [1, 2, 3]

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="mps"
)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
lens = OffsetLens.load("out/offset_lens.pt")
LAYERS = lens.target_layers


def ranks_in_rows(rows: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    picked = rows.gather(1, tokens)
    return (rows > picked).sum(-1)


acc: dict = {
    l: {
        mode: {b: {d: [] for d in DISTS} for b in PROBE_BINS}
        for mode in ("raw", "deflated", "base_raw", "base_deflated")
    }
    for l in LAYERS
}

eval_prompts = load_wikitext_prompts(N_FIT + N_EVAL)[N_FIT:]
embed = model.embed_tokens.weight
for pi, prompt in enumerate(eval_prompts):
    t0 = time.perf_counter()
    ids_full = model.encode(prompt, max_length=MAX_SEQ)
    with torch.no_grad(), ActivationRecorder(model.layers, at=LAYERS) as rec:
        model.forward(ids_full)
        hs = {l: rec.activations[l].detach()[0] for l in LAYERS}
    ids = ids_full[0].cpu()
    n = len(ids)
    ts = torch.arange(SKIP, n)
    T = len(ts)
    roll = torch.roll(torch.arange(T), T // 2)
    e_self = embed[ids[ts].to(embed.device)].float()  # [T, d]
    for l in LAYERS:
        h = hs[l][ts].float().to(embed.device)
        K0 = lens.K[(l, 0)].to(embed.device)
        h_def = h - e_self @ K0.T  # remove the bin-0 image of the actual token
        for b in PROBE_BINS:
            for name, res in (("raw", h), ("deflated", h_def)):
                logits = lens.ghost_logits(model, res, l, b)
                for d in DISTS:
                    el = (ts - d) >= SKIP
                    src = ids[(ts - d)[el]][:, None]
                    r = ranks_in_rows(logits[el], src)
                    acc[l][name][b][d].extend(r.tolist())
                    rb = ranks_in_rows(logits[roll][el], src)
                    acc[l]["base_" + name][b][d].extend(rb.tolist())
    print(f"prompt {pi + 1}/{N_EVAL}  {time.perf_counter() - t0:.0f}s", flush=True)


def med(xs: list[int]) -> float:
    return float(torch.tensor(xs, dtype=torch.float32).median())


out = {l: {} for l in LAYERS}
for l in LAYERS:
    for b in PROBE_BINS:
        for d in DISTS:
            raw, braw = med(acc[l]["raw"][b][d]), med(acc[l]["base_raw"][b][d])
            de, bde = med(acc[l]["deflated"][b][d]), med(acc[l]["base_deflated"][b][d])
            out[l][f"b{b}_d{d}"] = {
                "raw": raw, "raw_base": braw, "raw_ratio": braw / max(raw, 1),
                "def": de, "def_base": bde, "def_ratio": bde / max(de, 1),
            }
            print(
                f"L{l} bin Δ={b} dist {d}:  raw {braw / max(raw, 1):5.1f}x "
                f"({raw:.0f})   deflated {bde / max(de, 1):5.1f}x ({de:.0f})",
                flush=True,
            )

results = json.load(open("out/offset_results.json"))
results["deflate"] = out
with open("out/offset_results.json", "w") as f:
    json.dump(results, f)
print("appended deflate results to out/offset_results.json", flush=True)
