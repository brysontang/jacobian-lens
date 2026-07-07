"""Offset-resolved ghost lens (K_Δ) eval on held-out WikiText.

The summed backward lens said *what* a state is holding; K_Δ asks *from
where*. For each fitted (layer, lookback-bin) matrix this measures whether
the bin's readout actually recovers the token that far back — and only that
far back (distance selectivity), on the 8 held-out WikiText prompts.

Writes out/offset_results.json:
  norms        — ||K_Δ||_F/sqrt(d) per (layer, bin) + fit pair counts
  rank_matrix  — median rank of the token exactly d back under each bin's
                 readout, d = each bin's lower edge → per-layer 14×14
                 selectivity matrix, + shuffled-target baseline
  provenance   — per (layer, bin): where the readout's top-1 comes from
                 (bin window / this position / elsewhere in prompt / novel)
  french       — per (layer, bin, position) top-1 strings for the French
                 showcase prompt, classed by the same provenance

Run from repo root: .venv/bin/python out/eval_offset.py
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
SKIP = 16  # match fitting: sources p < 16 were never in any bin
N_FIT = 24  # keep the ghost-lens eval split: prompts [24:32) held out
N_EVAL = 8
MAX_SEQ = 128

FRENCH = (
    "Le vieux port de Marseille est le plus ancien quartier de la ville. "
    "Les bateaux de pêche rentrent chaque matin avec leur cargaison, et les "
    "marchands installent leurs étals le long du quai. Les touristes se "
    "promènent entre les cafés et les restaurants, tandis que les mouettes "
    "tournent au-dessus de l'eau claire du bassin."
)

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="mps"
)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)

lens = OffsetLens.load("out/offset_lens.pt")
LAYERS = lens.target_layers
BINS = lens.bins
NB = len(BINS)
D_REP = [lo for lo, _ in BINS]  # one probe distance per bin: its near edge
print(f"offset lens: layers {LAYERS}, {NB} bins, probing d={D_REP}", flush=True)


@torch.no_grad()
def hidden_at(prompt: str) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    ids = model.encode(prompt, max_length=MAX_SEQ)
    with ActivationRecorder(model.layers, at=LAYERS) as rec:
        model.forward(ids)
        return {l: rec.activations[l].detach()[0] for l in LAYERS}, ids[0].cpu()


def ranks_in_rows(
    sorted_rows: torch.Tensor, rows: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """Rank (0 = top) of tokens[i, j] within rows[i]; rows pre-sorted asc."""
    picked = rows.gather(1, tokens)
    vocab = rows.shape[1]
    return vocab - torch.searchsorted(sorted_rows, picked.contiguous(), right=False)


# accumulators: rank_lists[layer][bin][dist] -> list of ranks
rank_lists = {l: [[[] for _ in D_REP] for _ in range(NB)] for l in LAYERS}
base_lists = {l: [[[] for _ in D_REP] for _ in range(NB)] for l in LAYERS}
prov = {
    l: [dict(window=0, self=0, elsewhere=0, novel=0, n=0) for _ in range(NB)]
    for l in LAYERS
}

eval_prompts = load_wikitext_prompts(N_FIT + N_EVAL)[N_FIT:]
for pi, prompt in enumerate(eval_prompts):
    t0 = time.perf_counter()
    hs, ids = hidden_at(prompt)
    n = len(ids)
    ts = torch.arange(SKIP, n)
    T = len(ts)
    roll = torch.roll(torch.arange(T), T // 2)
    prompt_set = set(ids.tolist())
    for l in LAYERS:
        res = hs[l][ts]
        for b, (lo, hi) in enumerate(BINS):
            logits = lens.ghost_logits(model, res, l, b)  # [T, vocab] CPU
            sorted_rows = logits.sort(dim=1).values
            top1 = logits.argmax(-1)
            # -- selectivity: rank of the token exactly d back, per probe d
            for di, d in enumerate(D_REP):
                el = (ts - d) >= SKIP
                if not bool(el.any()):
                    continue
                src = ids[(ts - d)[el]][:, None]
                r = ranks_in_rows(sorted_rows[el], logits[el], src)
                rank_lists[l][b][di].extend(r[:, 0].tolist())
                rb = ranks_in_rows(
                    sorted_rows[roll][el], logits[roll][el], src
                )
                base_lists[l][b][di].extend(rb[:, 0].tolist())
            # -- provenance of the readout's own top-1
            for i, t in enumerate(ts.tolist()):
                p_lo, p_hi = max(SKIP, t - hi), t - lo
                if p_hi < p_lo:
                    continue
                v = int(top1[i])
                window = set(ids[p_lo : p_hi + 1].tolist())
                stats = prov[l][b]
                stats["n"] += 1
                if v in window:
                    stats["window"] += 1
                elif v == int(ids[t]):
                    stats["self"] += 1
                elif v in prompt_set:
                    stats["elsewhere"] += 1
                else:
                    stats["novel"] += 1
    print(
        f"prompt {pi + 1}/{N_EVAL}  seq={n}  {time.perf_counter() - t0:.0f}s",
        flush=True,
    )


def med(xs: list[int]) -> float | None:
    if not xs:
        return None
    return float(torch.tensor(xs, dtype=torch.float32).median())


rank_matrix = {
    l: [[med(rank_lists[l][b][di]) for di in range(len(D_REP))] for b in range(NB)]
    for l in LAYERS
}
baseline_matrix = {
    l: [[med(base_lists[l][b][di]) for di in range(len(D_REP))] for b in range(NB)]
    for l in LAYERS
}
n_matrix = [
    [len(rank_lists[LAYERS[0]][b][di]) for di in range(len(D_REP))]
    for b in range(NB)
]

# ---------------------------------------------------------------- french grid
hs, ids = hidden_at(FRENCH)
n = len(ids)
ts = list(range(SKIP, n))
tokens = [tok.decode([i]) for i in ids.tolist()]
prompt_set = set(ids.tolist())
french = {"tokens": tokens, "start": SKIP, "grids": {}}
for l in LAYERS:
    res = hs[l][torch.tensor(ts)]
    french["grids"][l] = {}
    for b, (lo, hi) in enumerate(BINS):
        logits = lens.ghost_logits(model, res, l, b)
        top1 = logits.argmax(-1)
        cells = []
        for i, t in enumerate(ts):
            v = int(top1[i])
            p_lo, p_hi = max(SKIP, t - hi), t - lo
            if p_hi < p_lo:
                cls = "na"
            elif v in set(ids[p_lo : p_hi + 1].tolist()):
                cls = "win"
            elif v == int(ids[t]):
                cls = "self"
            elif v in prompt_set:
                cls = "ctx"
            else:
                cls = "nv"
            cells.append({"t": tok.decode([v]), "c": cls})
        french["grids"][l][b] = cells

sqrt_d = lens.d_model**0.5
out = {
    "layers": LAYERS,
    "bins": [list(b) for b in BINS],
    "probe_distances": D_REP,
    "skip": SKIP,
    "norms": {
        l: [lens.K[(l, b)].norm().item() / sqrt_d for b in range(NB)]
        for l in LAYERS
    },
    "fit_pair_counts": [lens.pair_counts[b] for b in range(NB)],
    "rank_matrix": rank_matrix,
    "baseline_matrix": baseline_matrix,
    "n_matrix": n_matrix,
    "provenance": prov,
    "french": french,
    "n_eval_prompts": N_EVAL,
}
with open("out/offset_results.json", "w") as f:
    json.dump(out, f)
print("wrote out/offset_results.json", flush=True)
