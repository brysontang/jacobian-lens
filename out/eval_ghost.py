"""Ghost-token lens experiments on Qwen3.5-2B.

Produces out/ghost_results.json with:
  1. retention   — per-layer rank of the ACTUAL current input token under the
                   backward lens (l2), vs the embedding-lens baseline, vs the
                   forward lens's rank of the model's final top-1 NEXT token
                   (the crossover plot), on held-out wikitext.
  2. grids       — per-(layer, position) top-1 ghost tokens for the ASCII face
                   and a French prompt, plus forward-lens top-1 and per-cell
                   rank of the actual input token.
  3. cross_position — per-layer fractions: ghost top-1 is the token at this
                   position (self) / a token from elsewhere in the prompt
                   (moved) / a token not in the prompt at all (novel).

Run from repo root: .venv/bin/python out/eval_ghost.py
"""

import json
import time

import torch
import transformers

import jlens
from jlens.backward import BackwardLens
from jlens.examples import EXAMPLES, load_wikitext_prompts

jlens.configure_logging()

MODEL = "Qwen/Qwen3-0.6B"
SKIP = 16  # match fitting's valid-position convention
N_FIT = 24  # prompts consumed by fit_ghost.py; eval uses the ones after
N_EVAL = 8

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

backward = BackwardLens.load("out/backward_lens.pt")
forward = jlens.JacobianLens.load("out/forward_lens.pt")
B_LAYERS = backward.target_layers
F_LAYERS = forward.source_layers
print(f"backward layers {B_LAYERS}, forward layers {F_LAYERS}", flush=True)


def ranks_of(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Rank (0 = top) of targets[i] in logits[i]."""
    picked = logits.gather(1, targets[:, None])
    return (logits > picked).sum(-1)


# ---------------------------------------------------------------- retention
retention: dict[str, dict] = {"backward": {}, "baseline": {}, "forward": {}}
eval_prompts = load_wikitext_prompts(N_FIT + N_EVAL)[N_FIT:]
per_layer_b: dict[int, list] = {l: [] for l in B_LAYERS}
per_layer_base: dict[int, list] = {l: [] for l in B_LAYERS}
per_layer_f: dict[int, list] = {l: [] for l in F_LAYERS}

for i, prompt in enumerate(eval_prompts):
    t0 = time.perf_counter()
    ghost, ids = backward.apply(model, prompt, max_seq_len=128)
    base, _ = backward.apply(model, prompt, max_seq_len=128, use_jacobian=False)
    fwd, model_logits, _ = forward.apply(model, prompt, max_seq_len=128)
    ids = ids[0].cpu()
    n = len(ids)
    pos = torch.arange(SKIP, n - 1)
    actual = ids[pos]
    final_top1 = model_logits.argmax(-1)[pos]  # model's own next-token readout
    for l in B_LAYERS:
        per_layer_b[l].append(ranks_of(ghost[l][pos], actual))
        per_layer_base[l].append(ranks_of(base[l][pos], actual))
    for l in F_LAYERS:
        per_layer_f[l].append(ranks_of(fwd[l][pos], final_top1))
    print(f"eval prompt {i+1}/{N_EVAL} ({time.perf_counter()-t0:.0f}s)", flush=True)


def summarize(chunks: list[torch.Tensor]) -> dict:
    r = torch.cat(chunks).float()
    return {
        "top1_acc": float((r == 0).float().mean()),
        "top10_acc": float((r < 10).float().mean()),
        "median_rank": float(r.median()),
        "mean_log10_rank": float(torch.log10(r + 1).mean()),
        "n": int(r.numel()),
    }


for l in B_LAYERS:
    retention["backward"][l] = summarize(per_layer_b[l])
    retention["baseline"][l] = summarize(per_layer_base[l])
for l in F_LAYERS:
    retention["forward"][l] = summarize(per_layer_f[l])

# ------------------------------------------------- grids + cross-position
ascii_prompt = next(e for e in EXAMPLES if e.slug == "ascii-face").prompt


def analyze_prompt(prompt: str) -> dict:
    ghost, ids = backward.apply(model, prompt, max_seq_len=256)
    fwd, model_logits, _ = forward.apply(model, prompt, max_seq_len=256)
    ids = ids[0].cpu()
    n = len(ids)
    input_tokens = [tok.decode([t]) for t in ids]
    input_id_set = set(ids.tolist())
    grid: dict[str, list] = {}
    rank_grid: dict[str, list] = {}
    cross: dict[str, dict] = {}
    for l in B_LAYERS:
        top1 = ghost[l].argmax(-1)
        grid[str(l)] = [tok.decode([t]) for t in top1]
        rank_grid[str(l)] = ranks_of(ghost[l], ids).tolist()
        kinds = {"self": 0, "moved": 0, "novel": 0}
        for p in range(SKIP, n):
            t = int(top1[p])
            if t == int(ids[p]):
                kinds["self"] += 1
            elif t in input_id_set:
                kinds["moved"] += 1
            else:
                kinds["novel"] += 1
        total = max(1, n - SKIP)
        cross[str(l)] = {k: v / total for k, v in kinds.items()}
    fwd_grid = {
        str(l): [tok.decode([t]) for t in fwd[l].argmax(-1)] for l in F_LAYERS
    }
    fwd_grid["final"] = [tok.decode([t]) for t in model_logits.argmax(-1)]
    return {
        "input_tokens": input_tokens,
        "ghost_top1": grid,
        "actual_rank": rank_grid,
        "forward_top1": fwd_grid,
        "cross_position": cross,
    }


results = {
    "model": MODEL,
    "n_fit_prompts": backward.n_prompts,
    "backward_layers": B_LAYERS,
    "forward_layers": F_LAYERS,
    "retention": retention,
    "ascii_face": analyze_prompt(ascii_prompt),
    "french": analyze_prompt(FRENCH),
}
with open("out/ghost_results.json", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("wrote out/ghost_results.json", flush=True)
