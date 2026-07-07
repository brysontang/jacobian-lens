"""Causal test: is deep-layer ghost "junk" output-flavored?

Hypothesis: the L23 ghost top-1 at a dissolved position is "the input token
that would have caused this position's prediction". If so, *substituting* the
ghost token for the actual input token should preserve the model's next-token
prediction at that position far better than matched junk.

Per diverged position (ghost@23 top-1 != actual token) we rerun the model with
the position's token replaced by:
  ghost    — its own L23 ghost top-1
  shuffled — another diverged position's ghost top-1 (same prompt): junk with
             the same vocabulary flavor, wrong position
  random   — uniform-random vocab token

and measure, against the clean run at that position: top-1 agreement and the
rank of the clean top-1 under the substituted distribution.

Writes out/ghost_patch.json. Run from repo root:
  .venv/bin/python out/ghost_patch.py
"""

import json
import time

import torch
import transformers

import jlens
from jlens.backward import BackwardLens
from jlens.examples import load_wikitext_prompts

jlens.configure_logging()

SKIP = 16
N_FIT = 24
N_EVAL = 8
LAYER = 23
PER_PROMPT = 40  # diverged positions sampled per prompt
BATCH = 16
SEED = 0

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", dtype=torch.float16, device_map="mps"
)
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = jlens.from_hf(hf, tok)
backward = BackwardLens.load("out/backward_lens.pt")
gen = torch.Generator().manual_seed(SEED)


@torch.no_grad()
def logits_at(seqs: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Model logits at positions[i] for seqs[i]; batched."""
    out = []
    for i in range(0, len(seqs), BATCH):
        chunk = seqs[i : i + BATCH].to(model.input_device)
        logits = hf(chunk).logits.float().cpu()
        for j, p in enumerate(positions[i : i + BATCH]):
            out.append(logits[j, p])
    return torch.stack(out)


results = []
prompts = load_wikitext_prompts(N_FIT + N_EVAL)[N_FIT:]
for pi, prompt in enumerate(prompts):
    t0 = time.perf_counter()
    ghost, ids = backward.apply(model, prompt, max_seq_len=128)
    ids = ids[0].cpu()
    n = len(ids)
    g_top1 = ghost[LAYER].argmax(-1)

    diverged = [
        p for p in range(SKIP, n - 1) if int(g_top1[p]) != int(ids[p])
    ]
    perm = torch.randperm(len(diverged), generator=gen)
    test_pos = sorted(diverged[i] for i in perm[:PER_PROMPT])
    if len(test_pos) < 2:
        continue

    clean_logits = logits_at(ids[None, :].expand(len(test_pos), -1), test_pos)

    # shuffled-ghost assignment: derangement over the test positions
    while True:
        sh = torch.randperm(len(test_pos), generator=gen)
        if all(int(sh[i]) != i for i in range(len(test_pos))):
            break

    conds = {}
    conds["ghost"] = [int(g_top1[p]) for p in test_pos]
    conds["shuffled"] = [int(g_top1[test_pos[int(sh[i])]]) for i in range(len(test_pos))]
    conds["random"] = torch.randint(0, model.d_vocab if hasattr(model, "d_vocab") else int(hf.config.vocab_size), (len(test_pos),), generator=gen).tolist()

    for cond, subs in conds.items():
        seqs = ids[None, :].repeat(len(test_pos), 1).clone()
        for i, (p, s) in enumerate(zip(test_pos, subs)):
            seqs[i, p] = s
        sub_logits = logits_at(seqs, test_pos)
        for i, p in enumerate(test_pos):
            c = clean_logits[i]
            s = sub_logits[i]
            c_top1 = int(c.argmax())
            results.append({
                "prompt": pi,
                "pos": p,
                "cond": cond,
                "actual": tok.decode([int(ids[p])]),
                "sub": tok.decode([subs[i]]),
                "clean_top1": tok.decode([c_top1]),
                "agree": bool(int(s.argmax()) == c_top1),
                "rank_clean_top1": int((s > s[c_top1]).sum()),
                "ghost_rank_of_actual": int(
                    (ghost[LAYER][p] > ghost[LAYER][p][int(ids[p])]).sum()
                ),
            })
    print(
        f"prompt {pi + 1}/{N_EVAL}: {len(test_pos)} positions "
        f"({time.perf_counter() - t0:.0f}s)",
        flush=True,
    )

with open("out/ghost_patch.json", "w") as f:
    json.dump(results, f, ensure_ascii=False)

print(f"\n{len(results)} rows ({len(results) // 3} positions x 3 conditions)")
for cond in ("ghost", "shuffled", "random"):
    rows = [r for r in results if r["cond"] == cond]
    agree = sum(r["agree"] for r in rows) / len(rows)
    ranks = sorted(r["rank_clean_top1"] for r in rows)
    med = ranks[len(ranks) // 2]
    r10 = sum(r < 10 for r in ranks) / len(ranks)
    print(
        f"{cond:>9}: top-1 preserved {agree:5.1%}   "
        f"clean top-1 in sub top-10 {r10:5.1%}   median rank {med}"
    )
