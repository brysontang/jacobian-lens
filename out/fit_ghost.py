"""Fit the backward (ghost-token) lens + forward Jacobian lens on Qwen3.5-2B.

Run from the repo root:  .venv/bin/python out/fit_ghost.py
Checkpointed; safe to kill and re-run. Constants below are set from the MPS
benchmark (see out/README-ghost.md).
"""

import time

import torch
import transformers

import jlens
from jlens.backward import fit_backward
from jlens.examples import load_wikitext_prompts

jlens.configure_logging()

# Qwen3.5-2B was benchmarked and rejected: fp16 gradients NaN by layer 20
# (hybrid linear-attention, torch fallback), and at ~25s per block-depth unit
# a 6-layer/24-prompt fit is >24h on this M1 Pro. Dense Qwen3-0.6B is fp16-
# stable to layer 26, ~1.3s/unit at seq 49, fp32 == fp16 numerically.
MODEL = "Qwen/Qwen3-0.6B"
N_PROMPTS = 24  # README §9.3: quality saturates quickly; watch max_d_mean
MAX_SEQ_LEN = 128
DIM_BATCH = 16
BACKWARD_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26]
FORWARD_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23, 26]  # same set -> crossover plot

t0 = time.perf_counter()
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="mps"
)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
print(f"loaded in {time.perf_counter()-t0:.0f}s", flush=True)

prompts = load_wikitext_prompts(N_PROMPTS)
print(f"{len(prompts)} wikitext prompts", flush=True)

backward = fit_backward(
    model,
    prompts,
    target_layers=BACKWARD_LAYERS,
    dim_batch=DIM_BATCH,
    max_seq_len=MAX_SEQ_LEN,
    checkpoint_path="out/backward_ckpt.pt",
)
backward.save("out/backward_lens.pt")
print("backward lens saved", flush=True)

forward = jlens.fit(
    model,
    prompts,
    source_layers=FORWARD_LAYERS,
    dim_batch=DIM_BATCH,
    max_seq_len=MAX_SEQ_LEN,
    checkpoint_path="out/forward_ckpt.pt",
)
forward.save("out/forward_lens.pt")
print("forward lens saved", flush=True)
