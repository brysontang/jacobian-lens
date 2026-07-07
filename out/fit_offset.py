"""Fit the offset-resolved K_Δ lens ("what, and from where") on Qwen3-0.6B.

Run from the repo root:  .venv/bin/python out/fit_offset.py
Checkpointed; safe to kill and re-run. Layers bracket the L20-23 handoff band
found by the summed-K experiments (out/README-ghost.md): before / during /
after. Cost scales with n_targets x layers, so both are kept modest;
N_PROMPTS can be raised later — the checkpoint resumes and the prompt list is
a stable prefix.
"""

import time

import torch
import transformers

import jlens
from jlens.examples import load_wikitext_prompts
from jlens.offset import fit_offset

jlens.configure_logging()

MODEL = "Qwen/Qwen3-0.6B"
N_PROMPTS = 16  # prefix of the same fit set as fit_ghost.py; eval still holds out [24:32)
MAX_SEQ_LEN = 128
DIM_BATCH = 16
N_TARGETS = 8
LAYERS = [14, 20, 23]  # before / during / after the handoff band

t0 = time.perf_counter()
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, device_map="mps"
)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
print(f"loaded in {time.perf_counter()-t0:.0f}s", flush=True)

prompts = load_wikitext_prompts(N_PROMPTS)
print(f"{len(prompts)} wikitext prompts", flush=True)

lens = fit_offset(
    model,
    prompts,
    target_layers=LAYERS,
    n_targets=N_TARGETS,
    dim_batch=DIM_BATCH,
    max_seq_len=MAX_SEQ_LEN,
    checkpoint_path="out/offset_ckpt.pt",
)
lens.save("out/offset_lens.pt")
print("offset lens saved", flush=True)
