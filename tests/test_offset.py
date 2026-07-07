# Copyright 2026 Anthropic PBC (jlens); backward lens additions 2026.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from jlens.offset import OffsetLens, fit_offset, offset_jacobian_for_prompt

from .test_backward import _exact_K
from .tiny import TinyDecoder

BINS = [(0, 0), (1, 1), (2, 3), (4, 7)]


def test_offset_jacobian_matches_analytic():
    """No attention -> strictly position-local dynamics: bin 0 must recover the
    exact product of block Jacobians and every delta > 0 bin must be zero."""
    model = TinyDecoder(n_layers=4, d_model=8)
    for param in model.parameters():
        param.requires_grad_(False)
    prompt = "the quick brown fox " * 4
    sums, counts, seq_len, targets = offset_jacobian_for_prompt(
        model, prompt, [1, 3], n_targets=4, dim_batch=4, max_seq_len=64, bins=BINS
    )
    assert len(targets) == 4 and counts[0] == 4  # one delta=0 pair per target
    for layer in (1, 3):
        K0 = sums[layer][0] / counts[0]
        torch.testing.assert_close(K0, _exact_K(model, layer), rtol=0, atol=1e-5)
        for b in (1, 2, 3):
            assert counts[b] > 0
            zero = sums[layer][b] / counts[b]
            torch.testing.assert_close(
                zero, torch.zeros(8, 8), rtol=0, atol=1e-5
            )


def test_offset_lens_bin0_recovers_input():
    """Under linear position-local dynamics the bin-0 ghost readout must rank
    the actual input token first at every position, like the summed lens."""
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit_offset(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        target_layers=[2],
        n_targets=4,
        dim_batch=4,
        max_seq_len=64,
        bins=BINS,
    )
    assert lens.target_layers == [2] and lens.n_prompts == 2
    ghost, input_ids = lens.apply(
        model, "the quick brown fox jumps", layer=2, bin_indices=[0]
    )
    assert torch.equal(ghost[0].argmax(-1), input_ids[0].cpu())
    with pytest.raises(ValueError, match="not fitted"):
        lens.ghost_logits(model, torch.zeros(1, 8), layer=3, bin_idx=0)


def test_offset_save_load_and_checkpoint_resume(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "x", "klmnopqrst " * 5]  # "x" too short -> skip
    kwargs = dict(
        target_layers=[1], n_targets=3, dim_batch=4, max_seq_len=64, bins=BINS
    )
    reference = fit_offset(model, prompts, **kwargs)
    assert reference.n_prompts == 2

    checkpoint = str(tmp_path / "ckpt.pt")
    fit_offset(model, prompts, checkpoint_path=checkpoint, **kwargs)
    resumed = fit_offset(model, prompts, checkpoint_path=checkpoint, **kwargs)
    for key in reference.K:
        torch.testing.assert_close(resumed.K[key], reference.K[key])
    with pytest.raises(ValueError, match="n_targets"):
        fit_offset(
            model, prompts, checkpoint_path=checkpoint,
            **{**kwargs, "n_targets": 5},
        )

    path = tmp_path / "offset.pt"
    reference.save(str(path))
    reloaded = OffsetLens.load(str(path))
    assert reloaded.bins == reference.bins
    assert reloaded.pair_counts == reference.pair_counts
    for key in reference.K:
        torch.testing.assert_close(
            reloaded.K[key], reference.K[key], rtol=0, atol=2e-3
        )  # fp16 round-trip
    with pytest.raises(ValueError, match="not an OffsetLens"):
        OffsetLens.load(str(tmp_path / "ckpt.pt"))
