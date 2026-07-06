# Copyright 2026 Anthropic PBC (jlens); backward lens additions 2026.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from jlens.backward import (
    BackwardLens,
    backward_jacobian_for_prompt,
    fit_backward,
)

from .tiny import TinyDecoder


def _exact_K(model: TinyDecoder, layer: int) -> torch.Tensor:
    """TinyDecoder blocks are position-independent (no attention), so
    K_l = (I + w_l) @ ... @ (I + w_0) exactly."""
    K = torch.eye(model.d_model)
    for i in range(layer + 1):
        K = (torch.eye(model.d_model) + model.layers[i].linear.weight.detach()) @ K
    return K


def test_backward_jacobian_matches_analytic():
    """No attention -> no cross-position terms; the estimator must recover the
    exact product of block Jacobians, pinning orientation and indexing."""
    model = TinyDecoder(n_layers=4, d_model=8)
    for param in model.parameters():
        param.requires_grad_(False)  # embedding hook must root the graph itself
    prompt = "the quick brown fox " * 4
    jacobians, seq_len, n_valid = backward_jacobian_for_prompt(
        model, prompt, target_layers=[0, 1, 3], dim_batch=4, max_seq_len=64
    )
    assert set(jacobians) == {0, 1, 3}
    assert n_valid > 0 and seq_len > n_valid
    for layer in (0, 1, 3):
        assert jacobians[layer].shape == (8, 8)
        assert jacobians[layer].dtype == torch.float32
        torch.testing.assert_close(
            jacobians[layer], _exact_K(model, layer), rtol=0, atol=1e-5
        )


def test_negative_and_default_target_layers():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompt = "the quick brown fox " * 4
    neg, _, _ = backward_jacobian_for_prompt(
        model, prompt, target_layers=[-4, -1], dim_batch=4, max_seq_len=64
    )
    assert set(neg) == {0, 3}
    torch.testing.assert_close(neg[0], _exact_K(model, 0), rtol=0, atol=1e-5)
    all_layers, _, _ = backward_jacobian_for_prompt(
        model, prompt, target_layers=None, dim_batch=4, max_seq_len=64
    )
    assert set(all_layers) == {0, 1, 2, 3}
    with pytest.raises(ValueError, match="out of range"):
        backward_jacobian_for_prompt(model, prompt, target_layers=[9], dim_batch=4)


def test_ghost_top1_recovers_input_tokens():
    """Linear position-independent dynamics: h_l[p] == K_l e_{v(p)} exactly, so
    the l2 ghost readout must rank the actual input token first everywhere."""
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    lens = fit_backward(
        model, prompts, target_layers=[0, 2], dim_batch=4, max_seq_len=64
    )
    ghost, input_ids = lens.apply(model, "the quick brown fox jumps", layers=[0, 2])
    vocab_size = model.embed_tokens.weight.shape[0]
    seq_len = input_ids.shape[1]
    for layer in (0, 2):
        assert ghost[layer].shape == (seq_len, vocab_size)
        assert torch.equal(ghost[layer].argmax(-1), input_ids[0].cpu())
    # cosine is scale-invariant, so it must also recover the exact match.
    ghost_cos, _ = lens.apply(
        model, "the quick brown fox jumps", layers=[2], metric="cosine"
    )
    assert torch.equal(ghost_cos[2].argmax(-1), input_ids[0].cpu())


def test_apply_positions_metrics_and_baseline():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit_backward(
        model, ["abcdefghij " * 5], target_layers=[1], dim_batch=4, max_seq_len=64
    )
    vocab_size = model.embed_tokens.weight.shape[0]
    ghost, input_ids = lens.apply(
        model, "the quick brown fox jumps", layers=[1], positions=[0, -1]
    )
    assert ghost[1].shape == (2, vocab_size)
    # dot metric and embedding-lens baseline both produce logits.
    dot, _ = lens.apply(model, "hello world test", layers=[1], metric="dot")
    assert dot[1].shape[1] == vocab_size
    baseline, _ = lens.apply(
        model, "hello world test", layers=[3], use_jacobian=False
    )
    assert baseline[3].shape[1] == vocab_size
    with pytest.raises(ValueError, match="not in target_layers"):
        lens.apply(model, "x" * 30, layers=[3])
    with pytest.raises(ValueError, match="out of range"):
        lens.apply(model, "x" * 30, layers=[99], use_jacobian=False)
    with pytest.raises(ValueError, match="unknown metric"):
        lens.ghost_logits(model, torch.zeros(1, 8), 1, metric="euclidean")


def test_save_load_merge(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit_backward(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        target_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    assert lens.n_prompts == 2 and lens.target_layers == [0, 2]
    path = tmp_path / "backward.pt"
    lens.save(str(path))
    reloaded = BackwardLens.load(str(path))
    assert reloaded.target_layers == [0, 2] and reloaded.n_prompts == 2
    for layer in (0, 2):
        torch.testing.assert_close(
            reloaded.K[layer], lens.K[layer], rtol=0, atol=2e-3
        )  # fp16 round-trip
    merged = BackwardLens.merge([lens, reloaded])
    assert merged.n_prompts == 4
    mismatched = BackwardLens(K={1: torch.eye(8)}, n_prompts=1, d_model=8)
    with pytest.raises(ValueError, match="disagree"):
        BackwardLens.merge([lens, mismatched])
    with pytest.raises(ValueError, match="at least one"):
        BackwardLens.merge([])


def test_fit_backward_checkpoint_resume(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "x", "klmnopqrst " * 5]  # "x" too short -> skip
    checkpoint = str(tmp_path / "ckpt.pt")
    reference = fit_backward(
        model, prompts, target_layers=[0, 2], dim_batch=4, max_seq_len=64
    )
    assert reference.n_prompts == 2
    fit_backward(
        model,
        prompts,
        target_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    resumed = fit_backward(
        model,
        prompts,
        target_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    assert resumed.n_prompts == 2
    for layer in (0, 2):
        torch.testing.assert_close(resumed.K[layer], reference.K[layer])
    with pytest.raises(ValueError, match="target_layers"):
        fit_backward(
            model,
            prompts,
            target_layers=[0, 1],
            dim_batch=4,
            max_seq_len=64,
            checkpoint_path=checkpoint,
        )
