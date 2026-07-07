# Copyright 2026 Anthropic PBC (jlens); backward lens additions 2026.
# SPDX-License-Identifier: Apache-2.0
"""Offset-resolved backward lens: "what, and from where".

:mod:`jlens.backward` fits ``K_l = E[dh_l / d emb]`` with the cotangent seeded
at every valid position at once, so the gradient at embedding position ``p``
sums ``dh_l[p'] / d emb[p]`` over all targets ``p' >= p`` — the readout is
blind to *where* the as-if input sits relative to the state being decoded.

Here the cotangent is seeded at a single target position ``t`` per backward
pass, so the gradient at embedding position ``p`` is the clean single term
``dh_l[t] / d emb[p]``. Binning by lookback distance ``delta = t - p`` gives
one matrix per distance band::

    K_bin = E[ dh_l[p + delta] / d emb[p] ],  delta in bin

Decoding a residual against the whole family yields a ghost *profile over
distance*: what token the state is behaving as if it received here (bin 0),
one back, eight back, ... — the instrument for detecting cross-position
copying/pivots that the offset-summed lens cannot see.

Cost: passes scale with ``n_targets`` (per layer, per dim batch) instead of a
single all-positions pass, so fit few layers and modest ``n_targets``.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import torch

from jlens.backward import BackwardLens, _check_target_layers, _EmbeddingRecorder
from jlens.fitting import SKIP_FIRST_N_POSITIONS, _atomic_save
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Inclusive (lo, hi) lookback-distance bands. Single-distance bins where the
#: profile is sharpest (adjacent tokens), widening geometrically.
OFFSET_BINS: tuple[tuple[int, int], ...] = (
    (0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 7), (8, 11), (12, 15),
    (16, 23), (24, 31), (32, 47), (48, 63), (64, 95), (96, 127),
)


def _pick_targets(seq_len: int, n_targets: int, skip_first: int) -> list[int]:
    """Evenly spaced target positions in ``[skip_first, seq_len - 1]``."""
    if seq_len - 1 < skip_first:
        raise ValueError(
            f"prompt too short: seq_len={seq_len} <= skip_first={skip_first}"
        )
    spaced = torch.linspace(skip_first, seq_len - 1, n_targets).round().long()
    return sorted(set(spaced.tolist()))


def offset_jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    target_layers: Sequence[int],
    *,
    n_targets: int = 8,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    bins: Sequence[tuple[int, int]] = OFFSET_BINS,
) -> tuple[dict[int, dict[int, torch.Tensor]], dict[int, int], int, list[int]]:
    """Single-prompt sums for the offset estimator.

    For each target layer, each target position ``t`` gets its own backward
    passes (one-hot cotangents at ``h_l[t]`` only, dim-batched); the gradient
    row at embedding position ``p >= skip_first`` is accumulated into the bin
    containing ``t - p``.

    Returns:
        ``(sums, pair_counts, seq_len, targets)`` where
        ``sums[layer][bin_idx]`` is the ``[d_model, d_model]`` fp32 sum of
        ``dh_l[t]/d emb[p]`` over the ``pair_counts[bin_idx]`` pairs ``(t, p)``
        that fell in the bin (counts are layer-independent). Divide by the
        counts (across prompts) to get the mean ``K_bin``.
    """
    n_layers, d_model = model.n_layers, model.d_model
    targets_l = _check_target_layers(target_layers, n_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    positions = _pick_targets(seq_len, n_targets, skip_first)

    sums = {
        layer: {
            b: torch.zeros(d_model, d_model, dtype=torch.float32)
            for b in range(len(bins))
        }
        for layer in targets_l
    }
    pair_counts = {b: 0 for b in range(len(bins))}
    for t in positions:
        for b, (lo, hi) in enumerate(bins):
            p_lo, p_hi = max(skip_first, t - hi), t - lo
            if p_hi >= p_lo:
                pair_counts[b] += p_hi - p_lo + 1

    n_passes = math.ceil(d_model / dim_batch)
    with (
        _EmbeddingRecorder(model.embed_tokens) as emb_recorder,
        ActivationRecorder(model.layers, at=targets_l) as recorder,
        torch.enable_grad(),
    ):
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        embedding = emb_recorder.output
        if embedding is None:
            raise RuntimeError("embed_tokens hook never fired during forward()")
        batch_indices = torch.arange(dim_batch, device=embedding.device)

        # Deepest layer last; only the very last (layer, target, pass) may
        # free the graph.
        for layer_idx, layer in enumerate(targets_l):
            target_activation = recorder.activations[layer]
            cotangent = torch.zeros_like(target_activation)
            for t_idx, t in enumerate(positions):
                is_last_target = (
                    layer_idx == len(targets_l) - 1 and t_idx == len(positions) - 1
                )
                for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
                    n_dims = min(dim_batch, d_model - dim_start)
                    cotangent.zero_()
                    cotangent[
                        batch_indices[:n_dims], t, dim_start + batch_indices[:n_dims]
                    ] = 1.0
                    grads = torch.autograd.grad(
                        outputs=target_activation,
                        inputs=embedding,
                        grad_outputs=cotangent,
                        retain_graph=not (
                            is_last_target and pass_idx == n_passes - 1
                        ),
                    )
                    grad = grads[0]  # [dim_batch, seq_len, d_model]
                    for b, (lo, hi) in enumerate(bins):
                        p_lo, p_hi = max(skip_first, t - hi), t - lo
                        if p_hi < p_lo:
                            continue
                        rows = (
                            grad[:n_dims, p_lo : p_hi + 1, :].float().sum(dim=1)
                        )
                        sums[layer][b][dim_start : dim_start + n_dims, :] += (
                            rows.cpu()
                        )
                    del grads, grad
            del cotangent
            logger.debug(
                "  layer %d done (%d/%d)", layer, layer_idx + 1, len(targets_l)
            )

    return sums, pair_counts, seq_len, positions


class OffsetLens:
    """A fitted offset-resolved lens: ``K`` per (layer, distance bin).

    Attributes:
        K: ``{(layer, bin_idx): Tensor[d_model, d_model]}``.
        bins: The (lo, hi) inclusive lookback bands, indexed by ``bin_idx``.
        pair_counts: ``{bin_idx: n_pairs}`` the means were taken over.
    """

    def __init__(
        self,
        K: dict[tuple[int, int], torch.Tensor],
        *,
        bins: Sequence[tuple[int, int]],
        pair_counts: dict[int, int],
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.K = {key: k.float() for key, k in K.items()}
        self.bins = [tuple(b) for b in bins]
        self.pair_counts = dict(pair_counts)
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.target_layers = sorted({layer for layer, _ in self.K})
        # Ghost scoring (incl. ||K e_v||^2 caching) is delegated to one
        # BackwardLens per (layer, bin); built lazily.
        self._sub: dict[tuple[int, int], BackwardLens] = {}

    def __repr__(self) -> str:
        return (
            f"OffsetLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"layers={self.target_layers}, {len(self.bins)} bins)"
        )

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        torch.save(
            {
                "K_offset": {key: k.to(dtype) for key, k in self.K.items()},
                "bins": [list(b) for b in self.bins],
                "pair_counts": self.pair_counts,
                "n_prompts": self.n_prompts,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> OffsetLens:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "K_offset" not in checkpoint:
            raise ValueError(
                f"{path} is not an OffsetLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit checkpoint?)"
            )
        return cls(
            K=checkpoint["K_offset"],
            bins=[tuple(b) for b in checkpoint["bins"]],
            pair_counts=checkpoint["pair_counts"],
            n_prompts=checkpoint["n_prompts"],
            d_model=checkpoint["d_model"],
        )

    def _delegate(self, layer: int, bin_idx: int) -> BackwardLens:
        key = (layer, bin_idx)
        if key not in self.K:
            raise ValueError(
                f"(layer={layer}, bin={bin_idx}) not fitted; have layers "
                f"{self.target_layers}, bins 0..{len(self.bins) - 1}"
            )
        if key not in self._sub:
            self._sub[key] = BackwardLens(
                K={layer: self.K[key]}, n_prompts=self.n_prompts,
                d_model=self.d_model,
            )
        return self._sub[key]

    def ghost_logits(
        self,
        model: LensModel,
        residual: torch.Tensor,
        layer: int,
        bin_idx: int,
        *,
        metric: str = "l2",
    ) -> torch.Tensor:
        """Ghost logits for "the as-if token ``bins[bin_idx]`` back"."""
        return self._delegate(layer, bin_idx).ghost_logits(
            model, residual, layer, metric=metric
        )

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layer: int,
        bin_indices: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        metric: str = "l2",
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Ghost profile over distance at one layer.

        Returns ``({bin_idx: [n_positions, vocab]}, input_ids)``: for each
        requested bin, the readout "what token ``bins[bin_idx]`` positions
        back would explain this state".
        """
        if bin_indices is None:
            bin_indices = sorted(b for (l, b) in self.K if l == layer)
        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=[layer]) as recorder:
            model.forward(input_ids)
            full = recorder.activations[layer].detach()[0]
        residual = full if positions is None else full[list(positions)]
        ghost = {
            b: self.ghost_logits(model, residual, layer, b, metric=metric)
            for b in bin_indices
        }
        return ghost, input_ids


def fit_offset(
    model: LensModel,
    prompts: Sequence[str],
    *,
    target_layers: Sequence[int],
    n_targets: int = 8,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    bins: Sequence[tuple[int, int]] = OFFSET_BINS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> OffsetLens:
    """Fit ``K`` per (layer, distance bin) over prompts; see
    :func:`jlens.backward.fit_backward` for checkpointing semantics."""
    n_layers, d_model = model.n_layers, model.d_model
    targets_l = _check_target_layers(target_layers, n_layers)
    bins = [tuple(b) for b in bins]

    logger.info(
        "fit_offset: %d layers x %d bins, n_targets=%d, %d prompts",
        len(targets_l), len(bins), n_targets, len(prompts),
    )

    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("target_layers", targets_l),
            ("skip_first", skip_first),
            ("n_targets", n_targets),
            ("bins", [list(b) for b in bins]),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard"
                )
        K_sum = state["K_sum"]
        pair_counts = state["pair_counts"]
        n_done, next_idx = state["n_done"], state["next_idx"]
        logger.info("  resuming: %d/%d prompts processed", next_idx, len(prompts))
    else:
        K_sum = {
            (layer, b): torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in targets_l
            for b in range(len(bins))
        }
        pair_counts = {b: 0 for b in range(len(bins))}
        n_done, next_idx = 0, 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "K_sum": K_sum,
                    "pair_counts": pair_counts,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "target_layers": targets_l,
                    "skip_first": skip_first,
                    "n_targets": n_targets,
                    "bins": [list(b) for b in bins],
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    deepest = targets_l[-1]
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            sums, counts, seq_len, positions = offset_jacobian_for_prompt(
                model,
                prompt,
                targets_l,
                n_targets=n_targets,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
                bins=bins,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        for layer in targets_l:
            for b in range(len(bins)):
                K_sum[(layer, b)] += sums[layer][b]
        for b in range(len(bins)):
            pair_counts[b] += counts[b]
        n_done += 1
        next_idx = prompt_idx + 1

        k0 = K_sum[(deepest, 0)] / max(1, pair_counts[0])
        logger.info(
            "  prompt %d/%d  seq_len=%d n_targets=%d  %.0fs  "
            "||K_0(L%d)||/sqrt(d)=%.3f  pairs(bin0)=%d",
            prompt_idx + 1, len(prompts), seq_len, len(positions),
            time.perf_counter() - start_time, deepest,
            k0.norm().item() / sqrt_d, pair_counts[0],
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    K_mean = {
        key: K_sum[key] / pair_counts[key[1]]
        for key in K_sum
        if pair_counts[key[1]] > 0
    }
    logger.info("fit_offset: done, %d prompts", n_done)
    return OffsetLens(
        K=K_mean, bins=bins, pair_counts=pair_counts,
        n_prompts=n_done, d_model=d_model,
    )
