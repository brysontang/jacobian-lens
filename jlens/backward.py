# Copyright 2026 Anthropic PBC (jlens); backward lens additions 2026.
# SPDX-License-Identifier: Apache-2.0
"""Fitting and applying the backward Jacobian lens ("ghost-token" lens).

Where the forward lens (:mod:`jlens.fitting`) reads out what an activation is
disposed to make the model *say*, the backward lens reads out what input token
the activation looks like it *came from*: the context the model is holding, as
reconstructed in the input-token basis.

The transport is the average embedding-to-layer Jacobian::

    K_l = E[ dh_l / d emb ]

and the readout enumerates the vocabulary instead of inverting ``K_l``: every
token embedding ``e_v`` is pushed forward through ``K_l`` and scored by how
well it explains the observed residual ``h``::

    ghost_logits_l(h)[v] = <h, K_l e_v> - 0.5 * ||K_l e_v||**2
                         = -0.5 * ||K_l e_v - h||**2 + const(v-free)

so the top token is the single input token whose layer-``l`` image, under the
model's average linearised dynamics, lands nearest ``h``. Because the
embedding lookup is linear, this is exactly the Jacobian readout with respect
to the one-hot input, computed in the cheap (``d_model``-sized) basis.

Estimator (:func:`backward_jacobian_for_prompt`): the mirror of
:func:`jlens.fitting.jacobian_for_prompt` with the roles flipped. The autograd
graph is rooted at the output of ``model.embed_tokens``; for each target layer
``l``, a one-hot cotangent is injected at every valid position of ``h_l`` and
the gradient is read at the embedding. The gradient at embedding position
``p`` is ``sum_{p' >= p} dh_l[p'] / d emb[p]`` (causal attention zeroes the
rest); we take the mean over valid positions ``p``, matching the forward
lens's reduction.

Positional encodings: rooting at the embedding *module output* keeps the
decode basis position-free for any architecture. RoPE models never add
position to the residual stream; for additive-positional models (GPT-2) the
positional term is added downstream of the hook, so it is absorbed into the
dynamics rather than contaminating the basis.

Cost: unlike the forward lens, reverse mode cannot amortise across target
layers (one backward per layer per output dim), so fitting every layer costs
roughly ``n_layers / 2`` times the forward fit. Fit a subset of layers.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Sequence

import torch
from torch import nn

from jlens.fitting import (
    SKIP_FIRST_N_POSITIONS,
    _atomic_save,
    valid_position_mask,
)
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Vocab rows scored per chunk in :meth:`BackwardLens.ghost_logits`; bounds the
#: fp32 copy of the embedding matrix to ``chunk * d_model * 4`` bytes.
_VOCAB_CHUNK = 32768


class _EmbeddingRecorder:
    """Capture ``model.embed_tokens``'s forward output and make it the autograd
    graph root (mirrors ``ActivationRecorder(start_graph_at=...)``, one module
    earlier than ``layers[0]``)."""

    def __init__(self, embed_module: nn.Module) -> None:
        self._module = embed_module
        self.output: torch.Tensor | None = None
        self._handle: torch.utils.hooks.RemovableHandle | None = None

    def __enter__(self) -> _EmbeddingRecorder:
        def hook(module: nn.Module, inputs, output) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            tensor.requires_grad_(True)
            self.output = tensor

        self._handle = self._module.register_forward_hook(hook)
        return self

    def __exit__(self, *exc) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _check_target_layers(
    target_layers: Sequence[int] | None, n_layers: int
) -> list[int]:
    """Resolve None/negative layer indices and bounds-check."""
    if target_layers is None:
        return list(range(n_layers))
    targets = sorted({l + n_layers if l < 0 else l for l in target_layers})
    if not targets or targets[0] < 0 or targets[-1] >= n_layers:
        raise ValueError(
            f"target_layers {sorted(target_layers)} out of range for {n_layers} layers"
        )
    return targets


def backward_jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    target_layers: Sequence[int] | None = None,
    *,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer estimator ``K_l = dh_l / d emb`` for one prompt.

    Runs one forward pass on the prompt replicated ``dim_batch`` times, retains
    the graph, then for each target layer runs ``ceil(d_model / dim_batch)``
    backward passes. Batch element ``b`` carries a one-hot cotangent at
    ``h_l`` dimension ``dim_start + b``, set at every valid position; the
    gradient is read at the embedding output. See the module docstring for the
    resulting estimator.

    Args:
        model: The model to compute Jacobians for.
        prompt: Input text.
        target_layers: Layers ``l`` to compute ``K_l`` at. ``None`` fits every
            layer (expensive; see the module docstring). Negative indices
            count from the end.
        dim_batch: ``h_l`` dimensions computed per backward pass.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude; see
            :func:`jlens.fitting.valid_position_mask`.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
        target layer to a ``[d_model, d_model]`` fp32 CPU tensor ``K_l`` with
        ``K_l[i, j] = dh_l[i] / d emb[j]`` under the estimator's reduction.
    """
    n_layers, d_model = model.n_layers, model.d_model
    targets = _check_target_layers(target_layers, n_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in targets
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        _EmbeddingRecorder(model.embed_tokens) as emb_recorder,
        ActivationRecorder(model.layers, at=targets) as recorder,
        torch.enable_grad(),
    ):
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        embedding = emb_recorder.output  # [dim_batch, seq_len, d_model]
        if embedding is None:
            raise RuntimeError("embed_tokens hook never fired during forward()")

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            embedding.device
        )
        batch_indices = torch.arange(dim_batch, device=embedding.device)

        # Deepest layer last: its final pass drops retain_graph and frees the
        # whole graph; every earlier (layer, pass) must retain it.
        for layer_idx, layer in enumerate(targets):
            target_activation = recorder.activations[layer]
            cotangent = torch.zeros_like(target_activation)
            is_last_layer = layer_idx == len(targets) - 1
            for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
                n_dims_this_pass = min(dim_batch, d_model - dim_start)
                cotangent.zero_()
                cotangent[
                    batch_indices[:n_dims_this_pass, None],
                    valid_positions[None, :],
                    dim_start + batch_indices[:n_dims_this_pass, None],
                ] = 1.0
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=embedding,
                    grad_outputs=cotangent,
                    retain_graph=not (is_last_layer and pass_idx == n_passes - 1),
                )
                grad = grads[0]  # [dim_batch, seq_len, d_model]
                positions_on_device = valid_positions.to(
                    grad.device, non_blocking=True
                )
                rows = (
                    grad[:n_dims_this_pass, positions_on_device, :]
                    .float()
                    .mean(dim=1)
                )
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = (
                    rows.cpu()
                )
                del grads, grad
            del cotangent
            logger.debug(
                "  layer %d done (%d/%d)", layer, layer_idx + 1, len(targets)
            )

    return jacobians, seq_len, n_valid_positions


class BackwardLens:
    """A fitted backward lens: per-layer ``K_l`` matrices and the ghost readout.

    Attributes:
        K: ``{layer_index: Tensor[d_model, d_model]}``. Each ``K_l`` maps an
            embedding-basis vector to its layer-``l`` image under the average
            dynamics.
        target_layers: Sorted list of fitted layer indices.
        n_prompts: Number of prompts the lens was averaged over.
        d_model: Residual-stream width.
    """

    def __init__(
        self,
        K: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.K = {layer: k.float() for layer, k in K.items()}
        self.target_layers = sorted(self.K)
        self.n_prompts = n_prompts
        self.d_model = d_model
        # ||K_l e_v||^2 per vocab token, cached per layer on first use. Tied to
        # whichever model's embedding is first passed in; a BackwardLens is
        # fitted for exactly one model, so this is not keyed by model.
        self._sq_cache: dict[int | str, torch.Tensor] = {}

    def __repr__(self) -> str:
        return (
            f"BackwardLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"target_layers=[{self.target_layers[0]}..{self.target_layers[-1]}] "
            f"({len(self.target_layers)} layers))"
        )

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save to ``path`` (fp16 by default, matching ``JacobianLens.save``)."""
        torch.save(
            {
                "K": {layer: k.to(dtype) for layer, k in self.K.items()},
                "n_prompts": self.n_prompts,
                "target_layers": self.target_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> BackwardLens:
        """Load a lens previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "K" not in checkpoint:
            raise ValueError(
                f"{path} is not a BackwardLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit checkpoint?)"
            )
        return cls(
            K=checkpoint["K"],
            n_prompts=checkpoint["n_prompts"],
            d_model=checkpoint["d_model"],
        )

    @classmethod
    def merge(cls, lenses: Sequence[BackwardLens]) -> BackwardLens:
        """Combine lenses fitted on disjoint prompt subsets
        (``n_prompts``-weighted mean)."""
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.target_layers != first.target_layers
                or other.d_model != first.d_model
            ):
                raise ValueError("lenses disagree on target_layers / d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.target_layers:
            weighted_sum = sum(lens.K[layer] * lens.n_prompts for lens in lenses)
            merged[layer] = weighted_sum / n_total
        return cls(K=merged, n_prompts=n_total, d_model=first.d_model)

    def _squared_norms(
        self, embed_weight: torch.Tensor, layer: int | None
    ) -> torch.Tensor:
        """``||K_l e_v||**2`` for every vocab token (``||e_v||**2`` when
        ``layer`` is None), chunked over the vocab; cached fp32 on CPU."""
        key: int | str = "embed" if layer is None else layer
        if key not in self._sq_cache:
            chunks = []
            gram = (
                None
                if layer is None
                else (self.K[layer].T @ self.K[layer]).to(embed_weight.device)
            )
            for rows in embed_weight.split(_VOCAB_CHUNK):
                rows = rows.float()
                if gram is None:
                    chunks.append((rows * rows).sum(-1))
                else:
                    chunks.append(((rows @ gram) * rows).sum(-1))
            self._sq_cache[key] = torch.cat(chunks).cpu()
        return self._sq_cache[key]

    def ghost_logits(
        self,
        model: LensModel,
        residual: torch.Tensor,
        layer: int,
        *,
        metric: str = "l2",
        use_jacobian: bool = True,
    ) -> torch.Tensor:
        """Score every vocab token as the as-if input explaining ``residual``.

        Args:
            model: Supplies the embedding matrix (decode basis).
            residual: ``[n_positions, d_model]`` residual at ``layer``.
            layer: Which fitted ``K_l`` to use.
            metric: ``"l2"`` (default): ``<h, K e_v> - 0.5 ||K e_v||^2``, the
                nearest forward-image. ``"dot"``: ``<h, K e_v>``, the
                transpose/sensitivity readout. ``"cosine"``: dot normalised by
                ``||K e_v||`` (rank-equivalent to per-position cosine).
            use_jacobian: If ``False``, score against raw embeddings
                (``K = I``): the embedding-lens baseline.

        Returns:
            ``[n_positions, vocab_size]`` fp32 CPU tensor of ghost logits.
        """
        if metric not in ("l2", "dot", "cosine"):
            raise ValueError(f"unknown metric {metric!r}")
        embed_weight = model.embed_tokens.weight  # [vocab, d_model]
        h = residual.float().to(embed_weight.device)
        if use_jacobian:
            K = self.K[layer].to(embed_weight.device)
            g = h @ K  # [n, d]: <h, K e_v> == <g, e_v>
        else:
            g = h
        dot_chunks = [
            g @ rows.float().T for rows in embed_weight.split(_VOCAB_CHUNK)
        ]
        dot = torch.cat(dot_chunks, dim=-1).cpu()  # [n, vocab]
        if metric == "dot":
            return dot
        sq = self._squared_norms(embed_weight, layer if use_jacobian else None)
        if metric == "l2":
            return dot - 0.5 * sq
        return dot / sq.clamp_min(1e-12).sqrt()

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        metric: str = "l2",
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Run ``model`` on ``prompt`` and return ghost logits at ``positions``.

        Mirror of :meth:`jlens.lens.JacobianLens.apply`; the second element of
        the return is ``input_ids`` (``[1, seq_len]``) — the ground-truth
        input tokens the ghost readout can be compared against.
        """
        if layers is None:
            layers = self.target_layers
        out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
            )
        unknown = set(layers) - set(self.target_layers)
        if use_jacobian and unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not in target_layers; "
                f"fitted layers are {self.target_layers}"
            )

        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=sorted(set(layers))) as recorder:
            model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in set(layers)}

        ghost: dict[int, torch.Tensor] = {}
        for layer in layers:
            full = activations[layer][0]  # [seq_len, d_model]
            residual = full if positions is None else full[list(positions)]
            ghost[layer] = self.ghost_logits(
                model, residual, layer, metric=metric, use_jacobian=use_jacobian
            )
        return ghost, input_ids


def fit_backward(
    model: LensModel,
    prompts: Sequence[str],
    *,
    target_layers: Sequence[int] | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> BackwardLens:
    """Fit ``K_l`` over a list of prompts and return a :class:`BackwardLens`.

    Mirror of :func:`jlens.fitting.fit`; see it for checkpointing semantics.
    ``target_layers=None`` fits every layer, which costs ~``n_layers/2`` times
    the forward fit (see the module docstring) — prefer an explicit subset.
    """
    n_layers, d_model = model.n_layers, model.d_model
    targets = _check_target_layers(target_layers, n_layers)

    logger.info(
        "fit_backward: n_layers=%d d_model=%d, fitting %d target layers on %d prompts",
        n_layers,
        d_model,
        len(targets),
        len(prompts),
    )

    K_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("target_layers", targets),
            ("skip_first", skip_first),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard it"
                )
        K_sum, n_done, next_idx = (
            state["K_sum"],
            state["n_done"],
            state["next_idx"],
        )
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            next_idx,
            len(prompts),
        )
    else:
        K_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in targets
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "K_sum": K_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "target_layers": targets,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_K, seq_len, n_valid = backward_jacobian_for_prompt(
                model,
                prompt,
                targets,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        prompt_norm = max(per_prompt_K[l].norm().item() for l in targets) / sqrt_d
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_prompt_K[l] - K_sum[l] / n_done).norm()
                    / ((n_done + 1) * (K_sum[l] / n_done).norm())
                ).item()
                for l in targets
            )
        else:
            mean_rel_change = float("nan")

        for layer in targets:
            K_sum[layer] += per_prompt_K[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  "
            "max||K||/sqrt(d)=%.3f  max_d_mean=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    K_mean = {layer: K_sum[layer] / n_done for layer in targets}
    logger.info("fit_backward: done, %d prompts", n_done)
    return BackwardLens(K=K_mean, n_prompts=n_done, d_model=d_model)
