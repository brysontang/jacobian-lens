# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.backward import BackwardLens, backward_jacobian_for_prompt, fit_backward
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.offset import OffsetLens, fit_offset, offset_jacobian_for_prompt
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "BackwardLens",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "OffsetLens",
    "backward_jacobian_for_prompt",
    "configure_logging",
    "fit",
    "fit_backward",
    "fit_offset",
    "from_hf",
    "jacobian_for_prompt",
    "offset_jacobian_for_prompt",
]
