"""Fused elementwise helpers for step math: one kernel where the plain expression takes two."""

from __future__ import annotations

from typing import cast

import torch

from crematorium import Tensor

__all__ = ["mul_add"]


def mul_add(a: Tensor | float, b: Tensor | float, c: Tensor | float) -> Tensor:
    """
    ``a * b + c`` as a single ``addcmul`` when every operand is a tensor of
    one dtype on one device, else the plain expression (so Python scalars and
    mixed dtypes keep PyTorch's usual type promotion). Same value and the same
    gradients either way; in eager mode the fused form launches one kernel
    instead of two. Under ``torch.compile`` both forms fuse identically.
    """
    if (
        isinstance(a, Tensor)
        and isinstance(b, Tensor)
        and isinstance(c, Tensor)
        and a.dtype == b.dtype == c.dtype
        and a.device == b.device == c.device
    ):
        return torch.addcmul(c, a, b)
    return cast(Tensor, a * b + c)
