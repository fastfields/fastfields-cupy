"""Resample / restriction / spline-coefficient wrappers (cupy)."""

from __future__ import annotations

from typing import Any, Sequence

import fastfields_bind as _ff

from ._util import (
    as_gpu_contiguous,
    cupy,
    current_stream_ptr,
    require_gpu_contiguous,
)

__all__ = [
    "resample",
    "restriction",
    "spline_coeff",
    "spline_coeff_",
]


def resample(
    inp: Any,
    out_shape: Sequence[int],
    spline: int = 2,
    bound: int = 3,
    shift: float = 0.0,
    scale: Sequence[float] | None = None,
    ndim: int = 1,
) -> Any:
    """Spline resample (prolongation) of ``inp`` onto ``out_shape``.

    ``scale`` is a per-dim sequence of length ``ndim`` (input-index per
    output-index). Allocates and returns the output array.
    """
    cp = cupy()
    inp = as_gpu_contiguous(inp, name="inp")
    out = cp.empty(tuple(out_shape), dtype=inp.dtype)
    _ff.resample(
        out, inp, spline, bound, shift, scale, ndim, current_stream_ptr()
    )
    return out


def restriction(
    inp: Any,
    out_shape: Sequence[int],
    spline: int = 2,
    bound: int = 3,
    shift: float = 0.0,
    scale: Sequence[float] | None = None,
    ndim: int = 1,
) -> Any:
    """Restriction (adjoint of :func:`resample`) of ``inp`` onto ``out_shape``.

    The binding *accumulates* into the output, so the freshly allocated array
    is zero-initialised here.
    """
    cp = cupy()
    inp = as_gpu_contiguous(inp, name="inp")
    out = cp.zeros(tuple(out_shape), dtype=inp.dtype)
    _ff.restriction(
        out, inp, spline, bound, shift, scale, ndim, current_stream_ptr()
    )
    return out


def spline_coeff(inp: Any, spline: int = 3, bound: int = 3) -> Any:
    """Spline-coefficient prefilter along the last axis (functional).

    Orders 0/1 are no-ops. Returns a new array; ``inp`` is unmodified.
    """
    out = as_gpu_contiguous(inp, name="inp").copy()
    _ff.spline_coeff(out, spline, bound, current_stream_ptr())
    return out


def spline_coeff_(inp_out: Any, spline: int = 3, bound: int = 3) -> Any:
    """In-place spline-coefficient prefilter along the last axis. Returns ``inp_out``."""
    inp_out = require_gpu_contiguous(inp_out, name="inp_out")
    _ff.spline_coeff(inp_out, spline, bound, current_stream_ptr())
    return inp_out
