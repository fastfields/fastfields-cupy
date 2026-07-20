"""Compact-symmetric matrix operations (cupy).

The Hessian ``H`` is stored in a compact, diagonal-then-rows packing. For a
per-batch ``C x C`` symmetric matrix the packed length is ``C*(C+1)/2``:

* ``C == 2`` -> ``[h00, h11, h01]``
* ``C == 3`` -> ``[h00, h11, h22, h01, h02, h12]``

``inp`` / ``out`` vectors are shaped ``(..., C)``.
"""

from __future__ import annotations

from typing import Any

import fastfields_bind as _ff

from ._util import (
    as_gpu_contiguous,
    cupy,
    current_stream_ptr,
    require_gpu_contiguous,
)

__all__ = [
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
]


def sym_matvec(hessian: Any, inp: Any) -> Any:
    """Return ``out = H @ inp`` (functional). ``out`` shaped like ``inp``."""
    cp = cupy()
    hessian = as_gpu_contiguous(hessian, name="hessian")
    inp = as_gpu_contiguous(inp, name="inp")
    out = cp.empty_like(inp)
    _ff.sym_matvec(out, hessian, inp, current_stream_ptr())
    return out


def sym_matvec_backward(grd: Any, inp: Any) -> Any:
    """Backward of :func:`sym_matvec` w.r.t. the matrix.

    ``grd`` and ``inp`` are ``(..., C)``; returns the gradient in the same
    compact-symmetric packing as the Hessian, shape ``(..., C*(C+1)/2)``.
    """
    cp = cupy()
    grd = as_gpu_contiguous(grd, name="grd")
    inp = as_gpu_contiguous(inp, name="inp")
    c = inp.shape[-1]
    packed = c * (c + 1) // 2
    out = cp.empty(inp.shape[:-1] + (packed,), dtype=inp.dtype)
    _ff.sym_matvec_backward(out, grd, inp, current_stream_ptr())
    return out


def sym_addmatvec_(out: Any, hessian: Any, inp: Any) -> Any:
    """In-place accumulate: ``out += H @ inp``. Returns ``out``."""
    out = require_gpu_contiguous(out, name="out")
    hessian = as_gpu_contiguous(hessian, name="hessian")
    inp = as_gpu_contiguous(inp, name="inp")
    _ff.sym_addmatvec_(out, hessian, inp, current_stream_ptr())
    return out


def sym_submatvec_(out: Any, hessian: Any, inp: Any) -> Any:
    """In-place accumulate: ``out -= H @ inp``. Returns ``out``."""
    out = require_gpu_contiguous(out, name="out")
    hessian = as_gpu_contiguous(hessian, name="hessian")
    inp = as_gpu_contiguous(inp, name="inp")
    _ff.sym_submatvec_(out, hessian, inp, current_stream_ptr())
    return out


def sym_solve(hessian: Any, inp: Any, weight: Any = None) -> Any:
    """Return ``out = (H + diag(weight)) \\ inp`` (functional). ``weight`` optional."""
    cp = cupy()
    hessian = as_gpu_contiguous(hessian, name="hessian")
    inp = as_gpu_contiguous(inp, name="inp")
    if weight is not None:
        weight = as_gpu_contiguous(weight, name="weight")
    out = cp.empty_like(inp)
    _ff.sym_solve(out, hessian, inp, weight, current_stream_ptr())
    return out


def sym_solve_(inp_out: Any, hessian: Any, weight: Any = None) -> Any:
    """In-place solve: ``inp_out = (H + diag(weight)) \\ inp_out``. Returns ``inp_out``."""
    inp_out = require_gpu_contiguous(inp_out, name="inp_out")
    hessian = as_gpu_contiguous(hessian, name="hessian")
    if weight is not None:
        weight = as_gpu_contiguous(weight, name="weight")
    _ff.sym_solve_(inp_out, hessian, weight, current_stream_ptr())
    return inp_out


def sym_invert(hessian: Any) -> Any:
    """Return ``inv(H)`` (functional); both compact-symmetric."""
    cp = cupy()
    hessian = as_gpu_contiguous(hessian, name="hessian")
    out = cp.empty_like(hessian)
    _ff.sym_invert(out, hessian, current_stream_ptr())
    return out


def sym_invert_(hessian: Any) -> Any:
    """In-place invert: ``hessian = inv(hessian)``. Returns ``hessian``."""
    hessian = require_gpu_contiguous(hessian, name="hessian")
    _ff.sym_invert_(hessian, current_stream_ptr())
    return hessian
