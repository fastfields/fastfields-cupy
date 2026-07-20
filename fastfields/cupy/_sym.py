"""Compact-symmetric matrix operations (cupy).

The Hessian ``H`` is stored in a compact, diagonal-then-rows packing. For a
per-batch ``C x C`` symmetric matrix the packed length is ``C*(C+1)/2``:

* ``C == 2`` -> ``[h00, h11, h01]``
* ``C == 3`` -> ``[h00, h11, h22, h01, h02, h12]``

``inp`` / ``out`` vectors are shaped ``(..., C)``.
"""

from __future__ import annotations

from typing import Any

import fastfields.dlpack as _ff

from ._util import (
    as_gpu_array,
    broadcast_batch,
    broadcast_to_batch,
    cupy,
    current_stream_ptr,
    require_gpu_writethrough,
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
    """Return ``out = H @ inp`` (functional).

    Batch (leading) dims of ``hessian`` and ``inp`` are broadcast (zero-copy);
    ``out`` has the broadcast batch shape + ``inp``'s core dim ``(C,)``.
    """
    cp = cupy()
    hessian = as_gpu_array(hessian, name="hessian")
    inp = as_gpu_array(inp, name="inp")
    batch, (hessian_b, inp_b) = broadcast_batch([(hessian, 1), (inp, 1)])
    out = cp.empty(batch + (inp.shape[-1],), dtype=inp.dtype)
    _ff.sym_matvec(out, hessian_b, inp_b, current_stream_ptr())
    return out


def sym_matvec_backward(grd: Any, inp: Any) -> Any:
    """Backward of :func:`sym_matvec` w.r.t. the matrix.

    ``grd`` and ``inp`` are ``(..., C)`` (batch dims broadcast); returns the
    gradient in the same compact-symmetric packing, shape ``(..., C*(C+1)/2)``.
    """
    cp = cupy()
    grd = as_gpu_array(grd, name="grd")
    inp = as_gpu_array(inp, name="inp")
    c = inp.shape[-1]
    packed = c * (c + 1) // 2
    batch, (grd_b, inp_b) = broadcast_batch([(grd, 1), (inp, 1)])
    out = cp.empty(batch + (packed,), dtype=inp.dtype)
    _ff.sym_matvec_backward(out, grd_b, inp_b, current_stream_ptr())
    return out


def sym_addmatvec_(out: Any, hessian: Any, inp: Any) -> Any:
    """In-place accumulate: ``out += H @ inp``. Returns ``out``.

    ``out`` fixes the batch shape; ``hessian`` and ``inp`` are broadcast to it.
    """
    out = require_gpu_writethrough(out, name="out")
    hessian = as_gpu_array(hessian, name="hessian")
    inp = as_gpu_array(inp, name="inp")
    batch = out.shape[:-1]
    hessian_b = broadcast_to_batch(hessian, batch, 1)
    inp_b = broadcast_to_batch(inp, batch, 1)
    _ff.sym_addmatvec_(out, hessian_b, inp_b, current_stream_ptr())
    return out


def sym_submatvec_(out: Any, hessian: Any, inp: Any) -> Any:
    """In-place accumulate: ``out -= H @ inp``. Returns ``out``.

    ``out`` fixes the batch shape; ``hessian`` and ``inp`` are broadcast to it.
    """
    out = require_gpu_writethrough(out, name="out")
    hessian = as_gpu_array(hessian, name="hessian")
    inp = as_gpu_array(inp, name="inp")
    batch = out.shape[:-1]
    hessian_b = broadcast_to_batch(hessian, batch, 1)
    inp_b = broadcast_to_batch(inp, batch, 1)
    _ff.sym_submatvec_(out, hessian_b, inp_b, current_stream_ptr())
    return out


def sym_solve(hessian: Any, inp: Any, weight: Any = None) -> Any:
    """Return ``out = (H + diag(weight)) \\ inp`` (functional).

    ``weight`` is optional. Batch dims of ``hessian``/``inp``/``weight`` are
    broadcast (zero-copy).
    """
    cp = cupy()
    hessian = as_gpu_array(hessian, name="hessian")
    inp = as_gpu_array(inp, name="inp")
    if weight is None:
        batch, (hessian_b, inp_b) = broadcast_batch([(hessian, 1), (inp, 1)])
        out = cp.empty(batch + (inp.shape[-1],), dtype=inp.dtype)
        _ff.sym_solve(out, hessian_b, inp_b, None, current_stream_ptr())
    else:
        weight = as_gpu_array(weight, name="weight")
        batch, (hessian_b, inp_b, weight_b) = broadcast_batch(
            [(hessian, 1), (inp, 1), (weight, 1)]
        )
        out = cp.empty(batch + (inp.shape[-1],), dtype=inp.dtype)
        _ff.sym_solve(out, hessian_b, inp_b, weight_b, current_stream_ptr())
    return out


def sym_solve_(inp_out: Any, hessian: Any, weight: Any = None) -> Any:
    """In-place solve ``inp_out = (H + diag(weight)) \\ inp_out``.

    Returns ``inp_out``. It fixes the batch shape; ``hessian``/``weight`` are
    broadcast to it.
    """
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    hessian = as_gpu_array(hessian, name="hessian")
    batch = inp_out.shape[:-1]
    hessian_b = broadcast_to_batch(hessian, batch, 1)
    if weight is not None:
        weight = as_gpu_array(weight, name="weight")
        weight = broadcast_to_batch(weight, batch, 1)
    _ff.sym_solve_(inp_out, hessian_b, weight, current_stream_ptr())
    return inp_out


def sym_invert(hessian: Any) -> Any:
    """Return ``inv(H)`` (functional); both compact-symmetric."""
    cp = cupy()
    hessian = as_gpu_array(hessian, name="hessian")
    out = cp.empty_like(hessian)
    _ff.sym_invert(out, hessian, current_stream_ptr())
    return out


def sym_invert_(hessian: Any) -> Any:
    """In-place invert: ``hessian = inv(hessian)``. Returns ``hessian``."""
    hessian = require_gpu_writethrough(hessian, name="hessian")
    _ff.sym_invert_(hessian, current_stream_ptr())
    return hessian
