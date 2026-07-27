"""Spatial regularisers — cupy.

* **field** — multi-channel field ``(*batch, *spatial, C)``; per-channel
  ``absolute`` / ``membrane`` / ``bending`` (a scalar broadcasts to ``C``).
* **flow** — vector flow field; scalar penalties.

Signatures match the numpy/torch wrappers so ``fastfields.any`` dispatches
consistently.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import fastfields.dlpack as _ff
from fastfields.dlpack import as_bound

from ._sym import sym_matvec, sym_solve
from ._util import as_gpu_array, cupy, current_stream_ptr

__all__ = [
    "field_matvec",
    "field_diag",
    "flow_matvec",
    "flow_diag",
    "flow_relax",
    "flow_precond",
    "flow_forward",
]


def _per_channel(value, channels: int, name: str) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)] * channels
    out = [float(v) for v in value]
    if len(out) != channels:
        raise ValueError(
            f"{name} must be a scalar or a length-C={channels} sequence"
        )
    return out


def _voxel(value, ndim: int) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)] * ndim
    out = [float(v) for v in value]
    if len(out) != ndim:
        raise ValueError(
            f"voxel_size must be a scalar or a length-ndim={ndim} sequence"
        )
    return out


def field_matvec(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the field regulariser (same shape as ``inp``)."""
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    channels = inp.shape[-1]
    out = cp.zeros(inp.shape, dtype=inp.dtype)
    _ff.field_matvec(
        out,
        inp,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_matvec(
    inp: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the flow regulariser (scalar penalties; same shape as ``inp``).

    ``shears`` (Lamé mu) and ``div`` (Lamé lambda) add the linear-elastic
    penalty coupling the flow channels.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    out = cp.zeros(inp.shape, dtype=inp.dtype)
    _ff.flow_matvec(
        out,
        inp,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def field_diag(
    shape: Sequence[int],
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: Any = None,
) -> Any:
    """Diagonal (preconditioner) of the field regulariser, shaped ``shape``."""
    cp = cupy()
    out = cp.zeros(tuple(int(s) for s in shape), dtype=dtype or cp.float64)
    channels = out.shape[-1]
    _ff.field_diag(
        out,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_diag(
    shape: Sequence[int],
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    dtype: Any = None,
) -> Any:
    """Diagonal (preconditioner) of the flow regulariser, shaped ``shape``."""
    cp = cupy()
    out = cp.zeros(tuple(int(s) for s in shape), dtype=dtype or cp.float64)
    _ff.flow_diag(
        out,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return out


def flow_relax(
    flow: Any,
    hes: Any,
    grd: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Any:
    """Refine ``flow`` in place with ``nb_iter`` relaxation sweeps.

    Solves ``(H + L) x = g`` with per-voxel symmetric Hessian ``hes`` and
    gradient ``grd``; ``flow`` is the warm start, mutated and returned.
    """
    flow = as_gpu_array(flow, name="flow")
    hes = as_gpu_array(hes, name="hes")
    grd = as_gpu_array(grd, name="grd")
    _ff.flow_relax(
        flow,
        hes,
        grd,
        _voxel(voxel_size, ndim),
        float(absolute),
        float(membrane),
        float(bending),
        float(shears),
        float(div),
        as_bound(bound),
        ndim,
        int(nb_iter),
        current_stream_ptr(),
    )
    return flow


def flow_precond(
    mat: Any,
    vec: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the preconditioner ``(M + diag(R)) \\ vec``.

    ``M`` is the per-voxel compact-symmetric matrix ``mat``; ``diag(R)`` is the
    diagonal of the flow regulariser (same penalties as :func:`flow_matvec`).
    A composition of :func:`flow_diag` and ``sym_solve`` — no new kernel.
    """
    vec = as_gpu_array(vec, name="vec")
    diag = flow_diag(
        vec.shape, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim, dtype=vec.dtype,
    )
    return sym_solve(mat, vec, diag)


def flow_forward(
    mat: Any,
    vec: Any,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the forward matrix-vector product ``(M + R) @ vec``.

    ``M`` is the per-voxel compact-symmetric matrix ``mat`` and ``R`` the flow
    regulariser operator. A composition of ``sym_matvec`` and
    :func:`flow_matvec` — no new kernel.
    """
    vec = as_gpu_array(vec, name="vec")
    out = sym_matvec(mat, vec)
    out = out + flow_matvec(
        vec, absolute, membrane, bending, shears, div,
        voxel_size=voxel_size, bound=bound, ndim=ndim,
    )
    return out
