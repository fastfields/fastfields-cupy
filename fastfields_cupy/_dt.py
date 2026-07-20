"""Distance-transform wrappers (cupy)."""

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
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
]


# --------------------------------------------------------------------------- #
# Euclidean / L1 distance transform (in-place along the last axis)            #
# --------------------------------------------------------------------------- #
def dt_euclidean(inp: Any, voxel_spacing: float = 1.0) -> Any:
    """Euclidean distance transform along the last axis (functional).

    Features are marked with ``0`` and background with ``+inf``. Returns a new
    cupy array; ``inp`` is left unmodified.
    """
    out = as_gpu_contiguous(inp, name="inp").copy()
    _ff.dt_euclidean(out, voxel_spacing, current_stream_ptr())
    return out


def dt_euclidean_(inp_out: Any, voxel_spacing: float = 1.0) -> Any:
    """In-place Euclidean distance transform along the last axis."""
    inp_out = require_gpu_contiguous(inp_out, name="inp_out")
    _ff.dt_euclidean(inp_out, voxel_spacing, current_stream_ptr())
    return inp_out


def dt_l1(inp: Any, voxel_spacing: float = 1.0) -> Any:
    """L1 distance transform along the last axis (functional)."""
    out = as_gpu_contiguous(inp, name="inp").copy()
    _ff.dt_l1(out, voxel_spacing, current_stream_ptr())
    return out


def dt_l1_(inp_out: Any, voxel_spacing: float = 1.0) -> Any:
    """In-place L1 distance transform along the last axis."""
    inp_out = require_gpu_contiguous(inp_out, name="inp_out")
    _ff.dt_l1(inp_out, voxel_spacing, current_stream_ptr())
    return inp_out


# --------------------------------------------------------------------------- #
# Point-to-spline distance                                                    #
# --------------------------------------------------------------------------- #
def _alloc_spline_outputs(loc: Any, coeff: Any) -> tuple[Any, Any, Any, Any]:
    cp = cupy()
    loc = as_gpu_contiguous(loc, name="loc")
    coeff = as_gpu_contiguous(coeff, name="coeff")
    batch = loc.shape[:-1]
    time = cp.empty(batch, dtype=loc.dtype)
    dist = cp.empty(batch, dtype=loc.dtype)
    return time, dist, loc, coeff


def dt_spline_table(
    loc: Any,
    coeff: Any,
    times: Any,
    spline: int = 3,
    bound: int = 3,
) -> tuple[Any, Any]:
    """Point-to-spline distance via a dictionary of candidate ``times``.

    Returns ``(time, dist)``, each shaped ``loc.shape[:-1]``.
    """
    time, dist, loc, coeff = _alloc_spline_outputs(loc, coeff)
    times = as_gpu_contiguous(times, name="times")
    _ff.dt_spline_table(
        time, dist, loc, coeff, times, spline, bound, current_stream_ptr()
    )
    return time, dist


def dt_spline_brent(
    loc: Any,
    coeff: Any,
    max_iter: int,
    tol: float,
    step: float,
    spline: int = 3,
    bound: int = 3,
) -> tuple[Any, Any]:
    """Point-to-spline distance via Brent's method. Returns ``(time, dist)``."""
    time, dist, loc, coeff = _alloc_spline_outputs(loc, coeff)
    _ff.dt_spline_brent(
        time, dist, loc, coeff, max_iter, tol, step, spline, bound,
        current_stream_ptr(),
    )
    return time, dist


def dt_spline_gaussnewton(
    loc: Any,
    coeff: Any,
    max_iter: int,
    tol: float,
    spline: int = 3,
    bound: int = 3,
) -> tuple[Any, Any]:
    """Point-to-spline distance via Gauss-Newton. Returns ``(time, dist)``."""
    time, dist, loc, coeff = _alloc_spline_outputs(loc, coeff)
    _ff.dt_spline_gaussnewton(
        time, dist, loc, coeff, max_iter, tol, spline, bound,
        current_stream_ptr(),
    )
    return time, dist


# --------------------------------------------------------------------------- #
# Point-to-mesh distance                                                      #
# --------------------------------------------------------------------------- #
def dt_mesh(
    loc: Any,
    vertices: Any,
    faces: Any,
    signed: bool = True,
    naive: bool = False,
    return_nearest: bool = False,
) -> Any:
    """Point-to-triangular-mesh (squared) distance.

    ``loc`` is ``(..., D)``; ``vertices`` is ``(V, D)``; ``faces`` indexes
    vertices. Returns ``dist`` shaped ``loc.shape[:-1]``. If ``return_nearest``
    is set, returns ``(dist, nearest_vertex)`` with ``nearest_vertex`` holding
    the index of the closest vertex.
    """
    cp = cupy()
    loc = as_gpu_contiguous(loc, name="loc")
    vertices = as_gpu_contiguous(vertices, name="vertices")
    faces = cp.ascontiguousarray(faces)
    batch = loc.shape[:-1]
    dist = cp.empty(batch, dtype=loc.dtype)
    nearest = cp.empty(batch, dtype=cp.int64) if return_nearest else None
    _ff.dt_mesh(
        dist, nearest, loc, vertices, faces, signed, naive,
        current_stream_ptr(),
    )
    if return_nearest:
        return dist, nearest
    return dist
