"""Distance-transform wrappers (cupy)."""

from __future__ import annotations

from typing import Any

import fastfields.dlpack as _ff

from ._util import (
    as_gpu_array,
    broadcast_batch,
    cupy,
    current_stream_ptr,
    require_gpu_writethrough,
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
    out = as_gpu_array(inp, name="inp").copy()
    _ff.dt_euclidean(out, voxel_spacing, current_stream_ptr())
    return out


def dt_euclidean_(inp_out: Any, voxel_spacing: float = 1.0) -> Any:
    """In-place Euclidean distance transform along the last axis."""
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    _ff.dt_euclidean(inp_out, voxel_spacing, current_stream_ptr())
    return inp_out


def dt_l1(inp: Any, voxel_spacing: float = 1.0) -> Any:
    """L1 distance transform along the last axis (functional)."""
    out = as_gpu_array(inp, name="inp").copy()
    _ff.dt_l1(out, voxel_spacing, current_stream_ptr())
    return out


def dt_l1_(inp_out: Any, voxel_spacing: float = 1.0) -> Any:
    """In-place L1 distance transform along the last axis."""
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    _ff.dt_l1(inp_out, voxel_spacing, current_stream_ptr())
    return inp_out


# --------------------------------------------------------------------------- #
# Point-to-spline distance                                                    #
# --------------------------------------------------------------------------- #
def dt_spline_table(
    loc: Any,
    coeff: Any,
    times: Any,
    spline: int = 3,
    bound: int = 3,
) -> tuple[Any, Any]:
    """Point-to-spline distance via a dictionary of candidate ``times``.

    ``loc`` core ``(D,)``, ``coeff`` core ``(N, D)``, ``times`` core ``(K,)``;
    batch dims are broadcast (zero-copy) and ``time``/``dist`` are allocated
    with the broadcast batch shape. Returns ``(time, dist)``.
    """
    cp = cupy()
    loc = as_gpu_array(loc, name="loc")
    coeff = as_gpu_array(coeff, name="coeff")
    times = as_gpu_array(times, name="times")
    batch, (loc_b, coeff_b, times_b) = broadcast_batch(
        [(loc, 1), (coeff, 2), (times, 1)]
    )
    time = cp.empty(batch, dtype=loc.dtype)
    dist = cp.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_table(
        time,
        dist,
        loc_b,
        coeff_b,
        times_b,
        spline,
        bound,
        current_stream_ptr(),
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
    """Point-to-spline distance via Brent's method. Returns ``(time, dist)``.

    ``loc`` core ``(D,)``, ``coeff`` core ``(N, D)``; batch dims broadcast.
    """
    cp = cupy()
    loc = as_gpu_array(loc, name="loc")
    coeff = as_gpu_array(coeff, name="coeff")
    batch, (loc_b, coeff_b) = broadcast_batch([(loc, 1), (coeff, 2)])
    time = cp.empty(batch, dtype=loc.dtype)
    dist = cp.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_brent(
        time,
        dist,
        loc_b,
        coeff_b,
        max_iter,
        tol,
        step,
        spline,
        bound,
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
    """Point-to-spline distance via Gauss-Newton. Returns ``(time, dist)``.

    ``loc`` core ``(D,)``, ``coeff`` core ``(N, D)``; batch dims broadcast.
    """
    cp = cupy()
    loc = as_gpu_array(loc, name="loc")
    coeff = as_gpu_array(coeff, name="coeff")
    batch, (loc_b, coeff_b) = broadcast_batch([(loc, 1), (coeff, 2)])
    time = cp.empty(batch, dtype=loc.dtype)
    dist = cp.empty(batch, dtype=loc.dtype)
    _ff.dt_spline_gaussnewton(
        time,
        dist,
        loc_b,
        coeff_b,
        max_iter,
        tol,
        spline,
        bound,
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
    loc = as_gpu_array(loc, name="loc")
    vertices = as_gpu_array(vertices, name="vertices")
    faces = cp.asarray(faces)  # native strides preserved (stride-aware kernel)
    # cores: loc (D,), vertices (N, D), faces (M, D); batch dims broadcast.
    batch, (loc_b, vert_b, faces_b) = broadcast_batch(
        [(loc, 1), (vertices, 2), (faces, 2)]
    )
    dist = cp.empty(batch, dtype=loc.dtype)
    nearest = cp.empty(batch, dtype=cp.int64) if return_nearest else None
    _ff.dt_mesh(
        dist,
        nearest,
        loc_b,
        vert_b,
        faces_b,
        signed,
        naive,
        current_stream_ptr(),
    )
    if return_nearest:
        return dist, nearest
    return dist
