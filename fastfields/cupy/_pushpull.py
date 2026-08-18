"""Spline-interpolation gather / scatter (pushpull) — cupy.

Channel-last, x-first coordinate convention. The spatial rank ``D`` is taken
from ``grid``'s trailing axis; ``inp`` and ``grid`` must share the same rank.
Signatures match the numpy/torch wrappers so ``fastfields.auto`` dispatches
consistently.
"""

from __future__ import annotations

from typing import Any, Sequence

import fastfields.dlpack as _ff
from fastfields.helpers import as_bound, as_spline

from ._util import as_gpu_array, cupy, current_stream_ptr

__all__ = ["pull", "push", "count", "grad"]


def _spatial(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,) * ndim
    out = tuple(int(s) for s in shape)
    if len(out) != ndim:
        raise ValueError(f"shape must have length ndim={ndim}, got {shape!r}")
    return out


def pull(
    inp: Any,
    grid: Any,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Any:
    """Sample (pull) ``inp`` at ``grid`` -> ``(*B, *outshape, C)``."""
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    grid = as_gpu_array(grid, name="grid")
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    out = cp.zeros((*grid.shape[:-1], inp.shape[-1]), dtype=inp.dtype)
    _ff.pull(
        out,
        inp,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        current_stream_ptr(),
    )
    return out


def push(
    inp: Any,
    grid: Any,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Any:
    """Splat (push) ``inp`` into a volume of spatial size ``shape``.

    Adjoint of :func:`pull` -> ``(*B, *shape, C)``.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    grid = as_gpu_array(grid, name="grid")
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    if nbatch < 0:
        raise ValueError("grid rank is too small for the coordinate dim")
    spatial = _spatial(shape, ndim)
    out = cp.zeros(
        (*grid.shape[:nbatch], *spatial, inp.shape[-1]), dtype=inp.dtype
    )
    _ff.push(
        out,
        inp,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        current_stream_ptr(),
    )
    return out


def count(
    grid: Any,
    shape: int | Sequence[int],
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
) -> Any:
    """Splat ones into a volume of size ``shape`` -> ``(*B, *shape, 1)``."""
    cp = cupy()
    grid = as_gpu_array(grid, name="grid")
    ndim = grid.shape[-1]
    nbatch = grid.ndim - ndim - 1
    if nbatch < 0:
        raise ValueError("grid rank is too small for the coordinate dim")
    spatial = _spatial(shape, ndim)
    out = cp.zeros((*grid.shape[:nbatch], *spatial, 1), dtype=grid.dtype)
    _ff.count(
        out,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        current_stream_ptr(),
    )
    return out


def grad(
    inp: Any,
    grid: Any,
    order: int | str = 2,
    bound: int | str = "dct2",
    *,
    extrapolate: int = 1,
    abs: bool = False,
) -> Any:
    """Sample spatial gradients of ``inp`` at ``grid`` -> ``(...,C,D)``."""
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    grid = as_gpu_array(grid, name="grid")
    if grid.ndim != inp.ndim:
        raise ValueError("inp and grid must have the same rank")
    ndim = grid.shape[-1]
    out = cp.zeros((*grid.shape[:-1], inp.shape[-1], ndim), dtype=inp.dtype)
    _ff.grad(
        out,
        inp,
        grid,
        as_spline(order),
        as_bound(bound),
        int(extrapolate),
        bool(abs),
        current_stream_ptr(),
    )
    return out
