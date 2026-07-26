"""Resample / restriction / spline-coefficient wrappers (cupy)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import fastfields.dlpack as _ff
from fastfields.dlpack import (
    anchor_scale_shift,
    as_bound,
    as_spline,
    check_ndim,
    infer_ndim,
    resolve_out_spatial,
)

from ._util import (
    as_gpu_array,
    cupy,
    current_stream_ptr,
    require_gpu_writethrough,
)

__all__ = [
    "resample",
    "restriction",
    "spline_coeff",
    "spline_coeff_",
]


def _resolve(
    inp: Any,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
    ndim: Optional[int],
    anchor: str,
    scale: Optional[Sequence[float]],
    shift: Optional[float],
) -> tuple[tuple[int, ...], list[float], float]:
    """Resolve (full output shape, per-dim scale, scalar shift) for a call.

    The order/bound normalisation and the anchor/factor/shape resolution come
    from :mod:`fastfields.dlpack` so every backend shares one implementation.
    Raises ``ValueError`` if ``ndim`` is outside ``1..inp.ndim`` or an explicit
    ``scale`` has the wrong length.
    """
    ndim = infer_ndim(ndim, factor, shape)
    check_ndim(ndim, inp.ndim)
    spatial_in = tuple(inp.shape[-ndim:])
    out_spatial = resolve_out_spatial(spatial_in, ndim, factor, shape)
    a_scale, a_shift = anchor_scale_shift(
        anchor, spatial_in, out_spatial, ndim
    )
    if scale is not None:
        a_scale = [float(s) for s in scale]
        if len(a_scale) != ndim:
            raise ValueError(
                f"Expected scale of length ndim={ndim}, got {scale}."
            )
    if shift is not None:
        a_shift = float(shift)
    out_shape = tuple(int(n) for n in inp.shape[:-ndim]) + out_spatial
    return out_shape, a_scale, a_shift


def resample(
    inp: Any,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: Optional[int] = None,
    anchor: str = "centers",
    shift: Optional[float] = None,
    scale: Optional[Sequence[float]] = None,
) -> Any:
    """Spline resample (prolongation) of the last ``ndim`` axes.

    Allocates and returns the output array. The signature matches the
    numpy/torch wrappers so ``fastfields.any.resample`` dispatches
    consistently.

    Parameters
    ----------
    inp : cupy.ndarray
        Input array, shape ``(..., *inshape)``.
    factor : float or sequence of float, optional
        Per-axis resize multiplier (mutually exclusive with ``shape``; with
        neither, this is the identity).
    shape : int or sequence of int, optional
        Explicit output spatial size (the last ``ndim`` axes of the result).
    order : int or str, default=2
        Spline order (int ``0..7``, a :class:`Spline` enum, or a name such as
        ``"cubic"``).
    bound : int or str, default="dct2"
        Boundary condition (int, a :class:`Bound` enum, or a name such as
        ``"dct2"``/``"wrap"``).
    ndim : int, optional
        Number of trailing spatial dimensions (inferred when omitted).
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention, matching ``interpol.resize`` (see
        :func:`fastfields.dlpack.anchor_scale_shift`). Abbreviations
        accepted.
    shift : float, optional
        Sampling-shift override (default: the shift implied by ``anchor``).
    scale : sequence of float, optional
        Per-dim scale override (default: derived from ``anchor``).
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    out_shape, scale, shift = _resolve(
        inp, factor, shape, ndim, anchor, scale, shift
    )
    ndim = len(scale)
    out = cp.empty(out_shape, dtype=inp.dtype)
    _ff.resample(
        out,
        inp,
        as_spline(order),
        as_bound(bound),
        shift,
        scale,
        ndim,
        current_stream_ptr(),
    )
    return out


def restriction(
    inp: Any,
    factor: float | Sequence[float] | None = None,
    shape: int | Sequence[int] | None = None,
    *,
    order: int | str = 2,
    bound: int | str = "dct2",
    ndim: Optional[int] = None,
    anchor: str = "centers",
    shift: Optional[float] = None,
    scale: Optional[Sequence[float]] = None,
) -> Any:
    """Restriction (adjoint of :func:`resample`) of the last ``ndim`` axes.

    The binding *accumulates* into the output, so the freshly allocated array
    is zero-initialised here. Shares :func:`resample`'s ``factor``/``shape``/
    ``order`` signature; because the scale is derived from this call's own
    (input, output) shapes, a ``resample`` and a matching ``restriction`` use
    reciprocal scales and the same shift -- the adjoint the binding expects.

    Parameters
    ----------
    inp : cupy.ndarray
        Input array, shape ``(..., *inshape)``.
    factor : float or sequence of float, optional
        Per-axis resize multiplier (mutually exclusive with ``shape``).
    shape : int or sequence of int, optional
        Explicit output spatial size.
    order : int or str, default=2
        Spline order (see :func:`resample`).
    bound : int or str, default="dct2"
        Boundary condition (see :func:`resample`).
    ndim : int, optional
        Number of trailing spatial dimensions (inferred when omitted).
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention (see :func:`resample`).
    shift : float, optional
        Sampling-shift override (see :func:`resample`).
    scale : sequence of float, optional
        Per-dim scale override (see :func:`resample`).
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    out_shape, scale, shift = _resolve(
        inp, factor, shape, ndim, anchor, scale, shift
    )
    ndim = len(scale)
    out = cp.zeros(out_shape, dtype=inp.dtype)
    _ff.restriction(
        out,
        inp,
        as_spline(order),
        as_bound(bound),
        shift,
        scale,
        ndim,
        current_stream_ptr(),
    )
    return out


def spline_coeff(
    inp: Any, order: int | str = 3, bound: int | str = "dct2"
) -> Any:
    """Spline-coefficient prefilter along the last axis (functional).

    Orders 0/1 are no-ops. Returns a new array; ``inp`` is unmodified.
    """
    out = as_gpu_array(inp, name="inp").copy()
    _ff.spline_coeff(
        out, as_spline(order), as_bound(bound), current_stream_ptr()
    )
    return out


def spline_coeff_(
    inp_out: Any, order: int | str = 3, bound: int | str = "dct2"
) -> Any:
    """In-place spline-coeff prefilter (last axis); returns ``inp_out``."""
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    _ff.spline_coeff(
        inp_out, as_spline(order), as_bound(bound), current_stream_ptr()
    )
    return inp_out
