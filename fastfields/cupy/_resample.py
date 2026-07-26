"""Resample / restriction / spline-coefficient wrappers (cupy)."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import fastfields.dlpack as _ff

from ._util import (
    as_bound,
    as_gpu_array,
    as_spline,
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


# torch-interpol anchor conventions (see ``interpol.resize``). Each anchor is
# identified by its first (lower-cased) letter, so both the full name
# ("centers") and the abbreviation ("c") are accepted.
_ANCHORS = ("c", "e", "f", "l")
_ANCHOR_SHIFT = {"e": 0.5, "f": 0.0, "l": 1.0}


def _anchor_scale_shift(
    anchor: str,
    inshape: Sequence[int],
    outshape: Sequence[int],
    ndim: int,
) -> tuple[list[float], float]:
    """Map a torch-interpol ``anchor`` to a per-dim scale and scalar shift.

    The fastfields resize kernel samples input coordinate
    ``scale[d] * loc + shift * (scale[d] - 1)`` for output index ``loc``. The
    four anchors of ``interpol.resize`` map onto ``(scale, shift)`` as:

    ==========  =================  =======
    anchor      scale[d]           shift
    ==========  =================  =======
    ``centers`` ``(in-1)/(out-1)`` ``0.0``
    ``edges``   ``in/out``         ``0.5``
    ``first``   ``in/out``         ``0.0``
    ``last``    ``in/out``         ``1.0``
    ==========  =================  =======

    Parameters
    ----------
    anchor : str
        Anchor name or abbreviation (``centers``/``edges``/``first``/``last``
        or ``c``/``e``/``f``/``l``); matched case-insensitively on the first
        letter, mirroring ``interpol.resize``.
    inshape, outshape : sequence of int
        Input and output spatial sizes (length ``ndim``).
    ndim : int
        Number of spatial dimensions.

    Returns
    -------
    scale : list of float
        Per-dim input-index step per output-index step.
    shift : float
        Scalar sampling shift shared across dimensions.

    Raises
    ------
    ValueError
        If ``anchor`` is empty or its first letter is not one of ``c/e/f/l``.
    """
    key = str(anchor)[:1].lower()
    if key not in _ANCHORS:
        raise ValueError(
            f"anchor must be one of centers/edges/first/last, got {anchor!r}"
        )
    if key == "c":
        scale = [
            ((inshape[d] - 1) / (outshape[d] - 1))
            if (inshape[d] > 1 and outshape[d] > 1)
            else 1.0
            for d in range(ndim)
        ]
        return scale, 0.0
    scale = [float(inshape[d]) / float(outshape[d]) for d in range(ndim)]
    return scale, _ANCHOR_SHIFT[key]


def _infer_ndim(
    ndim: Optional[int],
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> int:
    """Infer the number of trailing spatial dimensions to resize.

    Mirrors the numpy/torch wrappers: an explicit ``ndim`` wins; otherwise a
    sequence ``shape`` or ``factor`` implies its length; failing that, ``1``.
    """
    if ndim is not None:
        return int(ndim)
    if shape is not None and not isinstance(shape, int):
        return len(list(shape))
    if factor is not None and not isinstance(factor, (int, float)):
        return len(list(factor))
    return 1


def _normalize_shape(shape: int | Sequence[int], ndim: int) -> list[int]:
    """Normalise a shape argument to a list of length ``ndim``."""
    if isinstance(shape, int):
        shape = [shape] * ndim
    shape = list(shape)
    if len(shape) != ndim:
        raise ValueError(f"Expected shape of length ndim={ndim}, got {shape}.")
    return [int(s) for s in shape]


def _resolve_out_spatial(
    spatial_in: Sequence[int],
    ndim: int,
    factor: float | Sequence[float] | None,
    shape: int | Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve the output spatial shape from ``factor`` or ``shape``.

    ``factor`` and ``shape`` are mutually exclusive; with neither, the output
    keeps the input spatial shape (identity). A scalar is broadcast to
    ``ndim`` entries.
    """
    if shape is not None:
        return tuple(_normalize_shape(shape, ndim))
    if factor is not None:
        if isinstance(factor, (int, float)):
            factors = [float(factor)] * ndim
        else:
            factors = [float(f) for f in factor]
        if len(factors) != ndim:
            raise ValueError(
                f"Expected factor of length ndim={ndim}, got {factors}."
            )
        return tuple(
            max(1, int(round(n * f))) for n, f in zip(spatial_in, factors)
        )
    return tuple(int(n) for n in spatial_in)  # identity


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

    Raises ``ValueError`` if ``ndim`` is outside ``1..inp.ndim`` or an
    explicit ``scale`` has the wrong length.
    """
    ndim = _infer_ndim(ndim, factor, shape)
    if ndim < 1 or ndim > inp.ndim:
        raise ValueError(f"ndim must be in 1..{inp.ndim}, got {ndim}")
    spatial_in = tuple(inp.shape[-ndim:])
    out_spatial = _resolve_out_spatial(spatial_in, ndim, factor, shape)
    a_scale, a_shift = _anchor_scale_shift(
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
        :func:`_anchor_scale_shift`). Abbreviations accepted.
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
