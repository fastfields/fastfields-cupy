"""Resample / restriction / spline-coefficient wrappers (cupy)."""

from __future__ import annotations

from typing import Any, Sequence

import fastfields.dlpack as _ff

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


def _resolve_scale_shift(
    inp: Any,
    out_shape: Sequence[int],
    anchor: str,
    scale: Sequence[float] | None,
    shift: float | None,
    ndim: int,
) -> tuple[list[float], float]:
    """Resolve the per-dim scale and scalar shift from ``anchor``/overrides."""
    a_scale, a_shift = _anchor_scale_shift(
        anchor, inp.shape[-ndim:], tuple(out_shape)[-ndim:], ndim
    )
    if scale is not None:
        a_scale = [float(s) for s in scale]
        if len(a_scale) != ndim:
            raise ValueError(
                f"Expected scale of length ndim={ndim}, got {scale}."
            )
    if shift is not None:
        a_shift = float(shift)
    return a_scale, a_shift


def resample(
    inp: Any,
    out_shape: Sequence[int],
    spline: int = 2,
    bound: int = 3,
    shift: float | None = None,
    scale: Sequence[float] | None = None,
    ndim: int = 1,
    anchor: str = "centers",
) -> Any:
    """Spline resample (prolongation) of ``inp`` onto ``out_shape``.

    Allocates and returns the output array.

    Parameters
    ----------
    inp : cupy.ndarray
        Input array, shape ``(..., *inshape)``.
    out_shape : sequence of int
        Full output shape (batch dims + the ``ndim`` spatial dims).
    spline : int, default=2
        Spline order.
    bound : int, default=3
        Boundary condition (default DCT2).
    shift : float, optional
        Sampling-shift override. When omitted the shift implied by ``anchor``
        is used; pass a value to override it (advanced use).
    scale : sequence of float, optional
        Per-dim scale (input-index per output-index), length ``ndim``. When
        omitted it is derived from ``anchor`` and the shapes; pass a value to
        override it (advanced use).
    ndim : int, default=1
        Number of spatial dimensions.
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention, matching ``interpol.resize``. Sets the
        default per-dim ``scale`` and ``shift`` (see
        :func:`_anchor_scale_shift` for the mapping). Abbreviations
        (``"c"``/``"e"``/``"f"``/``"l"``) are accepted.

        .. note::
           The default is ``"centers"``. Earlier releases behaved like
           ``"first"`` (scale ``in/out``, shift ``0``); pass ``anchor="first"``
           to recover that grid.
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    out = cp.empty(tuple(out_shape), dtype=inp.dtype)
    scale, shift = _resolve_scale_shift(
        inp, out_shape, anchor, scale, shift, ndim
    )
    _ff.resample(
        out, inp, spline, bound, shift, scale, ndim, current_stream_ptr()
    )
    return out


def restriction(
    inp: Any,
    out_shape: Sequence[int],
    spline: int = 2,
    bound: int = 3,
    shift: float | None = None,
    scale: Sequence[float] | None = None,
    ndim: int = 1,
    anchor: str = "centers",
) -> Any:
    """Restriction (adjoint of :func:`resample`) of ``inp`` onto ``out_shape``.

    The binding *accumulates* into the output, so the freshly allocated array
    is zero-initialised here. The ``anchor`` convention matches
    :func:`resample`; because the scale is derived from this call's own
    (input, output) shapes, a ``resample`` and a matching ``restriction`` use
    reciprocal scales and the same shift -- the adjoint relationship the
    binding expects.

    Parameters
    ----------
    inp : cupy.ndarray
        Input array, shape ``(..., *inshape)``.
    out_shape : sequence of int
        Full output shape (batch dims + the ``ndim`` spatial dims).
    spline : int, default=2
        Spline order.
    bound : int, default=3
        Boundary condition (default DCT2).
    shift : float, optional
        Sampling-shift override (see :func:`resample`).
    scale : sequence of float, optional
        Per-dim scale override (see :func:`resample`).
    ndim : int, default=1
        Number of spatial dimensions.
    anchor : {"centers", "edges", "first", "last"}, default="centers"
        Sampling-grid convention (see :func:`resample`).
    """
    cp = cupy()
    inp = as_gpu_array(inp, name="inp")
    out = cp.zeros(tuple(out_shape), dtype=inp.dtype)
    scale, shift = _resolve_scale_shift(
        inp, out_shape, anchor, scale, shift, ndim
    )
    _ff.restriction(
        out, inp, spline, bound, shift, scale, ndim, current_stream_ptr()
    )
    return out


def spline_coeff(inp: Any, spline: int = 3, bound: int = 3) -> Any:
    """Spline-coefficient prefilter along the last axis (functional).

    Orders 0/1 are no-ops. Returns a new array; ``inp`` is unmodified.
    """
    out = as_gpu_array(inp, name="inp").copy()
    _ff.spline_coeff(out, spline, bound, current_stream_ptr())
    return out


def spline_coeff_(inp_out: Any, spline: int = 3, bound: int = 3) -> Any:
    """In-place spline-coeff prefilter (last axis); returns ``inp_out``."""
    inp_out = require_gpu_writethrough(inp_out, name="inp_out")
    _ff.spline_coeff(inp_out, spline, bound, current_stream_ptr())
    return inp_out
