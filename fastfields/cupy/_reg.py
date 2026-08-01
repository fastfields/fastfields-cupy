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
    "field_addmatvec",
    "field_addmatvec_",
    "field_submatvec",
    "field_submatvec_",
    "field_diag",
    "field_adddiag",
    "field_adddiag_",
    "field_subdiag",
    "field_subdiag_",
    "field_kernel",
    "field_relax",
    "field_precond",
    "field_forward",
    "flow_matvec",
    "flow_addmatvec",
    "flow_addmatvec_",
    "flow_submatvec",
    "flow_submatvec_",
    "flow_diag",
    "flow_adddiag",
    "flow_adddiag_",
    "flow_subdiag",
    "flow_subdiag_",
    "flow_kernel",
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


# ---------------------------------------------------------------------------
# In-place accumulate ops (`out (+/-)= ...`)
# ---------------------------------------------------------------------------
#
# One kernel each: the in-place-only C primitive
# `ff::{field,flow}_{matvec,diag}_{add,sub}_`, restored from jitfields'
# `op='+'` / `op='-'` entry points. The out-of-place spelling copies the
# caller's array first and runs the same primitive on the copy.


def _field_matvec_acc(
    inp,
    field,
    absolute,
    membrane,
    bending,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = as_gpu_array(inp, name="inp")
    field = as_gpu_array(field, name="field")
    acc = inp if inplace else inp.copy()
    channels = field.shape[-1]
    fn = _ff.field_submatvec_ if sub else _ff.field_addmatvec_
    fn(
        acc,
        field,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return acc


def _field_diag_acc(
    inp, absolute, membrane, bending, voxel_size, bound, ndim, sub, inplace
):
    inp = as_gpu_array(inp, name="inp")
    acc = inp if inplace else inp.copy()
    channels = acc.shape[-1]
    fn = _ff.field_subdiag_ if sub else _ff.field_adddiag_
    fn(
        acc,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        current_stream_ptr(),
    )
    return acc


def _flow_matvec_acc(
    inp,
    flow,
    absolute,
    membrane,
    bending,
    shears,
    div,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = as_gpu_array(inp, name="inp")
    flow = as_gpu_array(flow, name="flow")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_submatvec_ if sub else _ff.flow_addmatvec_
    fn(
        acc,
        flow,
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
    return acc


def _flow_diag_acc(
    inp,
    absolute,
    membrane,
    bending,
    shears,
    div,
    voxel_size,
    bound,
    ndim,
    sub,
    inplace,
):
    inp = as_gpu_array(inp, name="inp")
    acc = inp if inplace else inp.copy()
    fn = _ff.flow_subdiag_ if sub else _ff.flow_adddiag_
    fn(
        acc,
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
    return acc


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


def _field_channels(channels, *penalties) -> int:
    """Infer C from an explicit value or the per-channel penalty lengths."""
    if channels is not None:
        return int(channels)
    for p in penalties:
        if p is not None and not isinstance(p, (int, float)):
            return len(p)
    return 1


def field_kernel(
    ndim: int,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    channels: int | None = None,
    voxel_size=None,
    bound: int | str = "dct2",
    dtype: Any = None,
) -> Any:
    """Materialise the field regulariser's per-channel Toeplitz stencil.

    Returns the small centred kernel that, convolved with a field, reproduces
    :func:`field_matvec`. The shape is ``(*k, C)`` (channels are independent),
    with ``k`` the stencil width per spatial dim: 1 (absolute), 3 (membrane) or
    5 (bending). ``C`` is ``channels`` if given, else inferred from the
    per-channel penalty lengths (default 1).
    """
    cp = cupy()
    ndim = int(ndim)
    channels = _field_channels(channels, absolute, membrane, bending)
    if bending is not None:
        width = 5
    elif membrane is not None:
        width = 3
    else:
        width = 1
    out = cp.zeros(
        tuple([width] * ndim + [channels]), dtype=dtype or cp.float64
    )
    _ff.field_kernel(
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


def field_relax(
    field: Any,
    hes: Any,
    grd: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
    nb_iter: int = 1,
) -> Any:
    """Refine ``field`` in place with ``nb_iter`` relaxation sweeps.

    Solves ``(H + L) x = g`` with per-voxel compact-symmetric Hessian ``hes``
    (packed ``C*(C+1)/2`` last axis), the per-channel field regulariser ``L``
    and gradient ``grd``; ``field`` is the warm start, mutated and returned.
    """
    field = as_gpu_array(field, name="field")
    hes = as_gpu_array(hes, name="hes")
    grd = as_gpu_array(grd, name="grd")
    channels = field.shape[-1]
    _ff.field_relax(
        field,
        hes,
        grd,
        _voxel(voxel_size, ndim),
        _per_channel(absolute, channels, "absolute"),
        _per_channel(membrane, channels, "membrane"),
        _per_channel(bending, channels, "bending"),
        as_bound(bound),
        ndim,
        int(nb_iter),
        current_stream_ptr(),
    )
    return field


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


def flow_kernel(
    ndim: int,
    absolute: float = 0.0,
    membrane: float = 0.0,
    bending: float = 0.0,
    shears: float = 0.0,
    div: float = 0.0,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    dtype: Any = None,
) -> Any:
    """Materialise the flow regulariser's Toeplitz convolution stencil.

    Returns the small centred kernel that, convolved with a flow field,
    reproduces :func:`flow_matvec`. The shape is ``(*k, ndim)`` for the
    per-channel vector stencil, or ``(*k, ndim, ndim)`` when ``shears``/``div``
    select the cross-channel (Lamé) matrix stencil, where ``k`` is the stencil
    width per spatial dim: 1 (absolute only), 3 (membrane/Lamé) or 5 (bending).
    """
    cp = cupy()
    ndim = int(ndim)
    is_matrix = shears != 0.0 or div != 0.0
    if shears == div == membrane == bending == 0.0:
        width = 1
    elif bending == 0.0:
        width = 3
    else:
        width = 5
    shape = [width] * ndim + [ndim]
    if is_matrix:
        shape += [ndim]
    out = cp.zeros(tuple(shape), dtype=dtype or cp.float64)
    _ff.flow_kernel(
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
        vec.shape,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
        dtype=vec.dtype,
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
        vec,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
    )
    return out


# --- accumulate variants -------------------------------------------------
#
# jitfields' ``_add`` / ``_sub`` (fresh array) and trailing-underscore in-place
# forms, as thin compositions ``inp ± op(...)`` over flow_matvec / flow_diag.


def flow_addmatvec(
    inp: Any,
    flow: Any,
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
    """Return ``inp + L @ flow`` as a **new** array.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def flow_submatvec(
    inp: Any,
    flow: Any,
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
    """Return ``inp - L @ flow`` as a **new** array.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def flow_addmatvec_(
    inp: Any,
    flow: Any,
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
    """In place ``inp += L @ flow``; returns ``inp``.

    Calls the fused in-place C primitive directly.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def flow_submatvec_(
    inp: Any,
    flow: Any,
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
    """In place ``inp -= L @ flow``; returns ``inp``.

    Calls the fused in-place C primitive directly.
    """
    return _flow_matvec_acc(
        inp,
        flow,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def flow_adddiag(
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
    """Return ``inp + diag(L)`` as a **new** array, shaped like ``inp``.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def flow_subdiag(
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
    """Return ``inp - diag(L)`` as a **new** array, shaped like ``inp``.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def flow_adddiag_(
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
    """In place ``inp += diag(L)``; returns ``inp``.

    Calls the fused in-place C primitive directly.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def flow_subdiag_(
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
    """In place ``inp -= diag(L)``; returns ``inp``.

    Calls the fused in-place C primitive directly.
    """
    return _flow_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        shears,
        div,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def field_precond(
    mat: Any,
    vec: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the preconditioner ``(M + diag(R)) \\ vec``."""
    vec = as_gpu_array(vec, name="vec")
    diag = field_diag(
        vec.shape,
        absolute,
        membrane,
        bending,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
        dtype=vec.dtype,
    )
    return sym_solve(mat, vec, diag)


def field_forward(
    mat: Any,
    vec: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Apply the forward matrix-vector product ``(M + R) @ vec``."""
    vec = as_gpu_array(vec, name="vec")
    out = sym_matvec(mat, vec)
    out = out + field_matvec(
        vec,
        absolute,
        membrane,
        bending,
        voxel_size=voxel_size,
        bound=bound,
        ndim=ndim,
    )
    return out


def field_addmatvec(
    inp: Any,
    field: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Return ``inp + L @ field`` as a **new** array.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def field_submatvec(
    inp: Any,
    field: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Return ``inp - L @ field`` as a **new** array.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def field_addmatvec_(
    inp: Any,
    field: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """In place ``inp += L @ field``; returns ``inp``.

    Calls the fused in-place C primitive directly.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def field_submatvec_(
    inp: Any,
    field: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """In place ``inp -= L @ field``; returns ``inp``.

    Calls the fused in-place C primitive directly.
    """
    return _field_matvec_acc(
        inp,
        field,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )


def field_adddiag(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Return ``inp + diag(L)`` as a **new** array, shaped like ``inp``.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=False,
    )


def field_subdiag(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """Return ``inp - diag(L)`` as a **new** array, shaped like ``inp``.

    Copies ``inp`` and runs the in-place accumulate primitive on the copy.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=False,
    )


def field_adddiag_(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """In place ``inp += diag(L)``; returns ``inp``.

    Calls the fused in-place C primitive directly.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=False,
        inplace=True,
    )


def field_subdiag_(
    inp: Any,
    absolute=None,
    membrane=None,
    bending=None,
    *,
    voxel_size=None,
    bound: int | str = "dct2",
    ndim: int = 1,
) -> Any:
    """In place ``inp -= diag(L)``; returns ``inp``.

    Calls the fused in-place C primitive directly.
    """
    return _field_diag_acc(
        inp,
        absolute,
        membrane,
        bending,
        voxel_size,
        bound,
        ndim,
        sub=True,
        inplace=True,
    )
