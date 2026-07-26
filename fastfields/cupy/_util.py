"""Shared helpers: lazy cupy import, array validation, and stream handling.

cupy is imported *lazily* so that ``import fastfields.cupy`` succeeds on
machines without cupy or a GPU. A clear error is raised only when a wrapper is
actually invoked without a working cupy/CUDA environment.
"""

from __future__ import annotations

from typing import Any, Sequence

from fastfields.dlpack import Bound, Spline

_cupy_module: Any = None

# Friendly string aliases for the spline-order and boundary-condition
# arguments, mirroring the numpy/torch wrappers so `order=`/`bound=` accept an
# int, a Spline/Bound enum, or a name on every backend.
_SPLINE_ALIASES = {
    "nearest": Spline.Nearest,
    "constant": Spline.Nearest,
    "linear": Spline.Linear,
    "quadratic": Spline.Quadratic,
    "cubic": Spline.Cubic,
    "fourth": Spline.FourthOrder,
    "fifth": Spline.FifthOrder,
    "sixth": Spline.SixthOrder,
    "seventh": Spline.SeventhOrder,
}

_BOUND_ALIASES = {
    "zero": Bound.Zero,
    "zeros": Bound.Zero,
    "replicate": Bound.Replicate,
    "nearest": Bound.Replicate,
    "dct1": Bound.DCT1,
    "dct2": Bound.DCT2,
    "neumann": Bound.DCT2,
    "reflect": Bound.DCT2,
    "dst1": Bound.DST1,
    "dst2": Bound.DST2,
    "dirichlet": Bound.DST2,
    "dft": Bound.DFT,
    "wrap": Bound.DFT,
    "circular": Bound.DFT,
    "nocheck": Bound.NoCheck,
}


def as_spline(value: int | str | Spline) -> int:
    """Normalise a spline-order argument to an ``int`` in ``0..7``.

    Accepts an integer, a :class:`Spline` enum, or a friendly string alias
    (e.g. ``"cubic"``). Raises ``ValueError`` for an unknown alias or an
    out-of-range integer.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _SPLINE_ALIASES:
            raise ValueError(
                f"unknown spline order {value!r}; "
                f"expected an int 0..7 or one of {sorted(_SPLINE_ALIASES)}"
            )
        return int(_SPLINE_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"spline order must be in 0..7, got {ivalue}")
    return ivalue


def as_bound(value: int | str | Bound) -> int:
    """Normalise a boundary-condition argument to an ``int`` in ``0..7``.

    Accepts an integer, a :class:`Bound` enum, or a friendly string alias
    (e.g. ``"dct2"``, ``"wrap"``). Raises ``ValueError`` for an unknown alias
    or an out-of-range integer.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in _BOUND_ALIASES:
            raise ValueError(
                f"unknown boundary condition {value!r}; "
                f"expected an int 0..7 or one of {sorted(_BOUND_ALIASES)}"
            )
        return int(_BOUND_ALIASES[key])
    ivalue = int(value)
    if not 0 <= ivalue <= 7:
        raise ValueError(f"boundary condition must be in 0..7, got {ivalue}")
    return ivalue


def cupy() -> Any:
    """Return the imported ``cupy`` module, importing it on first use.

    Raises
    ------
    ImportError
        If cupy is not installed. Installing the appropriate CUDA build (e.g.
        ``pip install fastfields-cupy[cupy]`` which pulls ``cupy-cuda12x``)
        resolves this.
    """
    global _cupy_module
    if _cupy_module is None:
        try:
            import cupy as _cp  # noqa: PLC0415  (deliberately lazy)
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "fastfields.cupy requires cupy, which is not installed. "
                "Install a CUDA-matched build, e.g. "
                "`pip install cupy-cuda12x` or "
                "`pip install fastfields-cupy[cupy]`."
            ) from exc
        _cupy_module = _cp
    return _cupy_module


def current_stream_ptr() -> int:
    """Return the pointer of cupy's *current* CUDA stream.

    fastfields_bind forwards its ``stream`` argument straight to the CUDA
    backend, so every wrapper passes this value. Because cupy queues all work
    on its current stream, forwarding that same stream keeps the binding's
    kernels ordered correctly with respect to the surrounding cupy operations
    (allocation of the output, later reads, etc.). Callers who want a specific
    stream simply wrap the call in a ``with stream:`` block.
    """
    return cupy().cuda.get_current_stream().ptr


def _is_gpu_array(arr: Any) -> bool:
    """Return whether ``arr`` is a cupy ``ndarray``."""
    return isinstance(arr, cupy().ndarray)


def as_gpu_array(arr: Any, *, name: str = "array") -> Any:
    """Validate a float32/float64 cupy array and return it **unchanged**.

    Use this for *read-only* inputs. The underlying C++/CUDA library is fully
    stride-aware (it receives the DLPack strides and indexes accordingly), so a
    read-only input does **not** need to be contiguous: passing it with its
    native strides is zero-copy and keeps big inputs from being duplicated. We
    therefore only validate the type/dtype and return the array as-is -- no
    ``cupy.ascontiguousarray`` copy is made.

    Parameters
    ----------
    arr : cupy.ndarray
        Candidate input array (must live in GPU memory).
    name : str, optional
        Argument name used in error messages.

    Returns
    -------
    cupy.ndarray
        ``arr`` unchanged.

    Raises
    ------
    TypeError
        If ``arr`` is not a cupy ``ndarray`` or is not float32/float64.
    """
    cp = cupy()
    if not isinstance(arr, cp.ndarray):
        raise TypeError(
            f"{name} must be a cupy.ndarray living on the GPU, got "
            f"{type(arr).__name__}."
        )
    if arr.dtype not in (cp.float32, cp.float64):
        raise TypeError(f"{name} must be float32 or float64, got {arr.dtype}.")
    return arr


# Backwards-compatible alias (contiguity is no longer forced on inputs).
as_gpu_contiguous = as_gpu_array


def require_gpu_writethrough(arr: Any, *, name: str = "array") -> Any:
    """Validate an in-place / output array and return it **unchanged**.

    The binding writes results through this array's memory. The underlying
    library is fully stride-aware (it receives the DLPack strides and indexes
    accordingly), so the write lands correctly regardless of memory layout --
    no contiguous copy is made. This keeps in-place ops zero-copy even for
    non-contiguous views, which is a core memory-efficiency feature of the
    library. We only reject non-arrays and wrong dtypes (where an in-place
    write could not land in the caller's buffer or would need a lossy cast).
    """
    cp = cupy()
    if not isinstance(arr, cp.ndarray):
        raise TypeError(
            f"{name} must be a cupy.ndarray living on the GPU, got "
            f"{type(arr).__name__}."
        )
    if arr.dtype not in (cp.float32, cp.float64):
        raise TypeError(f"{name} must be float32 or float64, got {arr.dtype}.")
    return arr


# Backwards-compatible alias (the contiguity requirement has been relaxed).
require_gpu_contiguous = require_gpu_writethrough


# --------------------------------------------------------------------------- #
# zero-copy batch-dim broadcasting                                            #
# --------------------------------------------------------------------------- #
#
# The raw bindings require every input tensor of an op to share the same batch
# (leading) dims -- they do not broadcast. We normalise inputs to a common
# broadcast batch shape *without copying*: each input is re-strided to the
# target shape (real stride on matching axes, 0-stride on broadcast axes) with
# ``as_strided``. The result shares device memory with the source, so big
# inputs are never duplicated; the stride-aware C++/CUDA kernels handle the
# 0-strides natively.


def _broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """Pure-python numpy-style broadcast of several shape tuples.

    Parameters
    ----------
    *shapes : tuple of int
        The shapes to broadcast together.

    Returns
    -------
    tuple of int
        The common broadcast shape.

    Raises
    ------
    ValueError
        If the shapes are not broadcast-compatible.
    """
    ndim = max((len(s) for s in shapes), default=0)
    out = [1] * ndim
    for s in shapes:
        for i in range(1, len(s) + 1):
            dim = s[-i]
            cur = out[-i]
            if dim == 1 or dim == cur:
                continue
            if cur == 1:
                out[-i] = dim
            else:
                raise ValueError(f"cannot broadcast batch shapes {shapes}")
    return tuple(out)


def bcast_view(arr: Any, shape: Sequence[int]) -> Any:
    """Return a zero-copy, DLPack-exportable broadcast of ``arr`` to ``shape``.

    Parameters
    ----------
    arr : cupy.ndarray
        Array to broadcast.
    shape : sequence of int
        Target shape.

    Returns
    -------
    cupy.ndarray
        A 0-stride ``as_strided`` view sharing memory with ``arr``.

    Raises
    ------
    ValueError
        If ``arr`` cannot be broadcast to ``shape``.
    """
    cp = cupy()
    shape = tuple(shape)
    if arr.shape == shape:
        return arr
    strides = [0] * len(shape)
    for i in range(1, arr.ndim + 1):
        dim = arr.shape[-i]
        if dim == shape[-i]:
            strides[-i] = arr.strides[-i]
        elif dim == 1:
            strides[-i] = 0
        else:
            raise ValueError(
                f"cannot broadcast array of shape {arr.shape} to {shape}"
            )
    return cp.lib.stride_tricks.as_strided(arr, shape, tuple(strides))


def broadcast_batch(
    specs: Sequence[tuple[Any, int]],
) -> tuple[tuple[int, ...], list[Any]]:
    """Broadcast the batch dims of several arrays to a common shape.

    Parameters
    ----------
    specs : sequence of (cupy.ndarray, int)
        ``(array, n_core)`` pairs, where ``n_core`` is the number of trailing
        (core) axes to leave untouched.

    Returns
    -------
    batch_shape : tuple of int
        The common broadcast batch shape.
    views : list of cupy.ndarray
        Each input broadcast (zero-copy) to ``batch_shape + its core dims``.
    """
    batch = _broadcast_shapes(*[a.shape[: a.ndim - nc] for a, nc in specs])
    views = [bcast_view(a, batch + a.shape[a.ndim - nc :]) for a, nc in specs]
    return batch, views


def broadcast_to_batch(arr: Any, batch: Sequence[int], n_core: int) -> Any:
    """Broadcast ``arr`` to ``batch + arr's core dims``.

    For in-place ops whose output already fixes the batch shape.

    Parameters
    ----------
    arr : cupy.ndarray
        Array to broadcast.
    batch : sequence of int
        Target batch (leading) shape.
    n_core : int
        Number of trailing core axes of ``arr`` to preserve.

    Returns
    -------
    cupy.ndarray
        A zero-copy broadcast view of ``arr``.
    """
    return bcast_view(arr, tuple(batch) + arr.shape[arr.ndim - n_core :])
