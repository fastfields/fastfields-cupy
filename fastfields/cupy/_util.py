"""Shared helpers: lazy cupy import, array validation, and stream handling.

cupy is imported *lazily* so that ``import fastfields.cupy`` succeeds on
machines without cupy or a GPU. A clear error is raised only when a wrapper is
actually invoked without a working cupy/CUDA environment.
"""

from __future__ import annotations

from typing import Any

_cupy_module: Any = None


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
                "Install a CUDA-matched build, e.g. `pip install cupy-cuda12x` "
                "or `pip install fastfields-cupy[cupy]`."
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
    return isinstance(arr, cupy().ndarray)


def as_gpu_contiguous(arr: Any, *, name: str = "array") -> Any:
    """Validate that ``arr`` is a float32/float64 cupy array; return it C-contiguous.

    A contiguous copy is made only when necessary. Use this for *read-only*
    inputs and for freshly allocated outputs.
    """
    cp = cupy()
    if not isinstance(arr, cp.ndarray):
        raise TypeError(
            f"{name} must be a cupy.ndarray living on the GPU, got "
            f"{type(arr).__name__}."
        )
    if arr.dtype not in (cp.float32, cp.float64):
        raise TypeError(
            f"{name} must be float32 or float64, got {arr.dtype}."
        )
    return cp.ascontiguousarray(arr)


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
        raise TypeError(
            f"{name} must be float32 or float64, got {arr.dtype}."
        )
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


def _broadcast_shapes(*shapes):
    """Pure-python numpy-style broadcast of several shape tuples."""
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
                raise ValueError(
                    f"cannot broadcast batch shapes {shapes}"
                )
    return tuple(out)


def bcast_view(arr, shape):
    """Return a zero-copy, DLPack-exportable broadcast of ``arr`` to ``shape``."""
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


def broadcast_batch(specs):
    """Broadcast the batch dims of several arrays to a common shape.

    ``specs`` is a list of ``(array, n_core)`` pairs, where ``n_core`` is the
    number of trailing (core) axes to leave untouched. Returns
    ``(batch_shape, [views...])`` with each view broadcast (zero-copy) to
    ``batch_shape + that array's core dims``.
    """
    batch = _broadcast_shapes(*[a.shape[: a.ndim - nc] for a, nc in specs])
    views = [bcast_view(a, batch + a.shape[a.ndim - nc:]) for a, nc in specs]
    return batch, views


def broadcast_to_batch(arr, batch, n_core):
    """Broadcast ``arr`` to ``batch + arr's core dims`` (for in-place ops whose
    output already fixes the batch shape)."""
    return bcast_view(arr, tuple(batch) + arr.shape[arr.ndim - n_core:])
