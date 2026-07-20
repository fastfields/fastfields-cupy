"""Shared helpers: lazy cupy import, array validation, and stream handling.

cupy is imported *lazily* so that ``import fastfields_cupy`` succeeds on
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
                "fastfields_cupy requires cupy, which is not installed. "
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
