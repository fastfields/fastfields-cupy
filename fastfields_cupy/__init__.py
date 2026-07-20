"""fastfields_cupy: a user-friendly cupy interface over ``fastfields_bind``.

This package mirrors the numpy-style fastfields interface but operates on
**cupy arrays living in CUDA device memory**. The underlying
``fastfields_bind`` functions accept any object exposing ``__dlpack__`` (cupy
arrays do, sharing device memory with zero copy) and write results in place or
through pre-allocated outputs.

Conventions
-----------
* Functional wrappers (``dt_euclidean``, ``sym_matvec``, ``resample`` ...)
  take cupy arrays, allocate their outputs, and return cupy arrays. Inputs are
  made C-contiguous via ``cupy.ascontiguousarray`` and must be float32/float64.
* Trailing-underscore wrappers (``dt_euclidean_``, ``sym_solve_`` ...) operate
  in place / through the caller's output array and return it.
* **Streams:** cupy queues work on its *current* CUDA stream. Every wrapper
  forwards ``cupy.cuda.get_current_stream().ptr`` to the binding's ``stream``
  argument, so the C++/CUDA kernels are correctly ordered w.r.t. the
  surrounding cupy operations. To run on a specific stream, wrap the call in a
  ``with my_stream:`` block.

cupy is imported lazily, so ``import fastfields_cupy`` succeeds even where cupy
or a GPU is unavailable; a clear ``ImportError`` is raised only when a wrapper
is actually called.
"""

from __future__ import annotations

# Spline / Bound are plain IntEnums in fastfields_bind (no cupy needed).
from fastfields_bind import Bound, Spline

from ._dt import (
    dt_euclidean,
    dt_euclidean_,
    dt_l1,
    dt_l1_,
    dt_mesh,
    dt_spline_brent,
    dt_spline_gaussnewton,
    dt_spline_table,
)
from ._resample import resample, restriction, spline_coeff, spline_coeff_
from ._sym import (
    sym_addmatvec_,
    sym_invert,
    sym_invert_,
    sym_matvec,
    sym_matvec_backward,
    sym_solve,
    sym_solve_,
    sym_submatvec_,
)
from ._util import current_stream_ptr

__version__ = "0.1.0"

__all__ = [
    # distance transforms
    "dt_euclidean",
    "dt_euclidean_",
    "dt_l1",
    "dt_l1_",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
    # symmetric-matrix ops
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
    # resampling
    "resample",
    "restriction",
    "spline_coeff",
    "spline_coeff_",
    # enums / helpers
    "Spline",
    "Bound",
    "current_stream_ptr",
    "__version__",
]
