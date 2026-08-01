"""fastfields.cupy: a user-friendly cupy interface over ``fastfields.dlpack``.

This package mirrors the numpy-style fastfields interface but operates on
**cupy arrays living in CUDA device memory**. The underlying
``fastfields.dlpack`` functions accept any object exposing ``__dlpack__`` (cupy
arrays do, sharing device memory with zero copy) and write results in place or
through pre-allocated outputs.

Conventions
-----------
* Functional wrappers (``dt_euclidean``, ``sym_matvec``, ``resample`` ...)
  take cupy arrays, allocate their outputs, and return cupy arrays. Inputs
  must be float32/float64 but are passed with their native strides (the
  stride-aware C++/CUDA library reads them zero-copy -- no contiguous copy is
  forced); freshly allocated outputs are always contiguous.
* Trailing-underscore wrappers (``dt_euclidean_``, ``sym_solve_`` ...) operate
  in place / through the caller's output array and return it.
* **Streams:** cupy queues work on its *current* CUDA stream. Every wrapper
  forwards ``cupy.cuda.get_current_stream().ptr`` to the binding's ``stream``
  argument, so the C++/CUDA kernels are correctly ordered w.r.t. the
  surrounding cupy operations. To run on a specific stream, wrap the call in a
  ``with my_stream:`` block.

cupy is imported lazily, so ``import fastfields.cupy`` succeeds even where cupy
or a GPU is unavailable; a clear ``ImportError`` is raised only when a wrapper
is actually called.
"""

from __future__ import annotations

# Spline / Bound are plain IntEnums in fastfields_bind (no cupy needed).
from fastfields.dlpack import Bound, Spline

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
from ._pushpull import count, grad, pull, push
from ._reg import (
    field_diag,
    field_adddiag,
    field_adddiag_,
    field_subdiag,
    field_subdiag_,
    field_forward,
    field_kernel,
    field_matvec,
    field_addmatvec,
    field_addmatvec_,
    field_submatvec,
    field_submatvec_,
    field_precond,
    flow_diag,
    flow_adddiag,
    flow_adddiag_,
    flow_subdiag,
    flow_subdiag_,
    flow_forward,
    flow_kernel,
    flow_matvec,
    flow_addmatvec,
    flow_addmatvec_,
    flow_submatvec,
    flow_submatvec_,
    flow_precond,
    flow_relax,
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
    # pushpull
    "pull",
    "push",
    "count",
    "grad",
    # regularisers
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
    # enums / helpers
    "Spline",
    "Bound",
    "current_stream_ptr",
    "__version__",
]
