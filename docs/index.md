# fastfields-cupy

**fastfields-cupy** runs the fastfields field operators on the **GPU**, straight
on your **CuPy arrays**. Same API as the NumPy package, but the work happens in
device memory and is ordered on CuPy's current CUDA stream.

CuPy itself isn't installed automatically — the right build depends on your CUDA
version — so pull it in with the `cupy` extra (or install a matching
`cupy-cuda12x` yourself).

## Install

```sh
pip install "fastfields-cupy[cupy]" \
    --extra-index-url https://fastfields.github.io/whl/cu128/
```

## Use it

```python
import cupy as cp
import fastfields.cupy as ff

mask = cp.zeros((256, 256), "float32")
mask[:, 128] = 1.0

dist = ff.dt_euclidean(mask)      # new array; mask is untouched
ff.dt_euclidean_(mask)            # in-place variant, writes into mask
```

## What's inside

| Operation | Functions |
|---|---|
| **Distance transforms** | `dt_euclidean`, `dt_l1` (along the last axis); point-to-spline `dt_spline_table` / `dt_spline_brent` / `dt_spline_gaussnewton`; point-to-mesh `dt_mesh` |
| **Positive-definite linear algebra** | `sym_matvec`, `sym_addmatvec_`, `sym_submatvec_`, `sym_solve`, `sym_invert` over whole fields of small symmetric matrices |
| **Resampling** | `resample` (spline up/down-sampling), `restriction` (its adjoint), `spline_coeff` (coefficient prefilter) |
| **Spline gather/scatter** | `pull`, `push`, `count`, `grad` |
| **Regularisers** | `field_*` (multi-channel field, per-channel penalties) and `flow_*` (vector flow, scalar penalties + the Lamé `shears`/`div` terms): the operator `*_matvec`, its diagonal `*_diag`, its stencil `*_kernel`, the `*_add…`/`*_sub…` accumulate forms of each, the RLS/JRLS-weighted `*_rls` variants, and the regularised system below |

Functional wrappers (`dt_euclidean`, `sym_matvec`, …) allocate their outputs and
return new arrays; the trailing-underscore variants (`dt_euclidean_`,
`sym_solve_`, …) write in place. To target a specific stream, wrap the call in a
`with my_stream:` block.

### Solving the regularised system

With `H` the per-voxel data-term Hessian (compact-symmetric packing) and `L` a
regulariser:

| call | what it does |
|---|---|
| `field_forward` / `flow_forward` | apply `(H + L) x` |
| `field_precond` / `flow_precond` | apply `(H + diag(L))⁻¹ g` (Jacobi) |
| `field_precond_` / `flow_precond_` | the same, in place |
| `field_relax` / `flow_relax` | Gauss-Seidel sweeps on `(H + L) x = g` |

Each is one fused kernel, so none of them allocates a temporary you then have
to keep alive until the stream is synchronised.

## What this wrapper does not expose

Of the 58 operations in the underlying C++ hub, this package wraps all but
five.

`pull_backward`, `push_backward`, `count_backward` and `grad_backward` are the
adjoints of the gather/scatter operators with respect to both the sampled
field and the sampling coordinates. They exist so an autograd engine can
differentiate through `grid` — which only `fastfields.torch` does. There is no
automatic differentiation on cupy, so they stay at the raw layer: call
`fastfields.dlpack.pull_backward(...)` if you need them.

`field_cg` — the conjugate-gradient solver — is **CPU-only** in the underlying
library: its dot products need a device-side reduction the CUDA backend does
not implement yet ([fastfields-lib#34](https://github.com/fastfields/fastfields-lib/issues/34)),
and the hub rejects CUDA arrays rather than quietly running on the host. A
`fastfields.cupy.field_cg` could therefore only ever raise, so there isn't
one. Use `field_relax`, which does run on CUDA and solves the same system by
Gauss-Seidel sweeps.

`tests/test_api_coverage.py` fails if that list ever changes by accident, and
it runs without a GPU.

See the [API reference](api/index.md) for full signatures and options.
