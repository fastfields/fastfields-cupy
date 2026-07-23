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

Functional wrappers (`dt_euclidean`, `sym_matvec`, …) allocate their outputs and
return new arrays; the trailing-underscore variants (`dt_euclidean_`,
`sym_solve_`, …) write in place. To target a specific stream, wrap the call in a
`with my_stream:` block.

See the [API reference](api/index.md) for full signatures and options.
