# fastfields-cupy

`fastfields-cupy` is a user-friendly [cupy](https://cupy.dev/) interface over the `fastfields.dlpack` bindings. It mirrors the numpy-style fastfields API but operates on **cupy arrays in CUDA device memory** (shared zero-copy via `__dlpack__`). `cupy` is not a hard dependency, since the correct wheel depends on your CUDA toolkit; install it via the `cupy` extra.

## Installation

```bash
pip install fastfields-cupy            # pulls fastfields-dlpack
pip install "fastfields-cupy[cupy]"    # also install a CUDA 12.x cupy build
```

## Usage

```python
import cupy as cp
import fastfields.cupy as ffc

# Euclidean distance transform along the last axis (functional).
x = cp.array([[0, cp.inf, cp.inf, 0, cp.inf]], dtype=cp.float32)
d = ffc.dt_euclidean(x)            # new array; x is untouched
ffc.dt_euclidean_(x)               # in-place variant
```

See the [API reference](api/index.md) for the full list of operations.
