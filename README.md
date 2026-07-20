# fastfields-cupy

A user-friendly [cupy](https://cupy.dev/) interface over the
[`fastfields.dlpack`](../fastfields-bind-py) nanobind bindings to the
`fastfields-lib` C++/CUDA library.

It mirrors the numpy-style fastfields API but operates on **cupy arrays in CUDA
device memory**. cupy arrays expose `__dlpack__`, so the bindings share device
memory with them at zero copy.

## Install

```bash
pip install fastfields-cupy            # pulls fastfields-dlpack
pip install "fastfields-cupy[cupy]"    # also install a CUDA 12.x cupy build
```

`cupy` is *not* a hard dependency because the correct wheel depends on your
CUDA toolkit. Install `cupy-cuda12x`, `cupy-cuda11x`, ... to match your system.

## Usage

```python
import cupy as cp
import fastfields.cupy as ffc

# Euclidean distance transform along the last axis (functional).
x = cp.array([[0, cp.inf, cp.inf, 0, cp.inf]], dtype=cp.float32)
d = ffc.dt_euclidean(x)            # new array; x is untouched
ffc.dt_euclidean_(x)               # in-place variant

# Compact-symmetric mat-vec:  out = H @ v
out = ffc.sym_matvec(hessian, v)
```

### Conventions

* **Functional** wrappers (`dt_euclidean`, `sym_matvec`, `resample`, ...) take
  cupy arrays, allocate their outputs, and return cupy arrays. Inputs are made
  C-contiguous and must be `float32`/`float64`.
* **Trailing-underscore** wrappers (`dt_euclidean_`, `sym_solve_`, ...) operate
  in place / through the caller's output and return it. They require
  C-contiguous arrays and never silently copy.

### Streams

cupy queues work on its *current* CUDA stream. Every wrapper forwards
`cupy.cuda.get_current_stream().ptr` to the binding's `stream` argument, so the
kernels are ordered correctly with respect to surrounding cupy operations. To
target a specific stream:

```python
s = cp.cuda.Stream()
with s:
    d = ffc.dt_euclidean(x)   # runs on stream `s`
```

## Testing

Runtime tests require a GPU and cupy; they are skipped otherwise:

```bash
python -m pytest tests/ -q
```
