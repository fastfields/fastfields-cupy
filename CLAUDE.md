# fastfields-cupy  (imports as `fastfields.cupy`)

A user-friendly **cupy** interface over the `fastfields.dlpack`
(`fastfields-bind-py`) bindings — the numpy-style API, but on cupy arrays in
**CUDA device memory**.

```
… ─ lib ─ bind-py ─ cupy ← (you are here) … ─ fastfields
```

## Philosophy / role
- Mirrors the numpy wrapper's surface but operates on cupy arrays. cupy arrays
  expose `__dlpack__`, so device memory is shared at **zero copy**.
- **Functional** wrappers (`dt_euclidean`, `sym_matvec`, `resample`, …) allocate
  outputs and return new arrays; **trailing-underscore** wrappers
  (`dt_euclidean_`, `sym_solve_`, …) operate in place and never silently copy.
- Inputs are made C-contiguous; dtypes must be float32/float64.
- **Lazy import of cupy**: cupy is *not* a hard dependency (the right wheel
  depends on the host CUDA toolkit). Install `fastfields-cupy[cupy]` or a
  matching `cupy-cuda1{1,2}x` manually.

## Streams
Every wrapper forwards `cupy.cuda.get_current_stream().ptr` to the binding's
`stream` argument, so kernels are ordered w.r.t. surrounding cupy work. Use a
`with stream:` block to target a specific stream.

## Layout
`fastfields/cupy/`: `__init__.py`, `_dt.py`, `_sym.py`, `_resample.py`,
`_util.py` (contiguity/stream helpers). `tests/test_cupy.py`.

## Build & test
```
pip install .                    # or pip install "fastfields-cupy[cupy]"
python -m pytest tests/ -q       # runtime tests need a GPU + cupy; skipped otherwise
```
Prefer a regular install over editable (native-namespace merge).

## Conventions & caveats
- **PEP 420 namespace**: ships only `fastfields/cupy/`, no
  `fastfields/__init__.py`.
- Requires a working CUDA `libfastfields-cuda` at runtime; the CUDA library is
  compile/link-validated only (no GPU in CI), so exercise on real hardware.
- Ruff: line-length 79, select B/E/F/I/W.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Status: `/home/user/fastfields-lib/MIGRATION.md`.
