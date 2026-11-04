"""The regularised system ``(H + L)`` and the stencil accumulate family — cupy.

The numeric checks here need cupy **and** a CUDA device, so they skip in CI
(there is no GPU anywhere in this stack). They exist so that the semantics of
the ops this package newly exposes -- and of the four it re-pointed at fused
kernels -- can be exercised on real hardware in one command. The structural
guarantees that *can* be checked without a GPU live in
``tests/test_reg_family_routing.py`` (which binding each wrapper reaches,
statically) and ``tests/test_api_coverage.py`` (which bindings are wrapped at
all); those run everywhere and are what CI relies on.

``field_forward`` / ``field_precond`` / ``flow_forward`` / ``flow_precond``
used to be assembled here in Python out of ``*_matvec`` + ``sym_matvec`` and
``*_diag`` + ``sym_solve``, because the hub entry points existed in C++ but
had never been bound. The composition is therefore the reference: it is a
different path through the library, and it is what a user upgrading across
this change is entitled to get the same answer from.
"""

from __future__ import annotations

import numpy as np
import pytest

import fastfields.cupy as ff

NDIM = 2
SPATIAL = (5, 6)
VS = [1.0, 1.2]
FIELD_PEN = {"absolute": [0.1, 0.2], "membrane": [0.5, 0.3],
             "bending": [0.05, 0.02]}
FLOW_PEN = {"absolute": 0.1, "membrane": 0.5, "bending": 0.0,
            "shears": 0.2, "div": 0.1}
KW = {"voxel_size": VS, "bound": "dct2", "ndim": NDIM}


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    try:
        ndev = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - depends on driver
        pytest.skip(f"cupy present but CUDA runtime unavailable: {exc}")
    if ndev < 1:
        pytest.skip("no CUDA device available")
    return cupy


def _operands(cupy, channels, seed=0):
    """A positive-definite compact-symmetric Hessian and a vector field."""
    rng = np.random.default_rng(seed)
    npack = channels * (channels + 1) // 2
    vec = rng.standard_normal(SPATIAL + (channels,))
    hes = np.zeros(SPATIAL + (npack,))
    for c in range(channels):
        hes[..., c] = 3.0 + c + rng.random(SPATIAL)
    hes[..., channels:] = 0.1 * rng.standard_normal(
        SPATIAL + (npack - channels,)
    )
    return cupy.asarray(hes), cupy.asarray(vec)


def _close(cupy, a, b):
    np.testing.assert_allclose(
        cupy.asnumpy(a), cupy.asnumpy(b), rtol=1e-12, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# forward / preconditioner vs. the composition they replaced                  #
# --------------------------------------------------------------------------- #


def test_field_forward_matches_sym_matvec_plus_field_matvec():
    cupy = _require_gpu()
    hes, vec = _operands(cupy, 2)
    got = ff.field_forward(hes, vec, **FIELD_PEN, **KW)
    ref = ff.sym_matvec(hes, vec) + ff.field_matvec(vec, **FIELD_PEN, **KW)
    _close(cupy, got, ref)


def test_flow_forward_matches_sym_matvec_plus_flow_matvec():
    cupy = _require_gpu()
    hes, vec = _operands(cupy, NDIM)
    got = ff.flow_forward(hes, vec, **FLOW_PEN, **KW)
    ref = ff.sym_matvec(hes, vec) + ff.flow_matvec(vec, **FLOW_PEN, **KW)
    _close(cupy, got, ref)


def test_field_precond_matches_sym_solve_with_field_diag():
    cupy = _require_gpu()
    hes, vec = _operands(cupy, 2)
    got = ff.field_precond(hes, vec, **FIELD_PEN, **KW)
    diag = ff.field_diag(vec.shape, **FIELD_PEN, **KW, dtype=vec.dtype)
    _close(cupy, got, ff.sym_solve(hes, vec, diag))


def test_flow_precond_matches_sym_solve_with_flow_diag():
    cupy = _require_gpu()
    hes, vec = _operands(cupy, NDIM)
    got = ff.flow_precond(hes, vec, **FLOW_PEN, **KW)
    diag = ff.flow_diag(vec.shape, **FLOW_PEN, **KW, dtype=vec.dtype)
    _close(cupy, got, ff.sym_solve(hes, vec, diag))


@pytest.mark.parametrize("op", ["forward", "precond"])
def test_leading_batch_dims_still_broadcast(op):
    """The composition broadcast batch dims via sym_*; the kernel must too."""
    cupy = _require_gpu()
    hes, vec = _operands(cupy, 2)
    single = getattr(ff, f"field_{op}")(hes, vec, **FIELD_PEN, **KW)
    batched = getattr(ff, f"field_{op}")(
        hes[None], cupy.broadcast_to(vec, (3,) + vec.shape).copy(),
        **FIELD_PEN, **KW,
    )
    assert batched.shape == (3,) + single.shape
    for i in range(3):
        _close(cupy, batched[i], single)


# --------------------------------------------------------------------------- #
# in-place preconditioners                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("family,pen", [("field", FIELD_PEN),
                                        ("flow", FLOW_PEN)])
def test_precond_inplace_agrees_and_mutates(family, pen):
    cupy = _require_gpu()
    hes, vec = _operands(cupy, 2 if family == "field" else NDIM)
    expected = getattr(ff, f"{family}_precond")(hes, vec, **pen, **KW)
    sol = vec.copy()
    returned = getattr(ff, f"{family}_precond_")(sol, hes, **pen, **KW)
    assert returned is sol
    _close(cupy, sol, expected)


def test_out_of_place_precond_leaves_its_input_alone():
    cupy = _require_gpu()
    hes, vec = _operands(cupy, 2)
    before = vec.copy()
    ff.field_precond(hes, vec, **FIELD_PEN, **KW)
    _close(cupy, vec, before)


# --------------------------------------------------------------------------- #
# stencil accumulate family                                                   #
# --------------------------------------------------------------------------- #


def _kernel(family):
    if family == "field":
        return ff.field_kernel(
            NDIM, **FIELD_PEN, channels=2, voxel_size=VS, bound="dct2"
        )
    return ff.flow_kernel(NDIM, **FLOW_PEN, voxel_size=VS, bound="dct2")


@pytest.mark.parametrize("family,pen", [("field", FIELD_PEN),
                                        ("flow", FLOW_PEN)])
def test_addkernel_on_zeros_is_the_kernel(family, pen):
    cupy = _require_gpu()
    k = _kernel(family)
    got = getattr(ff, f"{family}_addkernel")(cupy.zeros_like(k), **pen, **KW)
    _close(cupy, got, k)


@pytest.mark.parametrize("family,pen", [("field", FIELD_PEN),
                                        ("flow", FLOW_PEN)])
def test_subkernel_on_zeros_is_minus_the_kernel(family, pen):
    cupy = _require_gpu()
    k = _kernel(family)
    got = getattr(ff, f"{family}_subkernel")(cupy.zeros_like(k), **pen, **KW)
    _close(cupy, got, -k)


@pytest.mark.parametrize("family,pen", [("field", FIELD_PEN),
                                        ("flow", FLOW_PEN)])
@pytest.mark.parametrize("sign", ["add", "sub"])
def test_inplace_kernel_mutates_and_matches_functional(family, pen, sign):
    cupy = _require_gpu()
    start = _kernel(family) * 0.25
    expected = getattr(ff, f"{family}_{sign}kernel")(start, **pen, **KW)
    buf = start.copy()
    returned = getattr(ff, f"{family}_{sign}kernel_")(buf, **pen, **KW)
    assert returned is buf
    _close(cupy, buf, expected)


@pytest.mark.parametrize("family,pen", [("field", FIELD_PEN),
                                        ("flow", FLOW_PEN)])
def test_functional_kernel_accumulate_leaves_input_alone(family, pen):
    cupy = _require_gpu()
    start = _kernel(family) * 0.25
    before = start.copy()
    getattr(ff, f"{family}_addkernel")(start, **pen, **KW)
    _close(cupy, start, before)


def test_there_is_no_field_cg_on_cupy():
    """field_cg is CPU-only in fastfields-lib; see the package docstring."""
    assert not hasattr(ff, "field_cg"), (
        "field_cg appeared on the cupy backend: either fastfields-lib#34 "
        "landed a CUDA path (good -- update the docstring and "
        "tests/test_api_coverage.py's DELIBERATE), or this is a wrapper that "
        "can only raise"
    )
