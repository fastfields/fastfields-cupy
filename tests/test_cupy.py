"""Tests for fastfields.cupy.

Two groups:

1. Environment-independent tests that always run: they verify that
   ``import fastfields.cupy`` succeeds *without* cupy installed (the
   lazy-import requirement) and that calling a wrapper without cupy raises a
   clear error.

2. GPU correctness tests that require cupy *and* a CUDA device. They mirror the
   numpy-package checks (Euclidean DT vs brute force, sym_matvec vs dense) and
   are cleanly skipped when cupy/GPU is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# 1. Always-on tests (must pass even where cupy / GPU is absent)              #
# --------------------------------------------------------------------------- #
def test_import_without_cupy():
    """`import fastfields.cupy` must not hard-fail when cupy is missing."""
    import fastfields.cupy as ffc

    # Enums come straight from fastfields.dlpack (no cupy needed).
    assert int(ffc.Spline.Cubic) == 3
    assert int(ffc.Bound.DCT2) == 3
    for name in ("dt_euclidean", "sym_matvec", "resample", "spline_coeff"):
        assert callable(getattr(ffc, name))


def test_anchor_scale_shift_mapping():
    """The anchor->(scale, shift) mapping is pure Python (no cupy needed)."""
    from fastfields.dlpack import anchor_scale_shift as _anchor_scale_shift

    for name, abbr, exp_scale, exp_shift in [
        ("centers", "c", 7 / 3, 0.0),
        ("edges", "e", 2.0, 0.5),
        ("first", "f", 2.0, 0.0),
        ("last", "l", 2.0, 1.0),
    ]:
        scale, shift = _anchor_scale_shift(name, (8,), (4,), 1)
        assert shift == exp_shift
        assert scale == pytest.approx([exp_scale])
        # the abbreviation resolves to the same mapping
        assert _anchor_scale_shift(abbr, (8,), (4,), 1) == (scale, shift)


def test_anchor_unknown_raises():
    from fastfields.dlpack import anchor_scale_shift as _anchor_scale_shift

    with pytest.raises(ValueError, match="anchor"):
        _anchor_scale_shift("nope", (8,), (4,), 1)


def test_factor_shape_resolution_no_cupy():
    """factor/shape/ndim resolution is pure Python (no cupy needed)."""
    from fastfields.dlpack import (
        infer_ndim as _infer_ndim,
    )
    from fastfields.dlpack import (
        resolve_out_spatial as _resolve_out_spatial,
    )

    assert _infer_ndim(None, None, [4, 4]) == 2
    assert _infer_ndim(None, 2.0, None) == 1
    assert _infer_ndim(3, None, None) == 3
    # shape wins; scalar broadcasts to ndim
    assert _resolve_out_spatial((5, 5), 2, None, 10) == (10, 10)
    # factor rounds per-dim
    assert _resolve_out_spatial((5,), 1, 2, None) == (10,)
    # neither -> identity
    assert _resolve_out_spatial((7,), 1, None, None) == (7,)


def test_order_bound_aliases_no_cupy():
    """order/bound accept int, enum or name (no cupy needed)."""
    from fastfields.dlpack import Bound, as_bound, as_spline

    assert as_spline("linear") == 1
    assert as_spline(3) == 3
    assert as_bound("dct2") == 3
    assert as_bound("wrap") == int(Bound.DFT)
    with pytest.raises(ValueError, match="spline order"):
        as_spline("nope")
    with pytest.raises(ValueError, match="boundary"):
        as_bound("nope")


def _cupy_missing() -> bool:
    try:
        import cupy  # noqa: F401
    except ImportError:
        return True
    return False


@pytest.mark.skipif(not _cupy_missing(), reason="cupy is installed")
def test_call_without_cupy_raises():
    """Calling a wrapper without cupy raises a clear ImportError."""
    import fastfields.cupy as ffc

    with pytest.raises(ImportError, match="cupy"):
        ffc.current_stream_ptr()


# --------------------------------------------------------------------------- #
# 2. GPU correctness tests (skipped without cupy + CUDA device)              #
# --------------------------------------------------------------------------- #
def _require_gpu():
    """Skip unless cupy imports and at least one CUDA device is present."""
    cupy = pytest.importorskip("cupy")
    try:
        ndev = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - depends on driver
        pytest.skip(f"cupy present but CUDA runtime unavailable: {exc}")
    if ndev < 1:
        pytest.skip("no CUDA device available")
    return cupy


def _edt_reference(inp, voxel_spacing, cost):
    """Brute-force distance transform along the last axis (numpy)."""
    n = inp.shape[-1]
    flat = inp.reshape(-1, n)
    ref = np.full_like(flat, np.inf)
    for r in range(flat.shape[0]):
        for i in range(n):
            best = np.inf
            for j in range(n):
                best = min(best, flat[r, j] + cost(voxel_spacing * (i - j)))
            ref[r, i] = best
    return ref.reshape(inp.shape)


def _pack_symmetric(mats):
    """Dense (B,C,C) symmetric -> compact (B, C*(C+1)/2) diagonal-then-rows."""
    B, C, _ = mats.shape
    packed = np.zeros((B, C * (C + 1) // 2), dtype=mats.dtype)
    for b in range(B):
        idx = 0
        for k in range(C):
            packed[b, idx] = mats[b, k, k]
            idx += 1
        for i in range(C):
            for j in range(i + 1, C):
                packed[b, idx] = mats[b, i, j]
                idx += 1
    return packed


def test_dt_euclidean_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    inp = np.array(
        [
            [0, np.inf, np.inf, 0, np.inf, np.inf, np.inf],
            [np.inf, np.inf, 0, np.inf, np.inf, 0, np.inf],
        ],
        dtype=np.float32,
    )
    ref = _edt_reference(inp, 1.0, lambda d: d * d)
    out = ffc.dt_euclidean(cupy.asarray(inp), 1.0)
    np.testing.assert_allclose(cupy.asnumpy(out), ref, rtol=1e-5, atol=1e-5)


def test_dt_euclidean_inplace_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    inp = np.array(
        [[0, np.inf, np.inf, 0, np.inf, np.inf, np.inf]],
        dtype=np.float32,
    )
    ref = _edt_reference(inp, 1.0, lambda d: d * d)
    gpu = cupy.asarray(inp)
    ret = ffc.dt_euclidean_(gpu)
    assert ret is gpu  # written in place
    np.testing.assert_allclose(cupy.asnumpy(gpu), ref, rtol=1e-5, atol=1e-5)


def test_sym_matvec_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    for C in (2, 3):
        B = 4
        rng = np.random.default_rng(C)
        mats = rng.standard_normal((B, C, C))
        mats = mats + np.transpose(mats, (0, 2, 1))
        vec = rng.standard_normal((B, C))

        hessian = _pack_symmetric(mats)
        ref = np.einsum("bij,bj->bi", mats, vec)
        out = ffc.sym_matvec(cupy.asarray(hessian), cupy.asarray(vec))
        np.testing.assert_allclose(
            cupy.asnumpy(out), ref, rtol=1e-8, atol=1e-8
        )


def test_sym_matvec_broadcasts_batch_dims_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    # hessian batch (1,) vs vec batch (5,): must broadcast (zero-copy) and
    # match the manually-broadcast dense product.
    C, B = 3, 5
    rng = np.random.default_rng(0)
    mats = rng.standard_normal((1, C, C))
    mats = mats + np.transpose(mats, (0, 2, 1))  # batch (1,)
    vec = rng.standard_normal((B, C))  # batch (5,)
    hessian = _pack_symmetric(mats)

    out = ffc.sym_matvec(cupy.asarray(hessian), cupy.asarray(vec))
    assert out.shape == (B, C)
    ref = np.einsum("bij,bj->bi", np.broadcast_to(mats, (B, C, C)), vec)
    np.testing.assert_allclose(cupy.asnumpy(out), ref, rtol=1e-8, atol=1e-8)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------- #
# pushpull + regularisers                                                     #
# --------------------------------------------------------------------------- #


def test_pushpull_reg_surface_present():
    import fastfields.cupy as ffc

    for name in [
        "pull",
        "push",
        "count",
        "grad",
        "field_matvec",
        "field_matvec_add",
        "field_matvec_sub",
        "field_diag",
        "field_diag_add",
        "field_diag_sub",
        "field_kernel",
        "field_precond",
        "field_forward",
        "flow_matvec",
        "flow_matvec_add",
        "flow_matvec_add_",
        "flow_matvec_sub",
        "flow_matvec_sub_",
        "flow_diag",
        "flow_diag_add",
        "flow_diag_add_",
        "flow_diag_sub",
        "flow_diag_sub_",
        "flow_kernel",
        "flow_relax",
        "flow_precond",
        "flow_forward",
    ]:
        assert hasattr(ffc, name), name


def test_pull_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    inp = cupy.asarray([[0.0], [10.0], [20.0], [30.0]], dtype=cupy.float64)
    grid = cupy.asarray([[0.5], [1.5], [2.5]], dtype=cupy.float64)
    out = ffc.pull(inp, grid, order=1)
    assert cupy.allclose(out.ravel(), cupy.asarray([5.0, 15.0, 25.0]))


def test_push_is_pull_adjoint_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    grid = cupy.linspace(0, 5, 4, dtype=cupy.float64).reshape(4, 1)
    x = cupy.random.standard_normal((6, 1)).astype(cupy.float64)
    y = cupy.random.standard_normal((4, 1)).astype(cupy.float64)
    px = ffc.pull(x, grid, order=2)
    py = ffc.push(y, grid, shape=6, order=2)
    assert cupy.allclose((px * y).sum(), (x * py).sum())


def test_field_matvec_absolute_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    f = cupy.random.standard_normal((8, 2)).astype(cupy.float64)
    out = ffc.field_matvec(f, absolute=[2.0, 3.0], ndim=1)
    assert cupy.allclose(out[:, 0], 2.0 * f[:, 0])
    assert cupy.allclose(out[:, 1], 3.0 * f[:, 1])


def _flow_hessian_2d_gpu(cupy, H, W, seed):
    """Per-voxel SPD 2x2 Hessian, packed compact-symmetric -> (H, W, 3)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((H * W, 2, 2))
    mats = np.einsum("bij,bkj->bik", A, A) + 3.0 * np.eye(2)
    packed = _pack_symmetric(mats).reshape(H, W, 3)
    return cupy.asarray(packed)


def test_flow_kernel_is_matvec_impulse_response_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    cases = [
        (dict(absolute=2.5), False, 1),
        (dict(membrane=1.0), False, 3),
        (dict(bending=1.0), False, 5),
        (dict(shears=1.3, div=0.7), True, 3),
        (
            dict(absolute=0.3, membrane=0.5, bending=0.4, shears=1.3, div=0.7),
            True,
            5,
        ),
    ]
    C = 2
    for kw, is_matrix, width in cases:
        K = ffc.flow_kernel(2, **kw)
        assert K.shape == (
            (width, width, C, C) if is_matrix else (width, width, C)
        )
        kd = width
        N, cc, half = 2 * kd + 1, kd, kd // 2
        for j0 in range(C):
            x = cupy.zeros((N, N, C))
            x[cc, cc, j0] = 1.0
            o = ffc.flow_matvec(x, ndim=2, **kw)
            for a in range(kd):
                for b in range(kd):
                    for i in range(C):
                        got = float(o[cc + a - half, cc + b - half, i])
                        kern = (
                            float(K[a, b, i, j0])
                            if is_matrix
                            else (float(K[a, b, i]) if i == j0 else 0.0)
                        )
                        assert abs(got - kern) < 1e-10


def test_flow_forward_is_sym_matvec_plus_flow_matvec_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    H, W = 5, 6
    mat = _flow_hessian_2d_gpu(cupy, H, W, 11)
    vec = cupy.asarray(np.random.default_rng(11).standard_normal((H, W, 2)))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5)
    fwd = ffc.flow_forward(mat, vec, ndim=2, **kw)
    expect = ffc.sym_matvec(mat, vec) + ffc.flow_matvec(vec, ndim=2, **kw)
    assert cupy.allclose(fwd, expect, atol=1e-10)


def test_flow_precond_solves_diagonal_system_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    H, W = 5, 6
    mat = _flow_hessian_2d_gpu(cupy, H, W, 12)
    vec = cupy.asarray(np.random.default_rng(12).standard_normal((H, W, 2)))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5)
    x = ffc.flow_precond(mat, vec, ndim=2, **kw)
    diag = ffc.flow_diag(vec.shape, ndim=2, **kw)
    residual = ffc.sym_matvec(mat, x) + diag * x - vec
    assert cupy.allclose(residual, cupy.zeros_like(residual), atol=1e-5)


def test_flow_accumulate_variants_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    H, W = 5, 6
    rng = cupy.random.RandomState(21)
    flow = rng.standard_normal((H, W, 2))
    base = rng.standard_normal((H, W, 2))
    kw = dict(absolute=0.3, membrane=0.7, shears=1.0, div=0.5, ndim=2)
    L = ffc.flow_matvec(flow, **kw)
    d = ffc.flow_diag(base.shape, **kw)
    assert cupy.allclose(ffc.flow_matvec_add(base, flow, **kw), base + L)
    assert cupy.allclose(ffc.flow_matvec_sub(base, flow, **kw), base - L)
    assert cupy.allclose(ffc.flow_diag_add(base, **kw), base + d)
    assert cupy.allclose(ffc.flow_diag_sub(base, **kw), base - d)
    a = base.copy()
    assert ffc.flow_matvec_add_(a, flow, **kw) is a
    assert cupy.allclose(a, base + L)
    s = base.copy()
    assert ffc.flow_diag_sub_(s, **kw) is s
    assert cupy.allclose(s, base - d)


def test_field_precond_forward_accumulate_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    H, W, C = 5, 6, 2
    rng = np.random.default_rng(31)
    A = rng.standard_normal((H * W, C, C))
    mats = np.einsum("bij,bkj->bik", A, A) + (C + 1) * np.eye(C)
    mat = cupy.asarray(_pack_symmetric(mats).reshape(H, W, C * (C + 1) // 2))
    vec = cupy.asarray(rng.standard_normal((H, W, C)))
    kw = dict(absolute=[0.3, 0.4], membrane=[0.7, 0.5], ndim=2)
    fwd = ffc.field_forward(mat, vec, **kw)
    assert cupy.allclose(
        fwd, ffc.sym_matvec(mat, vec) + ffc.field_matvec(vec, **kw), atol=1e-10
    )
    x = ffc.field_precond(mat, vec, **kw)
    diag = ffc.field_diag(vec.shape, **kw)
    residual = ffc.sym_matvec(mat, x) + diag * x - vec
    assert cupy.allclose(residual, cupy.zeros_like(residual), atol=1e-5)
    base = cupy.asarray(rng.standard_normal((H, W, C)))
    L = ffc.field_matvec(vec, **kw)
    assert cupy.allclose(ffc.field_matvec_add(base, vec, **kw), base + L)


def test_field_kernel_is_matvec_impulse_response_gpu():
    cupy = _require_gpu()
    import fastfields.cupy as ffc

    cases = [
        (1, dict(absolute=[2.5, 1.5])),
        (3, dict(absolute=[0.3, 0.4], membrane=[1.0, 0.7])),
        (
            5,
            dict(absolute=[0.3, 0.4], membrane=[0.5, 0.6], bending=[1.0, 0.8]),
        ),
    ]
    C = 2
    for width, kw in cases:
        K = ffc.field_kernel(2, **kw)
        assert K.shape == (width, width, C)
        kd = width
        N, cc, half = 2 * kd + 1, kd, kd // 2
        for c0 in range(C):
            x = cupy.zeros((N, N, C))
            x[cc, cc, c0] = 1.0
            o = ffc.field_matvec(x, ndim=2, **kw)
            for a in range(kd):
                for b in range(kd):
                    for c in range(C):
                        got = float(o[cc + a - half, cc + b - half, c])
                        kern = float(K[a, b, c]) if c == c0 else 0.0
                        assert abs(got - kern) < 1e-10
