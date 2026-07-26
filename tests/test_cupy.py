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
    from fastfields.cupy._resample import _anchor_scale_shift

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
    from fastfields.cupy._resample import _anchor_scale_shift

    with pytest.raises(ValueError, match="anchor"):
        _anchor_scale_shift("nope", (8,), (4,), 1)


def test_factor_shape_resolution_no_cupy():
    """factor/shape/ndim resolution is pure Python (no cupy needed)."""
    from fastfields.cupy._resample import _infer_ndim, _resolve_out_spatial

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
    from fastfields.dlpack import Bound

    from fastfields.cupy._util import as_bound, as_spline

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
