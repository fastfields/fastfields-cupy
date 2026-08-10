"""Guard: every reg wrapper must reach its *own* family's C symbol.

The cupy sibling of ``fastfields-numpy``/``fastfields-torch``'s
``tests/test_reg_family_routing.py``. The ``field_*`` and ``flow_*``
regularisers are different operators: ``field_*`` applies a per-channel
scalar penalty (channels are independent), while ``flow_*`` treats the last
axis as a vector displacement in *voxel* units and additionally offers the
cross-channel Lame terms (``shears``/``div``).

Upstream ``jitfields`` shipped a copy-paste bug of exactly this shape
(``jitfields.field_kernel_add`` calls the low-level ``flow_kernel``), and with
an isotropic ``voxel_size`` and no Lame terms the two families produce
identical numbers -- so a swap is completely silent under default arguments
and has to be pinned down mechanically.

This file has to work where its numpy/torch siblings do not: there is no GPU
in CI and cupy is not even installed, so the runtime spy test they use cannot
run here. The primary guard is therefore **static** -- it reads the wrapper
source and checks which ``_ff.*`` binding each one can reach. That runs
everywhere and is what actually catches a family swap. The runtime spy test is
kept as well, mirroring the siblings exactly, and skips without cupy + device.
"""

from __future__ import annotations

import ast
import inspect
import math

import pytest

import fastfields.cupy as ff
from fastfields.cupy import _reg


def _public_reg_names():
    return {n for n in ff.__all__ if n.startswith(("field_", "flow_"))}


# --------------------------------------------------------------------------- #
# 1. Static routing guard (always runs -- no cupy, no GPU needed)             #
# --------------------------------------------------------------------------- #


def _function_defs():
    """Every ``def`` at module scope in ``_reg``, by name."""
    tree = ast.parse(inspect.getsource(_reg))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _reachable_bindings(name, defs, seen=None):
    """``_ff.*`` bindings reachable from ``defs[name]``, transitively.

    Wrappers delegate two ways: straight to ``_ff.<symbol>``, or through
    another function in the same module (``field_precond`` -> ``field_diag``,
    ``flow_addmatvec`` -> ``_flow_matvec_acc``). Both have to be followed, or
    a swap hidden one level down would go unnoticed. Names imported from
    elsewhere (``sym_solve``, ``sym_matvec``) are not reg-family symbols and
    are deliberately not followed.
    """
    seen = set() if seen is None else seen
    if name in seen:
        return set()
    seen.add(name)
    found = set()
    for sub in ast.walk(defs[name]):
        if (
            isinstance(sub, ast.Attribute)
            and isinstance(sub.value, ast.Name)
            and sub.value.id == "_ff"
        ):
            found.add(sub.attr)
        elif isinstance(sub, ast.Name) and sub.id in defs:
            found |= _reachable_bindings(sub.id, defs, seen)
    return found


def test_every_public_reg_wrapper_is_defined_in_reg():
    """Public reg names must come from ``_reg`` (so this file can read it)."""
    missing = sorted(_public_reg_names() - set(_function_defs()))
    assert not missing, f"public reg names not defined in _reg.py: {missing}"


@pytest.mark.parametrize("name", sorted(_public_reg_names()))
def test_wrapper_only_names_its_own_family(name):
    defs = _function_defs()
    family = name.split("_")[0]
    symbols = _reachable_bindings(name, defs)
    assert symbols, f"{name} reaches no fastfields.dlpack binding"
    wrong = sorted(s for s in symbols if not s.startswith(family + "_"))
    assert not wrong, (
        f"{name} is wired to the wrong regulariser family: it calls "
        f"{wrong} but must only call {family}_* symbols"
    )


@pytest.mark.parametrize("name", sorted(_public_reg_names()))
def test_public_wrapper_is_re_exported(name):
    """``_reg.__all__`` and the package ``__init__`` must not drift apart.

    Forgetting the ``__init__.py`` re-export is a real bug this stack has
    shipped before (fastfields-lib#33).
    """
    assert callable(getattr(ff, name, None)), f"{name} not re-exported"
    assert name in _reg.__all__, f"{name} missing from _reg.__all__"


def test_rls_wrappers_are_exposed():
    """RLS/JRLS parity with the C++ layer, for both families.

    ``fastfields-cupy#20`` added the field-side ones; the flow-side ones
    (dispatched and tested in C++ since fastfields-lib#41) followed for
    fastfields-lib#69.
    """
    expected = {
        "field_matvec_rls",
        "field_diag_rls",
        "field_relax_rls",
        "flow_matvec_rls",
        "flow_diag_rls",
        "flow_relax_rls",
    }
    missing = sorted(expected - _public_reg_names())
    assert not missing, f"RLS/JRLS wrappers not exposed: {missing}"


# --------------------------------------------------------------------------- #
# 2. Runtime routing guard (needs cupy + a CUDA device)                       #
# --------------------------------------------------------------------------- #

# Anisotropic on purpose: this is what makes field != flow observable.
VS = [1.0, 2.0, 3.0]
NDIM = 3
SHAPE = (6, 6, 6, NDIM)
KW = dict(membrane=1.0, voxel_size=VS, ndim=NDIM)


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    try:
        ndev = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - depends on driver
        pytest.skip(f"cupy present but CUDA runtime unavailable: {exc}")
    if ndev < 1:  # pragma: no cover - depends on hardware
        pytest.skip("no CUDA device available")
    return cupy


def _cases(cp):
    """Every public field_*/flow_* wrapper, with a call that reaches C."""

    def _x():
        n = math.prod(SHAPE)
        return cp.arange(n, dtype=cp.float64).reshape(SHAPE) / n

    def _hes():
        return cp.ones((6, 6, 6, NDIM * (NDIM + 1) // 2), dtype=cp.float64)

    def _g():
        return cp.ones(SHAPE, dtype=cp.float64)

    def _wgt():
        # JRLS weight map (trailing dim 1, shared across channels).
        return cp.ones((6, 6, 6, 1), dtype=cp.float64)

    return {
        "field_matvec": lambda: ff.field_matvec(_x(), **KW),
        "field_kernel": lambda: ff.field_kernel(
            NDIM, membrane=1.0, channels=NDIM, voxel_size=VS
        ),
        "field_diag": lambda: ff.field_diag(SHAPE, **KW),
        "field_relax": lambda: ff.field_relax(_x(), _hes(), _g(), **KW),
        "field_precond": lambda: ff.field_precond(_hes(), _x(), **KW),
        "field_forward": lambda: ff.field_forward(_hes(), _x(), **KW),
        "field_addmatvec": lambda: ff.field_addmatvec(_x(), _x(), **KW),
        "field_submatvec": lambda: ff.field_submatvec(_x(), _x(), **KW),
        "field_addmatvec_": lambda: ff.field_addmatvec_(_x(), _x(), **KW),
        "field_submatvec_": lambda: ff.field_submatvec_(_x(), _x(), **KW),
        "field_adddiag": lambda: ff.field_adddiag(_x(), **KW),
        "field_subdiag": lambda: ff.field_subdiag(_x(), **KW),
        "field_adddiag_": lambda: ff.field_adddiag_(_x(), **KW),
        "field_subdiag_": lambda: ff.field_subdiag_(_x(), **KW),
        "field_matvec_rls": lambda: ff.field_matvec_rls(_x(), _wgt(), **KW),
        "field_diag_rls": lambda: ff.field_diag_rls(
            _wgt(), membrane=1.0, channels=NDIM, voxel_size=VS, ndim=NDIM
        ),
        "field_relax_rls": lambda: ff.field_relax_rls(
            _x(), _hes(), _g(), _wgt(), **KW
        ),
        "flow_matvec": lambda: ff.flow_matvec(_x(), **KW),
        "flow_kernel": lambda: ff.flow_kernel(
            NDIM, membrane=1.0, voxel_size=VS
        ),
        "flow_diag": lambda: ff.flow_diag(SHAPE, **KW),
        "flow_relax": lambda: ff.flow_relax(_x(), _hes(), _g(), **KW),
        "flow_precond": lambda: ff.flow_precond(_hes(), _x(), **KW),
        "flow_forward": lambda: ff.flow_forward(_hes(), _x(), **KW),
        "flow_addmatvec": lambda: ff.flow_addmatvec(_x(), _x(), **KW),
        "flow_submatvec": lambda: ff.flow_submatvec(_x(), _x(), **KW),
        "flow_addmatvec_": lambda: ff.flow_addmatvec_(_x(), _x(), **KW),
        "flow_submatvec_": lambda: ff.flow_submatvec_(_x(), _x(), **KW),
        "flow_adddiag": lambda: ff.flow_adddiag(_x(), **KW),
        "flow_subdiag": lambda: ff.flow_subdiag(_x(), **KW),
        "flow_adddiag_": lambda: ff.flow_adddiag_(_x(), **KW),
        "flow_subdiag_": lambda: ff.flow_subdiag_(_x(), **KW),
        "flow_matvec_rls": lambda: ff.flow_matvec_rls(_x(), _wgt(), **KW),
        "flow_diag_rls": lambda: ff.flow_diag_rls(_wgt(), **KW),
        "flow_relax_rls": lambda: ff.flow_relax_rls(
            _x(), _hes(), _g(), _wgt(), **KW
        ),
    }


def test_every_public_reg_wrapper_has_a_runtime_case():
    """A newly exposed field_*/flow_* wrapper must be added to the table.

    Building the table only creates closures, so this forcing function needs
    cupy importable but no device.
    """
    cp = pytest.importorskip("cupy")
    assert _public_reg_names() == set(_cases(cp))


@pytest.mark.parametrize("name", sorted(_public_reg_names()))
def test_wrapper_only_calls_its_own_family(name, monkeypatch):
    cp = _require_gpu()
    import fastfields.dlpack as _fb

    seen = []
    for sym in [n for n in dir(_fb) if n.startswith(("field_", "flow_"))]:
        orig = getattr(_fb, sym)

        def spy(*args, _n=sym, _o=orig, **kwargs):
            seen.append(_n)
            return _o(*args, **kwargs)

        monkeypatch.setattr(_fb, sym, spy)
    _cases(cp)[name]()
    assert seen, f"{name} reached no fastfields.dlpack symbol"
    family = name.split("_")[0]
    wrong = [s for s in seen if not s.startswith(family + "_")]
    assert not wrong, (
        f"{name} is wired to the wrong regulariser family: it called "
        f"{wrong} but must only call {family}_* symbols"
    )
