"""Guard: the three ``dt_spline_*`` wrappers return ``(dist, time)``.

fastfields#38. The point-to-spline operators return a 2-tuple, and this
wrapper used to return it as ``(time, dist)`` while ``fastfields.numpy`` and
``fastfields.torch`` returned ``(dist, time)``. ``fastfields.auto`` forwards a
result unchanged, so the meaning of a dispatched call's return value depended
on which array type the caller passed -- a correctness hole in a dispatcher
whose whole purpose is backend independence. The owner's decision was
``(dist, time)`` everywhere, so cupy moved.

The flip is easy to make again, in either direction, because the raw binding
takes its output buffers the *other* way round
(``_ff.dt_spline_*(time, dist, ...)``, the order the C++ hub declares): a
reader who assumes the wrapper mirrors the binding will "fix" the return
statement, and nothing else in this repo would complain. The numeric suite
cannot: it needs a GPU, and there is none in CI -- which is exactly how the
divergence survived long enough to need an issue.

So this guard is **static**. It parses ``fastfields/cupy/_dt.py`` from the
checkout by path and never imports it, so it runs with no GPU, no cupy, and
no compiled ``fastfields.dlpack`` -- i.e. everywhere, including the
environments where every other check of this behaviour skips.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

#: Parsed by path rather than imported: see the module docstring.
_DT_SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fastfields"
    / "cupy"
    / "_dt.py"
)

SPLINE_OPS = ("dt_spline_table", "dt_spline_brent", "dt_spline_gaussnewton")

#: The canonical return order, shared with fastfields.numpy and
#: fastfields.torch (fastfields#38).
CANONICAL_ORDER = ("dist", "time")

#: The order the raw binding declares its two output buffers in. Deliberately
#: the reverse of the above; that is the whole trap.
BINDING_ORDER = ("time", "dist")


def _function_defs():
    """Every ``def`` at module scope in ``_dt.py``, by name."""
    assert _DT_SOURCE.is_file(), f"cannot find {_DT_SOURCE}"
    tree = ast.parse(_DT_SOURCE.read_text())
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _returned_names(node):
    """The names in the single ``return <tuple>`` of ``node``."""
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert len(returns) == 1, (
        f"{node.name} has {len(returns)} return statements; this guard "
        "assumes exactly one"
    )
    value = returns[0].value
    assert isinstance(value, ast.Tuple), (
        f"{node.name} does not return a tuple literal any more; update this "
        "guard together with the wrapper"
    )
    names = []
    for element in value.elts:
        assert isinstance(element, ast.Name), (
            f"{node.name} returns a non-name element; update this guard"
        )
        names.append(element.id)
    return tuple(names)


@pytest.mark.parametrize("name", SPLINE_OPS)
def test_wrapper_returns_dist_then_time(name):
    defs = _function_defs()
    assert name in defs, f"fastfields.cupy._dt has no {name!r}"
    assert _returned_names(defs[name]) == CANONICAL_ORDER, (
        f"fastfields.cupy.{name} must return {CANONICAL_ORDER} to match "
        "fastfields.numpy and fastfields.torch (fastfields#38). The binding "
        "takes (time, dist); the wrapper does not."
    )


@pytest.mark.parametrize("name", SPLINE_OPS)
def test_binding_call_keeps_the_hub_argument_order(name):
    """The *call* into the binding stays ``(time, dist, ...)``.

    If someone aligns the call site with the return statement, the wrapper
    still compiles and still runs -- and silently reports distance as time.
    """
    defs = _function_defs()
    calls = [
        node
        for node in ast.walk(defs[name])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == name
    ]
    assert len(calls) == 1, f"{name} does not call _ff.{name} exactly once"
    first_two = []
    for arg in calls[0].args[:2]:
        assert isinstance(arg, ast.Name), (
            f"{name}'s binding call no longer passes plain names; update "
            "this guard"
        )
        first_two.append(arg.id)
    assert tuple(first_two) == BINDING_ORDER, (
        f"fastfields.cupy.{name} must pass the binding's output buffers as "
        f"{BINDING_ORDER} -- that is the order fastfields.dlpack declares. "
        "Only the wrapper's *return* order is (dist, time)."
    )


@pytest.mark.parametrize("name", SPLINE_OPS)
def test_docstring_states_the_return_order(name):
    """A caller must not have to read the source to know the order."""
    doc = ast.get_docstring(_function_defs()[name]) or ""
    assert "``(dist, time)``" in doc, (
        f"fastfields.cupy.{name}'s docstring must state the "
        "``(dist, time)`` return order"
    )
    assert "``(time, dist)``" not in doc, (
        f"fastfields.cupy.{name}'s docstring still claims ``(time, dist)``"
    )
