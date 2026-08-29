"""The golden corpus, runtime lane.

A shim, and — unlike `test_cases.mojo` — **not** generated, because there is
nothing per-case to generate. Python injects each case into `globals()` at
import, so the file holds one line of wiring; the Mojo lane needs a real
`def test_*(` per case for the harness to collect, which is why that side has a
generated module and this one does not.

The cases are Mojo source. What runs here is the *same body* with `var`
dropped — the only rule, since `runner.transpile` writes the `def` line rather
than transcribing it — executed against `helpers.NAMESPACE`. A traceback
points into `golden/cases/<name>.mojo`, the file you edit.
"""

import pytest

import marrow

# The Python lane runs the corpus against marrow's *Python* frontend — `col`,
# `lit`, `count_star`, `if_else` and `LazyTable`, over the `Expr` / `Agg` /
# `Plan` binding types. All of that was removed on 2026-08-29 with the old
# expression package it was built on, and `marrow/expr/` has no Python bindings
# yet. Skip at module level rather than fail at import: a collection error here
# aborts the whole session, so the *Mojo* lane — which is ported and passing —
# would silently report zero cases run. Delete this guard when the bindings
# return; `helpers.py` is unchanged and still describes the intended lane.
if not hasattr(marrow, "col"):
    pytest.skip(
        "golden's Python lane needs marrow's expression frontend, which was "
        "removed with the old expression package",
        allow_module_level=True,
    )

import helpers
import runner

runner.install(helpers.NAMESPACE, helpers.check, globals())

# Infrastructure, not vocabulary: `table` is not marrow API, and it is not a
# place the two lanes disagree about how to spell an expression. `check` is no
# longer here at all — a case returns its plan and the harness checks it.
INFRASTRUCTURE = {"table"}


def test_golden_shims_are_declared():
    """Every case-vocabulary name is real marrow API or a declared shim.

    `helpers.SHIMS` is the convergence metric — one entry per spelling the two
    lanes still disagree about — and a metric nobody maintains is worse than
    none. This fails if a name is added to `NAMESPACE` without being either
    resolved against marrow or recorded as debt, and it fails the other way
    too: a shim that has become real marrow API must leave the set.
    """
    undeclared, stale = [], []
    for name, value in helpers.NAMESPACE.items():
        if name in INFRASTRUCTURE:
            continue
        real = getattr(marrow, name, None) is value
        if real and name in helpers.SHIMS:
            stale.append(name)
        elif not real and name not in helpers.SHIMS:
            undeclared.append(name)
    assert not undeclared, (
        f"not real marrow API and not declared in helpers.SHIMS: {sorted(undeclared)}"
    )
    assert not stale, (
        f"now real marrow API — delete from helpers.SHIMS: {sorted(stale)}"
    )
