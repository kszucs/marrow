"""The golden corpus, runtime lane.

A shim. Each case in `golden/cases/*.mojo` is transpiled by `runner.py` and
injected here as a real function, so item names match the Mojo lane's exactly
and `-k` selects the same cases in both.

The cases themselves are Mojo source. What runs here is the *same text* with
`raises` and `var` removed, executed against `helpers.NAMESPACE`.
"""

import marrow

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
