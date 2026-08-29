"""Parameter values for one execution.

Split out of `params.mojo`, which held two unrelated things: this alias, which
depends on nothing in the package, and `Param[T]`, a comptime-lane leaf node.
Sharing a file forced the alias to drag in `logical.Shape` and
`pruning.param_bounds`, and both of those import it back -- two of the
package's dependency cycles existed only because of the pairing. `Param` now
lives beside the other comptime leaves; this is a genuine leaf module.
"""

from std.collections import Dict

from ..scalars import DynScalar


comptime Bindings = Dict[String, DynScalar]
"""Parameter values for one execution: a plain name -> scalar map.

An alias, not a struct. It only ever wrapped a `Dict[String, DynScalar]` with
a `set`/`get` pair that `Dict` already provides — and `Dict.get` does not even
raise, where the wrapper's did. Being an alias means a caller writes a dict
literal:

    plan.execute(bindings={"min-a": Int64Scalar(4).to_dyn()})

Passed to `to_operator`, not stored on the plan, which is what keeps a plan
immutable and lets two executions use different values without interfering.

Missing names are not an error here — a parameter with a default is satisfied
without one, and `Param` raises naming itself when it has neither.
"""
