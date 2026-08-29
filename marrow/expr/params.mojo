"""Late-bound query parameters.

A parameter is **a literal whose value arrives later**: it has a dtype and a
shape when the plan is built, and a value only once something binds it. That is
why `Param` mirrors `Literal` — same families, same `Shape.scalar`, same
per-family split — rather than being a category of its own.

**A parameter is a description; its value belongs to an execution.** The node
holds a name, a dtype, help text and an optional default, and nothing else — no
cell, no mutable state. Values arrive through `Bindings` when the plan is
turned into operators:

    var min_a = param("min-a", int64)
    var plan = t.filter(col("a", int64) > min_a)

    plan.execute(bindings=Bindings().set("min-a", Int64Scalar(4).to_dyn()))

That is the layer's own rule — *a logical node is stateless* — applied here.
An earlier version of this module held the value in an `ArcPointer` cell shared
by every copy of the node, so `min_a.set(4)` reached into a built plan and
changed what it computed. It made a plan's result depend on hidden mutable
state, and it made executing one plan on two threads with two values a data
race: the same defect `expr/`'s process-global registry has, relocated into the
node rather than removed.

Passing the values *through* the execution instead means the plan stays
immutable, two executions with different values cannot interfere, and there is
no cell.

That one property removes an entire subsystem. `expr/` declares parameters
*inline* at each use — `col("a") > param("min-a", int64)` written twice must
still share a cell — so it needs a process-global registry keyed by name, a
second lookup table for the runtime lane, dedup on every declaration, and a
dtype-conflict check. It also inherits two limitations it records honestly: a
plan built but never executed leaks its declarations into the next plan's
`--help`, and the globals are unsynchronised, so building two plans on two
threads is a data race.

None of that exists here. Sharing is structural rather than name-keyed, so
there is no registry to leak and no global to race on, and one declaration
cannot conflict with itself.

There is deliberately **no `params()` traversal**. Asking a plan which
parameters it takes is a sixteen-method walk that only a `--help` surface
would use, and nothing outside a test ever asked. Add it back when something
does; until then a plan's parameters are discovered the way its columns are —
by binding it and being told, by name, which one is missing. `expr/`'s
`ParamCell` raises "parameter is not bound" *without* naming it, because a
cell cannot know the name it is read through. Here the node **is** the
parameter, so it can.
"""

from std.collections import Dict

from ..arrays import StructArray
from ..buffers import Bitmap
from ..dtypes import DynType, NumericType
from ..scalars import DynScalar
from ..schema import Schema
from ..tabular import RecordBatch
from .logical import Shape
from .`comptime`.core import NumericValue


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


struct Param[T: NumericType](NumericValue):
    """A late-bound numeric scalar — `Literal[T]` whose value arrives later.

    Immutable. It knows its name, dtype, help and default; the *value* arrives
    through `Bindings`, which the operator carries and hands back down to
    `bind`. `Shape.scalar`, so a predicate over a parameter costs one
    broadcast, exactly as a literal does.

    An unbound parameter with no default raises **naming itself** — the node is
    the parameter, so it can, where `expr/`'s cell explicitly cannot.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = Scalar[Self.T.native]
    """This execution's value, looked up once per batch — the same stage at
    which a column leaf resolves its column."""

    var _name: String
    var _help: String
    var _default: Optional[Scalar[Self.T.native]]
    """Used when nothing binds this name. Absent means the parameter is
    required, and `bind` says so naming it."""

    def __init__(
        out self,
        var name: String,
        var help: String = String(),
        var default: Optional[Scalar[Self.T.native]] = None,
    ):
        self._name = name^
        self._help = help^
        self._default = default^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return List[String]()

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        """Read this execution's value — the one leaf that reads `bindings`.

        **Here, rather than as a rewrite at `to_operator`.** Substituting at
        lowering would need every composite node to rebuild itself with
        resolved children, one method per node for a concern one node has.
        `bind` already walks the whole tree and already carries per-execution
        state, so this costs nothing that was not already being paid.
        """
        var got = bindings.get(self._name)
        if got:
            return got.value().as_primitive[Self.T]().value()
        if self._default:
            return self._default.value()
        raise Error(
            "parameter '",
            self._name,
            "' is not bound",
            (": " + self._help) if self._help else "",
        )

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](bound)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")
