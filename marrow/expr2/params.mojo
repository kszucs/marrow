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

Resolving at `to_operator` time instead means the plan stays immutable, two
executions with different values cannot interfere, and there is no cell.

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
does; until then a plan's parameters are discovered the way its columns are
resolved — by binding it and being told, by name, which one is missing.
`expr/`'s `ParamCell` raises "parameter is not bound" *without* naming it,
because a cell cannot know the name it is read through. Here the node **is**
the parameter, so it can.
"""

from std.collections import Dict

from ..buffers import Bitmap
from ..dtypes import DynType, NumericType
from ..scalars import DynScalar
from ..schema import Schema
from ..tabular import RecordBatch
from .logical import Shape
from .`comptime`.core import NumericValue


struct Bindings(Copyable, Movable):
    """Parameter values for one execution.

    Passed to `to_operator`, not stored on the plan, which is what keeps a plan
    immutable and lets two executions use different values without
    interfering.

    Missing names are not an error here — a parameter with a default is
    satisfied without one, and `Param` raises naming itself when it has
    neither.
    """

    var _values: Dict[String, DynScalar]

    def __init__(out self):
        self._values = Dict[String, DynScalar]()

    def set(var self, var name: String, var value: DynScalar) -> Self:
        """Bind one name. Returns self so bindings chain."""
        self._values[name^] = value^
        return self^

    def get(self, name: String) raises -> Optional[DynScalar]:
        if name in self._values:
            return Optional(self._values[name].copy())
        return None


struct Param[T: NumericType](NumericValue):
    """A late-bound numeric scalar — `Literal[T]` whose value arrives later.

    Immutable. It knows its name, dtype, help and default; the *value* arrives
    through `Bindings` at `to_operator`, and `resolve` returns a new `Param`
    holding it. `Shape.scalar`, so a predicate over a parameter costs one
    broadcast, exactly as a literal does.

    An unbound parameter with no default raises **naming itself** — the node is
    the parameter, so it can, where `expr/`'s cell explicitly cannot.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = Scalar[Self.T.native]
    """The value, already substituted. `bind` does not look anything up —
    `resolve` did, at lowering."""

    var _name: String
    var _help: String
    var _value: Optional[Scalar[Self.T.native]]
    """The value to use: the declared default until `resolve` replaces it with
    this execution's binding. Empty means neither exists, which `resolve`
    reports as an error naming this parameter."""

    def __init__(
        out self,
        var name: String,
        var help: String = String(),
        var default: Optional[Scalar[Self.T.native]] = None,
    ):
        self._name = name^
        self._help = help^
        self._value = default^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return List[String]()

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    def resolve(self, bindings: Bindings) raises -> Self:
        """Substitute this execution's value — the one node that does.

        Called once, from `to_operator`, so a parameter is gone before the
        first batch arrives and an unbound one fails at lowering rather than
        partway through a stream. Every composite node above this one rebuilds
        with resolved children, which is how a parameter nested inside
        `a > min_a` is reached at all; the leaf that is not a parameter takes
        `Value.resolve`'s default and costs nothing.
        """
        var got = bindings.get(self._name)
        if got:
            return Param[Self.T](
                self._name.copy(),
                self._help.copy(),
                Optional(got.value().as_primitive[Self.T]().value()),
            )
        if self._value:
            return self.copy()
        raise Error(
            "parameter '",
            self._name,
            "' is not bound",
            (": " + self._help) if self._help else "",
        )

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        # `resolve` filled this or raised; the guard is for a `Param` reached
        # without going through `to_operator`, which nothing does.
        if self._value:
            return self._value.value()
        raise Error("parameter '", self._name, "' was not resolved")

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](bound)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")
