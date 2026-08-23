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
cannot conflict with itself. What a plan's parameters *are* is answered by
walking the plan — `Value.params()`, a sibling to `columns()` — which makes
them a property of a plan rather than of a process.

The cell is an `ArcPointer[Optional[Scalar[...]]]` and not a struct: there is
nothing for one to hold beyond the value. `expr/`'s `ParamCell` raises
"parameter is not bound" *without naming it*, and its docstring explains that
the cell cannot know the name it is being read through. Here the node knows its
own name, so the error names it.
"""

from std.hashlib import Hasher
from std.memory import ArcPointer
from std.collections import Dict

from ..buffers import Bitmap
from ..dtypes import DynType, NumericType, StringLikeType
from ..scalars import DynScalar, PrimitiveScalar, StringScalar
from ..schema import Schema
from ..tabular import RecordBatch
from .logical import DynOperator, Shape, Value
from .`comptime`.core import NumericValue
from .physical import Datum, EvalOperator, Morsel, Operator


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


struct DynParam(Copyable, Equatable, Hashable, Movable, Writable):
    """A parameter's declaration, erased so a CLI can list parameters of
    different types together.

    Pure metadata: name, dtype, help, default. There is nothing to *bind*
    through it, because a value is no longer stored anywhere in the plan — a
    CLI reads these to render `--help` and to parse argv, then hands the parsed
    values to `execute` as `Bindings`.
    """

    var name: String
    var dtype: DynType
    var help: String
    var default: Optional[DynScalar]

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String,
        var default: Optional[DynScalar],
    ):
        self.name = name^
        self.dtype = dtype^
        self.help = help^
        self.default = default^

    def __eq__(self, other: Self) -> Bool:
        """A parameter's identity is its **name**.

        Not a shortcut: "one name is one parameter" is the semantics, so two
        declarations sharing a name are the same parameter however their help
        text differs. Saying so here is what lets `merged` be one generic
        function over anything with an identity, rather than one overload per
        key.
        """
        return self.name == other.name

    def __ne__(self, other: Self) -> Bool:
        return self.name != other.name

    def __hash__[H: Hasher](self, mut hasher: H):
        # Hashes the name and nothing else, so it agrees with `__eq__`.
        self.name.__hash__(hasher)

    def is_required(self) -> Bool:
        """A parameter with no default must be supplied at execution."""
        return not self.default

    def write_to[W: Writer](self, mut writer: W):
        writer.write("--", self.name, " <", self.dtype, ">")


struct Param[T: NumericType](NumericValue):
    """A late-bound numeric scalar — `Literal[T]` whose value arrives later.

    Immutable. It knows its name, dtype, help and default; the *value* is
    supplied to `to_operator` through `Bindings` and lives in the operator that
    results. `Shape.scalar`, so a predicate over a parameter costs one
    broadcast, exactly as a literal does.

    An unbound parameter with no default raises **naming itself** — the node is
    the parameter, so it can, where `expr/`'s cell explicitly cannot.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = Scalar[Self.T.native]
    """The resolved value. `bind` does not resolve it — `to_operator` did, and
    the operator carries it."""

    var _name: String
    var _help: String
    var _default: Optional[Scalar[Self.T.native]]

    def __init__(
        out self,
        var name: String,
        var help: String = String(),
        var default: Optional[Scalar[Self.T.native]] = None,
    ):
        self._name = name^
        self._help = help^
        self._default = default^

    def to_dyn_param(self) -> DynParam:
        var default = Optional[DynScalar](None)
        if self._default:
            default = PrimitiveScalar[Self.T](self._default.value()).to_dyn()
        return DynParam(
            self._name.copy(), DynType(Self.T()), self._help.copy(), default^
        )

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return List[String]()

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    def params(self) -> List[DynParam]:
        return [self.to_dyn_param()]

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: RecordBatch, bindings: Bindings) raises -> Self.Bound:
        """Resolve against this execution's values.

        **Here and not at `to_operator`.** `to_operator` copies a node without
        descending into it, so a parameter nested inside `a > min_a` would
        never be reached — resolving there needs every composite node to
        rebuild itself with resolved children, a second traversal of the whole
        tree. `bind` already walks it and already carries per-execution state,
        so this is where a value belongs.
        """
        var got = bindings.get(self._name)
        if got:
            return got.value().as_primitive[Self.T]().value()
        if self._default:
            return self._default.value()
        raise Error("parameter '", self._name, "' is not bound")

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](bound)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")
