"""The leaves of the comptime lane: a column reference and a constant.

A leaf is where a fused subtree touches the batch, and therefore where `bind`
does its work — every schema lookup and `Variant` unwrap happens here so the
lane loop above does none.
"""

from ...arrays import (
    BinaryLikeArray,
    BoolArray,
    ListLikeArray,
    PrimitiveArray,
)
from ...buffers import Bitmap
from ...dtypes import (
    BoolType,
    DynType,
    NumericType,
    ListLikeType,
    StringLikeType,
    TemporalType,
)
from ...scalars import PrimitiveScalar, StringScalar
from ...arrays import StructArray, Int32Array
from ...dtypes import Int32Type
from ...kernels.nested import ArrayLengthKernel
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape
from ..bindings import Bindings
from ...kernels.bounds import Bounds
from ..pruning import PruneStats, Truth, param_bounds
from ..physical import Datum
from .core import (
    BoolValue,
    ColumnBound,
    ListValue,
    NumericValue,
    StringValue,
    TemporalValue,
    Unnamed,
)


struct Column[T: NumericType](ColumnBound, NumericValue):
    """A numeric column, resolved by name once per batch."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return [self._name.copy()]

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        # The argument is ignored: this lane knows its type outright.
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        # A leaf returns its column as-is, validity included; the fused loop
        # above it decides what nulls mean.
        return batch.field(self._name).copy()

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        # `RecordBatch.column(name)` owns the missing-name diagnostic:
        # `get_field_index` answers -1, and indexing a column list with that
        # trips a bounds assert that aborts the process instead of naming the
        # column. Every leaf goes through it for that reason.
        return batch.field(self._name).as_primitive[Self.T]().copy()

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        """The typed unwrap of this column's statistic, guarded on dtype.

        `stats.bounds[T]` compares `DynType(T())` against the stored scalar's
        own type before unwrapping, because `as_primitive[T]` is a
        `debug_assert` that *aborts the process* on a mismatch rather than
        raising."""
        return stats.bounds[Self.T](self._name)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct TemporalColumn[T: TemporalType](ColumnBound, TemporalValue):
    """A date/time/timestamp/duration column, resolved by name once per batch.

    **Byte-for-byte the same lane as `Column[T]`** — temporal dtypes are
    fixed-width signed integers underneath, so `bind` and `lane[W]` are
    identical. It is a separate struct only because Mojo has no conditional
    conformance: one leaf cannot be a `NumericValue` when `T` is `int64` and a
    `TemporalValue` when `T` is `date32`, and the difference matters because
    `date + date` must not compile.

    That is the whole duplication, and the point of the split is that it stops
    at the leaf: everything above binds on `PrimitiveValue`, where the previous
    expression package
    needs `TemporalColumn` *plus* duplicated comparison arms.

    **Comparison works too, but through its own node.** `TemporalCompare` is
    separate from `NumericCompare` rather than a shared one, and the reason is
    named rather than hidden: `NumericCompare.ArgType` is
    `promote[L.Type, R.Type]`, and `promote` is bound on `NumericType` because
    it encodes *numeric* widening (signedness, int-to-float). Those rules do
    not generalise to temporal, and `wider[L.native, R.native]` is not a
    substitute: it picks by width and would silently change what `int32 <
    float32` compares in. Generalising `promote` is a decision about promotion
    semantics, not a bound to widen, so it is left to its own change.
    """

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return [self._name.copy()]

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        """Read from the schema, not built from `Self.T()`.

        The one place this leaf genuinely differs from `Column[T]`: a numeric
        dtype is `Defaultable` and can answer from its type, a temporal one
        cannot.

        `field(name=...)` rather than `fields[get_field_index(...)]`: the index
        form answers `-1` for an unknown column and `fields[-1]` is the *last*
        field, so a typo silently reported a neighbour's dtype.
        """
        return schema.field(name=self._name).dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._name).copy()

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_primitive[Self.T]().copy()

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct Literal[T: NumericType](NumericValue):
    """A numeric constant, splatted into every lane."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = NoneType
    """Nothing to resolve — the value is in the node, so the lane splats it."""

    var _value: Scalar[Self.Type.native]

    def __init__(out self, value: Scalar[Self.Type.native]):
        self._value = value

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return List[String]()

    def name(self) -> String:
        # SQL names `SELECT 1` as `1`; so does this.
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        # Stays a scalar. `Shape == 0` tells the caller so, and `Datum.to_array`
        # is the one place it stops being lazy — a predicate over a constant
        # never allocates a column.
        return PrimitiveScalar[Self.T](self._value).to_dyn()

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return NoneType()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # A constant is never null.
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](self._value)

    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        return Bounds[Self.Type.native].point(self._value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct BoolColumn(BoolValue, ColumnBound):
    """A boolean column, resolved by name once per batch.

    Separate from `Column[T]` because booleans are **bit-packed**: the `Bound`
    is a `BoolArray` and the lane loads through `values()`, the offset-applied
    `BitmapView`, rather than through a typed buffer. `Column[T]` is bound on
    `NumericType` and cannot take `BoolType` — the same reason `PrimitiveArray[bool_]`
    is not a thing anywhere in the tree.

    Without this leaf a fused expression could not read a `bool` column at all,
    so any three-valued-logic test would have to synthesise its operands from
    comparisons. the previous expression package shipped without it for exactly
    that reason and had to
    add it later.
    """

    comptime NativeType = DType.bool
    comptime shape = Shape.columnar
    comptime Bound = BoolArray

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return [self._name.copy()]

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(BoolType())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        # As with `Column[T]`: hand back the column rather than re-packing an
        # identical bitmap through the fused driver.
        return batch.field(self._name).copy()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_bool().copy()

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return bound.values().load[W](idx)

    def prune(self, stats: PruneStats, bindings: Bindings) -> Truth:
        """A bare bool column: `never` when all-null or when `max` is False."""
        return stats.bool_truth(self._name)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._name)


struct StringColumn[T: StringLikeType](ColumnBound, StringValue):
    """A string column, resolved by name once per batch.

    Parameterised on `StringLikeType` rather than fixed to `string`, so
    `large_string` is the same leaf with a different offset width rather than a
    second node type.
    """

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = BinaryLikeArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return [self._name.copy()]

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        # Hand back the column rather than copying every byte through a
        # builder — the whole reason the trait default is overridable.
        return batch.field(self._name).copy()

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_type[Self.Bound]().copy()

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # `unsafe_get` borrows from `bound.values`, a *field*; the trait
        # promises a borrow from `bound`. A field borrow is valid for at least
        # as long as the struct that holds it, so widening is sound -- Mojo
        # just will not do it implicitly.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct StringLiteral[T: StringLikeType](StringValue, Unnamed):
    """A constant string. Stays `Shape.scalar`, so it never materialises
    unless something asks it to."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = String
    """The value itself, so `lane` can borrow it.

    A `Bool` placeholder before, which forced `lane` to answer
    `self._value.copy()` -- one allocation **per row** for a constant. `bind`
    runs once per batch."""

    var _value: String

    def __init__(out self, var value: String):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return List[String]()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return Datum(StringScalar(self._value.copy()).to_dyn())

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        # One copy per batch, so every row of the loop borrows it.
        return self._value.copy()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        return StringSlice(bound)

    def write_to[W: Writer](self, mut writer: W):
        writer.write('"', self._value, '"')


struct ListColumn[T: ListLikeType](ColumnBound, ListValue):
    """A list column, resolved by name once per batch.

    Parameterised on `ListLikeType`, so `list`, `large_list` and `map` are the
    same leaf with a different offset width rather than three node types.

    It has no `lane` because `ListValue` has none — a list element is a whole
    sub-array. What reads it are nodes of other families: `ListLength` below is
    a `NumericValue` over this leaf's bound column.
    """

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = ListLikeArray[Self.T]
    """`ListValue` declares no `Bound` — a list is only ever read from a
    column, so naming it there would be a variable with one value. It is named
    *here* because `ColumnBound` needs something to narrow to `Array`, and this
    leaf is the one place the answer is fixed."""

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return [self._name.copy()]

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        # From the schema: a list dtype carries its child field, which cannot
        # be conjured from `Self.T()` any more than a timestamp's unit can.
        # `field(name=...)` raises on an unknown column; the index form answers
        # -1 and would report the last field's dtype instead.
        return schema.field(name=self._name).dtype.copy()

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._name).copy()

    # -- ListValue ----------------------------------------------------------

    def bind(
        self, batch: StructArray, bindings: Bindings
    ) raises -> ListLikeArray[Self.Type]:
        return batch.field(self._name).as_type[ListLikeArray[Self.T]]().copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct ListLength[A: ListValue](ColumnBound, NumericValue, Unnamed):
    """`array_length(list)` — a list consumed into a fixed-width column.

    The shape every list operation takes: it binds a `ListValue` and produces a
    lane of its own family. That is why `ListValue` needs no `lane` — nothing
    reads a list element as a value, only as something to measure or search.

    `bind` runs `ArrayLengthKernel` over the whole column and `lane` reads the
    result, as `CaseWhen` does: the work is offset arithmetic the kernel
    already vectorises, and a per-element lane would only re-derive it.
    """

    comptime Type = Int32Type
    comptime shape = Shape.columnar
    comptime Bound = Int32Array

    var a: Self.A

    def __init__(out self, var a: Self.A):
        self.a = a^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return self.a.columns()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Int32Type())

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return ArrayLengthKernel.apply(self.a.bind(batch, bindings))

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("array_length(", self.a, ")")


# ---------------------------------------------------------------------------
# Param -- a literal whose value arrives at execution time
# ---------------------------------------------------------------------------
# A parameter is **a literal whose value arrives later**: it has a dtype and a
# shape when the plan is built, and a value only once something binds it. That is
# why `Param` mirrors `Literal` — same families, same `Shape.scalar`, same
# per-family split — rather than being a category of its own.
#
# **A parameter is a description; its value belongs to an execution.** The node
# holds a name, a dtype, help text and an optional default, and nothing else — no
# cell, no mutable state. Values arrive through `Bindings` when the plan is
# turned into operators:
#
#     var min_a = param("min-a", int64)
#     var plan = t.filter(col("a", int64) > min_a)
#
#     plan.execute(bindings=Bindings().set("min-a", Int64Scalar(4).to_dyn()))
#
# That is the layer's own rule — *a logical node is stateless* — applied here.
# An earlier version of this module held the value in an `ArcPointer` cell shared
# by every copy of the node, so `min_a.set(4)` reached into a built plan and
# changed what it computed. It made a plan's result depend on hidden mutable
# state, and it made executing one plan on two threads with two values a data
# race: the same defect the previous expression package's process-global
# registry has, relocated into the
# node rather than removed.
#
# Passing the values *through* the execution instead means the plan stays
# immutable, two executions with different values cannot interfere, and there is
# no cell.
#
# That one property removes an entire subsystem. the previous expression
# package declares parameters
# *inline* at each use — `col("a") > param("min-a", int64)` written twice must
# still share a cell — so it needs a process-global registry keyed by name, a
# second lookup table for the runtime lane, dedup on every declaration, and a
# dtype-conflict check. It also inherits two limitations it records honestly: a
# plan built but never executed leaks its declarations into the next plan's
# `--help`, and the globals are unsynchronised, so building two plans on two
# threads is a data race.
#
# None of that exists here. Sharing is structural rather than name-keyed, so
# there is no registry to leak and no global to race on, and one declaration
# cannot conflict with itself.
#
# There is deliberately **no `params()` traversal**. Asking a plan which
# parameters it takes is a sixteen-method walk that only a `--help` surface
# would use, and nothing outside a test ever asked. Add it back when something
# does; until then a plan's parameters are discovered the way its columns are —
# by binding it and being told, by name, which one is missing. the previous
# expression package's
# `ParamCell` raises "parameter is not bound" *without* naming it, because a
# cell cannot know the name it is read through. Here the node **is** the
# parameter, so it can.


struct Param[T: NumericType](NumericValue):
    """A late-bound numeric scalar — `Literal[T]` whose value arrives later.

    Immutable. It knows its name, dtype, help and default; the *value* arrives
    through `Bindings`, which the operator carries and hands back down to
    `bind`. `Shape.scalar`, so a predicate over a parameter costs one
    broadcast, exactly as a literal does.

    An unbound parameter with no default raises **naming itself** — the node is
    the parameter, so it can, where the previous expression package's cell
    explicitly cannot.
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

    def bounds(
        self, stats: PruneStats, bindings: Bindings
    ) -> Bounds[Self.Type.native]:
        """A bound parameter prunes exactly as well as a literal, because
        pruning runs at *execution* time with the same `Bindings` `bind` will
        see. the previous expression package needed a process-global registry
        for this and still
        regressed a parameterised date filter to reading every row group.

        Unbound and undefaulted answers unknown and does not raise: the scan
        reads everything and `bind` then raises naming the parameter. Pruning
        degrades; binding raises.
        """
        return param_bounds[Self.T](bindings, self._name, self._default)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")
