"""The leaves of the comptime lane: a column reference and a constant.

A leaf is where a fused subtree touches the batch, and therefore where `bind`
does its work — every schema lookup and `Variant` unwrap happens here so the
lane loop above does none.
"""

from ...arrays import (
    BinaryLikeArray,
    BoolArray,
    DictionaryArray,
    FixedSizeBinaryArray,
    FixedSizeListArray,
    ListLikeArray,
    NullArray,
    PrimitiveArray,
    StructArray,
)
from ...builders import BinaryLikeBuilder
from ...buffers import Bitmap
from ...dtypes import (
    BinaryLikeType,
    BoolType,
    DecimalType,
    DynType,
    IntervalType,
    ListLikeType,
    NumericType,
    PrimitiveType,
    StringLikeType,
    TemporalType,
    bool_,
)
from ...scalars import (
    ArrowScalar,
    BinaryLikeScalar,
    BoolScalar,
    DictionaryScalar,
    DynScalar,
    FixedSizeBinaryScalar,
    ListScalar,
    NullScalar,
    PrimitiveScalar,
    StructScalar,
)
from ...schema import Schema
from ..logical import References, Shape
from ..bindings import (
    Bindings,
    ParamSpec,
    bool_from_text,
    numeric_from_text,
    string_from_text,
)
from ..index import Index
from ..physical import Datum
from .core import (
    BinaryValue,
    BoolValue,
    ColumnBound,
    DecimalValue,
    DictionaryValue,
    FixedSizeBinaryValue,
    FixedSizeListValue,
    IntervalValue,
    ListValue,
    NullValue,
    NumericValue,
    StringValue,
    StructValue,
    TemporalValue,
    Unnamed,
)


struct NumericColumn[T: NumericType](ColumnBound, NumericValue):
    """A numeric column, resolved by name once per batch."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

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

    def defined(self, index: Index) raises -> BoolArray:
        """Where this column holds at least one non-null value."""
        return index.defined(self._name)

    def statistics[
        Stat: NumericType
    ](
        self, index: Index, bindings: Bindings, upper: Bool
    ) raises -> PrimitiveArray[Stat]:
        """This column's per-chunk extremes, straight off the zone map.

        Nothing is erased on this path and nothing is checked here: `Stat` is
        what the reader asked for, and the index answers null for any chunk
        whose recorded statistic is some other type. A column the plan declares
        as `int64` in a file that wrote `int32` therefore prunes nothing rather
        than reading a scalar at the wrong type -- which is not a raise but a
        process abort, since `as_primitive` on the wrong type is unchecked in a
        release build.
        """
        if upper:
            return index.maxes[Stat](self._name, Stat())
        return index.mins[Stat](self._name, Stat())

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct TemporalColumn[T: TemporalType](ColumnBound, TemporalValue):
    """A date/time/timestamp/duration column, resolved by name once per batch.

    **Byte-for-byte the same lane as `NumericColumn[T]`** — temporal dtypes are
    fixed-width signed integers underneath, so `bind` and `lane[W]` are
    identical. It is a separate struct because one leaf cannot be a
    `NumericValue` when `T` is `int64` and a `TemporalValue` when `T` is
    `date32` — conditional conformance exists, but it cannot satisfy the
    families' narrowed `Type` or their `Bound` (see CLAUDE.md) — and the
    difference matters because `date + date` must not compile.

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

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        """Read from the schema, not built from `Self.T()`.

        The one place this leaf genuinely differs from `NumericColumn[T]`: a numeric
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


struct NumericLiteral[T: NumericType](NumericValue):
    """A numeric constant, splatted into every lane."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = NoneType
    """Nothing to resolve — the value is in the node, so the lane splats it."""

    var _value: Scalar[Self.Type.native]

    def __init__(out self, value: Scalar[Self.Type.native]):
        self._value = value

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

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
        return PrimitiveScalar[Self.T](self._value)

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

    def statistics[
        Stat: NumericType
    ](
        self, index: Index, bindings: Bindings, upper: Bool
    ) raises -> PrimitiveArray[Stat]:
        """A literal is the same value in every chunk, so both extremes are
        it — converted to `Stat` by a builtin SIMD cast, which is what makes a
        promoting comparison prune without linking `kernels::cast`."""
        return PrimitiveScalar[Stat](Scalar[Stat.native](self._value)).repeat(
            index.chunks
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct TemporalLiteral[T: TemporalType](TemporalValue, Unnamed):
    """A date/time/timestamp/duration constant, splatted into every lane.

    **The leaf the comptime lane was missing, and pruning is why it exists.**
    `TemporalCompare` compares two temporal operands, and until now the only
    temporal operand was a column — so `date_col > date_const`, the shape a
    zone map is most useful for, could not be written in this lane at all and
    every temporal predicate kept every chunk.

    It carries its dtype as well as its value, for the same reason
    `TemporalColumn` does: a unit is a value, so `Self.T` cannot build one.
    That is also what makes `lit(19000, date32())` read naturally — the dtype
    argument is already there, exactly as the numeric overload has it.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = NoneType
    """Nothing to resolve — the value is in the node, so the lane splats it."""

    var _value: Scalar[Self.Type.native]
    var _dtype: Self.T

    def __init__(out self, value: Scalar[Self.Type.native], dtype: Self.T):
        self._value = value
        self._dtype = dtype.copy()

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return PrimitiveScalar[Self.T](Optional(self._value), self._dtype)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return NoneType()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](self._value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct BoolColumn(BoolValue, ColumnBound):
    """A boolean column, resolved by name once per batch.

    Separate from `NumericColumn[T]` because booleans are **bit-packed**: the `Bound`
    is a `BoolArray` and the lane loads through `values()`, the offset-applied
    `BitmapView`, rather than through a typed buffer. `NumericColumn[T]` is bound on
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

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(BoolType())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        # As with `NumericColumn[T]`: hand back the column rather than re-packing an
        # identical bitmap through the fused driver.
        return batch.field(self._name).copy()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_bool().copy()

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return bound.values().load[W](idx)

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

    def references(self, mut into: References):
        into.column(self._name)

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

    def references(self, mut into: References):
        pass

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return BinaryLikeScalar[Self.T](self._value)

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


struct BoolLiteral(BoolValue):
    """A boolean constant, splatted into every lane."""

    comptime NativeType = DType.bool
    comptime shape = Shape.scalar
    comptime Bound = NoneType

    var _value: Bool

    def __init__(out self, value: Bool):
        self._value = value

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        # Stays a scalar; `BoolValue.evaluate` would pack a whole bitmap.
        return BoolScalar(self._value)

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return NoneType()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return SIMD[DType.bool, W](self._value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct DecimalColumn[T: DecimalType](ColumnBound, DecimalValue):
    """A decimal column, resolved by name once per batch.

    The lane of `TemporalColumn[T]`, for the same reason: a decimal is a
    fixed-width integer underneath, and its precision and scale are on the
    dtype instance, so `dtype` reads the schema rather than `Self.T()`.
    """

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
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


struct DecimalLiteral[T: DecimalType](DecimalValue):
    """A decimal constant: its unscaled integer and the dtype that scales it.

    Carries its dtype for the reason `TemporalLiteral` does — precision and
    scale are values, so `Self.T` cannot build one.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = NoneType

    var _value: Scalar[Self.Type.native]
    var _dtype: Self.T

    def __init__(out self, value: Scalar[Self.Type.native], dtype: Self.T):
        self._value = value
        self._dtype = dtype.copy()

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return PrimitiveScalar[Self.T](Optional(self._value), self._dtype)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return NoneType()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](self._value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct IntervalColumn[T: IntervalType](ColumnBound, IntervalValue):
    """An interval column, resolved by name once per batch.

    An interval dtype has no parameters, so unlike the decimal and temporal
    leaves this one answers `dtype` from `Self.T()`.
    """

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

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


struct IntervalLiteral[T: IntervalType](IntervalValue):
    """An interval constant, in its storage encoding."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = NoneType

    var _value: Scalar[Self.Type.native]

    def __init__(out self, value: Scalar[Self.Type.native]):
        self._value = value

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return PrimitiveScalar[Self.T](self._value)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return NoneType()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](self._value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


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

    def references(self, mut into: References):
        into.column(self._name)

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


# ---------------------------------------------------------------------------
# Param -- a literal whose value arrives at execution time
# ---------------------------------------------------------------------------
# A parameter is **a literal whose value arrives later**: it has a dtype and a
# shape when the plan is built, and a value only once something binds it. That is
# why `NumericParam` mirrors `NumericLiteral` — same families, same `Shape.scalar`, same
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
# None of that exists here. There is no registry to leak and no global to race
# on: a plan's parameters are found by walking the plan. Each parameter leaf
# declares a `ParamSpec` from `references` — the one walk that also answers
# `columns()` — and `DynRelation.params()` keeps one per name. A name read twice
# is one parameter, because `Bindings` is keyed by name; a read that disagrees
# about its dtype is refused when the value binds, by `_bound` below.
#
# An unbound parameter raises **naming itself**. The previous expression
# package's `ParamCell` could not, because a cell does not know the name it is
# read through; here the node is the parameter.
#
# Every family has one, and all of them read their value through `_bound`,
# which is where a binding is checked against the declared dtype. The typed
# downcast after it cannot check: `as_primitive[T]` on a scalar of another
# dtype is not a raise but a process abort in a release build.


def _shown[S: Writable](default: Optional[S]) -> Optional[String]:
    """A parameter's default as `--help` shows it, or `None` when required."""
    if default:
        return String(default.value())
    else:
        return None


def _shown_bool(default: Optional[BoolScalar]) -> Optional[String]:
    """A `bool` default as a command line spells it — `true` or `false`, the
    spellings `parse_bool` reads back."""
    if default:
        return String("true") if default.value().value() else String("false")
    else:
        return None


def _bound[
    S: ArrowScalar
](bindings: Bindings, name: String, dtype: DynType) raises -> Optional[S]:
    """This execution's value for the parameter `name`, or `None` when nothing
    binds it — refused unless it is a non-null `S` of `dtype`.

    Checked on the typed member, not through `DynScalar.type()`, which would
    link a walk over every scalar arm into a binary that names one. A null is
    refused, except for the `null` dtype whose only value it is: a fixed-width
    parameter's `Bound` is one value with no validity to carry it.
    """
    var got = bindings.get(name)
    if got:
        if not got.value().isa[S]():
            raise Error(
                "parameter '",
                name,
                "' is ",
                dtype,
                " but was bound to a scalar of another kind",
            )
        ref typed = got.value().as_type[S]()
        var actual = typed.type()
        if actual != dtype:
            raise Error(
                "parameter '",
                name,
                "' is ",
                dtype,
                " but was bound to ",
                actual,
            )
        if typed.is_null() and not dtype.is_null():
            raise Error("parameter '", name, "' was bound to null")
        return typed.copy()
    else:
        return None


def _unbound(name: String, help: String) -> Error:
    """The diagnostic for a parameter with no binding and no default."""
    return Error(
        "parameter '", name, "' is not bound", (": " + help) if help else ""
    )


def _param[
    S: ArrowScalar
](
    bindings: Bindings,
    name: String,
    help: String,
    dtype: DynType,
    default: Optional[S],
) raises -> S:
    """A parameter's value for this execution, or its default — which `param`
    checked against `dtype` when it built the node."""
    var got = _bound[S](bindings, name, dtype)
    if got:
        return got.value().copy()
    elif default:
        return default.value().copy()
    else:
        raise _unbound(name, help)


def _primitive_param[
    T: PrimitiveType
](
    bindings: Bindings,
    name: String,
    help: String,
    dtype: T,
    default: Optional[Scalar[T.native]],
) raises -> Scalar[T.native]:
    """A fixed-width parameter's value for this execution, or its default."""
    var got = _bound[PrimitiveScalar[T]](bindings, name, DynType(dtype))
    if got:
        return got.value().value()
    elif default:
        return default.value()
    else:
        raise _unbound(name, help)


struct NumericParam[T: NumericType](NumericValue):
    """A late-bound numeric scalar — `NumericLiteral[T]` whose value arrives later.

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

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                DynType(Self.T()),
                self._help.copy(),
                _shown(self._default),
                numeric_from_text[Self.T],
            )
        )

    def name(self) -> String:
        return self._name.copy()

    # -- PrimitiveValue -----------------------------------------------------

    def _bound(self, bindings: Bindings) raises -> Self.Bound:
        """This execution's value, or the default; raises **naming itself**
        when there is neither.

        Split out because two callers need it: `bind`, once per batch, and
        `statistics`, once per scan when this parameter sits in a predicate the
        source is pruning with.
        """
        return _primitive_param[Self.T](
            bindings, self._name, self._help, Self.T(), self._default
        )

    def statistics[
        Stat: NumericType
    ](
        self, index: Index, bindings: Bindings, upper: Bool
    ) raises -> PrimitiveArray[Stat]:
        """One value for the whole execution, so both extremes are it — the
        answer `NumericLiteral` gives, read from `bindings` rather than off the node.

        This is what makes the AOT lane prune at all: its plans are written
        `col("amount", int64) >= param("min-amount")`, and a parameter that
        said nothing would leave every such query reading the whole file.
        """
        return PrimitiveScalar[Stat](
            Scalar[Stat.native](self._bound(bindings))
        ).repeat(index.chunks)

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        """Read this execution's value.

        **Here, rather than as a rewrite at `to_operator`.** Substituting at
        lowering would need every composite node to rebuild itself with
        resolved children, one method per node for a concern one node has.
        `bind` already walks the whole tree and already carries per-execution
        state, so this costs nothing that was not already being paid.
        """
        return self._bound(bindings)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](bound)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


struct TemporalParam[T: TemporalType](TemporalValue):
    """A late-bound date/time/timestamp/duration — `TemporalLiteral[T]` whose
    value arrives later.

    Carries its dtype, as the literal does: a unit and a timezone are values,
    and a binding is checked against them, so a `timestamp[ms]` parameter
    refuses a `timestamp[s]` scalar rather than reading its ticks at the wrong
    unit.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = Scalar[Self.T.native]

    var _name: String
    var _dtype: Self.T
    var _help: String
    var _default: Optional[Scalar[Self.T.native]]

    def __init__(
        out self,
        var name: String,
        dtype: Self.T,
        var help: String = String(),
        var default: Optional[Scalar[Self.T.native]] = None,
    ):
        self._name = name^
        self._dtype = dtype.copy()
        self._help = help^
        self._default = default^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                DynType(self._dtype),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _primitive_param[Self.T](
            bindings, self._name, self._help, self._dtype, self._default
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


struct DecimalParam[T: DecimalType](DecimalValue):
    """A late-bound decimal — `DecimalLiteral[T]` whose value arrives later.

    A binding must match the declared precision and scale, since the unscaled
    integer means nothing without them.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = Scalar[Self.T.native]

    var _name: String
    var _dtype: Self.T
    var _help: String
    var _default: Optional[Scalar[Self.T.native]]

    def __init__(
        out self,
        var name: String,
        dtype: Self.T,
        var help: String = String(),
        var default: Optional[Scalar[Self.T.native]] = None,
    ):
        self._name = name^
        self._dtype = dtype.copy()
        self._help = help^
        self._default = default^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                DynType(self._dtype),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _primitive_param[Self.T](
            bindings, self._name, self._help, self._dtype, self._default
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


struct IntervalParam[T: IntervalType](IntervalValue):
    """A late-bound interval — `IntervalLiteral[T]` whose value arrives
    later."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = Scalar[Self.T.native]

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

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                DynType(Self.T()),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _primitive_param[Self.T](
            bindings, self._name, self._help, Self.T(), self._default
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


struct BoolParam(BoolValue):
    """A late-bound boolean — `BoolLiteral` whose value arrives later."""

    comptime NativeType = DType.bool
    comptime shape = Shape.scalar
    comptime Bound = Bool

    var _name: String
    var _help: String
    var _default: Optional[BoolScalar]

    def __init__(
        out self,
        var name: String,
        var help: String = String(),
        var default: Optional[BoolScalar] = None,
    ):
        self._name = name^
        self._help = help^
        self._default = default^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                DynType(bool_),
                self._help.copy(),
                _shown_bool(self._default),
                bool_from_text,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        # Stays a scalar; `BoolValue.evaluate` would pack a whole bitmap.
        return _param[BoolScalar](
            bindings, self._name, self._help, bool_, self._default
        )

    # -- BoolValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _param[BoolScalar](
            bindings, self._name, self._help, bool_, self._default
        ).value()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return SIMD[DType.bool, W](bound)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


struct StringParam[T: StringLikeType](StringValue):
    """A late-bound string — `StringLiteral[T]` whose value arrives later."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = String
    """The value itself, so `lane` borrows it — see `StringLiteral.Bound`."""

    var _name: String
    var _help: String
    var _default: Optional[BinaryLikeScalar[Self.T]]

    def __init__(
        out self,
        var name: String,
        var help: String = String(),
        var default: Optional[BinaryLikeScalar[Self.T]] = None,
    ):
        self._name = name^
        self._help = help^
        self._default = default^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                DynType(Self.T()),
                self._help.copy(),
                _shown(self._default),
                string_from_text[Self.T],
            )
        )

    def name(self) -> String:
        return self._name.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return _param[BinaryLikeScalar[Self.T]](
            bindings, self._name, self._help, DynType(Self.T()), self._default
        )

    def value(self, bindings: Bindings) raises -> String:
        """This execution's string, or the default; raises naming itself when
        there is neither. What a scan reads to resolve a parameter path."""
        return _param[BinaryLikeScalar[Self.T]](
            bindings, self._name, self._help, DynType(Self.T()), self._default
        ).value()

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self.value(bindings)

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        return StringSlice(bound)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


struct FixedSizeBinaryColumn(ColumnBound, FixedSizeBinaryValue):
    """A fixed-size binary column, resolved by name once per batch."""

    comptime shape = Shape.columnar
    comptime Bound = FixedSizeBinaryArray

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return schema.field(name=self._name).dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._name).copy()

    # -- FixedSizeBinaryValue -----------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_fixed_size_binary().copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct FixedSizeBinaryLiteral(ColumnBound, FixedSizeBinaryValue):
    """A fixed-size binary constant, held as a scalar and broadcast when bound.
    """

    comptime shape = Shape.scalar
    comptime Bound = FixedSizeBinaryArray

    var _value: FixedSizeBinaryScalar

    def __init__(out self, var value: FixedSizeBinaryScalar):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._value.type()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._value.copy()

    # -- FixedSizeBinaryValue -----------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._value.repeat(len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct FixedSizeBinaryParam(ColumnBound, FixedSizeBinaryValue):
    """A late-bound fixed-size binary — `FixedSizeBinaryLiteral` whose value arrives later.
    """

    comptime shape = Shape.scalar
    comptime Bound = FixedSizeBinaryArray

    var _name: String
    var _dtype: DynType
    var _help: String
    var _default: Optional[FixedSizeBinaryScalar]

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String = String(),
        var default: Optional[FixedSizeBinaryScalar] = None,
    ):
        self._name = name^
        self._dtype = dtype^
        self._help = help^
        self._default = default^

    def _scalar(self, bindings: Bindings) raises -> FixedSizeBinaryScalar:
        return _param[FixedSizeBinaryScalar](
            bindings, self._name, self._help, self._dtype, self._default
        )

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                self._dtype.copy(),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._scalar(bindings)

    # -- FixedSizeBinaryValue -----------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._scalar(bindings).repeat(len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


struct FixedSizeListColumn(ColumnBound, FixedSizeListValue):
    """A fixed-size list column, resolved by name once per batch."""

    comptime shape = Shape.columnar
    comptime Bound = FixedSizeListArray

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return schema.field(name=self._name).dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._name).copy()

    # -- FixedSizeListValue -------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_fixed_size_list().copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct FixedSizeListLiteral(ColumnBound, FixedSizeListValue):
    """A fixed-size list constant, held as a scalar and broadcast when bound."""

    comptime shape = Shape.scalar
    comptime Bound = FixedSizeListArray

    var _value: ListScalar

    def __init__(out self, var value: ListScalar):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._value.type()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._value.copy()

    # -- FixedSizeListValue -------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._value.repeat(len(batch)).as_fixed_size_list().copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct FixedSizeListParam(ColumnBound, FixedSizeListValue):
    """A late-bound fixed-size list — `FixedSizeListLiteral` whose value arrives later.
    """

    comptime shape = Shape.scalar
    comptime Bound = FixedSizeListArray

    var _name: String
    var _dtype: DynType
    var _help: String
    var _default: Optional[ListScalar]

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String = String(),
        var default: Optional[ListScalar] = None,
    ):
        self._name = name^
        self._dtype = dtype^
        self._help = help^
        self._default = default^

    def _scalar(self, bindings: Bindings) raises -> ListScalar:
        return _param[ListScalar](
            bindings, self._name, self._help, self._dtype, self._default
        )

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                self._dtype.copy(),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._scalar(bindings)

    # -- FixedSizeListValue -------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (
            self._scalar(bindings)
            .repeat(len(batch))
            .as_fixed_size_list()
            .copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


struct StructColumn(ColumnBound, StructValue):
    """A struct column, resolved by name once per batch."""

    comptime shape = Shape.columnar
    comptime Bound = StructArray

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return schema.field(name=self._name).dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._name).copy()

    # -- StructValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_struct().copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct StructLiteral(ColumnBound, StructValue):
    """A struct constant, held as a scalar and broadcast when bound."""

    comptime shape = Shape.scalar
    comptime Bound = StructArray

    var _value: StructScalar

    def __init__(out self, var value: StructScalar):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._value.type()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._value.copy()

    # -- StructValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._value.repeat(len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct StructParam(ColumnBound, StructValue):
    """A late-bound struct — `StructLiteral` whose value arrives later."""

    comptime shape = Shape.scalar
    comptime Bound = StructArray

    var _name: String
    var _dtype: DynType
    var _help: String
    var _default: Optional[StructScalar]

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String = String(),
        var default: Optional[StructScalar] = None,
    ):
        self._name = name^
        self._dtype = dtype^
        self._help = help^
        self._default = default^

    def _scalar(self, bindings: Bindings) raises -> StructScalar:
        return _param[StructScalar](
            bindings, self._name, self._help, self._dtype, self._default
        )

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                self._dtype.copy(),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._scalar(bindings)

    # -- StructValue --------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._scalar(bindings).repeat(len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


struct DictionaryColumn(ColumnBound, DictionaryValue):
    """A dictionary-encoded column, resolved by name once per batch."""

    comptime shape = Shape.columnar
    comptime Bound = DictionaryArray

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return schema.field(name=self._name).dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._name).copy()

    # -- DictionaryValue ----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_dictionary().copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct DictionaryLiteral(ColumnBound, DictionaryValue):
    """A dictionary-encoded constant, held as a scalar and broadcast when bound.
    """

    comptime shape = Shape.scalar
    comptime Bound = DictionaryArray

    var _value: DictionaryScalar

    def __init__(out self, var value: DictionaryScalar):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._value.type()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._value.copy()

    # -- DictionaryValue ----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._value.repeat(len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct DictionaryParam(ColumnBound, DictionaryValue):
    """A late-bound dictionary-encoded — `DictionaryLiteral` whose value arrives later.
    """

    comptime shape = Shape.scalar
    comptime Bound = DictionaryArray

    var _name: String
    var _dtype: DynType
    var _help: String
    var _default: Optional[DictionaryScalar]

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String = String(),
        var default: Optional[DictionaryScalar] = None,
    ):
        self._name = name^
        self._dtype = dtype^
        self._help = help^
        self._default = default^

    def _scalar(self, bindings: Bindings) raises -> DictionaryScalar:
        return _param[DictionaryScalar](
            bindings, self._name, self._help, self._dtype, self._default
        )

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                self._dtype.copy(),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._scalar(bindings)

    # -- DictionaryValue ----------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._scalar(bindings).repeat(len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


struct NullColumn(ColumnBound, NullValue):
    """A null-typed column, resolved by name once per batch."""

    comptime shape = Shape.columnar
    comptime Bound = NullArray

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return schema.field(name=self._name).dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._name).copy()

    # -- NullValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return batch.field(self._name).as_null().copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct NullLiteral(ColumnBound, NullValue):
    """A null-typed constant, held as a scalar and broadcast when bound."""

    comptime shape = Shape.scalar
    comptime Bound = NullArray

    var _value: NullScalar

    def __init__(out self, var value: NullScalar):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._value.type()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._value.copy()

    # -- NullValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._value.repeat(len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct NullParam(ColumnBound, NullValue):
    """A late-bound null-typed — `NullLiteral` whose value arrives later."""

    comptime shape = Shape.scalar
    comptime Bound = NullArray

    var _name: String
    var _dtype: DynType
    var _help: String
    var _default: Optional[NullScalar]

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String = String(),
        var default: Optional[NullScalar] = None,
    ):
        self._name = name^
        self._dtype = dtype^
        self._help = help^
        self._default = default^

    def _scalar(self, bindings: Bindings) raises -> NullScalar:
        return _param[NullScalar](
            bindings, self._name, self._help, self._dtype, self._default
        )

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                self._dtype.copy(),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._scalar(bindings)

    # -- NullValue ----------------------------------------------------------

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return self._scalar(bindings).repeat(len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


struct ListLiteral[T: ListLikeType](ColumnBound, ListValue):
    """A list constant, held as a scalar and broadcast when bound.

    `lit` checks the scalar's dtype against `T` when it builds one, which is
    what makes the unchecked `as_type` in `bind` sound.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = ListLikeArray[Self.T]

    var _value: ListScalar

    def __init__(out self, var value: ListScalar):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def name(self) -> String:
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return self._value.type()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._value.copy()

    # -- ListValue ----------------------------------------------------------

    def bind(
        self, batch: StructArray, bindings: Bindings
    ) raises -> ListLikeArray[Self.Type]:
        return (
            self._value.repeat(len(batch))
            .as_type[ListLikeArray[Self.T]]()
            .copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct ListParam[T: ListLikeType](ColumnBound, ListValue):
    """A late-bound list — `ListLiteral[T]` whose value arrives later."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = ListLikeArray[Self.T]

    var _name: String
    var _dtype: DynType
    var _help: String
    var _default: Optional[ListScalar]

    def __init__(
        out self,
        var name: String,
        var dtype: DynType,
        var help: String = String(),
        var default: Optional[ListScalar] = None,
    ):
        self._name = name^
        self._dtype = dtype^
        self._help = help^
        self._default = default^

    def _scalar(self, bindings: Bindings) raises -> ListScalar:
        return _param[ListScalar](
            bindings, self._name, self._help, self._dtype, self._default
        )

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                self._dtype.copy(),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return self._dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self._scalar(bindings)

    # -- ListValue ----------------------------------------------------------

    def bind(
        self, batch: StructArray, bindings: Bindings
    ) raises -> ListLikeArray[Self.Type]:
        return (
            self._scalar(bindings)
            .repeat(len(batch))
            .as_type[ListLikeArray[Self.T]]()
            .copy()
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")


def _repeat_bytes[
    T: BinaryLikeType
](value: Span[UInt8, _], times: Int) raises -> BinaryLikeArray[T]:
    """`times` copies of `value`, which need not be UTF-8.

    A binary leaf holds `List[UInt8]` rather than a `BinaryScalar`, whose
    `String` would promise UTF-8 the bytes do not keep, so it broadcasts them
    itself."""
    var builder = BinaryLikeBuilder[T](times, len(value) * times)
    for _ in range(times):
        builder.append(StringSlice(unsafe_from_utf8=value))
    return builder.finish()


struct BinaryColumn[T: BinaryLikeType](BinaryValue, ColumnBound):
    """A `binary` or `large_binary` column, resolved by name once per batch."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = BinaryLikeArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.column(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(self._name).copy()

    # -- BinaryValue --------------------------------------------------------

    def bind(
        self, batch: StructArray, bindings: Bindings
    ) raises -> BinaryLikeArray[Self.Type]:
        return batch.field(self._name).as_type[BinaryLikeArray[Self.T]]().copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct BinaryLiteral[T: BinaryLikeType](BinaryValue, ColumnBound, Unnamed):
    """A binary constant.

    **Evaluates to a column, not a scalar**, though its `shape` is
    `Shape.scalar`: the bytes need not be UTF-8, so they are not handed to a
    `BinaryScalar`, and `Datum.to_array` accepts a column of the batch's
    length either way.
    """

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = BinaryLikeArray[Self.T]

    var _value: List[UInt8]

    def __init__(out self, var value: List[UInt8]):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        pass

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self.bind(batch, bindings).to_dyn()

    # -- BinaryValue --------------------------------------------------------

    def bind(
        self, batch: StructArray, bindings: Bindings
    ) raises -> BinaryLikeArray[Self.Type]:
        return _repeat_bytes[Self.T](Span(self._value), len(batch))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(<", len(self._value), " bytes>)")


struct BinaryParam[T: BinaryLikeType](BinaryValue, ColumnBound):
    """A late-bound binary value — `BinaryLiteral[T]` whose value arrives
    later, bound as a `BinaryScalar` (or `LargeBinaryScalar`). Evaluates to a column
    for the reason the literal does."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = BinaryLikeArray[Self.T]

    var _name: String
    var _help: String
    var _default: Optional[List[UInt8]]

    def __init__(
        out self,
        var name: String,
        var help: String = String(),
        var default: Optional[List[UInt8]] = None,
    ):
        self._name = name^
        self._help = help^
        self._default = default^

    # -- Value --------------------------------------------------------------

    def references(self, mut into: References):
        into.param(
            ParamSpec(
                self._name.copy(),
                DynType(Self.T()),
                self._help.copy(),
                _shown(self._default),
                None,
            )
        )

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self.bind(batch, bindings).to_dyn()

    # -- BinaryValue --------------------------------------------------------

    def bind(
        self, batch: StructArray, bindings: Bindings
    ) raises -> BinaryLikeArray[Self.Type]:
        var got = _bound[BinaryLikeScalar[Self.T]](
            bindings, self._name, DynType(Self.T())
        )
        if got:
            return _repeat_bytes[Self.T](
                got.value().to_string().as_bytes(), len(batch)
            )
        elif self._default:
            return _repeat_bytes[Self.T](
                Span(self._default.value()), len(batch)
            )
        else:
            raise _unbound(self._name, self._help)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("param(", self._name, ")")
