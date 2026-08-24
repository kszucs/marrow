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
from ...arrays import Int32Array
from ...dtypes import Int32Type
from ...kernels.nested import ArrayLengthKernel
from ...schema import Schema
from ...tabular import RecordBatch
from ..logical import Shape
from ..params import Bindings
from ..physical import Datum
from .core import (
    BoolValue,
    ListValue,
    NumericValue,
    StringValue,
    TemporalValue,
)


struct Column[T: NumericType](NumericValue):
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

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        # A leaf returns its column as-is, validity included; the fused loop
        # above it decides what nulls mean.
        return batch.column(self._name).copy()

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        # `RecordBatch.column(name)` owns the missing-name diagnostic:
        # `get_field_index` answers -1, and indexing a column list with that
        # trips a bounds assert that aborts the process instead of naming the
        # column. Every leaf goes through it for that reason.
        return batch.column(self._name).as_primitive[Self.T]().copy()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # The bound column already carries it — no second lookup, no re-read.
        return bound.to_data().owned_validity()

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct TemporalColumn[T: TemporalType](TemporalValue):
    """A date/time/timestamp/duration column, resolved by name once per batch.

    **Byte-for-byte the same lane as `Column[T]`** — temporal dtypes are
    fixed-width signed integers underneath, so `bind` and `lane[W]` are
    identical. It is a separate struct only because Mojo has no conditional
    conformance: one leaf cannot be a `NumericValue` when `T` is `int64` and a
    `TemporalValue` when `T` is `date32`, and the difference matters because
    `date + date` must not compile.

    That is the whole duplication, and the point of the split is that it stops
    at the leaf: everything above binds on `PrimitiveValue`, where `expr/`
    needs `TemporalColumn` *plus* duplicated comparison arms *plus*
    `TemporalMinMax`.

    **What works today: projection and grouping.** Comparison does not — the
    node exists but is still bound on `NumericValue`. The blocker is named
    rather than hidden: `NumericCompare.ArgType` is
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
        """
        return schema.fields[schema.get_field_index(self._name)].dtype.copy()

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        return batch.column(self._name).copy()

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        return batch.column(self._name).as_primitive[Self.T]().copy()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return bound.to_data().owned_validity()

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

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        # Stays a scalar. `Shape == 0` tells the caller so, and `Datum.to_array`
        # is the one place it stops being lazy — a predicate over a constant
        # never allocates a column.
        return PrimitiveScalar[Self.T](self._value).to_dyn()

    # -- ComptimeValue ------------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        return NoneType()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # A constant is never null.
        return None

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return SIMD[Self.Type.native, W](self._value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("lit(", self._value, ")")


struct BoolColumn(BoolValue):
    """A boolean column, resolved by name once per batch.

    Separate from `Column[T]` because booleans are **bit-packed**: the `Bound`
    is a `BoolArray` and the lane loads through `values()`, the offset-applied
    `BitmapView`, rather than through a typed buffer. `Column[T]` is bound on
    `NumericType` and cannot take `BoolType` — the same reason `PrimitiveArray[bool_]`
    is not a thing anywhere in the tree.

    Without this leaf a fused expression could not read a `bool` column at all,
    so any three-valued-logic test would have to synthesise its operands from
    comparisons. `expr/` shipped without it for exactly that reason and had to
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

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        # As with `Column[T]`: hand back the column rather than re-packing an
        # identical bitmap through the fused driver.
        return batch.column(self._name).copy()

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        return batch.column(self._name).as_bool().copy()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return bound.to_data().owned_validity()

    @always_inline
    def lane[W: Int](self, bound: Self.Bound, idx: Int) -> SIMD[DType.bool, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._name)


struct StringColumn[T: StringLikeType](StringValue):
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

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        # Hand back the column rather than copying every byte through a
        # builder — the whole reason the trait default is overridable.
        return batch.column(self._name).copy()

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        return batch.column(self._name).as_type[Self.Bound]().copy()

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return bound.to_data().owned_validity()

    @always_inline
    def lane(self, bound: Self.Bound, idx: Int) -> String:
        return String(bound.unsafe_get(UInt(idx)))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct StringLiteral[T: StringLikeType](StringValue):
    """A constant string. Stays `Shape.scalar`, so it never materialises
    unless something asks it to."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = Bool

    var _value: String

    def __init__(out self, var value: String):
        self._value = value^

    # -- Value --------------------------------------------------------------

    def columns(self) -> List[String]:
        return List[String]()

    def name(self) -> String:
        return String("")

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        return Datum(StringScalar(self._value.copy()).to_dyn())

    # -- StringValue --------------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        # Nothing to resolve: a constant reads no column.
        return False

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        return None

    @always_inline
    def lane(self, bound: Self.Bound, idx: Int) -> String:
        return self._value.copy()

    def write_to[W: Writer](self, mut writer: W):
        writer.write('"', self._value, '"')


struct ListColumn[T: ListLikeType](ListValue):
    """A list column, resolved by name once per batch.

    Parameterised on `ListLikeType`, so `list`, `large_list` and `map` are the
    same leaf with a different offset width rather than three node types.

    It has no `lane` because `ListValue` has none — a list element is a whole
    sub-array. What reads it are nodes of other families: `ListLength` below is
    a `NumericValue` over this leaf's bound column.
    """

    comptime Type = Self.T
    comptime shape = Shape.columnar

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
        return schema.fields[schema.get_field_index(self._name)].dtype.copy()

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        return batch.column(self._name).copy()

    # -- ListValue ----------------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> ListLikeArray[Self.Type]:
        return batch.column(self._name).as_type[ListLikeArray[Self.T]]().copy()

    def validity(
        self, bound: ListLikeArray[Self.Type]
    ) raises -> Optional[Bitmap[mut=False]]:
        return bound.to_data().owned_validity()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col(", self._name, ")")


struct ListLength[A: ListValue](NumericValue):
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

    def name(self) -> String:
        return String()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Int32Type())

    def resolve(self, bindings: Bindings) raises -> Self:
        return Self(self.a.resolve(bindings))

    # -- PrimitiveValue -----------------------------------------------------

    def bind(self, batch: RecordBatch) raises -> Self.Bound:
        return ArrayLengthKernel.apply(self.a.bind(batch))

    def validity(self, bound: Self.Bound) raises -> Optional[Bitmap[mut=False]]:
        # A null list has no length, so validity is the list's own.
        return bound.to_data().owned_validity()

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("array_length(", self.a, ")")
