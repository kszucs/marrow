# Copyright 2024 Szűcs Krisztián
# SPDX-License-Identifier: Apache-2.0

"""Subqueries in the comptime lane: a column of the enclosing query, and the
value of a scalar subquery.

Both are leaves. A subquery is planned around the node that reads it —
`Decorrelate` adds a join, an aggregate or a binding — so what a leaf here does
at run time is read: `outer(name, dtype)` the enclosing query's column from the
row a subquery was joined to, `scalar(rel, dtype)` its slot, as a binding or a
column. Nothing is erased: each leaf is typed on its family, so a predicate
reading one fuses like one reading a column.

Existence and membership tests are not here. They become semi and anti joins
and are never evaluated, so `exists(rel)` and `is_in(value, rel)` build the
runtime lane's node in either lane; see `builders.mojo`.
"""

from ...arrays import BinaryLikeArray, PrimitiveArray, StructArray
from ...builders import nulls
from ...dtypes import (
    DecimalType,
    DynType,
    NumericType,
    PrimitiveType,
    StringLikeType,
    TemporalType,
)
from ...errors import TypeError
from ...scalars import BinaryLikeScalar, PrimitiveScalar
from ...schema import Schema
from ..index import Index
from ..logical import (
    Bindings,
    DynRelation,
    References,
    Shape,
    Subquery,
    SubqueryKind,
    Value,
)
from ..physical import Datum
from .core import (
    ColumnBound,
    DecimalValue,
    NumericValue,
    StringValue,
    TemporalValue,
    Unnamed,
)


# ---------------------------------------------------------------------------
# equality between two bare reads
# ---------------------------------------------------------------------------
def _bare_read[V: Value](value: V) -> Bool:
    """Whether `value` merely reads one column of its own query or of the
    enclosing one: it names exactly the one column it reads, and declares no
    parameter or subquery."""
    var refs = References()
    value.references(refs)
    if len(refs.params) != 0 or len(refs.subqueries) != 0:
        return False
    var name = value.name()
    if len(refs.columns) == 1 and len(refs.outer_columns) == 0:
        return refs.columns[0] == name
    if len(refs.columns) == 0 and len(refs.outer_columns) == 1:
        return refs.outer_columns[0] == name
    return False


def bare_reads[L: Value, R: Value](l: L, r: R) -> Bool:
    """Whether both operands are bare reads — what `Value.equates` asks of an
    equality, which a comparison node answers with this."""
    return _bare_read(l) and _bare_read(r)


# ---------------------------------------------------------------------------
# outer — a column of the enclosing query
# ---------------------------------------------------------------------------
# Read from `Subquery.outer_field(name)`, the field a correlated predicate that
# is not a join key finds the enclosing row under. An equality with an inner
# column never runs: it becomes a join key. Pruning learns nothing from one,
# so `statistics` and `defined` keep their defaults, which keep every chunk.


def _outer_column[
    T: PrimitiveType
](batch: StructArray, name: String) raises -> PrimitiveArray[T]:
    """The enclosing query's column `name`, where a subquery's row was joined
    to it."""
    return batch.field(Subquery.outer_field(name)).as_primitive[T]().copy()


struct OuterNumericColumn[T: NumericType](ColumnBound, NumericValue):
    """A numeric column of the query enclosing this subquery."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def references(self, mut into: References):
        into.outer(self._name)

    def name(self) -> String:
        return self._name.copy()

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(Subquery.outer_field(self._name)).copy()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _outer_column[Self.T](batch, self._name)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("outer(", self._name, ")")


struct OuterTemporalColumn[T: TemporalType](ColumnBound, TemporalValue):
    """A date/time/timestamp/duration column of the enclosing query. Carries
    its dtype, as `TemporalLiteral` does: the unit is a value."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String
    var _dtype: Self.T

    def __init__(out self, var name: String, dtype: Self.T):
        self._name = name^
        self._dtype = dtype.copy()

    def references(self, mut into: References):
        into.outer(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(Subquery.outer_field(self._name)).copy()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _outer_column[Self.T](batch, self._name)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("outer(", self._name, ")")


struct OuterDecimalColumn[T: DecimalType](ColumnBound, DecimalValue):
    """A decimal column of the enclosing query. Carries its dtype: precision
    and scale are values."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String
    var _dtype: Self.T

    def __init__(out self, var name: String, dtype: Self.T):
        self._name = name^
        self._dtype = dtype.copy()

    def references(self, mut into: References):
        into.outer(self._name)

    def name(self) -> String:
        return self._name.copy()

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(Subquery.outer_field(self._name)).copy()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _outer_column[Self.T](batch, self._name)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("outer(", self._name, ")")


struct OuterStringColumn[T: StringLikeType](ColumnBound, StringValue):
    """A string column of the enclosing query."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = BinaryLikeArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    def references(self, mut into: References):
        into.outer(self._name)

    def name(self) -> String:
        return self._name.copy()

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return batch.field(Subquery.outer_field(self._name)).copy()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return (
            batch.field(Subquery.outer_field(self._name))
            .get[Self.Bound]()
            .copy()
        )

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        # A field borrow outlives nothing the struct does not; see
        # `StringColumn.lane`.
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("outer(", self._name, ")")


# ---------------------------------------------------------------------------
# scalar — the one value of a subquery
# ---------------------------------------------------------------------------
# The value is a read of the subquery's slot, exactly as `Subquery.read`: the
# binding an uncorrelated subquery was computed into, broadcast like a
# parameter, else the column a correlated one was joined in as. The declared
# dtype is checked against what the slot holds, on the typed member or the
# batch's schema, never through an erased dispatch.


def _mismatch(sq: Subquery, declared: DynType) -> TypeError:
    return TypeError(
        t"subquery: {sq} is declared {declared} but answers another type"
    )


def _primitive_slot[
    T: PrimitiveType
](
    sq: Subquery, batch: StructArray, bindings: Bindings, dtype: T
) raises -> PrimitiveArray[T]:
    """`sq`'s value over `batch` as a `T` column."""
    var declared = DynType(dtype)
    var got = bindings.get(sq.slot())
    if got:
        ref value = got.value()
        if not value.isa[PrimitiveScalar[T]]():
            raise _mismatch(sq, declared)
        ref typed = value.get[PrimitiveScalar[T]]()
        if typed.type() != declared:
            raise _mismatch(sq, declared)
        return typed.repeat(len(batch))
    var i = sq.column_in(batch)
    if i == -1:
        raise sq.not_planned()
    if batch.dtype.as_struct().fields[i].dtype != declared:
        raise _mismatch(sq, declared)
    return batch.field(i).as_primitive[T]().copy()


struct NumericScalarSubquery[T: NumericType](
    ColumnBound, NumericValue, Unnamed
):
    """`(SELECT ...)` producing a numeric value.

    Prunes like a parameter once its binding is in hand: an uncorrelated
    subquery is computed before the scan below it reads a row group, so
    `col("a", int64) > scalar(rel, int64)` skips the groups a constant would.
    """

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _sq: Subquery

    def __init__(out self, var relation: DynRelation):
        self._sq = Subquery(SubqueryKind.scalar, relation^)

    def references(self, mut into: References):
        self._sq.references(into)

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self.bind(batch, bindings).to_dyn()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _primitive_slot[Self.T](self._sq, batch, bindings, Self.T())

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def statistics[
        Stat: NumericType
    ](
        self, index: Index, bindings: Bindings, upper: Bool
    ) raises -> PrimitiveArray[Stat]:
        """The binding in every chunk, as `NumericParam` answers; null —
        nothing proven — while the value is a per-row column or a null."""
        var got = bindings.get(self._sq.slot())
        if got and got.value().isa[PrimitiveScalar[Self.T]]():
            ref typed = got.value().get[PrimitiveScalar[Self.T]]()
            if typed.is_valid():
                return PrimitiveScalar[Stat](
                    Scalar[Stat.native](typed.value())
                ).repeat(index.chunks())
        return nulls[Stat](index.chunks(), Stat())

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._sq)


struct TemporalScalarSubquery[T: TemporalType](
    ColumnBound, TemporalValue, Unnamed
):
    """`(SELECT ...)` producing a date/time/timestamp/duration. Carries its
    dtype, so a binding of another unit is refused rather than misread."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _sq: Subquery
    var _dtype: Self.T

    def __init__(out self, var relation: DynRelation, dtype: Self.T):
        self._sq = Subquery(SubqueryKind.scalar, relation^)
        self._dtype = dtype.copy()

    def references(self, mut into: References):
        self._sq.references(into)

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self.bind(batch, bindings).to_dyn()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _primitive_slot[Self.T](self._sq, batch, bindings, self._dtype)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._sq)


struct DecimalScalarSubquery[T: DecimalType](
    ColumnBound, DecimalValue, Unnamed
):
    """`(SELECT ...)` producing a decimal. Carries its dtype: a binding must
    match its precision and scale."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _sq: Subquery
    var _dtype: Self.T

    def __init__(out self, var relation: DynRelation, dtype: Self.T):
        self._sq = Subquery(SubqueryKind.scalar, relation^)
        self._dtype = dtype.copy()

    def references(self, mut into: References):
        self._sq.references(into)

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(self._dtype)

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self.bind(batch, bindings).to_dyn()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        return _primitive_slot[Self.T](self._sq, batch, bindings, self._dtype)

    @always_inline
    def lane[
        W: Int
    ](self, bound: Self.Bound, idx: Int) -> SIMD[Self.Type.native, W]:
        return bound.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._sq)


struct StringScalarSubquery[T: StringLikeType](
    ColumnBound, StringValue, Unnamed
):
    """`(SELECT ...)` producing a string."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = BinaryLikeArray[Self.T]

    var _sq: Subquery

    def __init__(out self, var relation: DynRelation):
        self._sq = Subquery(SubqueryKind.scalar, relation^)

    def references(self, mut into: References):
        self._sq.references(into)

    def evaluate(self, batch: StructArray, bindings: Bindings) raises -> Datum:
        return self.bind(batch, bindings).to_dyn()

    def bind(self, batch: StructArray, bindings: Bindings) raises -> Self.Bound:
        var declared = DynType(Self.T())
        var got = bindings.get(self._sq.slot())
        if got:
            ref value = got.value()
            if not value.isa[BinaryLikeScalar[Self.T]]():
                raise _mismatch(self._sq, declared)
            return value.get[BinaryLikeScalar[Self.T]]().repeat(len(batch))
        var i = self._sq.column_in(batch)
        if i == -1:
            raise self._sq.not_planned()
        if batch.dtype.as_struct().fields[i].dtype != declared:
            raise _mismatch(self._sq, declared)
        return batch.field(i).get[Self.Bound]().copy()

    @always_inline
    def lane(
        self, ref bound: Self.Bound, idx: Int
    ) -> StringSlice[origin_of(bound)]:
        return rebind[StringSlice[origin_of(bound)]](
            bound.unsafe_get(UInt(idx))
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self._sq)
