"""The leaves of the comptime lane: a column reference and a constant.

A leaf is where a fused subtree touches the batch, and therefore where `bind`
does its work — every schema lookup and `Variant` unwrap happens here so the
lane loop above does none.
"""

from ...arrays import PrimitiveArray
from ...buffers import Bitmap
from ...dtypes import DynType, NumericType
from ...scalars import PrimitiveScalar
from ...schema import Schema
from ...tabular import RecordBatch
from ..core import Datum, Shape
from .core import NumericValue


struct Column[T: NumericType](NumericValue):
    """A numeric column, resolved by name once per batch."""

    comptime Type = Self.T
    comptime shape = Shape.columnar
    comptime Bound = PrimitiveArray[Self.T]

    var _name: String

    def __init__(out self, var name: String):
        self._name = name^

    # -- Analyzable ---------------------------------------------------------

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
        return Datum(batch.column(self._name).copy())

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


struct Literal[T: NumericType](NumericValue):
    """A numeric constant, splatted into every lane."""

    comptime Type = Self.T
    comptime shape = Shape.scalar
    comptime Bound = NoneType
    """Nothing to resolve — the value is in the node, so the lane splats it."""

    var _value: Scalar[Self.Type.native]

    def __init__(out self, value: Scalar[Self.Type.native]):
        self._value = value

    # -- Analyzable ---------------------------------------------------------

    def columns(self) -> List[String]:
        return List[String]()

    def name(self) -> String:
        # SQL names `SELECT 1` as `1`; so does this.
        return String(self._value)

    def dtype(self, schema: Schema) raises -> DynType:
        return DynType(Self.T())

    # -- Evaluable ----------------------------------------------------------

    def evaluate(self, batch: RecordBatch) raises -> Datum:
        # Stays a scalar. `Shape == 0` tells the caller so, and `into_array`
        # is the one place it stops being lazy — a predicate over a constant
        # never allocates a column.
        return Datum(PrimitiveScalar[Self.T](self._value).to_dyn())

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
