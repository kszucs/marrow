"""Prototype: typed expression tree with kernel fusion.

Two separate layers:

- **Static layer** (`Value` / `NumericValue` / `BoolValue` / `StringValue`
  traits, `Column` / `Negate` / `Add` structs): each node implements
  `core(idx) -> SIMD[...]` (numeric/bool) or `core(idx) -> StringScalar`.
  `execute()` drives a single fused `vectorize` loop -- the compiler inlines
  the full tree, producing zero intermediate arrays.

- **Runtime layer** (`RuntimeExpr`): lazy tree built at construction time,
  evaluated on `.execute()` via the existing typed/`AnyArray` kernel overloads
  in `marrow/kernels/arithmetic.mojo`. Does NOT implement `Value`; no static
  SIMD type guarantees. This is the Python-facing path.
"""

from std.algorithm.backend.vectorize import vectorize
from std.sys import size_of
from std.sys.info import simd_byte_width
from std.sys.intrinsics import _type_is_eq

import marrow.dtypes as dt
from marrow.arrays import AnyArray, BoolArray, PrimitiveArray, StringArray
from marrow.buffers import Bitmap, Buffer
from marrow.builders import arange
from marrow.dtypes import Int32Type
from marrow.kernels.arithmetic import (
    add,
    neg,
    AddKernel,
    NegKernel,
    SubKernel,
    MulKernel,
)
from marrow.kernels.boolean import AndKernel, OrKernel, NotKernel
from marrow.kernels.compare import (
    EqKernel,
    NeKernel,
    LtKernel,
    LeKernel,
    GtKernel,
    GeKernel,
)
from marrow.kernels.execution import ExecutionContext
from marrow.kernels.filter import filter as filter_kernel
from marrow.tabular import RecordBatch
from marrow.views import BufferView
from std.memory import OwnedPointer
from std.utils.index import IndexList
from std.reflection import reflect


# ===========================================================================
# Static layer -- kernel fusion via comptime core()
# ===========================================================================


trait Value(Copyable, ImplicitlyDestructible, Movable, Writable):
    """Base for statically-typed expression nodes.

    OutType is the broader dt.DataType so that Value can be implemented
    by all data types (numeric, boolean, string).  Numeric-specific and
    boolean-specific execution live on the derived traits.
    """

    comptime OutType: dt.DataType

    def bind(mut self, batch: RecordBatch) raises:
        pass


trait NumericValue(Value):
    """Marker: OutType is numeric. Inherits bind() from Value.

    NativeType is the Mojo scalar type for SIMD operations; OutType is the
    broader dt.DataType used for array construction.  core[W](idx) returns a
    SIMD lane of NativeType; execute() drives a single fused vectorize loop.
    Operator overloads build the expression tree without executing it.
    """

    comptime OutType: dt.NumericType
    comptime NativeType: DType

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[Self.NativeType, W]:
        ...

    def __neg__(self) -> Negate[Self]:
        return Negate(self.copy())

    def negate(self) -> Negate[Self]:
        return Negate(self.copy())

    def __add__[RHS: NumericValue](self, rhs: RHS) -> Add[Self, RHS]:
        return Add(self.copy(), rhs.copy())

    def __sub__[RHS: NumericValue](self, rhs: RHS) -> Sub[Self, RHS]:
        return Sub(self.copy(), rhs.copy())

    def __mul__[RHS: NumericValue](self, rhs: RHS) -> Mul[Self, RHS]:
        return Mul(self.copy(), rhs.copy())

    def __eq__[RHS: NumericValue](self, rhs: RHS) -> Equal[Self, RHS]:
        return Equal(self.copy(), rhs.copy())

    def __ne__[RHS: NumericValue](self, rhs: RHS) -> NotEqual[Self, RHS]:
        return NotEqual(self.copy(), rhs.copy())

    def __lt__[RHS: NumericValue](self, rhs: RHS) -> Less[Self, RHS]:
        return Less(self.copy(), rhs.copy())

    def __le__[RHS: NumericValue](self, rhs: RHS) -> LessEq[Self, RHS]:
        return LessEq(self.copy(), rhs.copy())

    def __gt__[RHS: NumericValue](self, rhs: RHS) -> Greater[Self, RHS]:
        return Greater(self.copy(), rhs.copy())

    def __ge__[RHS: NumericValue](self, rhs: RHS) -> GreaterEq[Self, RHS]:
        return GreaterEq(self.copy(), rhs.copy())

    def execute(self, length: Int) raises -> PrimitiveArray[Self.OutType]:
        comptime width = simd_byte_width() // size_of[Scalar[Self.NativeType]]()
        var buf = Buffer.alloc_uninit[Self.NativeType](length)
        var view = buf.view[Self.NativeType](0, length)

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]) -> None:
            var i = idx[0]
            view.store[W](i, self.core[W](i))

        _vectorize_dispatch[Self.NativeType, width, fill](length)
        return PrimitiveArray[Self.OutType](
            dtype=Self.OutType(),
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=buf.to_immutable(),
        )


trait BoolValue(Copyable, ImplicitlyDestructible, Movable, Writable):
    """Base for bool-output expression nodes (comparisons).

    InNative carries the numeric dtype of the leaves so execute() can choose
    the correct SIMD width.  core[W](idx) returns W booleans that
    execute() bit-packs into a BoolArray via BitmapView.store[W].
    """

    comptime InNative: DType

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        ...

    def bind(mut self, batch: RecordBatch) raises:
        pass

    def __and__[RHS: BoolValue](self, rhs: RHS) -> And[Self, RHS]:
        return And(self.copy(), rhs.copy())

    def __or__[RHS: BoolValue](self, rhs: RHS) -> Or[Self, RHS]:
        return Or(self.copy(), rhs.copy())

    def __invert__(self) -> Not[Self]:
        return Not(self.copy())

    def execute(self, length: Int) raises -> BoolArray:
        comptime width = max(
            8, simd_byte_width() // size_of[Scalar[Self.InNative]]()
        )
        var bm = Bitmap.alloc_uninit(length)
        var view = bm.view()

        @parameter
        @always_inline
        def fill[
            W: Int, rank: Int, alignment: Int = 1
        ](idx: IndexList[rank]) -> None:
            var i = idx[0]
            view.store[W](i, self.core[W](i))

        _vectorize_dispatch[Self.InNative, width, fill](length)
        return BoolArray(
            length=length,
            nulls=0,
            offset=0,
            bitmap=None,
            buffer=bm.to_immutable(),
        )


# ---------------------------------------------------------------------------
# Leaf value nodes -- Column (owned) and ColumnRef (deferred by name)
# ---------------------------------------------------------------------------


struct Column[T: dt.NumericType](NumericValue, Value):
    """Owned column -- data provided at construction time.

    Use when you have an actual array and want to build a composite typed
    SIMD tree without deferred binding overhead.
    """

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var arr: PrimitiveArray[Self.T]

    def __init__(out self, var arr: PrimitiveArray[Self.T]):
        self.arr = arr^

    def __init__(out self, *, copy: Self):
        self.arr = copy.arr.copy()

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[Self.NativeType, W]:
        return self.arr.values().load[W](idx)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Col[", len(self.arr), "]")


struct ColumnRef[name: StaticString, T: dt.NumericType](NumericValue, Value):
    """Named column placeholder resolved from a RecordBatch at execute time.

    `name` and `T` are compile-time constants, so each distinct (name, T) pair
    is a unique type -- full AOT specialization and DCE are preserved.
    """

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var _arr: Optional[PrimitiveArray[Self.T]]

    def __init__(out self):
        self._arr = None

    def __init__(out self, *, copy: Self):
        if copy._arr:
            self._arr = copy._arr.unsafe_value().copy()
        else:
            self._arr = None

    def bind(mut self, batch: RecordBatch) raises:
        self._arr = batch.column(Self.name).as_primitive[Self.T]().copy()

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[Self.NativeType, W]:
        return self._arr.unsafe_value().values().load[W](idx)

    def col_name(self) -> String:
        """Column name."""
        return Self.name

    def col_dtype(self) -> Self.T:
        """Runtime DataType instance for this column."""
        return Self.T()

    def write_to[W: Writer](self, mut writer: W):
        writer.write("col['", Self.name, "', ", Self.T(), "]")


@always_inline
def lit[T: dt.NumericType](value: Scalar[T.native], dtype: T) -> Literal[T]:
    """Create a scalar constant for use in AOT expression trees.

    Usage: ``lit(Int32(5), dt.int32)`` or ``lit[dt.int32](5)``.
    Returns a ``Literal[T]`` broadcast to all SIMD lanes.
    """
    return Literal[T](value)


@always_inline
def lit[T: dt.NumericType](dtype: T, value: Int) -> Literal[T]:
    """Create a scalar constant from an ``Int`` literal.

    Usage: ``lit(dt.int32, 5)`` -- more ergonomic when the type is the lead arg.
    """
    return Literal[T](Scalar[T.native](value))


@always_inline
def col[name: StaticString, T: dt.NumericType](dtype: T) -> ColumnRef[name, T]:
    """Create a named column placeholder for use in AOT expression trees.

    Usage: ``col['price'](dt.float32)``
    The return type is ``ColumnRef['price', Float32Type]`` -- a unique
    compile-time type that preserves full AOT specialization.
    """
    return ColumnRef[name, T]()


struct Negate[T: NumericValue](NumericValue):
    comptime OutType = Self.T.OutType
    comptime NativeType = Self.T.NativeType

    var arg: Self.T

    def __init__(out self, var arg: Self.T):
        self.arg = arg^

    def __init__(out self, *, copy: Self):
        self.arg = copy.arg.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.arg.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[Self.NativeType, W]:
        return NegKernel.core[Self.NativeType, W](self.arg.core[W](idx))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Negate(", self.arg, ")")


struct Add[L: NumericValue, R: NumericValue](NumericValue):
    comptime OutType = Self.L.OutType
    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Add requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](idx)
        var r = self.right.core[W](idx).cast[Self.NativeType]()
        return AddKernel.core[Self.NativeType, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Add(", self.left, ", ", self.right, ")")


struct Literal[T: dt.NumericType](NumericValue):
    """Scalar constant broadcast to all SIMD lanes."""

    comptime OutType = Self.T
    comptime NativeType = Self.T.native

    var value: Scalar[Self.NativeType]

    def __init__(out self, value: Scalar[Self.NativeType]):
        self.value = value

    def __init__(out self, *, copy: Self):
        self.value = copy.value

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[Self.NativeType, W]:
        return SIMD[Self.NativeType, W](self.value)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Lit[", self.value, "]")


struct Sub[L: NumericValue, R: NumericValue](NumericValue):
    comptime OutType = Self.L.OutType
    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Sub requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](idx)
        var r = self.right.core[W](idx).cast[Self.NativeType]()
        return SubKernel.core[Self.NativeType, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Sub(", self.left, ", ", self.right, ")")


struct Mul[L: NumericValue, R: NumericValue](NumericValue):
    comptime OutType = Self.L.OutType
    comptime NativeType = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Mul requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[Self.NativeType, W]:
        var l = self.left.core[W](idx)
        var r = self.right.core[W](idx).cast[Self.NativeType]()
        return MulKernel.core[Self.NativeType, W](l, r)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Mul(", self.left, ", ", self.right, ")")


# ---------------------------------------------------------------------------
# Comparison expression nodes -- BoolValue
# ---------------------------------------------------------------------------


struct Equal[L: NumericValue, R: NumericValue](BoolValue):
    comptime InNative = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Equal requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return EqKernel.core[Self.L.NativeType, W](
            self.left.core[W](idx),
            self.right.core[W](idx).cast[Self.L.NativeType](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " == ", self.right, ")")


struct NotEqual[L: NumericValue, R: NumericValue](BoolValue):
    comptime InNative = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "NotEqual requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return NeKernel.core[Self.L.NativeType, W](
            self.left.core[W](idx),
            self.right.core[W](idx).cast[Self.L.NativeType](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " != ", self.right, ")")


struct Less[L: NumericValue, R: NumericValue](BoolValue):
    comptime InNative = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Less requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return LtKernel.core[Self.L.NativeType, W](
            self.left.core[W](idx),
            self.right.core[W](idx).cast[Self.L.NativeType](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " < ", self.right, ")")


struct LessEq[L: NumericValue, R: NumericValue](BoolValue):
    comptime InNative = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "LessEq requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return LeKernel.core[Self.L.NativeType, W](
            self.left.core[W](idx),
            self.right.core[W](idx).cast[Self.L.NativeType](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " <= ", self.right, ")")


struct Greater[L: NumericValue, R: NumericValue](BoolValue):
    comptime InNative = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "Greater requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return GtKernel.core[Self.L.NativeType, W](
            self.left.core[W](idx),
            self.right.core[W](idx).cast[Self.L.NativeType](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " > ", self.right, ")")


struct GreaterEq[L: NumericValue, R: NumericValue](BoolValue):
    comptime InNative = Self.L.NativeType

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        comptime assert _type_is_eq[
            Self.L.OutType, Self.R.OutType
        ](), "GreaterEq requires matching output types"
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return GeKernel.core[Self.L.NativeType, W](
            self.left.core[W](idx),
            self.right.core[W](idx).cast[Self.L.NativeType](),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " >= ", self.right, ")")


# ---------------------------------------------------------------------------
# Boolean combinator nodes -- BoolValue x BoolValue -> BoolValue
# ---------------------------------------------------------------------------


struct And[L: BoolValue, R: BoolValue](BoolValue):
    comptime InNative = Self.L.InNative

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return AndKernel.core[W](
            self.left.core[W](idx),
            self.right.core[W](idx),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " AND ", self.right, ")")


struct Or[L: BoolValue, R: BoolValue](BoolValue):
    comptime InNative = Self.L.InNative

    var left: Self.L
    var right: Self.R

    def __init__(out self, var left: Self.L, var right: Self.R):
        self.left = left^
        self.right = right^

    def __init__(out self, *, copy: Self):
        self.left = copy.left.copy()
        self.right = copy.right.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.left.bind(batch)
        self.right.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return OrKernel.core[W](
            self.left.core[W](idx),
            self.right.core[W](idx),
        )

    def write_to[W: Writer](self, mut writer: W):
        writer.write("(", self.left, " OR ", self.right, ")")


struct Not[E: BoolValue](BoolValue):
    comptime InNative = Self.E.InNative

    var expr: Self.E

    def __init__(out self, var expr: Self.E):
        self.expr = expr^

    def __init__(out self, *, copy: Self):
        self.expr = copy.expr.copy()

    def bind(mut self, batch: RecordBatch) raises:
        self.expr.bind(batch)

    @always_inline
    def core[W: Int](self, idx: Int) -> SIMD[DType.bool, W]:
        return NotKernel.core[W](self.expr.core[W](idx))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("NOT(", self.expr, ")")


# ---------------------------------------------------------------------------
# StringValue trait and string leaf nodes
# ---------------------------------------------------------------------------


trait StringValue(Copyable, ImplicitlyDestructible, Movable, Writable):
    """Base for string expression nodes.

    core(idx) returns the StringScalar at the given index;
    execute() materialises a StringArray.
    """

    comptime OutType = dt.string

    @always_inline
    def core(self, idx: Int) -> StringScalar:
        ...

    def bind(mut self, batch: RecordBatch) raises:
        pass


struct StringColumn(StringValue):
    """Owned string column -- data provided at construction time."""

    var arr: StringArray

    def __init__(out self, var arr: StringArray):
        self.arr = arr^

    def __init__(out self, *, copy: Self):
        self.arr = copy.arr.copy()

    @always_inline
    def core(self, idx: Int) -> StringScalar:
        return self.arr.unsafe_get(UInt(idx))

    def write_to[W: Writer](self, mut writer: W):
        writer.write("StrCol[", len(self.arr), "]")


# ---------------------------------------------------------------------------
# Relation -- base trait for compile-time relational nodes
# ---------------------------------------------------------------------------


trait Relation(Copyable, ImplicitlyDestructible, Movable, Writable):
    """Base for compile-time relational nodes that consume a RecordBatch."""

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        ...


# ---------------------------------------------------------------------------
# Filter -- multi-column relational filter returning RecordBatch
# ---------------------------------------------------------------------------


struct Filter[
    Pred: BoolValue,
    *Fields: FieldDescriptor,
](Relation):
    """Relational filter: evaluate Pred, apply selection mask to all batch columns.

    *Fields defines the schema made available to the predicate (via ColumnRef
    bindings). All columns of the input RecordBatch are filtered and returned.

    Usage::

        var t = Schema[Field['a', Int32Type], Field['b', Int32Type], Field['data', Int32Type]]()
        var result: RecordBatch = t.filter(t.a + t.b > lit(int32, 5)).execute(batch)
        var col_data = result.column("data")
    """

    var _pred: Self.Pred

    def __init__(out self, var pred: Self.Pred):
        self._pred = pred^

    def __init__(out self, *, copy: Self):
        self._pred = copy._pred.copy()

    def execute(self, batch: RecordBatch) raises -> RecordBatch:
        """Execute the filter, binding a fresh copy of the predicate each call.

        Works on both rvalues (``t.filter(pred).execute(batch)``) and stored
        vars (``rel.execute(batch1); rel.execute(batch2)``).
        """
        var pred = self._pred.copy()
        pred.bind(batch)
        var sel: AnyArray = pred.execute(batch.num_rows())
        var out_cols = List[AnyArray]()
        for i in range(batch.num_columns()):
            out_cols.append(filter_kernel(batch.column(i), sel.copy()))
        return RecordBatch(schema=batch.schema, columns=out_cols^)

    def write_to[W: Writer](self, mut writer: W):
        writer.write("Filter WHERE ", self._pred)


# ---------------------------------------------------------------------------
# Schema -- Field type tags + __getattr__ for t.col_name syntax
# ---------------------------------------------------------------------------


trait FieldDescriptor:
    """Compile-time trait for schema field descriptors."""

    comptime dtype: dt.NumericType

    @staticmethod
    def _name_matches[other: StaticString]() -> Bool:
        ...


struct Field[name: StaticString, T: dt.NumericType](
    Copyable, FieldDescriptor, Movable
):
    """Compile-time schema field descriptor. Carries no runtime data.

    Pass as a type parameter or as a value with dtype inference::

        Schema[Field['price', Float32Type], Field['qty', Int32Type]]()  # explicit
        Schema(Field['price'](float32), Field['qty'](int32))             # inferred T
    """

    comptime dtype = Self.T

    def __init__(out self, _dtype: Self.T):
        """Construct with dtype value; T is inferred, name must be in brackets.

        Enables ``Field['price'](float32)`` -> ``Field['price', Float32Type]``.
        """
        pass

    @staticmethod
    def _name_matches[other: StaticString]() -> Bool:
        return Self.name == other


@always_inline
def field[name: StaticString, T: dt.NumericType](dtype: T) -> Field[name, T]:
    """Convenience shorthand for ``Field["name"](dtype)``.

    Usage: ``field["a"](int32)`` -- same as ``Field["a"](int32)`` but shorter.
    """
    return Field[name](dtype)


def _schema_find_idx[
    name: StaticString,
    *Fields: FieldDescriptor,
    start: Int = 0,
]() -> Int:
    """Compile-time recursion: return the index of the field named ``name``."""
    comptime if start >= Fields.size:
        comptime assert False, "Schema has no field named '" + name + "'"
        return -1
    comptime if Fields[start]._name_matches[name]():
        return start
    return _schema_find_idx[name, *Fields, start=start + 1]()


struct Schema[*Fields: FieldDescriptor](Copyable, Movable):
    """Schema parameterised by ``Field[name, T]`` type tags; no runtime fields.

    Every attribute access (``t.col_name``) goes through ``__getattr_param__``,
    which returns a ``ColumnRef[name, T]`` for the matching field.

    Usage::

        var t = Schema[Field['a', Int32Type], Field['data', Int32Type]]()
        var t = Schema(Field("a", int32), Field("data", int32))  # inferred
        var result = t.data.where(t.a + t.b > lit(int32, 5)).execute(batch)
    """

    def __init__(out self):
        pass

    def __init__(out self, *fields: *Self.Fields):
        """Infer *Fields from Field value arguments.

        Enables ``Schema(Field("a", int32), Field("b", float32))`` syntax.
        """
        pass

    def __init__(out self, *, copy: Self):
        pass

    @always_inline
    def __getattr_param__[
        name: StaticString,
        idx: Int = _schema_find_idx[name, *Self.Fields](),
    ](self) -> ColumnRef[name, Self.Fields[idx].dtype]:
        """Return ``ColumnRef[name, T]`` for the matching field in the schema.
        """
        return ColumnRef[name, Self.Fields[idx].dtype]()

    def filter[Pred: BoolValue](self, pred: Pred) -> Filter[Pred, *Self.Fields]:
        """Build a Filter over all fields in this schema.

        Usage::

            var t = Schema[Field['a', Int32Type], Field['data', Int32Type]]()
            var result: RecordBatch = t.filter(t.a > lit(int32, 5)).execute(batch)
        """
        return Filter[Pred, *Self.Fields](pred.copy())


struct Table[T: AnyType](Copyable, Movable):
    """Schema parameterised by a struct type; fields discovered via Mojo reflection.

    Every attribute access (``t.col_name``) goes through ``__getattr_param__``,
    which returns a deferred field descriptor resolved from the struct's
    field type via ``reflect[Self.T].field_type[name]``.

    Note: Mojo's reflection system returns an unconstrained type that cannot
    be proven to satisfy ``NumericType`` at compile time, so this struct is
    a placeholder for future use.  For now, use ``Schema[Field...]`` which
    provides full AOT expression support.

    Usage::

        struct MySchema:
            var a: Int32Type
            var b: Float64Type

        var t = Table[MySchema]()  # placeholder -- use Schema instead
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass


@always_inline
def table[*Fields: FieldDescriptor](*fields: *Fields) -> Schema[*Fields]:
    """Convenience alias for Schema().

    Usage: ``table(field["a"](int32), field["b"](int32))``
    """
    return Schema[*Fields](copy=Schema[*Fields]())


def _vectorize_dispatch[
    native: DType,
    cpu_width: Int,
    process: def[W: Int, rank: Int, alignment: Int = 1](
        IndexList[rank]
    ) capturing -> None,
](length: Int):
    """Run process over [0, length) using vectorize. Mirrors _apply_dispatch."""

    @always_inline
    def lane[W: Int](i: Int):
        process[W, rank=1](IndexList[1](i))

    vectorize[cpu_width](length, lane)


# ===========================================================================
# Runtime layer -- lazy tree, NOT implementing Value
# ===========================================================================

# sizeof[RuntimeExpr] = sizeof(OwnedPointer) = pointer width, so
# List[RuntimeExpr] inside _NodeContent has a known element size and the
# recursive size dependency resolves to a finite constant.

comptime _RT_COLUMN = 0
comptime _RT_NEGATE = 1
comptime _RT_ADD = 2


struct _NodeContent(Copyable, ImplicitlyDestructible, Movable):
    var _kind: Int
    var _name: String
    var _args: List[RuntimeExpr]

    def __init__(
        out self,
        kind: Int,
        name: String,
        var args: List[RuntimeExpr],
    ):
        self._kind = kind
        self._name = name
        self._args = args^

    def __init__(out self, *, copy: Self):
        self._kind = copy._kind
        self._name = copy._name
        self._args = List[RuntimeExpr]()
        for i in range(len(copy._args)):
            self._args.append(RuntimeExpr(copy=copy._args[i]))


struct RuntimeExpr(Copyable, ImplicitlyDestructible, Movable):
    """Lazy runtime expression tree. Call execute() to materialise.

    sizeof[RuntimeExpr] = sizeof(OwnedPointer) = pointer width, allowing
    List[RuntimeExpr] children in _NodeContent without infinite size recursion.
    Does NOT implement Value -- no static SIMD type guarantees.
    """

    var _node: OwnedPointer[_NodeContent]

    def __init__(out self, var node: _NodeContent):
        self._node = OwnedPointer[_NodeContent](node^)

    def __init__(out self, *, copy: Self):
        var inner = _NodeContent(copy=copy._node[])
        self._node = OwnedPointer[_NodeContent](inner^)

    @staticmethod
    def column(name: String) -> Self:
        return Self(_NodeContent(_RT_COLUMN, name, List[RuntimeExpr]()))

    def negate(self) raises -> Self:
        var args = List[RuntimeExpr]()
        args.append(self.copy())
        return Self(_NodeContent(_RT_NEGATE, "", args^))

    def add(self, other: Self) raises -> Self:
        var args = List[RuntimeExpr]()
        args.append(self.copy())
        args.append(other.copy())
        return Self(_NodeContent(_RT_ADD, "", args^))

    def execute(self, data: Dict[String, AnyArray]) raises -> AnyArray:
        if self._node[]._kind == _RT_COLUMN:
            return data[self._node[]._name].copy()
        elif self._node[]._kind == _RT_NEGATE:
            return neg(self._node[]._args[0].execute(data))
        else:
            return add(
                self._node[]._args[0].execute(data),
                self._node[]._args[1].execute(data),
            )


def main() raises:
    var a = arange[Int32Type](1, 9)  # [1, 2, 3, 4, 5, 6, 7, 8]
    var b = arange[Int32Type](10, 18)  # [10, 11, 12, 13, 14, 15, 16, 17]
    var c = arange[Int32Type](1, 9)  # [1, 2, 3, 4, 5, 6, 7, 8]

    # -(a) + b -- numeric fusion (existing)
    var num_expr = Add(Negate(Column(a.copy())), Column(b.copy()))
    print(num_expr)  # Add(Negate(Col[8]), Col[8])
    print(execute(num_expr, 8))  # [9, 9, 9, 9, 9, 9, 9, 9]

    # (a + b) == (c + 1) -- fused: 3 ops, 1 pass, 0 intermediate arrays
    var eq_expr = Equal(
        Add(Column(a.copy()), Column(b.copy())),
        Add(Column(c.copy()), Literal[Int32Type](1)),
    )
    print(eq_expr)
    print(execute(eq_expr, 8))

    # a > 0 AND b < 15 -- compound predicate, 1 fused pass
    var and_expr = And(
        Greater(Column(a.copy()), Literal[Int32Type](0)),
        Less(Column(b.copy()), Literal[Int32Type](15)),
    )
    print(and_expr)  # (Col[8] > Lit[0]) AND (Col[8] < Lit[15])
    print(
        execute(and_expr, 8)
    )  # [true, true, true, true, true, false, false, false]

    # NOT (a > 5)
    var not_expr = Not(Greater(Column(a.copy()), Literal[Int32Type](5)))
    print(not_expr)  # NOT((Col[8] > Lit[5]))
    print(
        execute(not_expr, 8)
    )  # [true, true, true, true, true, false, false, false]

    # Schema-based Filter -- multi-column relational filter (print-only demo)
    var t = Schema[
        Field["a", Int32Type], Field["b", Int32Type], Field["data", Int32Type]
    ]()
    var filter_rel = t.filter(
        Greater(
            Add(col["a"](Int32Type()), col["b"](Int32Type())),
            Literal[Int32Type](15),
        )
    )
    print(filter_rel)  # Filter WHERE (col['a', ...] + col['b', ...] > Lit[15])

    # Runtime path (unchanged)
    var rt_expr = RuntimeExpr.column("a").negate().add(RuntimeExpr.column("b"))
    var data = Dict[String, AnyArray]()
    data["a"] = a^
    data["b"] = b^
    print(rt_expr.execute(data))
